import copy
import math
import random
from typing import List, Optional, Tuple

from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler


# HELPER: Modified gram schmidt orthogonalisation
def _modified_gram_schmidt(vectors: List[List[float]]) -> List[List[float]]:
    """Return a list of orthonormal vectors via Modified Gram-Schmidt."""
    ortho = []
    for v in vectors:
        v = list(v)  # copy
        for u in ortho:
            dot = sum(a * b for a, b in zip(v, u))
            v = [vi - dot * ui for vi, ui in zip(v, u)]
        norm = math.sqrt(sum(x ** 2 for x in v))
        if norm < 1e-12:
            continue  # discard near-zero vector
        ortho.append([x / norm for x in v])
    return ortho


def _generate_seed_set(dimension: int, u: int) -> List[List[float]]:
    """Generate u orthonormal vectors in R^dimension using Modified Gram-Schmidt."""
    candidates = [
        [random.gauss(0, 1) for _ in range(dimension)]
        for _ in range(u)
    ]
    return _modified_gram_schmidt(candidates)


#HELPER: Subspace initialisation, generate one point per particle
def _subspace_point(
    seed_set: List[List[float]],
    center: List[float],
    lower_bounds: List[float],
    upper_bounds: List[float],
    margin_fraction: float = 0.1,
) -> List[float]:
    """
    Generate one position using the subspace initialisation strategy
    (Van Zyl & Engelbrecht, 2015).

    A random linear combination of the seed vectors forms a direction d.
    A scalar t is sampled uniformly along the line p = t*d + c so that
    p stays approximately within the search bounds (with MARGIN allowance).
    """
    dimension = len(center)

    # Direction: random linear combination of seed vectors
    d = [0.0] * dimension
    for b in seed_set:
        c_i = random.random()  # U(0, 1)
        for j in range(dimension):
            d[j] += c_i * b[j]

    # Normalise direction to avoid scale issues
    norm = math.sqrt(sum(x ** 2 for x in d))
    if norm < 1e-12:
        # Degenerate direction — fall back to uniform random
        return [random.uniform(lower_bounds[j], upper_bounds[j])
                for j in range(dimension)]
    d = [x / norm for x in d]

    # Find tmin, tmax: the range of t such that p = t*d + center stays within
    # bounds (+/- margin in each dimension).
    margin = [margin_fraction * (upper_bounds[j] - lower_bounds[j])
              for j in range(dimension)]

    t_candidates = []
    for j in range(dimension):
        if abs(d[j]) < 1e-12:
            continue
        t1 = (upper_bounds[j] - center[j]) / d[j]
        t2 = (lower_bounds[j] - center[j]) / d[j]
        # Check that the generated point is within-bounds-with-margin for all dims
        for t_val in (t1, t2):
            p = [t_val * d[k] + center[k] for k in range(dimension)]
            if all(lower_bounds[k] - margin[k] <= p[k] <= upper_bounds[k] + margin[k]
                   for k in range(dimension)):
                t_candidates.append(t_val)

    if len(t_candidates) < 2:
        # Fallback: small perturbation around centre
        return [center[j] + random.uniform(-margin[j], margin[j])
                for j in range(dimension)]

    t_min = min(t_candidates)
    t_max = max(t_candidates)
    t = random.uniform(t_min, t_max)

    point = [t * d[j] + center[j] for j in range(dimension)]
    # Clamp to hard bounds
    point = [max(lower_bounds[j], min(point[j], upper_bounds[j]))
             for j in range(dimension)]
    return point


# 1: Stochastic Scaling PSO (Oldewage et al., 2019)
class StochasticScalingPSO(Solver[List[float], float]):
    """ 
    PSO with Stochastic Scaling and linearly increasing k groups to scale
    For every iteration, the d dimensions are randomly partitioned into k groups. 
    A separate random scaling factor [0,1] is drawn for each group and applied to the velocity update
    for every dimension in that group. This is stochastic damping. It helps particles avoid overshooting
    desirable search regions in high dimensional spaces
    
    The number of k groups increases linearly from 1 to k max. To nearly fine grained (nearly per dimension scaling)
    References:
        Oldewage, E.T., Cleghorn, C.W., & Engelbrecht, A.P. (2019).
        Solving large scale optimisation problems using particle swarm
        optimisation. SSCI 2019.
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        name: str,
        swarm_size: int,
        w: float,
        c1: float,
        c2: float,
        max_iterations: int,
        k_max: Optional[int] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs,
    ):
        """
        Args:
            swarm_size:     Number of particles.
            w:              Inertia weight.
            c1:             Cognitive coefficient.
            c2:             Social coefficient.
            max_iterations: Total iterations (needed to schedule k linearly).
            k_max:          Maximum number of groups. Defaults to dimension.
        """
        super().__init__(problem, name, constraint_handler)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.k_max = k_max if k_max is not None else problem.dimension
        self.iteration = 0

        lower_bounds, upper_bounds = self.problem.bounds

        # Standard uniform random initialisation
        self.population = [
            [random.uniform(lower_bounds[d], upper_bounds[d])
             for d in range(self.problem.dimension)]
            for _ in range(self.swarm_size)
        ]
        self.velocities = [
            [random.uniform(
                -abs(upper_bounds[d] - lower_bounds[d]),
                abs(upper_bounds[d] - lower_bounds[d])
            ) * 0.1 for d in range(self.problem.dimension)]
            for _ in range(self.swarm_size)
        ]

        self.evaluations = [self.problem.evaluate(pos) for pos in self.population]
        self.pbest_positions = copy.deepcopy(self.population)
        self.pbest_evaluations = copy.deepcopy(self.evaluations)

        best_idx = min(range(self.swarm_size),
                       key=lambda i: self.evaluations[i].fitness)
        self.gbest_position = copy.deepcopy(self.population[best_idx])
        self.gbest_evaluation = copy.deepcopy(self.evaluations[best_idx])

    def _current_k(self) -> int:
        """Linearly interpolate k from 1 to k_max over max_iterations."""
        if self.max_iterations <= 1:
            return self.k_max
        t = self.iteration / (self.max_iterations - 1)
        return max(1, round(1 + t * (self.k_max - 1)))

    def _partition_dimensions(self, k: int) -> List[List[int]]:
        """Randomly partition dimensions into k groups."""
        dims = list(range(self.problem.dimension))
        random.shuffle(dims)
        groups = [[] for _ in range(k)]
        for idx, d in enumerate(dims):
            groups[idx % k].append(d)
        return groups

    def step(self) -> None:
        lower_bounds, upper_bounds = self.problem.bounds
        k = self._current_k()
        groups = self._partition_dimensions(k)

        # Build per-dimension scaling map
        scale = [0.0] * self.problem.dimension
        for group in groups:
            s_g = random.random()  # U(0, 1) stochastic scaling factor
            for d in group:
                scale[d] = s_g

        for i in range(self.swarm_size):
            for d in range(self.problem.dimension):
                r1 = random.random()
                r2 = random.random()
                cognitive = self.c1 * r1 * (self.pbest_positions[i][d]
                                            - self.population[i][d])
                social = self.c2 * r2 * (self.gbest_position[d]
                                         - self.population[i][d])
                inertia = self.w * self.velocities[i][d]
                self.velocities[i][d] = scale[d] * (inertia + cognitive + social)

            for d in range(self.problem.dimension):
                self.population[i][d] += self.velocities[i][d]
                self.population[i][d] = max(
                    lower_bounds[d],
                    min(self.population[i][d], upper_bounds[d])
                )

            self.evaluations[i] = self.problem.evaluate(self.population[i])

            if self.comparator.is_better(self.evaluations[i],
                                         self.pbest_evaluations[i]):
                self.pbest_positions[i] = copy.deepcopy(self.population[i])
                self.pbest_evaluations[i] = copy.deepcopy(self.evaluations[i])

                if self.comparator.is_better(self.pbest_evaluations[i],
                                             self.gbest_evaluation):
                    self.gbest_position = copy.deepcopy(self.pbest_positions[i])
                    self.gbest_evaluation = copy.deepcopy(self.pbest_evaluations[i])

        self.iteration += 1

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        return [(self.gbest_position, self.gbest_evaluation)]

    def get_population(self) -> List[List[float]]:
        return self.population

    def get_population_evaluations(self) -> List[Evaluation[float]]:
        return self.evaluations
    
class SubspaceInitPSO(Solver[List[float], float]):
    """ 
    PSO with subspace based intitialisation
    
    We dont initialise particles uniformly at random across the full search space, instead
    each particles intitial position (and personal best) is generated along a random line through the center
    of the search space. The lines direction is a linear combination of u orthonormal seed vectors drawn using modified gram-schmidt.
    This forces the swarm to focus on a small subspace, reducing initial momentum and pushing for fine grained exploitation
    beneficial in high dimensional space.
    
    References:
        Van Zyl, E.T. & Engelbrecht, A.P. (2015). A Subspace-Based Method for
        PSO Initialization. IEEE SSCI 2015.
    
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        name: str,
        swarm_size: int,
        w: float,
        c1: float,
        c2: float,
        seed_set_size: int = 1,
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs,
    ):
        """
        Args:
            swarm_size:    Number of particles.
            w:             Inertia weight.
            c1:            Cognitive coefficient.
            c2:            Social coefficient.
            seed_set_size: Number of orthonormal seed vectors (u).
                           u=1 (default) performs best for very high dimensions.
        """
        super().__init__(problem, name, constraint_handler)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.seed_set_size = seed_set_size

        lower_bounds, upper_bounds = self.problem.bounds
        center = [(lower_bounds[d] + upper_bounds[d]) / 2.0
                  for d in range(self.problem.dimension)]

        # Generate one seed set shared across the swarm
        seed_set = _generate_seed_set(self.problem.dimension, seed_set_size)

        # For each particle, generate two subspace points; assign the better
        # one as pbest and the other as the initial position (as per the paper).
        self.population = []
        self.pbest_positions = []
        self.pbest_evaluations = []
        self.evaluations = []

        for _ in range(self.swarm_size):
            p1 = _subspace_point(seed_set, center, lower_bounds, upper_bounds,
                                 margin_fraction=0.1)
            p2 = _subspace_point(seed_set, center, lower_bounds, upper_bounds,
                                 margin_fraction=0.0)  # pbest must stay in bounds
            e1 = self.problem.evaluate(p1)
            e2 = self.problem.evaluate(p2)

            if e2.fitness < e1.fitness:
                pbest_pos, pbest_eval = p2, e2
                pos, pos_eval = p1, e1
            else:
                pbest_pos, pbest_eval = p1, e1
                pos, pos_eval = p2, e2

            self.pbest_positions.append(pbest_pos)
            self.pbest_evaluations.append(pbest_eval)
            self.population.append(pos)
            self.evaluations.append(pos_eval)

        # Initial velocities set to zero (as per the paper)
        self.velocities = [
            [0.0] * self.problem.dimension
            for _ in range(self.swarm_size)
        ]

        best_idx = min(range(self.swarm_size),
                       key=lambda i: self.pbest_evaluations[i].fitness)
        self.gbest_position = copy.deepcopy(self.pbest_positions[best_idx])
        self.gbest_evaluation = copy.deepcopy(self.pbest_evaluations[best_idx])

    def step(self) -> None:
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(self.swarm_size):
            for d in range(self.problem.dimension):
                r1 = random.random()
                r2 = random.random()
                cognitive = self.c1 * r1 * (self.pbest_positions[i][d]
                                            - self.population[i][d])
                social = self.c2 * r2 * (self.gbest_position[d]
                                         - self.population[i][d])
                inertia = self.w * self.velocities[i][d]
                self.velocities[i][d] = inertia + cognitive + social

            for d in range(self.problem.dimension):
                self.population[i][d] += self.velocities[i][d]
                self.population[i][d] = max(
                    lower_bounds[d],
                    min(self.population[i][d], upper_bounds[d])
                )

            self.evaluations[i] = self.problem.evaluate(self.population[i])

            if self.comparator.is_better(self.evaluations[i],
                                         self.pbest_evaluations[i]):
                self.pbest_positions[i] = copy.deepcopy(self.population[i])
                self.pbest_evaluations[i] = copy.deepcopy(self.evaluations[i])

                if self.comparator.is_better(self.pbest_evaluations[i],
                                             self.gbest_evaluation):
                    self.gbest_position = copy.deepcopy(self.pbest_positions[i])
                    self.gbest_evaluation = copy.deepcopy(self.pbest_evaluations[i])

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        return [(self.gbest_position, self.gbest_evaluation)]

    def get_population(self) -> List[List[float]]:
        return self.population

    def get_population_evaluations(self) -> List[Evaluation[float]]:
        return self.evaluations
    
class HybridPSO(Solver[List[float], float]):
    """
    Hybrid PSO that combines sub space initialisation and stochastic scaling
    
    The solver intitialises the swarm using sub space strategy from Van Zyl and Engelbrecht (2015) 
    and then runs the stochastic scaling velocity from Oldewage et al. (2019)
    
    The hypothesis is this hybrid approach inherits the low momentum start from subspace initialisation
    (enabling finer exploitation) while also benefiting from the stochastic dimension group scaling that 
    prevents teh swarm from stagnating in high dimensions
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        name: str,
        swarm_size: int,
        w: float,
        c1: float,
        c2: float,
        max_iterations: int,
        seed_set_size: int = 1,
        k_max: Optional[int] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs,
    ):
        """
        Args:
            swarm_size:     Number of particles.
            w:              Inertia weight.
            c1:             Cognitive coefficient.
            c2:             Social coefficient.
            max_iterations: Total iterations (for linear k schedule).
            seed_set_size:  Number of orthonormal seed vectors for init (u).
            k_max:          Maximum number of dimension groups. Defaults to D.
        """
        super().__init__(problem, name, constraint_handler)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.seed_set_size = seed_set_size
        self.k_max = k_max if k_max is not None else problem.dimension
        self.iteration = 0

        lower_bounds, upper_bounds = self.problem.bounds
        center = [(lower_bounds[d] + upper_bounds[d]) / 2.0
                  for d in range(self.problem.dimension)]

        # --- Subspace Initialisation ---
        seed_set = _generate_seed_set(self.problem.dimension, seed_set_size)

        self.population = []
        self.pbest_positions = []
        self.pbest_evaluations = []
        self.evaluations = []

        for _ in range(self.swarm_size):
            p1 = _subspace_point(seed_set, center, lower_bounds, upper_bounds,
                                 margin_fraction=0.1)
            p2 = _subspace_point(seed_set, center, lower_bounds, upper_bounds,
                                 margin_fraction=0.0)
            e1 = self.problem.evaluate(p1)
            e2 = self.problem.evaluate(p2)

            if e2.fitness < e1.fitness:
                pbest_pos, pbest_eval = p2, e2
                pos, pos_eval = p1, e1
            else:
                pbest_pos, pbest_eval = p1, e1
                pos, pos_eval = p2, e2

            self.pbest_positions.append(pbest_pos)
            self.pbest_evaluations.append(pbest_eval)
            self.population.append(pos)
            self.evaluations.append(pos_eval)

        # Initial velocities set to zero (as per Van Zyl & Engelbrecht)
        self.velocities = [
            [0.0] * self.problem.dimension
            for _ in range(self.swarm_size)
        ]

        best_idx = min(range(self.swarm_size),
                       key=lambda i: self.pbest_evaluations[i].fitness)
        self.gbest_position = copy.deepcopy(self.pbest_positions[best_idx])
        self.gbest_evaluation = copy.deepcopy(self.pbest_evaluations[best_idx])

    def _current_k(self) -> int:
        if self.max_iterations <= 1:
            return self.k_max
        t = self.iteration / (self.max_iterations - 1)
        return max(1, round(1 + t * (self.k_max - 1)))

    def _partition_dimensions(self, k: int) -> List[List[int]]:
        dims = list(range(self.problem.dimension))
        random.shuffle(dims)
        groups = [[] for _ in range(k)]
        for idx, d in enumerate(dims):
            groups[idx % k].append(d)
        return groups

    def step(self) -> None:
        lower_bounds, upper_bounds = self.problem.bounds
        k = self._current_k()
        groups = self._partition_dimensions(k)

        scale = [0.0] * self.problem.dimension
        for group in groups:
            s_g = random.random()
            for d in group:
                scale[d] = s_g

        for i in range(self.swarm_size):
            for d in range(self.problem.dimension):
                r1 = random.random()
                r2 = random.random()
                cognitive = self.c1 * r1 * (self.pbest_positions[i][d]
                                            - self.population[i][d])
                social = self.c2 * r2 * (self.gbest_position[d]
                                         - self.population[i][d])
                inertia = self.w * self.velocities[i][d]
                self.velocities[i][d] = scale[d] * (inertia + cognitive + social)

            for d in range(self.problem.dimension):
                self.population[i][d] += self.velocities[i][d]
                self.population[i][d] = max(
                    lower_bounds[d],
                    min(self.population[i][d], upper_bounds[d])
                )

            self.evaluations[i] = self.problem.evaluate(self.population[i])

            if self.comparator.is_better(self.evaluations[i],
                                         self.pbest_evaluations[i]):
                self.pbest_positions[i] = copy.deepcopy(self.population[i])
                self.pbest_evaluations[i] = copy.deepcopy(self.evaluations[i])

                if self.comparator.is_better(self.pbest_evaluations[i],
                                             self.gbest_evaluation):
                    self.gbest_position = copy.deepcopy(self.pbest_positions[i])
                    self.gbest_evaluation = copy.deepcopy(self.pbest_evaluations[i])

        self.iteration += 1

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        return [(self.gbest_position, self.gbest_evaluation)]

    def get_population(self) -> List[List[float]]:
        return self.population

    def get_population_evaluations(self) -> List[Evaluation[float]]:
        return self.evaluations