# cilpy/problem/unconstrained.py
"""
Benchmark Optimization Functions.

This module provides concrete implementations of common benchmark functions
for single-objective, unconstrained optimization, adhering to the `Problem`
interface.
"""
import math
from typing import List, Tuple

from cilpy.problem import Problem, Evaluation


class Sphere(Problem[List[float], float]):
    """The Sphere function.

    A continuous, convex, and unimodal benchmark function. It is one of the
    simplest benchmark problems. The global minimum is at the origin (0, ..., 0)
    with a fitness value of 0.

    The function is defined as: f(x) = Sum(x_i^2) for i = 1 to n.
    """

    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-100, 100)):
        """Initializes a Sphere function problem instance.

        Args:
            dimension: The number of decision variables (dimensions). Defaults
                to 2.
            domain: A tuple `(min_val, max_val)` defining the symmetric search
                space boundary for each dimension. Defaults to (-100, 100).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Sphere"
        )

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Sphere function for a given solution.

        Args:
            solution (List[float]): A list of floats representing the decision
                variables of the candidate solution.

        Returns:
            Evaluation[float]: An Evaluation object containing the fitness
                (a single float) and no constraint violations.
        """
        fitness = sum(x**2 for x in solution)
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Sphere function is not dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(False, False)` as the function is
                static.
        """
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:
        """
        Returns the known theoretical min and max fitness values for the
        problem.

        This is used for calculating normalized performance metrics.

        Returns:
            A tuple containing (global_minimum_fitness, global_maximum_fitness).
        """
        # --- Best fitness ---
        best_fitness = 0.0

        # --- Worst fitness ---
        lower_bound = self.bounds[0][0]
        upper_bound = self.bounds[1][0]
        limit = max(abs(lower_bound), abs(upper_bound))
        worst_fitness = self.dimension * (limit**2)

        return (best_fitness, worst_fitness)


class Quadratic(Problem[List[float], float]):
    """The Quadratic function.

    Also known as the Schwefel 1.2 or Double Sum function. It is a continuous,
    convex, and unimodal benchmark function. It is more difficult than the
    Sphere  function because the variables are dependent on each other due to
    the cumulative summation.

    The global minimum is at the origin (0, ..., 0) with a fitness value of 0.

    The function is defined as: f(x) = Sum_{j=1}^n ( Sum_{k=1}^j x_k )^2
    """

    def __init__(
        self, dimension: int = 2, domain: Tuple[float, float] = (-100.0, 100.0)
    ):
        """Initializes a Quadric function problem instance.

        Args:
            dimension: The number of decision variables (dimensions). Defaults
                to 2.
            domain: A tuple `(min_val, max_val)` defining the symmetric search
                space boundary for each dimension. Defaults to (-100, 100).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Quadric"
        )

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Quadric function for a given solution.

        Implements f(x) = Σ(Σ(x_k)^2) efficiently using a running sum.

        Args:
            solution (List[float]): A list of floats representing the decision
                variables.

        Returns:
            Evaluation[float]: An Evaluation object containing the fitness.
        """
        fitness = 0.0
        cumulative_sum = 0.0
        for x in solution:
            cumulative_sum += x
            fitness += cumulative_sum**2

        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Quadric function is not dynamic.

        Returns:
            Tuple[bool, bool]: (False, False).
        """
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False


class Ackley(Problem[List[float], float]):
    """The Ackley function.

    A widely used multimodal benchmark function characterized by a nearly flat
    outer region and a large number of local minima. Its global minimum is at
    the origin (0, ..., 0) with a fitness value of 0.

    The function is defined as:
    f(x) = -a * exp(-b * sqrt(1/n * Sum(x_i^2)))
          - exp(1/n * Sum(cos(c*x_i)))
          + a
          + exp(1)
    """

    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-32, 32)):
        """Initializes an Ackley function problem instance.

        The standard parameters `a=20`, `b=0.2`, and `c=2pi` are used.

        Args:
            dimension: The number of decision variables (dimensions). Defaults
                to 2.
            domain: A tuple `(min_val, max_val)` defining the symmetric search
                space boundary for each dimension. Defaults to (-32, 32).
        """
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Ackley"
        )
        # Standard parameters for the Ackley function
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the Ackley function for a given solution.

        Args:
            solution (List[float]): A list of floats representing the decision
                variables of the candidate solution.

        Returns:
            Evaluation[float]: An Evaluation object containing the fitness
                (a single float) and no constraint violations.
        """
        sum_sq = sum(x**2 for x in solution)
        sum_cos = sum(math.cos(self.c * x) for x in solution)

        term1 = -self.a * math.exp(-self.b * math.sqrt(sum_sq / self.dimension))
        term2 = -math.exp(sum_cos / self.dimension)

        fitness = term1 + term2 + self.a + math.e
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the Ackley function is not dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(False, False)` as the function is
                static.
        """
        return (False, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the function is not multi-objective."""
        return False


if __name__ == "__main__":
    my_sphere = Sphere(2, (-5.12, 5.12))
    print(my_sphere.evaluate([0, 0]))

    my_quadratic = Quadratic(2, (-5.12, 5.12))
    print(my_quadratic.evaluate([0, 0]))

    my_ackley = Ackley(2, (-5.12, 5.12))
    print(my_ackley.evaluate([0, 0]))
