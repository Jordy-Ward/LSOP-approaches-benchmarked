# cilpy/problem/cmpb.py
"""
The Constrained Moving Peaks Benchmark (CMPB).

This module provides an implementation of the CMPB generator, a dynamic and
constrained optimization problem.
"""

from typing import Any, Dict, Tuple

import numpy as np

from cilpy.problem import Evaluation, Problem
from cilpy.problem.mpb import MovingPeaksBenchmark, generate_mpb_configs


class ConstrainedMovingPeaksBenchmark(Problem[np.ndarray, float]):
    """An implementation of the Constrained Moving Peaks Benchmark (CMPB).

    This class generates a dynamic constrained optimization problem by composing
    two independent `MovingPeaksBenchmark` instances: one for the objective
    function landscape (`f`) and one for the constraint landscape (`g`).

    The problem is naturally a maximization problem defined as:
    Maximize: h(x) = f(x) - g(x)
    A solution is considered feasible if h(x) >= 0 (i.e., f(x) >= g(x)).

    To align with standard minimization solvers, this class formulates the
    problem as:
    Minimize: g(x) - f(x)
    Subject to: g(x) - f(x) <= 0

    This formulation correctly models the problem, where the objective function
    and the single inequality constraint are the same.

    Attributes:
        f_landscape (MovingPeaksBenchmark): The MPB instance for the objective
            function landscape.
        g_landscape (MovingPeaksBenchmark): The MPB instance for the constraint
            function landscape.
    """

    def __init__(
        self,
        f_params: Dict[str, Any],
        g_params: Dict[str, Any],
        name: str = "ConstrainedMovingPeaksBenchmark",
    ):
        """Initializes the Constrained Moving Peaks Benchmark generator.

        Args:
            f_params (Dict[str, Any]): A dictionary of parameters for the
                objective landscape (f), which will be passed to the
                `MovingPeaksBenchmark` constructor.
            g_params (Dict[str, Any]): A dictionary of parameters for the
                constraint landscape (g), which will be passed to the
                `MovingPeaksBenchmark` constructor.
            name (str): The name for the problem instance.

        Raises:
            ValueError: If the 'dimension' parameter is not specified or is
                different for `f_params` and `g_params`.
        """
        f_dim = f_params.get("dimension")
        g_dim = g_params.get("dimension")

        if f_dim is None or g_dim is None or f_dim != g_dim:
            raise ValueError(
                "The 'dimension' parameter must be specified and identical for "
                "both f_params and g_params."
            )

        self.f_landscape = MovingPeaksBenchmark(**f_params)
        self.g_landscape = MovingPeaksBenchmark(**g_params)

        # The problem's domain is defined by the 'f' landscape.
        super().__init__(
            dimension=self.f_landscape.dimension,
            bounds=self.f_landscape.bounds,
            name=name,
        )

        # Determine if landscapes are dynamic based on their change frequency.
        self._is_f_dynamic = f_params.get("change_frequency", 0) > 0
        self._is_g_dynamic = g_params.get("change_frequency", 0) > 0

    def evaluate(self, solution: np.ndarray) -> Evaluation[float]:
        """Evaluates a solution against the composed objective and constraint.

        This method calls the `evaluate` method of the underlying `f` and `g`
        landscapes exactly once, ensuring that their internal evaluation
        counters are updated correctly. It then composes the results to form the
        final fitness and constraint violation.

        Args:
            solution (np.ndarray): The candidate solution to be evaluated.

        Returns:
            Evaluation[float]: An Evaluation object containing the composed
                fitness and the single inequality constraint violation.
        """
        # Evaluate each landscape once. This triggers their internal update
        # logic and returns the negated maximization value.
        f_eval = self.f_landscape.evaluate(solution)
        g_eval = self.g_landscape.evaluate(solution)

        # The MPB implementation in cilpy is already a minimization solver,
        # hence we convert evaluations back to the original maximization values.
        f_val = -f_eval.fitness
        g_val = -g_eval.fitness

        # The objective for maximization is h(x) = f(x) - g(x).  Infeasible
        # areas are indicated where h(x) < 0.
        composed_fitness = f_val - g_val

        # Fitness is negated for minimization solvers
        return Evaluation(
            fitness=-composed_fitness, constraints_inequality=[-composed_fitness]
        )

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates whether the objective or constraint landscapes can change.

        The composed objective `g(x) - f(x)` is dynamic if either `f` or `g`
        is dynamic. Similarly, the composed constraint is dynamic if either
        `f` or `g` is dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(is_objective_dynamic, is_constraint_dynamic)`.
        """
        return (self._is_f_dynamic, self._is_g_dynamic)

    def is_multi_objective(self) -> bool:
        return False

    def begin_iteration(self) -> None:
        """
        Notifies the underlying landscapes that a new solver iteration is
        beginning.

        This method acts as a delegate, calling the `begin_iteration` method on
        both the objective (f) and constraint (g) landscapes. This ensures
        that their internal iteration counters are incremented and environmental
        changes are triggered correctly and in sync.
        """
        self.f_landscape.begin_iteration()
        self.g_landscape.begin_iteration()

    def get_fitness_bounds(self) -> Tuple[float, float]:
        """
        Returns the known theoretical min and max fitness values for the
        problem.

        This is used for calculating normalized performance metrics.

        Returns:
            A tuple containing (global_minimum_fitness, global_maximum_fitness).
        """
        return (-self.f_landscape._max_height, -0.0)


if __name__ == "__main__":

    def demonstrate_cmpb(name: str, f_params: Dict[str, Any], g_params: Dict[str, Any]):
        """Helper function to run and print a constrained scenario."""
        print("=" * 60)
        print(f"Demonstration: {name}")
        print("=" * 60)

        # Instantiate the constrained problem
        problem = ConstrainedMovingPeaksBenchmark(f_params, g_params)
        change_frequency = max(
            f_params.get("change_frequency", 0), g_params.get("change_frequency", 0)
        )

        if change_frequency == 0:
            print("Both landscapes are static. No changes will occur.")
            return

        # Define a few points to track their feasibility and fitness over time
        test_points = {
            "Center": np.array([50.0, 50.0]),
            "Corner": np.array([10.0, 10.0]),
        }

        num_changes_to_observe = 5
        total_evaluations = change_frequency * num_changes_to_observe

        for i in range(total_evaluations + 1):
            # 1. Notify the problem that a new iteration is beginning.
            problem.begin_iteration()

            # 2. Evaluate points in the (potentially new) landscape.
            evals = {name: problem.evaluate(pos) for name, pos in test_points.items()}

            if i > 0 and i % (change_frequency / 2) == 0:
                print(
                    f"\n--- Environment Change #{i // change_frequency} (at iteration {i}) ---"
                )
                for name, evaluation in evals.items():
                    violation = evaluation.constraints_inequality[0]  # type: ignore
                    is_feasible = violation <= 0
                    print(
                        f"  - Point '{name}': Fitness = {evaluation.fitness:.2f}, "
                        f"Violation = {violation:.2f}, Feasible = {is_feasible}"
                    )
        print("\n")

    all_problems = generate_mpb_configs(dimension=2)

    # --- Scenario 1: Dynamic Objective, Static Constraints ---
    # Instantiate the problem generator
    objective_params = all_problems["A2R"]
    constraint_params = all_problems["STA"]
    demonstrate_cmpb("A2R/STA", objective_params, constraint_params)

    # --- Scenario 2: Static Objective, Dynamic Constraints ---
    objective_params = all_problems["STA"]
    constraint_params = all_problems["A2R"]
    demonstrate_cmpb("STA/A2R", objective_params, constraint_params)
