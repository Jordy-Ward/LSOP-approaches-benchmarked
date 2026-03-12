# cilpy/problem/__init__.py
"""The problem module: Defines the optimization problem interface.

This module provides the abstract "contract" for all optimization problems
within the `cilpy` library. It consists of two main components:

1. `Evaluation`: A dataclass that standardizes the return value from any problem
   evaluation, capturing fitness and constraint information.
2. `Problem`: An abstract base class that defines the required methods and
   attributes for a problem to be compatible with `cilpy` solvers.

By implementing the `Problem` interface, users can define custom optimization
landscapes that any solver in the library can operate on.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

# Generic types for candidate solutions and fitness values
SolutionType = TypeVar("SolutionType")
FitnessType = TypeVar("FitnessType")


@dataclass
class Evaluation(Generic[FitnessType]):
    """A container for the results of a single problem evaluation.

    This dataclass standardizes the output of a problem's `evaluate` method,
    providing a consistent structure for fitness values and constraint
    violations that all `cilpy` solvers can understand.

    Attributes:
        fitness: The objective function value(s) of the solution.
            This can be a single float for single-objective problems or a list
            of floats for multi-objective problems.
        constraints_inequality: A list of values for inequality
            constraints. For a constraint `g(x) <= 0`, a positive value
            indicates a violation. `None` if the problem has no inequality
            constraints.
        constraints_equality: A list of values for equality
            constraints. For a constraint `h(x) == 0`, any non-zero value
            indicates a violation. `None` if the problem has no equality
            constraints.

    Example:
        .. code-block:: python

            # For a single-objective, unconstrained problem
            eval_unconstrained = Evaluation(fitness=10.5)

            # For a multi-objective, constrained problem
            eval_constrained = Evaluation(
                fitness=[10.5, -2.1],
                constraints_inequality=[0.5, -0.1], # First constraint violated
                constraints_equality=[-0.01, 0.0]
            )
    """

    fitness: FitnessType
    constraints_inequality: Optional[List[float]] = None
    constraints_equality: Optional[List[float]] = None


class Problem(ABC, Generic[SolutionType, FitnessType]):
    """An abstract interface for an optimization problem.

    This class serves as the blueprint for all problems in `cilpy`. To create a
    new problem, you must inherit from this class and implement its abstract
    methods. The interface is generic, allowing for various solution types
    (e.g., `List[float]`, `np.ndarray`) and fitness structures.

    Attributes:
        name (str): The name of the problem instance.
        dimension (int): The number of decision variables in the solution space.
        bounds (Tuple[SolutionType, SolutionType]): A tuple `(lower_bounds,
        upper_bounds)` defining the search space for each dimension.

    Example:
        A minimal implementation for a 2D Sphere function problem.

        .. code-block:: python

            from cilpy.problem import Problem, Evaluation

            class SphereProblem(Problem[list[float], float]):
                def __init__(self, dimension: int):
                    super().__init__(
                        dimension=dimension,
                        bounds=([-5.12] * dimension, [5.12] * dimension),
                        name="Sphere"
                    )

                def evaluate(self, solution: list[float]) -> Evaluation[float]:
                    fitness = sum(x**2 for x in solution)
                    return Evaluation(fitness=fitness)

                def is_dynamic(self) -> tuple[bool, bool]:
                    return (False, False)
    """

    @abstractmethod
    def __init__(
        self, dimension: int, bounds: Tuple[SolutionType, SolutionType], name: str
    ) -> None:
        """Initializes a Problem instance.

        Subclasses must call `super().__init__(...)` to ensure these core
        attributes are set.

        Args:
            name: The name of the optimization problem.
            dimension: The dimensionality of the solution space.
            bounds: A tuple `(lower_bounds, upper_bounds)` defining the
                feasible range for each decision variable. For a real-valued
                problem, this is typically `([L1, L2, ...], [U1, U2, ...])`.
        """
        self.name = name
        self.dimension = dimension
        self.bounds = bounds

    @abstractmethod
    def evaluate(self, solution: SolutionType) -> Evaluation[FitnessType]:
        """Evaluates a candidate solution.

        This is the core method of a problem. It takes a solution from a solver
        and returns its performance and feasibility.

        Args:
            solution: The candidate solution to be evaluated. Its type
                (e.g., `List[float]`, `np.ndarray`) must be consistent
                with the `SolutionType` used in the class definition.

        Returns:
            An `Evaluation` object containing the fitness and constraint
            violation information for the given solution.
        """
        pass

    @abstractmethod
    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates if the problem's landscape changes over time.

        Dynamic problems require specialized solvers that can adapt to
        changes in the objective function or constraints.

        Note:
            Solvers can query this method to decide whether to employ
            change-detection mechanisms or re-evaluate their archives of
            best-so-far solutions.

        Returns:
            A tuple `(is_objective_dynamic, is_constraint_dynamic)` where:
                - `is_objective_dynamic` is `True` if the objective
                  function(s) change over time.
                - `is_constraint_dynamic` is `True` if the constraint
                  function(s) change over time.
        """
        pass

    @abstractmethod
    def is_multi_objective(self) -> bool:
        """Indicates if the problem has multiple objectives."""
        pass

    def begin_iteration(self) -> None:
        """
        A notification called by the ExperimentRunner before each solver
        iteration.

        Dynamic problems should override this method to update their internal
        state, such as an iteration counter, and to trigger environmental
        changes. The default implementation does nothing.
        """
        pass

    def get_fitness_bounds(self) -> Tuple[FitnessType, FitnessType]:
        """
        Returns the known theoretical min and max fitness values for the
        problem.

        This is used for calculating normalized performance metrics.

        Returns:
            A tuple containing (global_minimum_fitness, global_maximum_fitness).
        """
        raise NotImplementedError(
            f"The problem '{self.name}' does not have a known optimum value. "
            "Implement get_fitness_bounds() to use metrics that require it."
        )
