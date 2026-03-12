# cilpy/solver/chm/__init__.py
"""
The constraint handling mechanism module: Defines the constraint handling
mechanism interface.

This module provides the abstract "contract" for all constraint handling
mechanisms within the `cilpy` library.

In the future, this module could be modified to be a comparator for
multi-objective optimization problems.
"""
from abc import ABC, abstractmethod
from typing import Generic

from cilpy.problem import Evaluation, FitnessType


class ConstraintHandler(ABC, Generic[FitnessType]):
    """
    An abstract interface for a constraint handling mechanism.

    This class defines the strategy for comparing two evaluations in the
    context of a constrained optimization problem. Solvers will delegate
    comparison logic to an implementation of this class.
    """

    @abstractmethod
    def is_better(
        self, eval_a: Evaluation[FitnessType], eval_b: Evaluation[FitnessType]
    ) -> bool:
        """
        Compares two evaluations to determine if `eval_a` is better than `eval_b`.

        Args:
            eval_a: The first evaluation.
            eval_b: The second evaluation.

        Returns:
            True if `eval_a` is considered superior to `eval_b` according to the
            specific constraint handling strategy, False otherwise.
        """
        pass


class DefaultComparator(ConstraintHandler[float]):
    """
    A default comparator for single-objective, unconstrained problems.

    This handler performs a simple fitness comparison, assuming that a lower
    fitness value is better. It does not consider constraints.
    """

    def is_better(self, eval_a: Evaluation[float], eval_b: Evaluation[float]) -> bool:
        return eval_a.fitness < eval_b.fitness
