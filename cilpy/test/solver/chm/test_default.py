# test/solver/chm/test_chm.py
"""
Unit tests for the default constraint handling mechanisms.

This suite tests the comparators defined in the base `chm` module,
ensuring they correctly implement their respective comparison strategies.
"""
import pytest

from cilpy.problem import Evaluation
from cilpy.solver.chm import DefaultComparator


class TestDefaultComparator:
    """
    Tests the `DefaultComparator`.

    This comparator should behave as if it were for an unconstrained problem,
    meaning it only compares the `fitness` attribute of two evaluations
    and completely ignores any constraint data. The tests confirm this behavior.
    """

    @pytest.fixture
    def comparator(self) -> DefaultComparator:
        """Provides a fresh instance of the DefaultComparator for each test."""
        return DefaultComparator()

    # --- Test Scenarios for is_better(a, b) ---

    def test_a_is_better_simple(self, comparator: DefaultComparator):
        """Tests the case where A's fitness is clearly better (lower) than B's."""
        eval_a = Evaluation(fitness=10.0)
        eval_b = Evaluation(fitness=20.0)
        assert comparator.is_better(eval_a, eval_b) is True

    def test_a_is_worse_simple(self, comparator: DefaultComparator):
        """Tests the case where A's fitness is clearly worse (higher) than B's."""
        eval_a = Evaluation(fitness=20.0)
        eval_b = Evaluation(fitness=10.0)
        assert comparator.is_better(eval_a, eval_b) is False

    def test_fitness_is_equal(self, comparator: DefaultComparator):
        """Tests the case where fitness values are identical."""
        eval_a = Evaluation(fitness=15.0)
        eval_b = Evaluation(fitness=15.0)
        # is_better should return False if not strictly better
        assert comparator.is_better(eval_a, eval_b) is False

    def test_a_is_better_with_negative_fitness(self, comparator: DefaultComparator):
        """Tests comparison with negative fitness values."""
        eval_a = Evaluation(fitness=-100.0)
        eval_b = Evaluation(fitness=-50.0)
        assert comparator.is_better(eval_a, eval_b) is True

    # --- Tests to ensure constraints are IGNORED ---

    def test_a_is_better_despite_being_infeasible(self, comparator: DefaultComparator):
        """
        Tests that A is chosen if its fitness is better, even if it violates
        constraints and B does not. This confirms constraints are ignored.
        """
        # A is infeasible but has better fitness
        eval_a = Evaluation(
            fitness=5.0,
            constraints_inequality=[1.0]  # Violated: g(x) > 0
        )
        # B is feasible but has worse fitness
        eval_b = Evaluation(
            fitness=10.0,
            constraints_inequality=[-1.0] # Not violated: g(x) <= 0
        )
        assert comparator.is_better(eval_a, eval_b) is True

    def test_a_is_worse_despite_being_feasible(self, comparator: DefaultComparator):
        """
        Tests that A is NOT chosen if its fitness is worse, even if it is
        the only feasible solution. This confirms constraints are ignored.
        """
        # A is feasible but has worse fitness
        eval_a = Evaluation(
            fitness=10.0,
            constraints_equality=[0.0]  # Not violated
        )
        # B is infeasible but has better fitness
        eval_b = Evaluation(
            fitness=5.0,
            constraints_equality=[0.1] # Violated
        )
        assert comparator.is_better(eval_a, eval_b) is False

    def test_equal_fitness_different_feasibility(self, comparator: DefaultComparator):
        """
        Tests that A is NOT chosen if fitness is equal, regardless of
        constraint violations.
        """
        # A is infeasible
        eval_a = Evaluation(
            fitness=10.0,
            constraints_inequality=[1.0]
        )
        # B is feasible
        eval_b = Evaluation(
            fitness=10.0,
            constraints_inequality=[-1.0]
        )
        assert comparator.is_better(eval_a, eval_b) is False
