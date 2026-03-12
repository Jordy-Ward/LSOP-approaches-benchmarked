# test/solver/chm/test_alpha_constraint.py
"""
Unit tests for the AlphaConstraintHandler.

This suite is divided into three sections:
1.  Initialization: Tests the constructor for valid and invalid parameters.
2.  Satisfaction Calculation: Rigorously tests the `_calculate_satisfaction`,
    covering all cases for inequality and equality constraints.
3.  Comparison Logic: Tests the `is_better` method to ensure it correctly
    applies the lexicographic comparison rules based on the alpha threshold,
    satisfaction levels, and fitness values.
"""
import pytest

from cilpy.problem import Evaluation
from cilpy.solver.chm.alpha_constraint import AlphaConstraintHandler

# --- Test Initialization ---

def test_handler_initialization():
    """Tests that the handler initializes correctly with valid parameters."""
    handler = AlphaConstraintHandler(alpha=0.9, b_inequality=10.0, b_equality=0.1)
    assert handler.alpha == 0.9
    assert handler.b_inequality == 10.0
    assert handler.b_equality == 0.1

@pytest.mark.parametrize("invalid_alpha", [-0.1, 1.1, -10, 10])
def test_handler_init_raises_error_for_invalid_alpha(invalid_alpha):
    """Tests that a ValueError is raised for an alpha value outside the [0, 1] range."""
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        AlphaConstraintHandler(alpha=invalid_alpha)


# --- Test Satisfaction Calculation (`_calculate_satisfaction`) ---

class TestCalculateSatisfaction:
    """
    A dedicated class to test the `_calculate_satisfaction` method in isolation.
    Uses b_inequality=10.0 and b_equality=1.0 for predictable calculations.
    """
    @pytest.fixture
    def handler(self) -> AlphaConstraintHandler:
        """Provides a handler with standard `b` values for testing."""
        return AlphaConstraintHandler(alpha=0.9, b_inequality=10.0, b_equality=1.0)

    def test_satisfaction_with_no_constraints(self, handler: AlphaConstraintHandler):
        """If there are no constraints, satisfaction should be maximal (1.0)."""
        evaluation = Evaluation(fitness=100.0)
        assert handler._calculate_satisfaction(evaluation) == 1.0

    @pytest.mark.parametrize("g_val, expected_mu_g", [
        (-5.0, 1.0),  # Feasible: g(x) <= 0 -> mu = 1
        (0.0,  1.0),   # Boundary feasible: g(x) = 0 -> mu = 1
        (5.0,  0.5),   # Partially satisfied: 0 < g <= b -> mu = 1 - g/b
        (10.0, 0.0),   # Boundary unsatisfied: g = b -> mu = 1 - 10/10 = 0
        (15.0, 0.0),   # Unsatisfied: g > b -> mu = 0
    ])
    def test_satisfaction_for_single_inequality_constraint(
        self, handler: AlphaConstraintHandler, g_val, expected_mu_g
    ):
        """Tests the satisfaction calculation for various inequality constraint values."""
        evaluation = Evaluation(fitness=0.0, constraints_inequality=[g_val])
        assert handler._calculate_satisfaction(evaluation) == pytest.approx(expected_mu_g)

    @pytest.mark.parametrize("h_val, expected_mu_h", [
        (0.0,   1.0),  # Perfectly satisfied: |h| = 0 -> mu = 1
        (0.5,   0.5),  # Partially satisfied: 0 < |h| <= b -> mu = 1 - |h|/b
        (-0.5,  0.5),  # Test with negative h value
        (1.0,   0.0),  # Boundary unsatisfied: |h| = b -> mu = 1 - 1/1 = 0
        (-1.0,  0.0),  # Test with negative h value at boundary
        (2.0,   0.0),  # Unsatisfied: |h| > b -> mu = 0
        (-2.0,  0.0),  # Test with negative h value
    ])
    def test_satisfaction_for_single_equality_constraint(
        self, handler: AlphaConstraintHandler, h_val, expected_mu_h
    ):
        """Tests the satisfaction calculation for various equality constraint values."""
        evaluation = Evaluation(fitness=0.0, constraints_equality=[h_val])
        assert handler._calculate_satisfaction(evaluation) == pytest.approx(expected_mu_h)

    def test_satisfaction_is_minimum_of_all_constraints(self, handler: AlphaConstraintHandler):
        """
        Tests that the final mu is the minimum of all individual constraint
        satisfaction levels.
        """
        # mu_g values: [1.0 (for g=-5), 0.5 (for g=5)]
        # mu_h values: [0.8 (for h=0.2), 0.9 (for h=-0.1)]
        # The minimum of {1.0, 0.5, 0.8, 0.9} is 0.5.
        evaluation = Evaluation(
            fitness=0.0,
            constraints_inequality=[-5.0, 5.0],
            constraints_equality=[0.2, -0.1]
        )
        assert handler._calculate_satisfaction(evaluation) == pytest.approx(0.5)


# --- Test Comparison Logic (`is_better`) ---

class TestIsBetterLogic:
    """
    A dedicated class to test the `is_better` method's comparison logic.
    Uses alpha=0.9 for all tests.
    """
    @pytest.fixture
    def handler(self) -> AlphaConstraintHandler:
        """Provides a handler with alpha=0.9 for testing comparison rules."""
        return AlphaConstraintHandler(alpha=0.9)

    # Helper Evaluations for various scenarios
    # mu is calculated with default b=1.0
    # A is "better" than B in all cases
    eval_A_feasible_good_f = Evaluation(fitness=10) # mu = 1.0
    eval_B_feasible_bad_f = Evaluation(fitness=20)  # mu = 1.0

    eval_A_high_mu_good_f = Evaluation(fitness=10, constraints_inequality=[0.1]) # mu = 0.9
    eval_B_low_mu_good_f = Evaluation(fitness=10, constraints_inequality=[0.2])  # mu = 0.8

    eval_A_high_mu_any_f = Evaluation(fitness=100, constraints_inequality=[0.1]) # mu = 0.9
    eval_B_low_mu_any_f = Evaluation(fitness=5, constraints_inequality=[0.2])   # mu = 0.8

    # --- Rule 1: Both solutions are "alpha-feasible" (mu >= alpha) ---
    def test_rule1_both_alpha_feasible_A_better(self, handler: AlphaConstraintHandler):
        """
        If both mu_a and mu_b >= alpha, decision is based on fitness.
        Here, f_a < f_b, so A should be better.
        """
        # mu_a = 1.0, mu_b = 0.9. Both >= 0.9.
        # f_a = 10, f_b = 20.
        eval_b = Evaluation(fitness=20, constraints_inequality=[0.1])
        assert handler.is_better(self.eval_A_feasible_good_f, eval_b) is True
        assert handler.is_better(eval_b, self.eval_A_feasible_good_f) is False

    def test_rule1_both_alpha_feasible_equal_fitness(self, handler: AlphaConstraintHandler):
        """If both are alpha-feasible and fitness is equal, neither is better."""
        eval_b = Evaluation(fitness=10, constraints_inequality=[0.05]) # mu=0.95
        assert handler.is_better(self.eval_A_high_mu_good_f, eval_b) is False
        assert handler.is_better(eval_b, self.eval_A_high_mu_good_f) is False

    # --- Rule 2: Satisfaction levels are equal ---
    def test_rule2_equal_mu_A_better(self, handler: AlphaConstraintHandler):
        """
        If mu_a == mu_b, decision is based on fitness.
        Here, f_a < f_b, so A should be better.
        """
        # Both have mu = 0.8
        eval_a = Evaluation(fitness=10, constraints_inequality=[0.2])
        eval_b = Evaluation(fitness=20, constraints_inequality=[0.2])
        assert handler.is_better(eval_a, eval_b) is True
        assert handler.is_better(eval_b, eval_a) is False

    def test_rule2_isclose_mu_A_better(self, handler: AlphaConstraintHandler):
        """Tests that `math.isclose` is used for mu comparison."""
        eval_a = Evaluation(fitness=10, constraints_inequality=[0.2000000001])
        eval_b = Evaluation(fitness=20, constraints_inequality=[0.2])
        assert handler.is_better(eval_a, eval_b) is True
        assert handler.is_better(eval_b, eval_a) is False

    # --- Rule 3: Otherwise, higher satisfaction level wins ---
    def test_rule3_higher_mu_wins_regardless_of_fitness(self, handler: AlphaConstraintHandler):
        """
        If mu levels differ and they are not both alpha-feasible, the higher
        mu wins, even if its fitness is much worse.
        """
        # mu_a = 0.9, mu_b = 0.8. Neither or only one might be >= alpha.
        # f_a = 100 (worse), f_b = 5 (better).
        # A should win because mu_a > mu_b.
        assert handler.is_better(self.eval_A_high_mu_any_f, self.eval_B_low_mu_any_f) is True
        assert handler.is_better(self.eval_B_low_mu_any_f, self.eval_A_high_mu_any_f) is False

    def test_rule3_one_alpha_feasible_one_not(self, handler: AlphaConstraintHandler):
        """An alpha-feasible solution should always beat a non-alpha-feasible one."""
        # A is just alpha-feasible (mu=0.9), B is just not (mu=0.89)
        # Fitness of A is worse, but it should still win.
        eval_a = Evaluation(fitness=100, constraints_inequality=[0.1]) # mu=0.9
        eval_b = Evaluation(fitness=5, constraints_inequality=[0.11])  # mu=0.89
        assert handler.is_better(eval_a, eval_b) is True
        assert handler.is_better(eval_b, eval_a) is False
