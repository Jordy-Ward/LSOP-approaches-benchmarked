# test/problem/test_cmpb.py
"""
Unit tests for the Constrained Moving Peaks Benchmark (CMPB).

This suite tests the CMPB problem, focusing on:
1.  Correct initialization and handling of parameters for the `f` and `g`
    landscapes.
2.  The composition logic in the `evaluate` method, ensuring fitness and
    constraint values are calculated correctly for feasible, infeasible, and
    boundary solutions.
3.  The dynamic behavior, confirming that the problem state changes if either
    of the underlying landscapes is dynamic.
4.  Correct propagation of calls, ensuring that `CMPB.evaluate` calls the
    `evaluate` method of each underlying landscape exactly once.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from cilpy.problem import Evaluation
from cilpy.problem.mpb import MovingPeaksBenchmark
from cilpy.problem.cmpb import ConstrainedMovingPeaksBenchmark, generate_mpb_configs


# --- Test Fixtures ---

@pytest.fixture(scope="module")
def mpb_configs():
    """Provides a standard set of MPB configurations for testing."""
    return generate_mpb_configs(dimension=2)

@pytest.fixture
def static_params(mpb_configs):
    """Parameters for a static MPB landscape."""
    return mpb_configs['STA'].copy()

@pytest.fixture
def dynamic_params(mpb_configs):
    """Parameters for a dynamic MPB landscape."""
    params = mpb_configs['A1L'].copy()
    # Many peaks to increase likelihood of changed evaluation
    params['num_peaks'] = 50
    # A low change frequency to speed up tests
    params['change_frequency'] = 10
    return params


# --- Main Test Class ---

class TestConstrainedMovingPeaksBenchmark:
    """Tests for the ConstrainedMovingPeaksBenchmark problem."""

    # --- Initialization Tests ---

    def test_cmpb_initialization(self, static_params, dynamic_params):
        """Tests successful initialization of the CMPB problem."""
        problem = ConstrainedMovingPeaksBenchmark(
            f_params=static_params,
            g_params=dynamic_params,
            name="TestCMPB"
        )

        assert isinstance(problem.f_landscape, MovingPeaksBenchmark)
        assert isinstance(problem.g_landscape, MovingPeaksBenchmark)
        assert problem.dimension == static_params['dimension']
        assert np.array_equal(problem.bounds[0], problem.f_landscape.bounds[0])
        assert problem.name == "TestCMPB"

        # Check if dynamic flags are set correctly based on params
        assert not problem._is_f_dynamic
        assert problem._is_g_dynamic

    def test_init_raises_error_on_dimension_mismatch(self, static_params):
        """Tests that a ValueError is raised if dimensions do not match."""
        g_params_mismatch = static_params.copy()
        g_params_mismatch['dimension'] = static_params['dimension'] + 1

        with pytest.raises(ValueError, match="must be specified and identical"):
            ConstrainedMovingPeaksBenchmark(
                f_params=static_params,
                g_params=g_params_mismatch
            )

    def test_init_raises_error_on_missing_dimension(self, static_params):
        """Tests that a ValueError is raised if 'dimension' is missing."""
        g_params_missing = static_params.copy()
        del g_params_missing['dimension']

        with pytest.raises(ValueError, match="must be specified and identical"):
            ConstrainedMovingPeaksBenchmark(
                f_params=static_params,
                g_params=g_params_missing
            )

    # --- Evaluation Logic Tests ---

    def test_evaluate_return_structure(self, static_params, dynamic_params):
        """Tests that `evaluate` returns an Evaluation object with the correct structure."""
        problem = ConstrainedMovingPeaksBenchmark(static_params, dynamic_params)
        solution = np.array([50.0, 50.0])
        result = problem.evaluate(solution)

        assert isinstance(result, Evaluation)
        assert isinstance(result.fitness, float) or result.fitness == 0
        assert isinstance(result.constraints_inequality, list)
        assert len(result.constraints_inequality) == 1
        assert isinstance(result.constraints_inequality[0], float)
        assert result.constraints_equality is None

    @patch('cilpy.problem.cmpb.MovingPeaksBenchmark')
    def test_evaluate_calls_underlying_landscapes_once(self, MockMPB):
        """
        Ensures CMPB.evaluate calls f.evaluate and g.evaluate exactly once, which is
        critical for correctly updating the dynamic counters.
        """
        # Configure mock instances to return mock Evaluation objects
        mock_f = MagicMock()
        mock_f.evaluate.return_value = Evaluation(fitness=-80.0) # f_val = 80
        mock_g = MagicMock()
        mock_g.evaluate.return_value = Evaluation(fitness=-50.0) # g_val = 50

        # When the CMPB constructor calls MovingPeaksBenchmark(), return our mocks
        MockMPB.side_effect = [mock_f, mock_g]

        problem = ConstrainedMovingPeaksBenchmark({'dimension': 2}, {'dimension': 2})
        solution = np.array([1.0, 1.0])
        problem.evaluate(solution)

        # Assert that each evaluate method was called exactly once with the solution
        mock_f.evaluate.assert_called_once_with(solution)
        mock_g.evaluate.assert_called_once_with(solution)

    @pytest.mark.parametrize("f_val, g_val, expected_fitness, expected_violation", [
        # Feasible case: f(x) > g(x) => g(x) - f(x) is negative
        # Objective = 30.0 - 100.0 = -70.0
        (100.0, 30.0, -70.0, -70.0),
        # Infeasible case: g(x) > f(x) => g(x) - f(x) is positive
        # Objective = 90.0 - 40.0 = 50.0
        (40.0, 90.0, 50.0, 50.0),
        # Boundary case: g(x) = f(x) => g(x) - f(x) is zero
        # Objective = 50.0 - 50.0 = 0.0
        (50.0, 50.0, 0.0, 0.0),
    ])
    def test_evaluate_composition_logic(self, f_val, g_val, expected_fitness, expected_violation):
        """
        Tests the core evaluation logic for feasible, infeasible, and boundary cases.
        The problem is to minimize `min(g(x) - f(x), 0)` subject to `g(x) - f(x) <= 0`.
        """
        # Use mocks to create a predictable scenario
        with patch('cilpy.problem.cmpb.MovingPeaksBenchmark') as MockMPB:
            mock_f = MagicMock()
            # Remember, MPB returns negated fitness
            mock_f.evaluate.return_value = Evaluation(fitness=-f_val)
            mock_g = MagicMock()
            mock_g.evaluate.return_value = Evaluation(fitness=-g_val)
            MockMPB.side_effect = [mock_f, mock_g]

            problem = ConstrainedMovingPeaksBenchmark({'dimension': 1}, {'dimension': 1})
            result = problem.evaluate(np.array([0.0]))

            assert result.fitness == pytest.approx(expected_fitness)
            assert result.constraints_inequality[0] == pytest.approx(expected_violation)

    # --- Dynamic Behavior and `is_dynamic` Tests ---

    def test_dynamic_behavior_when_one_landscape_is_dynamic(self, static_params, dynamic_params):
        """Tests that the overall problem is dynamic if just one component is dynamic."""
        problem = ConstrainedMovingPeaksBenchmark(static_params, dynamic_params)
        solution = np.array([25.0, 75.0])
        problem.begin_iteration()
        initial_eval = problem.evaluate(solution)

        # Evaluate until a change is triggered in the dynamic_g landscape
        for _ in range(dynamic_params['change_frequency']):
            problem.begin_iteration()
            problem.evaluate(solution)

        problem.begin_iteration()
        final_eval = problem.evaluate(solution)

        # The fitness and constraint violation should have changed
        # It is possible that they have not, due to chance. Just re-run the
        # test case if that is the case
        assert initial_eval.fitness != final_eval.fitness
        assert initial_eval.constraints_inequality[0] != final_eval.constraints_inequality[0]

    def test_no_dynamic_behavior_when_both_static(self, static_params):
        """Tests that the problem is static if both underlying landscapes are static."""
        problem = ConstrainedMovingPeaksBenchmark(static_params, static_params.copy())
        solution = np.array([50.0, 50.0])
        initial_eval = problem.evaluate(solution)

        # Evaluate many times; no change should occur
        for _ in range(50):
            problem.evaluate(solution)

        final_eval = problem.evaluate(solution)
        assert initial_eval.fitness == final_eval.fitness
        assert initial_eval.constraints_inequality[0] == final_eval.constraints_inequality[0]

    @pytest.mark.parametrize("f_is_dyn, g_is_dyn, expected_tuple", [
        (True,  True,  (True, True)),
        (True,  False, (True, False)),
        (False, True,  (False, True)),
        (False, False, (False, False)),
    ])
    def test_is_dynamic_method(self, f_is_dyn, g_is_dyn, expected_tuple):
        """
        Tests that the `is_dynamic` method correctly reports the status of the
        underlying f and g landscapes based on their configuration.
        """
        f_freq = 10 if f_is_dyn else 0
        g_freq = 10 if g_is_dyn else 0
        f_params = {'dimension': 1, 'change_frequency': f_freq}
        g_params = {'dimension': 1, 'change_frequency': g_freq}

        problem = ConstrainedMovingPeaksBenchmark(f_params, g_params)
        assert problem.is_dynamic() == expected_tuple
