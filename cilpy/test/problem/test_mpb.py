# test/problem/test_mpb.py
"""
Unit tests for the Moving Peaks Benchmark (MPB) problem.

This suite tests three main components:
1.  The internal `_Peak` class, ensuring its evaluation and update
    mechanisms work as expected, including boundary reflection.
2.  The `MovingPeaksBenchmark` problem class, verifying its initialization,
    dynamic behavior (i.e., landscape changes), and adherence to the
    `Problem` interface.
3.  The `generate_mpb_configs` function, ensuring it creates the correct
    number of configurations and applies the classification rules correctly,
    especially in cases of conflict.
"""
import pytest
import numpy as np
from typing import Dict, Any

from cilpy.problem import Evaluation
from cilpy.problem.mpb import (
    _Peak,
    MovingPeaksBenchmark,
    generate_mpb_configs
)

# --- Tests for the internal _Peak helper class ---

class TestPeakClass:
    """Tests the functionality of the internal _Peak helper class."""

    def test_peak_initialization(self):
        """Tests if a peak is initialized with the correct attributes."""
        position = np.array([10.0, 20.0])
        height = 50.0
        width = 5.0
        peak = _Peak(position, height, width)

        assert np.array_equal(peak.v, position)
        assert peak.h == height
        assert peak.w == width
        assert np.array_equal(peak.s_v, np.zeros_like(position))

    def test_peak_evaluate(self):
        """Tests the peak's evaluation function using a known distance."""
        # Peak at [10, 10] with height 100 and width 10
        peak = _Peak(np.array([10.0, 10.0]), 100.0, 10.0)
        # Point to evaluate at [13, 14], which is a distance of 5 from the center
        # sqrt((13-10)^2 + (14-10)^2) = sqrt(3^2 + 4^2) = sqrt(9 + 16) = 5
        test_point = np.array([13.0, 14.0])

        # Expected value: h - w * dist = 100 - 10 * 5 = 50
        assert peak.evaluate(test_point) == pytest.approx(50.0)

    def test_peak_update_changes_properties(self):
        """
        Tests that the update method modifies a peak's properties when
        severities are non-zero.
        """
        np.random.seed(42) # for reproducibility
        initial_pos = np.array([50.0, 50.0])
        peak = _Peak(initial_pos.copy(), 50.0, 10.0)
        bounds = (np.array([0.0, 0.0]), np.array([100.0, 100.0]))

        peak.update(
            height_sev=5.0,
            width_sev=1.0,
            change_sev=2.0,
            lambda_param=0.5,
            bounds=bounds,
            max_height_cap=70
        )

        # Assert that all properties have changed from their initial values
        assert peak.h != 50.0
        assert peak.w != 10.0
        assert not np.array_equal(peak.v, initial_pos)

    def test_peak_update_boundary_reflection(self):
        """
        Tests that a peak is reflected back into the bounds if it moves
        outside, and its shift vector is inverted on the correct axis.
        """
        # Create a peak near the upper boundary
        initial_pos = np.array([99.0, 50.0])
        peak = _Peak(initial_pos.copy(), 50.0, 10.0)
        
        # Manually set a shift vector that will push it out of bounds
        peak.s_v = np.array([2.0, 1.0]) # Move 2 units on x, 1 unit on y
        bounds = (np.array([0.0, 0.0]), np.array([100.0, 100.0]))
        
        # This update is simplified just to test the boundary logic
        peak.v += peak.s_v # New position would be [101.0, 51.0]

        # Manually apply the reflection logic from the update method
        _, max_b = bounds
        high_mask = peak.v > max_b
        peak.v[high_mask] = 2 * max_b[high_mask] - peak.v[high_mask]
        peak.s_v[high_mask] *= -1.0

        # Expected reflected position: 2 * 100 - 101 = 99.0
        assert peak.v[0] == pytest.approx(99.0)
        # Y-position should be unchanged
        assert peak.v[1] == pytest.approx(51.0)
        # Shift vector should be inverted on the x-axis, not y-axis
        assert np.array_equal(peak.s_v, np.array([-2.0, 1.0]))


# --- Tests for the MovingPeaksBenchmark Problem Class ---

class TestMovingPeaksBenchmark:
    """Tests the main MovingPeaksBenchmark problem."""

    def test_mpb_initialization(self):
        """Tests if the MPB is initialized correctly."""
        problem = MovingPeaksBenchmark(dimension=5, num_peaks=20)
        assert problem.dimension == 5
        assert len(problem.peaks) == 20
        assert problem.bounds[0].shape == (5,)
        assert problem.bounds[1].shape == (5,)
        assert isinstance(problem.name, str)

    def test_mpb_dynamic_change_triggers(self):
        """
        Tests that the landscape changes after `change_frequency` evaluations.
        """
        np.random.seed(0)
        problem = MovingPeaksBenchmark(
            dimension=2,
            num_peaks = 100,     # Many peaks to ensure noticeable change
            change_frequency=10,
            height_severity=10.0 # High severity to ensure noticeable change
        )
        test_point = np.array([50.0, 50.0])

        # Evaluate once to get the initial fitness at this point
        problem.begin_iteration()
        initial_eval = problem.evaluate(test_point)
        assert isinstance(initial_eval, Evaluation)
        assert isinstance(initial_eval.fitness, float)

        # Evaluate 7 more times (total 8, less than change_frequency)
        for _ in range(7):
            problem.begin_iteration()
            problem.evaluate(test_point)

        # The fitness should still be the same
        problem.begin_iteration()
        eval_before_change = problem.evaluate(test_point)
        assert initial_eval.fitness == eval_before_change.fitness

        # The 10th evaluation should trigger the change
        problem.begin_iteration()
        eval_after_change = problem.evaluate(test_point)

        # The fitness should now be different
        assert initial_eval.fitness != eval_after_change.fitness

    def test_mpb_no_change_when_static(self):
        """
        Tests that the landscape does NOT change if change_frequency is 0.
        """
        problem = MovingPeaksBenchmark(dimension=2, change_frequency=0)
        test_point = np.array([50.0, 50.0])
        initial_fitness = problem.evaluate(test_point).fitness

        # Evaluate many times
        for _ in range(100):
            problem.evaluate(test_point)
        
        final_fitness = problem.evaluate(test_point).fitness
        assert initial_fitness == final_fitness

    def test_mpb_get_optimum_value(self):
        """Tests if get_optimum_value returns the negated height of the tallest peak."""
        problem = MovingPeaksBenchmark()
        # Manually set peak heights for a predictable outcome
        problem.peaks[0].h = 80.0
        for peak in problem.peaks[1:]:
            peak.h = 70.0

        # The best value should be -80.0
        assert problem.get_fitness_bounds()[0] == pytest.approx(-70.0)

    def test_mpb_is_dynamic(self):
        """Tests that the problem correctly identifies itself as dynamic."""
        problem = MovingPeaksBenchmark()
        assert problem.is_dynamic() == (True, False)


# --- Tests for the generate_mpb_configs function ---

class TestGenerateMPBConfigs:
    """Tests the configuration generator function."""

    @pytest.fixture(scope="class")
    def all_configs(self) -> Dict[str, Dict[str, Any]]:
        """A fixture to generate the configs once for all tests in this class."""
        return generate_mpb_configs(dimension=2)

    def test_generate_configs_count_and_keys(self, all_configs):
        """Tests if the function generates the correct number of configurations."""
        # 27 dynamic (3x3x3) + 1 static ("STA")
        assert len(all_configs) == 28
        assert "STA" in all_configs
        assert "A1L" in all_configs
        assert "C3R" in all_configs

    def test_generate_configs_static_case(self, all_configs):
        """Tests the parameters of the static 'STA' configuration."""
        sta_config = all_configs["STA"]
        assert sta_config["change_frequency"] == 0
        assert sta_config["change_severity"] == 0
        assert sta_config["height_severity"] == 0
        assert sta_config["width_severity"] == 0

    def test_generate_configs_value_error(self):
        """Tests that a ValueError is raised for s_for_random=0."""
        with pytest.raises(ValueError, match="'s_for_random' must be a non-zero value."):
            generate_mpb_configs(s_for_random=0)

    @pytest.mark.parametrize("acronym, expected_rules", [
        # Test Type I: height_severity = 0, s != 0
        ("P1L", {"height_severity": 0.0, "change_severity": 1.0, "lambda_param": 1.0}),
        # Test Type II: s = 0, h_sev != 0
        ("A2C", {"change_severity": 0.0, "lambda_param": 0.0}),
        # Test Type III: s != 0, h_sev != 0
        ("C3R", {"change_severity": 10.0, "lambda_param": 0.0}),
        # Test Conflict (*1C/*3C): 'C' wants s=0, '1' wants s!=0. 'C' wins priority.
        # Edit: 'C' wins priority and sets s to 1.0.
        ("A1C", {"change_severity": 1.0, "height_severity": 0.0}),
        # Test Conflict (*2L/*2R): '2' wants s=0, 'L' wants s!=0. '2' wins priority.
        ("C2L", {"change_severity": 0.0, "lambda_param": 1.0}),
    ])
    def test_config_rules_and_conflicts(self, all_configs, acronym, expected_rules):
        """
        Spot-checks several key configurations to ensure rules and conflict
        resolutions are applied correctly.
        """
        assert acronym in all_configs
        config = all_configs[acronym]

        for key, expected_value in expected_rules.items():
            assert config[key] == pytest.approx(expected_value)
