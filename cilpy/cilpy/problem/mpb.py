# cilpy/problem/mpb.py
"""
The Moving Peaks Benchmark (MPB) for dynamic optimization problems.

This module provides an implementation of the Moving Peaks Benchmark (MPB)
generator, a widely used tool for creating dynamic, multi-peaked optimization
landscapes. It is designed to test the ability of optimization algorithms to
adapt to changing environments.

The MPB landscape is defined by a number of peaks, each with its own height,
width, and position. At specified intervals (controlled by `change_frequency`),
these peak properties are updated, causing the landscape to shift, change shape,
or both.

While the benchmark is naturally a maximization problem, this implementation
negates the fitness value upon evaluation, allowing it to be used directly with
standard minimization algorithms.

--------------------------------------------------------------------------------
The 28 Standard Problem Classes & The `generate_mpb_configs` Function
--------------------------------------------------------------------------------
This module also includes the `generate_mpb_configs` helper function, which
programmatically creates parameter dictionaries for the 28 standard problem
classes defined by Duhain and Engelbrecht.

These classes are identified by a 3-letter acronym (e.g., 'A1L'), which combines
one code from each of the three classification schemes detailed below.

### 1. Duhain & Engelbrecht: Spatial and Temporal Severity (First Letter)
Defines the magnitude and frequency of changes.

-   **Progressive ('P')**: Frequent, small changes.
    -   `change_frequency`: Low value (high temporal change).
    -   `change_severity` (`s`): Low value.
    -   `height_severity`: Low value.

-   **Abrupt ('A')**: Infrequent, large changes.
    -   `change_frequency`: High value (low temporal change).
    -   `change_severity` (`s`): High value.
    -   `height_severity`: High value.

-   **Chaotic ('C')**: Frequent, large changes.
    -   `change_frequency`: Low value (high temporal change).
    -   `change_severity` (`s`): High value.
    -   `height_severity`: High value.

### 2. Hu & Eberhart / Shi & Eberhart: Optima Modification (Second Letter)
Defines *what* changes about the peaks (position, value, or both).

-   **Type I ('1')**: Locations change, heights are constant.
    -   `height_severity`: Set to 0.0.
    -   Requires `change_severity` (`s`) != 0 for movement.

-   **Type II ('2')**: Locations are static, heights change.
    -   `height_severity`: Set to a non-zero value.
    -   Requires `change_severity` (`s`) = 0 to prevent movement.

-   **Type III ('3')**: Both locations and heights change.
    -   `height_severity`: Set to a non-zero value.
    -   Requires `change_severity` (`s`) != 0 for movement.

### 3. Angeline: Optima Trajectory (Third Letter)
Defines the *pattern* of peak movement.

-   **Linear ('L')**: Peaks move in a straight, correlated line.
    -   `lambda_param`: Set to 1.0.
    -   Requires `change_severity` (`s`) != 0 for movement.

-   **Circular ('C')**: Peaks have a periodic movement pattern. This is achieved in
    the parameterization by preventing translational movement.
    -   Requires `change_severity` (`s`) = 0.

-   **Random ('R')**: Peaks move randomly without a discernible pattern.
    -   `lambda_param`: Set to 0.0.
    -   Requires `change_severity` (`s`) != 0 for movement.

### Conflict Resolution
Some combinations are impossible (e.g., a Type II problem, which requires
`s = 0`, cannot have Linear movement, which requires `s != 0`). The
`generate_mpb_configs` function marks these impossible configurations by setting
the `change_severity` parameter to the string 'XXX'.
"""

import itertools
import numpy as np
import random

from typing import Any, Dict, List, Tuple

from cilpy.problem import Evaluation, Problem


class _Peak:
    """Represents a single peak within the Moving Peaks Benchmark landscape.

    This is a helper class that encapsulates the state and behavior of one peak,
    including its position, height, width, and how it changes over time.

    Attributes:
        v (np.ndarray): The vector representing the peak's location (center).
        h (float): The scalar value for the peak's height.
        w (float): The scalar value for the peak's width.
        s_v (np.ndarray): The shift vector, influencing the peak's movement.
    """

    def __init__(self, position: np.ndarray, height: float, width: float):
        """Initializes a _Peak instance.

        Args:
            position (np.ndarray): The initial location vector of the peak.
            height (float): The initial height of the peak.
            width (float): The initial width of the peak.
        """
        self.v = position
        self.h = height
        self.w = width
        self.s_v = np.zeros_like(position)

    def evaluate(self, x: np.ndarray) -> float:
        """Calculates the peak's value at a given position `x`.

        The peak function is defined as: p_i(x) = h - w * ||x - v||, where
        ||.|| is the Euclidean distance.

        Args:
            x (np.ndarray): The candidate solution's position vector.

        Returns:
            float: The value of the peak function at position `x`.
        """
        dist = np.linalg.norm(x - self.v)
        return float(self.h - self.w * dist)

    def update(
        self,
        height_sev: float,
        width_sev: float,
        change_sev: float,
        lambda_param: float,
        bounds: Tuple[np.ndarray, np.ndarray],
        max_height_cap: float,
    ):
        """Updates the peak's parameters for the next environment.

        Parameters are updated as follows:
            - height = height + (hSeverity * normal(0, 1))
            - width = width + (wSeverity * normal(0, 1))
            - position = position + s_v

        The shift vector (s_v) is controlled by:
            - change_sev: how much positional change occurs
            - lambda: how much random movement (lower -> more random movement)
            - s_v: the current value for the shift vector

        The update equation for the shift vector can be found in section 4.1.3
        of Gary Pamparà's PhD thesis, equation 4.4.

        Args:
            height_sev (float): The severity of height changes.
            width_sev (float): The severity of width changes.
            change_sev (float): The severity of positional changes (shift length).
            lambda_param (float): A correlation coefficient for peak movement.
                A value of 0.0 implies fully random movement.
            bounds (Tuple[np.ndarray, np.ndarray]): The problem's lower and
                upper boundaries for enforcing constraints.
        """
        # Update height with Gaussian noise
        self.h += height_sev * np.random.normal(0, 1)
        self.h = min(self.h, max_height_cap)  # Cap the height

        # Update width with Gaussian noise
        self.w += width_sev * np.random.normal(0, 1)
        self.w = max(0.1, self.w)  # ensure width remains a positive value

        # Update shift vector
        dim = len(self.v)
        # Generate a random vector p_r and normalize to length 'change_sev'
        p_r = np.random.uniform(-1, 1, size=dim)
        mag_pr = np.linalg.norm(p_r)
        if mag_pr > 0:
            p_r *= change_sev / mag_pr

        combined_move = (1.0 - lambda_param) * p_r + lambda_param * self.s_v
        mag_move = np.linalg.norm(p_r + self.s_v)
        if mag_move > 0:
            self.s_v = (change_sev / mag_move) * combined_move
        else:
            self.s_v = np.zeros(dim)

        # Update position
        self.v += self.s_v

        # Enforce boundary conditions via reflection
        min_b, max_b = bounds
        low_mask = self.v < min_b
        self.v[low_mask] = 2 * min_b[low_mask] - self.v[low_mask]
        self.s_v[low_mask] *= -1.0
        high_mask = self.v > max_b
        self.v[high_mask] = 2 * max_b[high_mask] - self.v[high_mask]
        self.s_v[high_mask] *= -1.0


class MovingPeaksBenchmark(Problem[np.ndarray, float]):
    """An implementation of the Moving Peaks Benchmark (MPB) generator.

    This class conforms to the `Problem` interface and produces dynamic,
    unconstrained optimization problems. The objective is to find the maximum
    value in a landscape composed of several moving peaks.

    Note:
        Since most solvers are minimizers, the `evaluate` method returns the
        *negated* value of the MPB function. Minimizing this value is
        equivalent to maximizing the original function.

    Attributes:
        peaks (List[_Peak]): A list of the peak objects in the landscape.
    """

    def __init__(
        self,
        dimension: int = 2,
        num_peaks: int = 10,
        domain: Tuple[float, float] = (0.0, 100.0),
        min_height: float = 30.0,
        max_height: float = 70.0,
        min_width: float = 1.0,
        max_width: float = 12.0,
        change_frequency: int = 5000,
        change_severity: float = 1.0,
        height_severity: float = 7.0,
        width_severity: float = 1.0,
        lambda_param: float = 0.0,
        name: str = "MovingPeaksBenchmark",
    ):
        """Initializes the Moving Peaks Benchmark problem.

        Args:
            dimension (int): The dimensionality of the search landscape.
            num_peaks (int): The number of peaks in the landscape.
            domain (Tuple[float, float]): The `(min, max)` coordinates for the
                symmetric search space.
            min_height (float): The minimum initial height of a peak.
            max_height (float): The maximum initial height of a peak.
            min_width (float): The minimum initial width of a peak.
            max_width (float): The maximum initial width of a peak.
            change_frequency (int): The number of evaluations between landscape changes.
            change_severity (float): Controls how severely peak positions change.
            height_severity (float): Controls how severely peak heights change.
            width_severity (float): Controls how severely peak widths change.
            lambda_param (float): Correlates peak movement over time. A value of
                0.0 results in random movement direction at each change.
            name (str): The name of the problem instance.
        """
        min_bounds = np.array([domain[0]] * dimension)
        max_bounds = np.array([domain[1]] * dimension)
        super().__init__(dimension, (min_bounds, max_bounds), name)

        self._change_frequency = change_frequency
        self._change_sev = change_severity
        self._height_sev = height_severity
        self._width_sev = width_severity
        self._lambda = lambda_param
        self._max_height = max_height

        self.peaks: List[_Peak] = []
        for _ in range(num_peaks):
            pos = np.random.uniform(domain[0], domain[1], size=dimension)
            height = random.uniform(min_height, max_height)
            width = random.uniform(min_width, max_width)
            self.peaks.append(_Peak(pos, height, width))

        self._base_value = 0.0  # As per Equation 4.2
        self._eval_count = 0
        self._iteration_count = 0

    def evaluate(self, solution: np.ndarray) -> Evaluation[float]:
        """Evaluates a solution and returns its fitness.

        This method checks if the environment should change based on the
        evaluation count. It then calculates the function value as the maximum
        of all peak evaluations.

        Args:
            solution (np.ndarray): The candidate solution to be evaluated.

        Returns:
            Evaluation[float]: An Evaluation object containing the negated
                fitness value for use with minimization solvers.
        """
        self._eval_count += 1

        peak_values = [p.evaluate(solution) for p in self.peaks]
        fitness = float(max([self._base_value] + peak_values))
        return Evaluation(fitness=-fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that the problem's objectives are dynamic.

        Returns:
            Tuple[bool, bool]: A tuple `(True, False)` as the objective
                function changes over time but there are no constraints.
        """
        return (True, False)

    def is_multi_objective(self) -> bool:
        """Indicates that the problem is not multi-objective."""
        return False

    def begin_iteration(self) -> None:
        """
        This method is called by the runner once per iteration.
        It handles the logic for changing the environment.
        """
        self._iteration_count += 1

        if (
            self._change_frequency > 0
            and self._iteration_count > 0
            and self._iteration_count % self._change_frequency == 0
        ):
            self.update_all_peaks()

    def get_fitness_bounds(self) -> Tuple[float, float]:
        """
        Returns the known theoretical min and max fitness values for the
        problem.

        This is used for calculating normalized performance metrics.

        Returns:
            A tuple containing (global_minimum_fitness, global_maximum_fitness).
        """
        return (-self._max_height, -self._base_value)

    def update_all_peaks(self) -> None:
        """Updates all peaks of the mpb."""
        for peak in self.peaks:
            peak.update(
                height_sev=self._height_sev,
                width_sev=self._width_sev,
                change_sev=self._change_sev,
                lambda_param=self._lambda,
                bounds=self.bounds,
                max_height_cap=self._max_height,
            )


def generate_mpb_configs(
    dimension: int = 5,
    num_peaks: int = 10,
    domain: Tuple[float, float] = (0.0, 100.0),
    min_height: float = 30.0,
    max_height: float = 70.0,
    min_width: float = 1.0,
    max_width: float = 12.0,
    width_severity: float = 0.05,
    s_for_random: float = 1.0,  # s value for s != 0
) -> Dict[str, Dict[str, Any]]:
    """
    Programmatically generates parameter dictionaries for all 28 MPB classes.

    This function combines the rules from three classification schemes to generate
    27 dynamic problem configurations and 1 static configuration. It handles
    contradictions between rules as specified.

    Args:
        s_for_random (float): The non-zero value to use for the change_severity
            parameter `s` when a non-zero value is required. Defaults to 1.0.

    Returns:
        Dict[str, Dict]: A dictionary where keys are the 3-letter acronyms
            (e.g., "A1C", "P3L") and values are the corresponding parameter
            dictionaries for the MovingPeaksBenchmark constructor. A "STA" key
            is included for the static case.
    """
    if s_for_random == 0:
        raise ValueError("'s_for_random' must be a non-zero value.")

    # 1. Base Configuration (common to all classes)
    base_params = {
        "dimension": dimension,
        "num_peaks": num_peaks,
        "domain": domain,
        "min_height": min_height,
        "max_height": max_height,
        "min_width": min_width,
        "max_width": max_width,
        "width_severity": width_severity,  # Often kept low
    }

    # 2. Define "Low" vs. "High" Values for severity and frequency
    LOW_S, HIGH_S = s_for_random, 10.0
    LOW_H, HIGH_H = 1.0, 10.0

    # High temporal frequency = low number of evaluations between changes
    FREQ_PROGRESSIVE = 20
    FREQ_ABRUPT = 100
    FREQ_CHAOTIC = 30

    # 3. Classification Rules
    # Duhain & Engelbrecht: Severity (Spatial & Temporal)
    # Acronyms: P (Progressive), A (Abrupt), C (Chaotic)
    SEVERITY_CLASSES = {
        "P": {
            "change_severity": LOW_S,
            "height_severity": LOW_H,
            "change_frequency": FREQ_PROGRESSIVE,
        },
        "A": {
            "change_severity": HIGH_S,
            "height_severity": HIGH_H,
            "change_frequency": FREQ_ABRUPT,
        },
        "C": {
            "change_severity": HIGH_S,
            "height_severity": HIGH_H,
            "change_frequency": FREQ_CHAOTIC,
        },
    }

    # Hu & Eberhart / Shi & Eberhart: Optima Modification
    # Acronyms: 1 (Type I), 2 (Type II), 3 (Type III)
    # We use 's_req' to define the requirement for the change_severity (s)
    MODIFICATION_CLASSES = {
        "1": {"height_severity": 0.0, "s_req": "!=0"},
        "2": {"s_req": "=0"},  # height_severity will be taken from SEVERITY_CLASSES
        "3": {"s_req": "!=0"},  # height_severity will be taken from SEVERITY_CLASSES
    }

    # Angeline: Optima Trajectory
    # Acronyms: L (Linear), C (Circular), R (Random)
    MOVEMENT_CLASSES = {
        "L": {"lambda_param": 1.0, "s_req": "!=0"},
        "C": {"lambda_param": 0.0, "s_req": "=0"},  # lambda is irrelevant when s=0
        "R": {"lambda_param": 0.0, "s_req": "!=0"},
    }

    # --- Generation Logic ---
    all_configs = {}

    # 4. Iterate through all 3*3*3 = 27 combinations
    severity_codes = SEVERITY_CLASSES.keys()
    modification_codes = MODIFICATION_CLASSES.keys()
    movement_codes = MOVEMENT_CLASSES.keys()

    for sev_code, mod_code, mov_code in itertools.product(
        severity_codes, modification_codes, movement_codes
    ):
        acronym = f"{sev_code}{mod_code}{mov_code}"

        # Start with base and add severity parameters
        config = base_params.copy()
        config.update(SEVERITY_CLASSES[sev_code])
        config["name"] = acronym

        mod_rules = MODIFICATION_CLASSES[mod_code]
        mov_rules = MOVEMENT_CLASSES[mov_code]

        # 5. Resolve Conflicts for `change_severity` (s)
        s_req_mod = mod_rules["s_req"]
        s_req_mov = mov_rules["s_req"]

        is_conflict = (s_req_mod == "!=0" and s_req_mov == "=0") or (
            s_req_mod == "=0" and s_req_mov == "!=0"
        )

        if is_conflict:
            # *2L/*2R: 2 requires s = 0, but L&R requires s != 0
            #          2 gets priority since *3* requires s != 0
            if mod_code == "2" and (mov_code == "L" or mov_code == "R"):
                config["change_severity"] = 0.0
            # *1C/*3C: C requires s = 0, but 1&3 requires s != 0
            #          C gets priority since *2* requires s = 0
            elif mov_code == "C" and (mod_code == "1" or mod_code == "3"):
                config["change_severity"] = 1.0
            else:
                # This is not supposed to happen, but is kept for clarity
                config["change_severity"] = "XXX"  # Assign placeholder on conflict
        elif s_req_mod == "=0" or s_req_mov == "=0":
            # If either requires s=0 and there's no conflict, it must be 0
            config["change_severity"] = 0.0
        else:
            # Otherwise, s must be non-zero. Use the value from the severity class.
            # This is already set, but we make it explicit for clarity.
            pass

        # 6. Apply overrides from modification and movement rules
        # C1*: C requires hSeverity high, but 1 requires hSeverity = 0
        #      1 (mod rule) gets priority since *2*/*3* requires hSeverity != 0
        if "height_severity" in mod_rules:
            config["height_severity"] = mod_rules["height_severity"]

        if "lambda_param" in mov_rules:
            config["lambda_param"] = mov_rules["lambda_param"]

        all_configs[acronym] = config

    # Add the static problem class
    static_config = base_params.copy()
    static_config.update(
        {
            "change_frequency": 0,
            "change_severity": 0,
            "height_severity": 0,
            "width_severity": 0,
            "lambda_param": 0,
            "name": "STA",
        }
    )
    all_configs["STA"] = static_config

    return all_configs


if __name__ == "__main__":

    def demonstrate_mpb(params: dict):
        """Helper function to run and print a scenario."""
        print("-" * 50)
        print(f"Demonstration: {params.get('name')}")
        print("-" * 50)

        # Instantiate the problem
        problem = MovingPeaksBenchmark(**params)

        # We will track the position and value of a single peak to see how it moves.
        tracked_peak_index = 0

        # We will also evaluate a static point to see how the landscape changes underneath it.
        static_point_to_test = np.array([50.0] * params.get("dimension"))  # type: ignore

        num_changes_to_observe = 5
        total_evaluations = params["change_frequency"] * num_changes_to_observe

        for i in range(total_evaluations + 1):
            # The actual evaluation triggers the internal counter
            evaluation = problem.evaluate(static_point_to_test)
            problem.begin_iteration()

            # Check if the environment just changed
            if i > 0 and i % (params["change_frequency"] / 2) == 0:
                change_num = i // params["change_frequency"]
                peak_pos = problem.peaks[tracked_peak_index].v
                peak_evaluation = problem.evaluate(peak_pos)

                print(f"\nEnvironment Change #{change_num} (at evaluation {i}):")
                print(f"  - Position of Peak {tracked_peak_index}: {peak_pos}")
                print(
                    f"  - Value of Peak {tracked_peak_index}: [{peak_evaluation.fitness}]"
                )
                print(f"  - Value at static point: {evaluation.fitness:.2f}")

        print("\n")

    # Get all problem configurations
    all_problems = generate_mpb_configs(dimension=3)

    # --- Example ---
    # Select a specific problem, for example, "A2R" (Abrupt, Type II, Random)
    params = all_problems["A2R"]
    params["change_frequency"] = 10

    # Instantiate the problem generator
    demonstrate_mpb(params)

    # Parameters can also be specified at creation:
    all_problems = generate_mpb_configs(dimension=2)
    config = all_problems.get("A1C")
    print(f"MPB config for A1C: {config}")

    # And modified:
    config["num_peaks"] = 2  # type: ignore
    print(f"Modified A1C: {config}")
