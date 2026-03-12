# cilpy/runner.py
"""The experiment runner: Orchestrates computational intelligence experiments.

This module provides the `ExperimentRunner` class, which is the primary tool for
setting up, executing, and logging benchmark experiments in a structured and
reproducible way.
"""

import time
import csv
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Sequence

from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver


class ExperimentRunner:
    """Orchestrates the execution of computational intelligence experiments.

    This class is the main entry point for running experiments in `cilpy`. It
    automates the process of applying multiple solver configurations to a set of
    problems, handling independent runs, iteration loops, and results logging.

    The runner systematically pairs each solver with each problem, creating a
    dedicated output file for each combination. This declarative approach allows
    users to define complex experiments with minimal boilerplate code.

    Example:
        .. code-block:: python

            from cilpy.problem.unconstrained import Sphere
            from cilpy.solver.ga import GA
            from cilpy.runner import ExperimentRunner

            # 1. Define the problems to test on
            problems = [Sphere(dimension=10)]

            # 2. Define the solver configurations to test
            solver_configs = [
                {
                    "class": GA,
                    "params": {
                        "name": "GA_HighMutation",
                        "population_size": 50,
                        "mutation_rate": 0.2,
                        "crossover_rate": 0.8,
                    }
                },
                {
                    "class": GA,
                    "params": {
                        "name": "GA_LowMutation",
                        "population_size": 50,
                        "mutation_rate": 0.05,
                        "crossover_rate": 0.8,
                    }
                }
            ]

            # 3. Initialize and run the experiment
            runner = ExperimentRunner(
                problems=problems,
                solver_configurations=solver_configs,
                num_runs=30,
                max_iterations=1000
            )
            runner.run_experiments()
    """

    def __init__(
        self,
        problems: Sequence[Problem],
        solver_configurations: List[Dict[str, Any]],
        num_runs: int,
        max_iterations: int,
    ):
        """
        Initializes the ExperimentRunner.

        Args:
            problems: A sequence of problem instances to be solved.
                Each object must implement the `Problem` interface.
            solver_configurations: A list of solver configurations.
                Each configuration is a dictionary specifying the solver class
                and its parameters. The `problem` parameter is injected
                automatically by the runner and should not be included.
            num_runs: The number of independent runs for each
                solver-problem pair.
            max_iterations: The number of iterations (`solver.step()` calls)
                per run.

        Solver Configuration Format:
            .. code-block:: python

                [
                    {
                        "class": YourSolverClass,
                        "params": {
                            "name": "UniqueSolverName",
                            "param1": value1,
                            # ... other solver hyperparameters
                        }
                    },
                    # ... more configurations
                ]
        """
        self.problems = problems
        self.solver_configurations = solver_configurations
        self.num_runs = num_runs
        self.max_iterations = max_iterations

    def run_experiments(self):
        """Executes the full suite of defined experiments.

        This method iterates through each problem and applies every configured
        solver. For each problem-solver pair, it performs `num_runs` independent
        runs, each lasting for `max_iterations`.

        Results are logged to separate CSV files, with each file named using the
        pattern: `{problem.name}_{solver_name}.out.csv`.
        """

        total_start_time = time.time()
        print("======== Starting All Experiments ========")

        # Ensure output directory exists
        os.makedirs("out", exist_ok=True)

        for problem in self.problems:
            print(f"\n--- Processing Problem: {problem.name} ---")
            for config in self.solver_configurations:
                solver_class = config["class"]
                solver_params = config["params"].copy()
                constraint_handler_config = config.get("constraint_handler")

                # Add the current problem to the solver's parameters
                current_solver_params = solver_params
                current_solver_params["problem"] = problem

                solver_name = current_solver_params.get("name")
                output_file_path = os.path.join(
                    "out", f"{problem.name}_{solver_name}.out.csv"
                )

                print(f"\n  -> Starting Experiment: {solver_name} on {problem.name}")
                print(
                    f"     Configuration: {self.num_runs} runs, {self.max_iterations} iterations/run."
                )
                print(f"     Results will be saved to: {output_file_path}")

                self._run_single_experiment(
                    solver_class,
                    current_solver_params,
                    output_file_path,
                    constraint_handler_config,
                )

        total_end_time = time.time()
        print("\n======== All Experiments Finished ========")
        print(f"Total execution time: {total_end_time - total_start_time:.2f}s")

    def _is_solution_feasible(self, evaluation: Evaluation, tolerance=1e-6) -> bool:
        """
        Checks if an evaluation corresponds to a feasible solution.

        A solution is feasible if all inequality constraints are <= 0 and all
        equality constraints are approximately == 0.

        Args:
            evaluation (Evaluation): The evaluation object to check.
            tolerance (float): The tolerance for checking equality constraints.

        Returns:
            bool: True if the solution is feasible, False otherwise.
        """
        if evaluation is None:
            return False

        # Check inequality constraints: g(x) <= 0
        if evaluation.constraints_inequality:
            if any(v > 0 for v in evaluation.constraints_inequality):
                return False

        # Check equality constraints: h(x) == 0
        if evaluation.constraints_equality:
            if any(abs(v) > tolerance for v in evaluation.constraints_equality):
                return False

        return True

    def _run_single_run(
        self,
        run_id: int,
        constraint_handler_config: Optional[Dict],
        solver_params: Dict,
        solver_class: Type[Solver],
        writer,
        summary_file_path: str,
    ):
        run_start_time = time.time()
        print(f"     --- Starting Run {run_id}/{self.num_runs} ---")

        constraint_handler = None
        if constraint_handler_config:
            handler_class = constraint_handler_config["class"]
            handler_params = constraint_handler_config.get("params", {})
            constraint_handler = handler_class(**handler_params)

        # Add the handler to the solver's parameters
        current_solver_params = solver_params.copy()
        current_solver_params["constraint_handler"] = constraint_handler

        # Re-instantiate the solver for each run to ensure independence
        solver = solver_class(**current_solver_params)

        # --- Safely get fitness bounds and set RE flag ---
        bounds_known = False  # Flag to track if we can calculate relative error
        f_max, fitness_range = 0, 0

        try:
            f_min, f_max = solver.problem.get_fitness_bounds()
            fitness_range = f_max - f_min
            if fitness_range > 0:
                bounds_known = True
            else:
                print(
                    "Warning: Fitness range is zero or invalid. Relative Error will not be calculated."
                )
        except NotImplementedError:
            # The method is not implemented, so we leave bounds_known as False
            print(
                "Info: get_fitness_bounds() not implemented for this problem. Skipping Relative Error."
            )

        # --- Run iterations ---
        relative_error_history = []
        for iteration in range(1, self.max_iterations + 1):
            solver.problem.begin_iteration()
            solver.step()
            result = solver.get_result()

            # --- Measure Accuracy and Relative Error ---
            if solver.problem.is_multi_objective():
                # --- Multi-Objective Case ---
                accuracy = []  # This will hold the Pareto front
                for objective in result:
                    evaluation = objective[1]
                    accuracy.append(evaluation.fitness)

                # Relative Error is not applicable for a list of objectives
                relative_error = ""
            else:
                # --- Single-Objective Case ---
                # Get the single best fitness value
                accuracy = result[0][1].fitness

                # Calculate Relative Error, knowing 'accuracy' is a float
                if bounds_known:
                    # This is the correct formula for MINIMIZATION where a value
                    # approaching f_min is better, and the result should approach 1.
                    relative_error = (f_max - accuracy) / fitness_range
                    relative_error_history.append(relative_error)
                else:
                    # If bounds are not known for this problem
                    relative_error = ""

            # --- Measure Feasibility ---
            # Safely get population evaluations
            try:
                all_evaluations = solver.get_population_evaluations()
                if all_evaluations:
                    num_feasible = sum(
                        1 for e in all_evaluations if self._is_solution_feasible(e)
                    )
                    feasibility = (num_feasible / len(all_evaluations)) * 100
                else:
                    feasibility = ""
            except NotImplementedError:
                # If the problem doesn't implement it, log empty strings
                feasibility = ""

            # --- Measure Diversity ---
            # Safely get population
            try:
                population = solver.get_population()
                diversity = 0.0
                if population:
                    pop_array = np.array(population)
                    ns = pop_array.shape[0]

                    # Calculate the centroid (mean vector) of the population
                    centroid = np.mean(pop_array, axis=0)

                    # Calculate the sum of squared Euclidean distances from the centroid
                    sum_of_squared_diffs = np.sum((pop_array - centroid) ** 2)
                    diversity = (1 / ns) * np.sqrt(sum_of_squared_diffs)
            except NotImplementedError:
                # If the problem doesn't implement it, log empty strings
                diversity = ""

            # Log the data for the current iteration
            writer.writerow(
                [run_id, iteration, accuracy, feasibility, diversity, relative_error]
            )

        # --- Calculate Relative Error Distance (P_RED) conditionally ---
        if bounds_known:
            b_vector = np.array(relative_error_history)
            nv = len(b_vector)
            sum_of_squares = np.sum((1 - b_vector) ** 2)
            relative_error_distance = (
                np.sqrt(sum_of_squares) / np.sqrt(nv) if nv > 0 else 0.0
            )
        else:
            # If we couldn't calculate P_RE, we can't calculate P_RED either
            relative_error_distance = ""

        run_end_time = time.time()
        final_result = solver.get_result()
        print(
            f"     Run {run_id} finished in {run_end_time - run_start_time:.2f}s. "
            f"Best fitness: {final_result[0][1].fitness if final_result else 'N/A'}"
        )

        # --- Log the final P_RED for the entire run ---
        red_output = (
            f"{relative_error_distance:.6f}"
            if isinstance(relative_error_distance, float)
            else "N/A (bounds unknown)"
        )
        print(f"     Relative Error Distance (P_RED) for Run {run_id}: {red_output}")

        with open(summary_file_path, "a", newline="") as f:
            summary_writer = csv.writer(f)
            problem_name = solver.problem.name
            solver_name = solver.name
            summary_writer.writerow(
                [problem_name, solver_name, run_id, relative_error_distance]
            )

    def _run_single_experiment(
        self,
        solver_class: Type[Solver],
        solver_params: Dict,
        output_file: str,
        constraint_handler_config: Optional[Dict] = None,
    ):
        """Runs and logs a single experiment for a given solver on a problem.

        This internal method is called by `run_experiments`. It handles the
        instantiation of the solver for each of the `num_runs` and manages the
        iteration loop and CSV writing for a single problem-solver pair.

        The output CSV file contains the following columns:
        - `run_id`: The ID of the independent run (from 1 to `num_runs`).
        - `iteration`: The current iteration number (from 1 to `max_iterations`).
        - `accuracy`: Fitness of best solution(s) found so far.
        - `feasibility`: The percentage of solutions that are feasible.
        - `diversity`: A measure of the diversity at this iteration.
        - `relative_error`: The relative error at this iteration.

        Args:
            solver_class: The solver class to be instantiated.
            solver_params: The parameters for initializing the solver (including
                the `problem` instance).
            output_file: The path to the output CSV file.
        """
        # Prepare output files
        summary_file_path = output_file.replace(".out.csv", ".summary.out.csv")
        main_header = [
            "run_id",
            "iteration",
            "accuracy",
            "feasibility",
            "diversity",
            "relative_error",
        ]
        summary_header = [
            "problem_name",
            "solver_name",
            "run_id",
            "relative_error_distance",
        ]

        with open(summary_file_path, "w", newline="") as f_summary:
            summary_writer = csv.writer(f_summary)
            summary_writer.writerow(summary_header)

        experiment_start_time = time.time()

        with open(output_file, "w", newline="") as f_main:
            writer = csv.writer(f_main)
            writer.writerow(main_header)

            # Run experiments
            for run_id in range(1, self.num_runs + 1):
                self._run_single_run(
                    run_id,
                    constraint_handler_config,
                    solver_params,
                    solver_class,
                    writer,
                    summary_file_path,
                )

        experiment_end_time = time.time()
        solver_name = solver_params.get("name", solver_class.__name__)
        problem_name = solver_params["problem"].name
        print(
            f"  -> Experiment for {solver_name} on {problem_name} "
            f"finished in {experiment_end_time - experiment_start_time:.2f}s."
        )
        print(f"     Summary results saved to: {summary_file_path}")


if __name__ == "__main__":
    from cilpy.problem.unconstrained import Sphere, Ackley
    from cilpy.solver.pso import PSO
    from cilpy.solver.ga import GA

    # # --- 1. Define the Problems ---
    problems_to_run = [Sphere(dimension=3)]

    # --- 2. Define the Solver Configurations ---
    solver_configs = [
        {
            "class": PSO,
            "params": {
                "name": "PSO_Standard",
                "swarm_size": 30,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
        }
    ]

    # --- 3. Define the Experiment parameters ---
    number_of_runs = 5
    max_iter = 1000

    # --- 4. Create and run the experiments ---
    runner = ExperimentRunner(
        problems=problems_to_run,
        solver_configurations=solver_configs,
        num_runs=number_of_runs,
        max_iterations=max_iter,
    )
    runner.run_experiments()
