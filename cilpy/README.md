<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/willieloea/cilpy/refs/heads/master/docs/logo_cilpy_light.svg">
  <img src="https://raw.githubusercontent.com/willieloea/cilpy/refs/heads/master/docs/logo_cilpy_dark.svg" alt="cilpy logo" width="50%">
</picture>

# `cilpy`: A Computational Intelligence Library for Python.

[![PyPI Version](https://img.shields.io/pypi/v/cilpy.svg)](https://pypi.org/project/cilpy/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![documentation](https://img.shields.io/github/actions/workflow/status/willieloea/cilpy/pages/pages-build-deployment?label=documentation)](https://willieloea.github.io/cilpy/)

</div>

---

`cilpy` is a Python library for computational intelligence, with a focus on
nature-inspired optimization algorithms. It provides a flexible and extensible
framework for researchers and practitioners to experiment with various
optimization problems, including constrained, multi-objective, and many-
objective optimization.

`cilpy` is structured into distinct components for problem generation
(`cilpy.problem`), solving (`cilpy.solver`), and algorithm comparison
(`cilpy.compare`). Users can easily introduce new problems or solvers by
implementing the provided abstract interfaces. These interfaces were designed
with the ability to solve a wide variety of optimization problems in mind,
including:
  * Single and multi-objective optimization
  * Constrained and unconstrained problems
  * Static and dynamic optimization problems

### Installing `cilpy`
`cilpy` can be installed using pip:

```
pip install cilpy
```

### Usage Example
Here is a basic pseudocode example of how to set up and run an experiment with
`cilpy`. This demonstrates how to define a problem, configure a solver, and use
the `ExperimentRunner` (which orchestrates interaction between the problem and
the solver) to execute the optimization task.

```python
from cilpy.runner import ExperimentRunner
from cilpy.problem import SomeProblem
from cilpy.solver import SomeSolver

# --- 1. Define the Problems ---
dim = 3
dom = (-5.12, 5.12)
problems_to_run = [
    SomeProblem(dimension=dim, domain=dom)
]

# --- 2. Define the Solvers and their parameters ---
# Note: Although the `solver` interface requires the `problem` to be specified
# during solver initialization, it is omitted here as the runner will assign it.
solver_configs = [
    {
        "class": SomeSolver,
        "params": {
            "name": "MySolver",
            # Other solver parameters
        }
    },
]

# --- 3. Define the Experiment parameters ---
number_of_runs = 30 # how many times an experiment should be repeated
max_iter = 5000     # how many iterations per experiment

# --- 4. Create and run the experiments ---
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()
```

See [examples](examples/) for actual examples of how `cilpy` is used.

### Documentation

The full documentation can be found in the `docs/` folder and can be served
locally:
```bash
mkdocs serve
```

### Contributing

We welcome contributions to `cilpy`! If you are interested in contributing, please read our [contributing guidelines](./docs/dev/index.md) for details on our code of conduct and the process for submitting pull requests.
