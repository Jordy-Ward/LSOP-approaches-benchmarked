"""
PSO for large scale Optimisation

Compares four PSO variants across five LSOP benchmark functions at three dimensionalities (D=100, 500, 100)
with 30 indpendent runs each

The algorithms:
1. Standard PSO                     - Cononical PSO with uniform random initialisation
2. Stochastic Scaling PSO           - Stochastic scaling with linearly increasing groups
3. Subspace initialisation PSO      - subspace based initialisation
4. Hybrid PSO                       - subsapce initialisation + stochastic scaling

Problems:
1. Sphere                           - (unimodal, separable)
2. Elliptic                         - (unimodal, non-saperable)
3. Rastrigin                        - (multimodal, separable)
4. Rosenbrock                       - (unimodal, non-separable, narrow valley)
5. Schwefel 1.2                     - (unimodal, non-separable)
"""

from cilpy.runner import ExperimentRunner
from cilpy.problem.unconstrained import Sphere
from cilpy.problem.lsop import Elliptic, Rastrigin, Rosenbrock, Schwefel12
from cilpy.solver.pso import PSO
from cilpy.solver.lsop_pso import StochasticScalingPSO, SubspaceInitPSO, HybridPSO

# experiment parameters
NUM_RUNS                = 3
MAX_ITERATIONS          = 100
SWARM_SIZE              = 30
W                       = 0.729844
C1                      = 1.49618
C2                      = 1.49618
DIMENSIONS              = [10]

# build the problem
problems = []

#for each dimension create all 5 problems to solve.
for D in DIMENSIONS:
    problems = problems + [
        Sphere(dimension=D,         domain=(-100.0, 100.0)),
        Elliptic(dimension=D,       domain=(-100.0, 100.0)),
        Rastrigin(dimension=D,      domain=(-5.12, 5.12)),
        Rosenbrock(dimension=D,     domain=(-30.0, 30.0)),
        Schwefel12(dimension=D,     domain=(-100.0, 100.0))
    ]
    
# name all problems uniquely
for p in problems:
    p.name = f"{p.name}_D{p.dimension}"
    
# individual configs for each solver
# clean data driven way to manage multiple algorithm variants
solver_configs = [
    # 1. Standard PSO (baseline)
    {
        "class": PSO,
        "params": {
            "name": "StandardPSO",
            "swarm_size": SWARM_SIZE,
            "w": W, "c1": C1, "c2": C2,
        },
    },
    # 2. Stochastic Scaling PSO
    {
        "class": StochasticScalingPSO,
        "params": {
            "name": "StochasticScalingPSO",
            "swarm_size": SWARM_SIZE,
            "w": W, "c1": C1, "c2": C2,
            "max_iterations": MAX_ITERATIONS,
            # k_max defaults to D; the runner injects 'problem' before __init__,
            # so D is available at construction time.
        },
    },
    # 3. Subspace Initialisation PSO (u = 1 seed vector)
    {
        "class": SubspaceInitPSO,
        "params": {
            "name": "SubspaceInitPSO",
            "swarm_size": SWARM_SIZE,
            "w": W, "c1": C1, "c2": C2,
            "seed_set_size": 1,
        },
    },
    # 4. Hybrid PSO
    {
        "class": HybridPSO,
        "params": {
            "name": "HybridPSO",
            "swarm_size": SWARM_SIZE,
            "w": W, "c1": C1, "c2": C2,
            "max_iterations": MAX_ITERATIONS,
            "seed_set_size": 1,
        },
    },
]
# run
if __name__ == "__main__":
    runner = ExperimentRunner(
        problems=problems,
        solver_configurations=solver_configs,
        num_runs=NUM_RUNS,
        max_iterations=MAX_ITERATIONS,
    )
    runner.run_experiments()
    print("\nResults saved to out/ directory")

