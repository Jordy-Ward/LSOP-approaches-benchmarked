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