import math
from typing import List, Tuple
from cilpy.problem import Problem, Evaluation

class Elliptic(Problem[List[float], float]):
    
    """High conditioned elliptic function
    
    Unimodal, benchmark function for LSOP PSO. Contains high condition number
    """
    
    # Constrcutor for Elliptic class and hands config of problem up to parent class Problem
    # we make a tuple with two lists for the bounds. Where each tuple represents bounds per dimension. Allows for non uniform bounds per dimension
    def __init__(self, dimension: int = 100, domain: Tuple[float,float] = (-100.0, 100.0)):
        lower_bounds = [domain[0] * dimension]
        upper_bounds = [domain[1] * dimension]
        super().__init__(
            dimension = dimension,
            bounds=(lower_bounds, upper_bounds),
            name="Elliptic"
        )
    
    # evaluate the current candidate solution for all dimensions
    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        n = self.dimension
        fitness = 0.0
        for i, x in enumerate(solution):
            exponent = 6.0 * i / (n - 1) if n > 1 else 0.0
            fitness = fitness + (10.0 ** exponent) * (x ** 2)
        return Evaluation(fitness=fitness)
    
    # tells the problem solver and experiment runner if this problem changes over time
    # the first boolean tells you if the objective function itself changes between peaks
    # the second boolean tells you if the bounds of the search space change over time
    # the elliptic function has a completely static landscape
    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)
    
    # tells the framework if this problem has multiple competing objective functions or only one
    def is_multi_objective(self) -> bool:
        return False
    
    # tells the framework what range of fitness values this problem can produce. A lower and upper bound
    # 0.0 is the best possible fitness
    # the worst case takes the largest abs value possible in any dimension
    # imagine every single dimension sits at the furthest from the origin. Replace the x with limit
    # this is the worst possible case
    def get_fitness_bounds(self) -> Tuple[float, float]:
        best = 0.0
        lb = self.bounds[0][0]
        ub = self.bounds[1][0]
        limit = max(abs(lb), abs(ub))
        #in the worst case all x i's are at max, last term dominates with factor 10^6
        worst = sum((10.0 ** (6.0 * i / (self.dimension - 1))) * (limit ** 2)
                    for i in range(self.dimension))
        return (best, worst)
    
class Rosenbrock(Problem[List[float]], float):
    """Rosenbrock (Banana) function
    
    Unimodal for n<=3, multimodal for n>3, non seperable benchmark function. The global minimum is very hard to converge to. 
    Global minimum f(1,....,1) = 0
    """
    
    def __init__(self, dimension: int = 100, domain: Tuple[float,float] = (-30.0, 30.0)):
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension,
            bounds=(lower_bounds, upper_bounds),
            name="Rosenbrock"
        )
        
    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        fitness = sum(
            100.00 * (solution[i + 1] - solution[i] ** 2) ** 2 + (solution[i] - 1.0) ** 2
            for i in range(self.dimension - 1)
        )
        return Evaluation(fitness=fitness)
    
    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False

    # estimate of the worst possible value
    def get_fitness_bounds(self) -> Tuple[float, float]:
        best = 0.0
        lb = self.bounds[0][0]
        ub = self.bounds[1][0]
        limit = max(abs(lb), abs(ub))
        worst = (self.dimension - 1) * (100.0 * limit ** 4 + limit ** 2)
        return (best, worst)
    
class Rastrigin(Problem[List[float], float]):
    """ Rastrigin function.
    
    High multi modal function with large number local minima. One of the most
    challenging test functions for LSOP's

    Global minimum: f(0,...,0) = 0.
    """

    def __init__(self, dimension: int = 100, domain: Tuple[float, float] = (-5.12, 5.12)):
        lower_bounds = [domain[0]] * dimension
        upper_bounds = [domain[1]] * dimension
        super().__init__(
            dimension=dimension,
            bounds=(lower_bounds, upper_bounds),
            name="Rastrigin"
        )

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        n = self.dimension
        fitness = 10.0 * n + sum(
            x ** 2 - 10.0 * math.cos(2.0 * math.pi * x)
            for x in solution
        )
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False

    def get_fitness_bounds(self) -> Tuple[float, float]:
        best = 0.0
        lb = self.bounds[0][0]
        ub = self.bounds[1][0]
        limit = max(abs(lb), abs(ub))
        worst = self.dimension * (limit ** 2 + 10.0)
        return (best, worst)