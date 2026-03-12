# cilpy/solver/chm/alpha_constraint.py
import math

from cilpy.solver.chm import ConstraintHandler
from cilpy.problem import Evaluation


class AlphaConstraintHandler(ConstraintHandler[float]):
    """
    Implements the alpha-constraint handling approach.

    This method compares solutions based on their satisfaction level `mu(x)`
    and a user-defined threshold `alpha`. It prioritizes feasible solutions
    (or those with a high satisfaction level) over fitness:
    - If both violate constraints beyond a certain level (alpha), the solution
      with the better objective score is better.
    - If both don't violate constraints beyond a certain level (alpha), and the
      degree of violation is equal, the solution with the better objective score
      is better.
    - Otherwise, if one violates constraints less, it's solution is better.
    """
    def __init__(self,
                 alpha: float,
                 b_inequality: float = 1.0,
                 b_equality: float = 1.0):
        """
        Initializes the AlphaConstraintHandler.

        Args:
            alpha (float): The satisfaction level threshold. Solutions with
                mu(x) >= alpha are treated as feasible.
            b_inequality (float): Normalization factor for inequality
                constraints.
            b_equality (float): Normalization factor for equality constraints.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha
        self.b_inequality = b_inequality
        self.b_equality = b_equality

    def _calculate_satisfaction(self, evaluation: Evaluation[float]) -> float:
        """Calculates the satisfaction level mu(x) for a given evaluation."""
        satisfaction_levels = []

        # Inequality constraints g(x) <= 0
        if evaluation.constraints_inequality:
            for g in evaluation.constraints_inequality:
                if g <= 0:
                    mu_g = 1.0
                elif 0 < g <= self.b_inequality:
                    mu_g = 1.0 - (g / self.b_inequality)
                else:
                    mu_g = 0.0
                satisfaction_levels.append(mu_g)

        # Equality constraints h(x) == 0 (relaxed to |h(x)| <= b_equality)
        if evaluation.constraints_equality:
            for h in evaluation.constraints_equality:
                h_abs = abs(h)
                if h_abs <= self.b_equality:
                    mu_h = 1.0 - (h_abs / self.b_equality)
                else:
                    mu_h = 0.0
                satisfaction_levels.append(mu_h)

        # If there are no constraints, satisfaction is maximal
        if not satisfaction_levels:
            return 1.0

        # mu(x) is the minimum of all individual satisfaction levels
        return min(satisfaction_levels)

    def is_better(
            self, eval_a: Evaluation[float], eval_b: Evaluation[float]
    ) -> bool:
        """Compares two solutions using the alpha-constraint rule."""
        mu_a = self._calculate_satisfaction(eval_a)
        mu_b = self._calculate_satisfaction(eval_b)

        # Case 1: Both solutions meet the alpha threshold (are "feasible")
        if mu_a >= self.alpha and mu_b >= self.alpha:
            return eval_a.fitness < eval_b.fitness

        # Case 2: They have the same satisfaction level
        if math.isclose(mu_a, mu_b):
            return eval_a.fitness < eval_b.fitness

        # Case 3: Otherwise, the one with the higher satisfaction level wins
        return mu_a > mu_b
