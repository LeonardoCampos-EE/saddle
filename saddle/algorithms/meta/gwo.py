import numpy as np
import pandas as pd
from ..core.base import BaseMetaheuristicOptimizer


class GreyWolfOptimizer(BaseMetaheuristicOptimizer):
    def populate(self, size: int) -> None:
        super().populate(size)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        # Array containing the number of iterations
        iter_array = np.arange(self.iterations)

        # a(t) -> goes from 2 to 0 over the iterations
        # shape -> (iterations,)
        self.a = 2 - (iter_array * (2.0 / self.iterations))

        # r1 and r2 -> random numbers between 0 and 1
        # shape -> (iterations, variables, size)
        self.r1 = np.random.random_sample(
            size=(self.iterations, len(self.variables), self.size)
        )
        self.r2 = np.random.random_sample(
            size=(self.iterations, len(self.variables), self.size)
        )

        # A(t) -> controls the step size of each wolf in the search space
        # shape -> (iterations, variables, size)
        a = self.a[:, np.newaxis, np.newaxis]
        self.A = (2 * a * self.r1) - a

        # C(t) -> controls the movement of each wolf towards the best solutions
        # shape -> (iterations, variables, size)
        self.C = 2 * self.r2

    def optimize(self) -> None:
        for t in range(self.iterations):
            # Get parameters for current iteration
            A = self.A[t]
            C = self.C[t]

            self.calc_fn_obj()
            self.population = self.population.sort(by='fn_obj')
            alpha = self.population.iloc[0]
            beta = self.population.iloc[1]
            delta = self.population.iloc[2]

            self.update(A, C, alpha, beta, delta)

    def update(
        self,
        A: np.ndarray,
        C: np.ndarray,
        alpha: pd.Series,
        beta: pd.Series,
        delta: pd.Series,
    ) -> None:
        # Calculate D_alpha, D_beta, D_delta
        population = self.population.loc[:, self.variables]

        D_alpha = (alpha * C - population).abs()
        D_beta = (beta * C - population).abs()
        D_delta = (delta * C - population).abs()

        # Calculate X_alpha, X_beta, X_delta
        X_alpha = alpha - (A * D_alpha)
        X_beta = beta - (A * D_beta)
        X_delta = delta - (A * D_delta)

        # Update population
        population = (X_alpha + X_beta + X_delta) / 3.0

        self.population[:, self.variables] = population
        return
