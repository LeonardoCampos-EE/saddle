import numpy as np
import pandas as pd
from ..core.base import BaseMetaheuristicOptimizer
from ...functions.utils import clip_dataframe


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
        # shape -> (iterations, size, variables)
        self.r1 = np.random.random_sample(
            size=(self.iterations, self.size, len(self.variables))
        )
        self.r2 = np.random.random_sample(
            size=(self.iterations, self.size, len(self.variables))
        )

        # A(t) -> controls the step size of each wolf in the search space
        # shape -> (iterations, size, variables)
        a = self.a[:, np.newaxis, np.newaxis]
        self.A = (2 * a * self.r1) - a

        # C(t) -> controls the movement of each wolf towards the best solutions
        # shape -> (iterations, size, variables)
        self.C = 2 * self.r2

    def optimize(self) -> None:
        for t in range(self.iterations):
            # Get parameters for current iteration
            A = self.A[t]
            C = self.C[t]

            self.calculate_metric()
            best_indexes = self.population['metric'].argsort()
            alpha = self.population.loc[best_indexes[0], self.variables].to_numpy()
            beta = self.population.loc[best_indexes[1], self.variables].to_numpy()
            delta = self.population.loc[best_indexes[2], self.variables].to_numpy()

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

        D_alpha = np.abs(np.multiply(C, alpha) - population)
        D_beta = np.abs(np.multiply(C, beta) - population)
        D_delta = np.abs(np.multiply(C, delta) - population)

        # Calculate X_alpha, X_beta, X_delta
        X_alpha = alpha - np.multiply(A, D_alpha)
        X_beta = beta - np.multiply(A, D_beta)
        X_delta = delta - np.multiply(A, D_delta)

        # Update population
        population = (X_alpha + X_beta + X_delta) / 3.0

        population = clip_dataframe(
            population, upper=self.upper_bounds, lower=self.lower_bounds
        )
        self.population.loc[:, self.variables] = population
        return
