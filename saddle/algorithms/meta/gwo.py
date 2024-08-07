import numpy as np
import pandas as pd
from ..core.base import BaseMetaheuristicOptimizer
from ...functions.utils import clip_dataframe


class GreyWolfOptimizer(BaseMetaheuristicOptimizer):
    def populate(self, size: int) -> None:
        super().populate(size)
        self._initialize_parameters()
        self._initialize_history()

    def _initialize_parameters(self) -> None:
        # Array containing the number of iterations
        iter_array = np.arange(self.iterations)

        # a(t) -> goes from 2 to 0 over the iterations
        # shape -> (iterations,)
        self.a = 2 - (iter_array * (2.0 / self.iterations))

        # r1 and r2 -> random numbers between 0 and 1
        # shape -> (iterations, size, variables)
        def r():
            return np.random.random_sample(
                size=(self.iterations, self.size, len(self.variables))
            )

        # A(t) -> controls the step size of each wolf in the search space
        # shape -> (iterations, size, variables)
        a = self.a[:, np.newaxis, np.newaxis]
        self.A_alpha = (2 * a * r()) - a
        self.A_beta = (2 * a * r()) - a
        self.A_delta = (2 * a * r()) - a

        # C(t) -> controls the movement of each wolf towards the best solutions
        # shape -> (iterations, size, variables)
        self.C = 2 * r()

    def _initialize_history(self) -> None:
        self.alpha_history = pd.DataFrame()
        self.beta_history = pd.DataFrame()
        self.delta_history = pd.DataFrame()
        self.population_history = pd.DataFrame()

    def _update_history(self, t: int, best_indexes: np.ndarray) -> None:
        alpha = self.population.iloc[[best_indexes[0]]].copy()
        beta = self.population.iloc[[best_indexes[1]]].copy()
        delta = self.population.iloc[[best_indexes[2]]].copy()

        alpha["iteration"] = t
        self.alpha_history = pd.concat([self.alpha_history, alpha])

        beta["iteration"] = t
        self.beta_history = pd.concat([self.beta_history, beta])

        delta["iteration"] = t
        self.delta_history = pd.concat([self.delta_history, delta])

        pop = self.population.copy()
        pop["iteration"] = t
        self.population_history = pd.concat([self.population_history, pop])

    def optimize(self) -> None:
        for t in range(self.iterations):
            self.calculate_metric(t=t)
            best_indexes = self.population["metric"].argsort().to_numpy()
            alpha = self.population.loc[best_indexes[0], self.variables].to_numpy()
            beta = self.population.loc[best_indexes[1], self.variables].to_numpy()
            delta = self.population.loc[best_indexes[2], self.variables].to_numpy()
            self._update_history(t, best_indexes)

            self.update(alpha, beta, delta, t)

    def update(
        self, alpha: np.ndarray, beta: np.ndarray, delta: np.ndarray, t: int
    ) -> None:
        # Calculate D_alpha, D_beta, D_delta
        population = self.population.loc[:, self.variables]

        D_alpha = np.abs(np.multiply(self.C[t], alpha) - population)
        D_beta = np.abs(np.multiply(self.C[t], beta) - population)
        D_delta = np.abs(np.multiply(self.C[t], delta) - population)

        # Calculate X_alpha, X_beta, X_delta
        X_alpha = alpha - np.multiply(self.A_alpha[t], D_alpha)
        X_beta = beta - np.multiply(self.A_beta[t], D_beta)
        X_delta = delta - np.multiply(self.A_delta[t], D_delta)

        # Update population
        population = (X_alpha + X_beta + X_delta) / 3.0

        # if not self.constraints:
        population = clip_dataframe(
            population, upper=self.upper_bounds, lower=self.lower_bounds
        )
        self.population.loc[:, self.variables] = population
        return
