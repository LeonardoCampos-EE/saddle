from typing import Callable
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ..core.base import BaseMetaheuristicOptimizer
from ...types import ArrayLike, ParametricFunction
import numpy as np
import pandas as pd

from ...functions.utils import clip_dataframe
from ...functions.visualization import plot_contour


class ParticleSwarmOptimizer(BaseMetaheuristicOptimizer):
    velocity: np.ndarray
    w: float  # intertia weight constant
    c1: float  # cognitive constant
    c2: float  # social constant
    p_best: pd.DataFrame  # best historical points for the whole population
    g_best: pd.DataFrame  # best solution found

    def __init__(
        self,
        variables: list[str],
        upper_bounds: ArrayLike,
        lower_bounds: ArrayLike,
        iterations: int,
        size: int,
        fn_obj: Callable | ParametricFunction,
        constraints: dict[str, Callable | ParametricFunction] | None = None,
        seed: int = 42,
        w: float = 0.8,
        c1: float = 0.1,
        c2: float = 0.1,
    ) -> None:
        self.w = w
        self.c1 = c1
        self.c2 = c2
        super().__init__(
            variables,
            upper_bounds,
            lower_bounds,
            iterations,
            size,
            fn_obj,
            constraints,
            seed,
        )

    def populate(self, size: int) -> None:
        super().populate(size)
        self.velocity = np.zeros_like(self.population.loc[:, self.variables].to_numpy())
        self._initialize_parameters()
        self._initialize_history()

    def _initialize_history(self) -> None:
        self.velocity_history: list[np.ndarray] = []
        self.g_best_history = pd.DataFrame()
        self.p_best_history = pd.DataFrame()
        self.population_history = pd.DataFrame()

    def _update_history(self, t: int) -> None:
        self.velocity_history.append(self.velocity.copy())
        gbest = self.g_best.copy()
        gbest['iteration'] = t
        self.g_best_history = pd.concat([self.g_best_history, gbest])

        pbest = self.p_best.copy()
        pbest['iteration'] = t
        self.p_best_history = pd.concat([self.p_best_history, pbest])

        pop = self.population.copy()
        pop['iteration'] = t
        self.population_history = pd.concat([self.population_history, pop])

    def _initialize_parameters(self) -> None:
        # r1 and r2 -> random numbers between 0 and 1
        # shape -> (iterations, size, variables)
        self.r1 = np.random.random_sample(
            size=(self.iterations, self.size, len(self.variables))
        )
        self.r2 = np.random.random_sample(
            size=(self.iterations, self.size, len(self.variables))
        )

        # P best -> best historical points for the whole population
        self.p_best = self.population.copy()
        self.p_best.loc[:, 'metric'] = np.inf

        # G best -> best solution so far
        self.g_best = self.population.loc[[0], :].copy()
        self.g_best.loc[:, 'metric'] = np.inf

    def optimize(self) -> None:
        for t in range(self.iterations):
            # Get parameters for current iteration
            self.calculate_metric()
            self._get_p_best()
            # Get the index of the best member of the population
            best_index = self.population['metric'].argsort()[0]
            self._get_g_best(best_index=best_index)

            self.update(t=t)

    def _get_p_best(self) -> None:
        # Current population metric
        pop_metric = self.population.loc[:, 'metric'].to_numpy()

        # P best metric
        p_best_metric = self.p_best.loc[:, 'metric'].to_numpy()

        mask = pop_metric < p_best_metric

        # Change p_best particles with better particles from current population
        self.p_best.loc[mask] = self.population.loc[mask]

    def _get_g_best(self, best_index: int) -> None:
        best = self.population.iloc[[best_index]]
        if best['metric'].iloc[0] < self.g_best['metric'].iloc[0]:
            self.g_best = best.reset_index(drop=True)

    def update(self, t: int) -> None:
        self._update_velocity(t=t)
        self._update_history(t)
        self._update_population()

        self.population.loc[:, self.variables] = clip_dataframe(
            self.population.loc[:, self.variables],
            upper=self.upper_bounds,
            lower=self.lower_bounds,
        )

    def _update_velocity(self, t: int) -> None:
        p_best = self.p_best.loc[:, self.variables].to_numpy()
        pop = self.population.loc[:, self.variables].to_numpy()
        g_best = self.g_best.loc[:, self.variables].to_numpy()

        self.velocity = (
            self.w * self.velocity
            + np.multiply(self.c1 * self.r1[t], p_best - pop)
            + np.multiply(self.c2 * self.r2[t], g_best - pop)
        )

    def _update_population(self) -> None:
        self.population.loc[:, self.variables] += self.velocity

    def plot_contours(self, optima: list[float]) -> Figure:
        fig, ax = plot_contour(
            fun=self.fn_obj,
            lower=self.lower_bounds.to_list(),
            upper=self.upper_bounds.to_list(),
            optima=optima,
        )

        gbest_plot = ax.scatter(
            x=self.g_best_history[self.variables[0]].iloc[0],
            y=self.g_best_history[self.variables[1]].iloc[0],
            c='red',
            marker='x',
            s=10,
            label='G Best',
        )
        pop = self.population_history.loc[self.population_history['iteration'] == 0][
            self.variables
        ].to_numpy()
        pop_plot = ax.scatter(
            pop[:, 0],
            pop[:, 1],
            c='blue',
            marker='o',
            alpha=0.5,
            label='Population',
        )
        vel = self.velocity_history[0]
        vel_plot = ax.quiver(
            pop[:, 0],
            pop[:, 1],
            vel[:, 0],
            vel[:, 1],
            color='blue',
            width=0.005,
            angles='xy',
            scale_units='xy',
            scale=1,
        )

        def animate(i):
            ax.set_title(f'Iteration {i}')
            pop = self.population_history.loc[
                self.population_history['iteration'] == i
            ][self.variables].to_numpy()
            vel = self.velocity_history[i]
            pop_plot.set_offsets(pop)
            vel_plot.set_offsets(pop)
            vel_plot.set_UVC(vel[:, 0], vel[:, 1])
            gbest_plot.set_offsets(self.g_best_history.iloc[i][self.variables])
            plt.legend()
            return ax, pop_plot, vel_plot, gbest_plot

        anim = FuncAnimation(
            fig,
            animate,
            frames=list(range(1, len(self.velocity_history))),
            interval=500,
            blit=False,
            repeat=True,
        )
        anim.save('pso.gif', dpi=120, writer='pillow')

        return fig
