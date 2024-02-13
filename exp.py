from collections.abc import Callable
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from saddle.core.parametric_function import ParametricFunction
from saddle.core.types import ArrayLike
from saddle.examples.power_dispatch.problem import ThreeGenerators
from saddle.functions.converters import convert_bounds_to_series


def cost(pop: pd.DataFrame, params: pd.DataFrame) -> pd.Series:
    p = pop.values.T
    a = np.expand_dims(params["a"].values, axis=1)  # type: ignore
    b = np.expand_dims(params["b"].values, axis=1)  # type: ignore
    c = np.expand_dims(params["c"].values, axis=1)  # type: ignore
    e = np.expand_dims(params["e"].values, axis=1)  # type: ignore
    f = np.expand_dims(params["f"].values, axis=1)  # type: ignore
    _min = np.expand_dims(params["min"].values, axis=1)  # type: ignore
    _cost = a * p**2 + b * p + c + np.abs(e * np.sin(f * (_min - p)))
    return pd.Series(
        np.sum(
            _cost,
            axis=0,
        ),
    )


def demand_constraint(
    pop: pd.DataFrame,
    multipliers: pd.DataFrame,
    demand: float,
) -> pd.Series:
    p = pop.values
    v = multipliers["v"].values
    return pd.Series((p.sum(axis=1) - demand) * v)


def min_power_constraint(
    pop: pd.DataFrame,
    multipliers: pd.DataFrame,
    params: pd.DataFrame,
) -> pd.Series:
    u_cols = [col for col in multipliers.columns if "u_min" in col]
    u = multipliers[u_cols].values.T
    p = pop.values.T
    _min = np.expand_dims(params["min"].values, axis=1)  # type: ignore
    violation = (_min - p) * u
    violation = np.sum(violation.T, axis=1)
    return pd.Series(violation)


def max_power_constraint(
    pop: pd.DataFrame,
    multipliers: pd.DataFrame,
    params: pd.DataFrame,
) -> pd.Series:
    u_cols = [col for col in multipliers.columns if "u_max" in col]
    u = multipliers[u_cols].values.T
    p = pop.values.T
    _max = np.expand_dims(params["max"].values, axis=1)  # type: ignore
    violation = (p - _max) * u
    violation = np.sum(violation.T, axis=1)
    return pd.Series(violation)


class PSOPrimalDual:
    population: pd.DataFrame = pd.DataFrame()
    seed: int
    penalties: dict[str, np.ndarray]
    lagrangian: pd.Series
    w: float = 0.8
    c1: float = 0.1
    c2: float = 0.1

    def __init__(
        self,
        variables: list[str],
        upper_bounds: ArrayLike,
        lower_bounds: ArrayLike,
        iterations: int,
        size: int,
        fn_obj: Callable | ParametricFunction,
        constraints: dict[str, Callable | ParametricFunction],
        params: dict[str, float],
        seed: int = 42,
        w: float = 0.8,
        c1: float = 0.1,
        c2: float = 0.1,
    ) -> None:
        self.w = w
        self.c1 = c1
        self.c2 = c2

        u_min = [f"u_min_{i}" for i in range(len(variables))]
        u_max = [f"u_max_{i}" for i in range(len(variables))]

        self.lagrangian_variables = u_min + u_max + ["v"]
        self.params = params

        self.variables = variables
        self.constraints = constraints
        self.columns = (
            self.variables
            + ["fn_obj"]
            + list(self.constraints.keys())
            + ["lagrangian"]
            + ["gap"]
            + ["metric"]
        )
        self.upper_bounds = convert_bounds_to_series(upper_bounds, self.variables)
        self.lower_bounds = convert_bounds_to_series(lower_bounds, self.variables)
        self.iterations = iterations
        self.fn_obj = fn_obj
        self.seed = seed
        self.size = size
        if seed:
            np.random.seed(self.seed)

        self.f_obj_penalties = np.linspace(10000, 100, num=iterations)
        self.gap_penalties = np.linspace(1000, 10000, num=iterations)
        self.lagrangian_penalties = np.linspace(1000, 1, num=iterations)
        self.populate(self.size)
        self._initialize_history()

    def handle_u_min(self):
        u_cols = [col for col in self.lagrangian_multipliers.columns if "u_min" in col]
        p = self.population.loc[:, self.variables].values.T
        _min = np.expand_dims(self.params["params"]["min"].values, axis=1)  # type: ignore
        violation = _min - p

        for i, col in enumerate(u_cols):
            _violation = violation[i]
            mask = _violation != 0
            self.lagrangian_multipliers.loc[mask, col] = 0.0

    def handle_u_max(self):
        u_cols = [col for col in self.lagrangian_multipliers.columns if "u_max" in col]
        p = self.population.loc[:, self.variables].values.T
        _max = np.expand_dims(self.params["params"]["max"].values, axis=1)  # type: ignore
        violation = p - _max
        for i, col in enumerate(u_cols):
            _violation = violation[i]
            mask = _violation != 0
            self.lagrangian_multipliers.loc[mask, col] = 0.0

    def populate(self, size: int) -> None:
        self.size = size
        upp = self.upper_bounds.to_numpy()
        low = self.lower_bounds.to_numpy()
        population: np.ndarray = (upp - low) * np.random.uniform(
            size=(size, len(self.variables)),
        ) + low

        fn_obj = np.zeros(shape=(size, 1))
        lagrangian = np.zeros(shape=(size, 1))
        gap = np.zeros(shape=(size, 1))

        metric = fn_obj.copy()
        constraints = np.zeros(shape=(size, len(self.constraints)))
        population = np.hstack(
            [population, fn_obj, constraints, lagrangian, gap, metric],
        )
        self.population = pd.DataFrame(population, columns=self.columns)
        self.lagrangian_multipliers = pd.DataFrame(
            10 * np.abs(np.random.uniform(size=(size, len(self.lagrangian_variables)))),
            columns=self.lagrangian_variables,
        )
        self._initialize_parameters()

    def _calculate_fn_obj(self) -> None:
        fn_values: pd.Series = self.fn_obj(self.population.loc[:, self.variables])
        self.population.loc[:, "fn_obj"] = fn_values

    def _calculate_constraints(self) -> None:
        for name, constraint in self.constraints.items():
            constraint_values: pd.Series = constraint(
                self.population.loc[:, self.variables],
            )
            pen = 10000 if "demand" in name else 1
            self.population.loc[:, name] = pen * constraint_values

    def calculate_metric(self, t: int) -> None:
        self._calculate_fn_obj()
        self._calculate_constraints()
        self._calculate_lagrangian()
        metric = (
            2 * self.population.loc[:, ["fn_obj"]].to_numpy()
            + 1 / self.population.loc[:, ["lagrangian"]].to_numpy()
            + 1 * self.population.loc[:, ["gap"]].to_numpy()
            + np.expand_dims(
                self.population.loc[:, list(self.constraints.keys())]
                .to_numpy()
                .sum(axis=1),
                axis=1,
            )
        )
        self.population.loc[:, "metric"] = metric

    def _calculate_lagrangian(self) -> None:
        self.handle_u_min()
        self.handle_u_max()
        demand = demand_constraint(
            self.population.loc[:, self.variables],
            self.lagrangian_multipliers,
            self.params["demand"],
        )
        p_min = min_power_constraint(
            self.population.loc[:, self.variables],
            self.lagrangian_multipliers,
            self.params["params"],  # type: ignore
        )
        p_max = max_power_constraint(
            self.population.loc[:, self.variables],
            self.lagrangian_multipliers,
            self.params["params"],  # type: ignore
        )
        obj = self.fn_obj(self.population.loc[:, self.variables])

        lagrangian = obj + demand + p_min + p_max

        self.population.loc[:, "lagrangian"] = lagrangian
        self.population.loc[:, "gap"] = np.abs(obj - lagrangian)

    def _initialize_history(self) -> None:
        self.f_obj_history = []
        self.lagrangian_history = []
        self.gap_history = []
        self.metric_history = []

    def _initialize_parameters(self) -> None:
        # r1 and r2 -> random numbers between 0 and 1
        self.r1 = np.random.random_sample(size=(self.iterations, 1))
        self.r2 = np.random.random_sample(size=(self.iterations, 1))

        # P best -> best historical points for the whole population
        self.p_best = self.population.copy()
        self.p_best_lagrangian = self.lagrangian_multipliers.copy()
        self.p_best.loc[:, "metric"] = np.inf

        # G best -> best solution so far
        self.g_best = self.population.loc[[0], :].copy()
        self.g_best_lagrangian = self.lagrangian_multipliers.iloc[[0]].copy()
        self.g_best.loc[:, "metric"] = np.inf

        self.velocity = np.zeros_like(
            self.population.loc[:, self.variables].to_numpy(),
        )
        self.lagrangian_velocity = np.zeros_like(self.lagrangian_multipliers.to_numpy())

    def _get_p_best(self) -> None:
        # Current population metric
        pop_metric = self.population.loc[:, "metric"].to_numpy()

        # P best metric
        p_best_metric = self.p_best.loc[:, "metric"].to_numpy()

        mask = pop_metric < p_best_metric

        # Change p_best particles with better particles from current population
        self.p_best.loc[mask] = self.population.loc[mask]
        self.p_best_lagrangian[mask] = self.lagrangian_multipliers[mask]

    def _get_g_best(self, best_index: int) -> None:
        best = self.population.iloc[[best_index]]
        if best["metric"].iloc[0] < self.g_best["metric"].iloc[0]:
            self.g_best = best.reset_index(drop=True)
            self.g_best_lagrangian = self.lagrangian_multipliers.iloc[
                [best_index]
            ].reset_index(drop=True)
            self.best_multipliers = self.lagrangian_multipliers.iloc[
                [best_index]
            ].reset_index(drop=True)

        self.f_obj_history.append(self.g_best["fn_obj"])
        self.lagrangian_history.append(self.g_best["lagrangian"])
        self.gap_history.append(self.g_best["gap"])
        self.metric_history.append(self.g_best["metric"])

    def _update_velocity(self, t: int) -> None:
        p_best = self.p_best.loc[:, self.variables].to_numpy()
        p_best_lag = self.p_best_lagrangian.to_numpy()
        pop = self.population.loc[:, self.variables].to_numpy()
        pop_lag = self.lagrangian_multipliers.to_numpy()
        g_best = self.g_best.loc[:, self.variables].to_numpy()
        g_best_lag = self.g_best_lagrangian.to_numpy()

        self.velocity = (
            self.w * self.velocity
            + np.multiply(self.c1 * self.r1[t], p_best - pop)
            + np.multiply(self.c2 * self.r2[t], g_best - pop)
        )
        self.lagrangian_velocity = (
            self.w * self.lagrangian_velocity
            + np.multiply(self.c1 * self.r1[t], p_best_lag - pop_lag)
            + np.multiply(self.c2 * self.r2[t], g_best_lag - pop_lag)
        )

    def _update_population(self) -> None:
        self.population.loc[:, self.variables] += self.velocity
        self.lagrangian_multipliers += self.lagrangian_velocity

        cols = [col for col in self.lagrangian_multipliers.columns if "u" in col]
        self.lagrangian_multipliers.loc[:, cols] = self.lagrangian_multipliers.loc[
            :,
            cols,
        ].clip(0.0)

    def update(self, t: int) -> None:
        self._update_velocity(t=t)
        self._update_population()

    def optimize(self) -> None:
        for t in range(self.iterations):
            # Get parameters for current iteration
            self.calculate_metric(t=t)
            self._get_p_best()
            # Get the index of the best member of the population
            best_index = self.population["metric"].argsort()[0]
            self._get_g_best(best_index=best_index)

            self.update(t=t)


if __name__ == "__main__":
    pop_size = 1000
    problem_3 = ThreeGenerators()
    t = 100

    alg = PSOPrimalDual(
        variables=problem_3.variables,
        upper_bounds=problem_3.params["max"].to_list(),
        lower_bounds=problem_3.params["min"].to_list(),
        iterations=t,
        size=pop_size,
        fn_obj=problem_3.fn_obj,
        constraints=problem_3.constraints,
        params={"demand": problem_3.demand, "params": problem_3.params},  # type: ignore
    )
    alg.optimize()
    pprint(alg.population)
    pprint(alg.lagrangian_multipliers)
    pprint(alg.best_multipliers)
    pprint(alg.g_best)
    alg.g_best.to_csv("g_best.csv")

    plt.figure()
    plt.plot(np.arange(0, t), alg.f_obj_history, label="F obj")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    plt.plot(np.arange(0, t), alg.gap_history, label="Gap")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    plt.plot(np.arange(0, t), alg.metric_history, label="Metric")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(np.arange(0, t), alg.lagrangian_history, label="Lagrangian")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
