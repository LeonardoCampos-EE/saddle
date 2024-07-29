from pprint import pprint
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from saddle.core.types import ArrayLike
from saddle.core.parametric_function import ParametricFunction
from saddle.functions.converters import convert_bounds_to_series
from saddle.examples.power_dispatch.problem import ThreeGenerators
from tqdm import tqdm


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
        )
    )


def demand_constraint(
    pop: pd.DataFrame, multipliers: pd.DataFrame, demand: float
) -> pd.Series:
    p = pop.values
    v = multipliers["v"].values
    return pd.Series((p.sum(axis=1) - demand) * v)

def norm(value: float, min: float, max: float) -> float:
    return (value - min) / (max - min)

def min_power_constraint(
    pop: pd.DataFrame, multipliers: pd.DataFrame, params: pd.DataFrame
) -> pd.Series:
    u_cols = [col for col in multipliers.columns if "u_min" in col]
    u = multipliers[u_cols].values.T
    p = pop.values.T
    _min = np.expand_dims(params["min"].values, axis=1)  # type: ignore
    violation = (_min - p) * u
    violation = np.sum(violation.T, axis=1)
    return pd.Series(violation)


def max_power_constraint(
    pop: pd.DataFrame, multipliers: pd.DataFrame, params: pd.DataFrame
) -> pd.Series:
    u_cols = [col for col in multipliers.columns if "u_max" in col]
    u = multipliers[u_cols].values.T
    p = pop.values.T
    _max = np.expand_dims(params["max"].values, axis=1)  # type: ignore
    violation = (p - _max) * u
    violation = np.sum(violation.T, axis=1)
    return pd.Series(violation)


class GWOPrimalDual:
    population: pd.DataFrame = pd.DataFrame()
    seed: int
    penalties: dict[str, np.ndarray]
    lagrangian: pd.Series

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
    ) -> None:
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
            size=(size, len(self.variables))
        ) + low

        fn_obj = np.zeros(shape=(size, 1))
        lagrangian = np.zeros(shape=(size, 1))
        gap = np.zeros(shape=(size, 1))

        metric = fn_obj.copy()
        constraints = np.zeros(shape=(size, len(self.constraints)))
        population = np.hstack(
            [population, fn_obj, constraints, lagrangian, gap, metric]
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
                self.population.loc[:, self.variables]
            )
            pen = 1_000 if "demand" in name else 1
            self.population.loc[:, name] = pen * constraint_values

    def calculate_metric(self, t: int) -> None:
        self._calculate_fn_obj()
        self._calculate_constraints()
        self._calculate_lagrangian()
        metric = (
            1 * self.population.loc[:, ["fn_obj"]].to_numpy()
            - 1 * self.population.loc[:, ["lagrangian"]].to_numpy()
            + 2 * self.population.loc[:, ["gap"]].to_numpy()
            + np.expand_dims(
                self.population.loc[:, list(self.constraints.keys())]
                .to_numpy()
                .sum(axis=1),
                axis=1,
            )
        )
        self.population.loc[:, "metric"] = metric.flatten()

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

        def r_lagrangian():
            return np.random.random_sample(
                size=(self.iterations, self.size, len(self.lagrangian_variables))
            )

        # A(t) -> controls the step size of each wolf in the search space
        # shape -> (iterations, size, variables)
        a = self.a[:, np.newaxis, np.newaxis]
        self.A_alpha = (2 * a * r()) - a
        self.A_beta = (2 * a * r()) - a
        self.A_delta = (2 * a * r()) - a

        self.A_alpha_lagrangian = (2 * a * r_lagrangian()) - a
        self.A_beta_lagrangian = (2 * a * r_lagrangian()) - a
        self.A_delta_lagrangian = (2 * a * r_lagrangian()) - a


        # C(t) -> controls the movement of each wolf towards the best solutions
        # shape -> (iterations, size, variables)
        self.C = 2 * r()
        self.C_lagrangian = 2 * r_lagrangian()

        # Alpha
        self.alpha = self.population.loc[[0], :].copy()
        self.alpha_lagrangian = self.lagrangian_multipliers.iloc[[0]].copy()
        self.alpha.loc[:, "metric"] = np.inf

        # Beta
        self.beta = self.population.loc[[0], :].copy()
        self.beta_lagrangian = self.lagrangian_multipliers.iloc[[0]].copy()
        self.beta.loc[:, "metric"] = np.inf

        # Delta
        self.delta = self.population.loc[[0], :].copy()
        self.delta_lagrangian = self.lagrangian_multipliers.iloc[[0]].copy()
        self.delta.loc[:, "metric"] = np.inf

        return

    def _get_best_wolves(self) -> None:
        # sourcery skip: extract-method, inline-immediately-returned-variable
        # Current population metric
        pop_metric = self.population.loc[:, "metric"].to_numpy().flatten()

        # Get the three best
        best_indexes = np.argsort(pop_metric).flatten()

        start_index = 0
        if pop_metric[best_indexes[0]] < self.alpha["metric"].item():
            self.alpha = self.population.loc[[best_indexes[0]], :].copy()
            self.alpha_lagrangian = self.lagrangian_multipliers.loc[[best_indexes[0]], :].copy()
            start_index = 1

        beta_updated = False
        for i in range(start_index, len(best_indexes)):
            current_index = best_indexes[i]
            current_metric = pop_metric[current_index]

            # Update Beta
            if current_metric < self.beta["metric"].item() and not beta_updated:
                self.beta = self.population.loc[[current_index], :].copy()
                self.beta_lagrangian = self.lagrangian_multipliers.loc[[current_index], :].copy()
                beta_updated = True
                continue  # Move to next iteration to avoid assigning beta as delta


            # Update Delta
            if current_metric < self.delta["metric"].item():
                self.delta = self.population.loc[[current_index], :].copy()
                self.delta_lagrangian = self.lagrangian_multipliers.loc[[current_index], :].copy()
                break  # Delta updated, no need to continue
        
        self.f_obj_history.append(self.alpha["fn_obj"].item())
        self.lagrangian_history.append(self.alpha["lagrangian"].item())
        self.gap_history.append(self.alpha["gap"].item())
        self.metric_history.append(self.alpha["metric"].item())
        return
        

    def _update_population(self, t: int) -> None:
        population = self.population.loc[:, self.variables].to_numpy()

        alpha = self.alpha[self.variables].to_numpy()
        beta = self.beta[self.variables].to_numpy()
        delta = self.delta[self.variables].to_numpy()

        # Calculate D_alpha, D_beta, D_delta
        D_alpha = np.abs(np.multiply(self.C[t], alpha) - population)
        D_beta = np.abs(np.multiply(self.C[t], beta) - population)
        D_delta = np.abs(np.multiply(self.C[t], delta) - population)

        # Calculate X_alpha, X_beta, X_delta
        X_alpha = alpha - np.multiply(self.A_alpha[t], D_alpha)
        X_beta = beta - np.multiply(self.A_beta[t], D_beta)
        X_delta = delta - np.multiply(self.A_delta[t], D_delta)

        # Update population
        population = (X_alpha + X_beta + X_delta) / 3.0

        self.population.loc[:, self.variables] = population

        return

    def _update_lagrangian_multipliers(self, t: int) -> None:
        lagrangian_multipliers = self.lagrangian_multipliers.loc[:, self.lagrangian_variables].to_numpy()

        alpha = self.alpha_lagrangian[self.lagrangian_variables].to_numpy()
        beta = self.beta_lagrangian[self.lagrangian_variables].to_numpy()
        delta = self.delta_lagrangian[self.lagrangian_variables].to_numpy()

        # Calculate D_alpha, D_beta, D_delta
        D_alpha = np.abs(np.multiply(self.C_lagrangian[t], alpha) - lagrangian_multipliers)
        D_beta = np.abs(np.multiply(self.C_lagrangian[t], beta) - lagrangian_multipliers)
        D_delta = np.abs(np.multiply(self.C_lagrangian[t], delta) - lagrangian_multipliers)

        # Calculate X_alpha, X_beta, X_delta
        X_alpha = alpha - np.multiply(self.A_alpha_lagrangian[t], D_alpha)
        X_beta = beta - np.multiply(self.A_beta_lagrangian[t], D_beta)
        X_delta = delta - np.multiply(self.A_delta_lagrangian[t], D_delta)

        # Update population
        lagrangian_multipliers = (X_alpha + X_beta + X_delta) / 3.0


        self.lagrangian_multipliers.loc[:, self.lagrangian_variables] = lagrangian_multipliers

        cols = [col for col in self.lagrangian_multipliers.columns if "u" in col]
        self.lagrangian_multipliers.loc[:, cols] = self.lagrangian_multipliers.loc[
            :, cols
        ].clip(0.0)

        return
 
    def update(self, t: int) -> None:
        self._update_population(t=t)
        self._update_lagrangian_multipliers(t=t)

    def optimize(self) -> None:
        for t in tqdm(range(self.iterations)):
            # Get parameters for current iteration
            self.calculate_metric(t=t)
            self._get_best_wolves()
            self.update(t=t)


if __name__ == "__main__":
    num_runs = 200
    res = []

    seeds = np.random.randint(
        0, 1000, size=num_runs
    )
    pop_size = 1000
    problem_3 = ThreeGenerators()
    t = 100

    # for i in range(num_runs):
    #     seed = seeds[i]
    #     np.random.seed(seed)
    #     alg = GWOPrimalDual(
    #         variables=problem_3.variables,
    #         upper_bounds=problem_3.params["max"].to_list(),
    #         lower_bounds=problem_3.params["min"].to_list(),
    #         iterations=t,
    #         size=pop_size,
    #         fn_obj=problem_3.fn_obj,
    #         constraints=problem_3.constraints,
    #         params={"demand": problem_3.demand, "params": problem_3.params},  # type: ignore
    #         seed=seed,
    #     )
    #     alg.optimize()
    #     res.append(alg.alpha["fn_obj"].item())

    # best = np.argmin(res)
    # best_seed = seeds[best]
    # np.random.seed(best_seed)
    # df = pd.DataFrame(
    #     {
    #         "obj": res,
    #         "seed": seeds
    #     }
    # ).to_csv("gwo_res.csv")

    best_seed = 42
    alg = GWOPrimalDual(
        variables=problem_3.variables,
        upper_bounds=problem_3.params["max"].to_list(),
        lower_bounds=problem_3.params["min"].to_list(),
        iterations=t,
        size=pop_size,
        fn_obj=problem_3.fn_obj,
        constraints=problem_3.constraints,
        params={"demand": problem_3.demand, "params": problem_3.params},  # type: ignore
        seed=best_seed,
    )
    alg.optimize()
    
    pprint(alg.population)
    pprint(alg.lagrangian_multipliers)
    pprint(alg.alpha)
    pprint(alg.alpha_lagrangian)
    alg.alpha.to_csv("alpha.csv")

    plt.figure()
    plt.plot(np.arange(0, t), alg.f_obj_history, label="F obj")
    plt.grid(True)
    plt.show()

    plt.plot(np.arange(0, t), alg.gap_history, label="Gap")
    plt.grid(True)
    plt.savefig("gap.png")
    plt.show()

    plt.plot(np.arange(0, t), alg.metric_history, label="Metric")
    plt.grid(True)
    plt.savefig("metric.png")
    plt.show()

    plt.figure()
    plt.plot(np.arange(0, t), alg.lagrangian_history, label="Lagrangian")
    plt.grid(True)
    plt.show()

    print(res)
