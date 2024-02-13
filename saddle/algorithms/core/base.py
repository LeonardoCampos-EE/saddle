from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd

from saddle.core.parametric_function import ParametricFunction
from saddle.core.types import ArrayLike
from saddle.functions.converters import convert_bounds_to_series


class BaseOptimizer(ABC):
    variables: list[str]
    upper_bounds: pd.Series
    lower_bounds: pd.Series
    iterations: int
    fn_obj: Callable | ParametricFunction
    constraints: dict[str, Callable | ParametricFunction]

    def __init__(
        self,
        variables: list[str],
        upper_bounds: ArrayLike,
        lower_bounds: ArrayLike,
        iterations: int,
        fn_obj: Callable | ParametricFunction,
        constraints: dict[str, Callable | ParametricFunction] | None = None,
    ) -> None:
        self.variables = variables
        self.constraints = constraints or {}
        self.columns = (
            self.variables + ["fn_obj"] + list(self.constraints.keys()) + ["metric"]
        )

        self.upper_bounds = convert_bounds_to_series(upper_bounds, self.variables)
        self.lower_bounds = convert_bounds_to_series(lower_bounds, self.variables)
        self.iterations = iterations
        self.fn_obj = fn_obj

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Updates the algorithm's position on each iteration
        """
        raise NotImplementedError

    @abstractmethod
    def optimize(self) -> None:
        """Optimization main loop
        """
        raise NotImplementedError


class BaseMetaheuristicOptimizer(BaseOptimizer):
    population: pd.DataFrame = pd.DataFrame()
    seed: int
    penalties: dict[str, np.ndarray]

    def __init__(
        self,
        variables: list[str],
        upper_bounds: ArrayLike,
        lower_bounds: ArrayLike,
        iterations: int,
        size: int,
        fn_obj: Callable | ParametricFunction,
        constraints: dict[str, Callable | ParametricFunction] | None = None,
        penalties: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.size = size
        if seed:
            np.random.seed(self.seed)
        if constraints:
            if penalties:
                assert len(constraints) == len(penalties)
                self.penalties = {
                    name: np.linspace(10, penalties[name], num=iterations)
                    for name in penalties
                }
            else:
                self.penalties = {
                    name: np.linspace(10, 100000, num=iterations)
                    for name in constraints
                }
        super().__init__(
            variables, upper_bounds, lower_bounds, iterations, fn_obj, constraints,
        )
        self.populate(self.size)

    def populate(self, size: int) -> None:
        """Initialize the population
        """
        self.size = size
        upp = self.upper_bounds.to_numpy()
        low = self.lower_bounds.to_numpy()
        population: np.ndarray = (upp - low) * np.random.uniform(
            size=(size, len(self.variables)),
        ) + low

        fn_obj = np.zeros(shape=(size, 1))
        metric = fn_obj.copy()
        if self.constraints:
            constraints = np.zeros(shape=(size, len(self.constraints)))
            population = np.hstack([population, fn_obj, constraints, metric])
        else:
            population = np.hstack([population, fn_obj, metric])

        self.population = pd.DataFrame(population, columns=self.columns)

    def _calculate_fn_obj(self) -> None:
        fn_values: pd.Series = self.fn_obj(self.population.loc[:, self.variables])
        self.population.loc[:, "fn_obj"] = fn_values

    def _calculate_constraints(self, t: int) -> None:
        for name, constraint in self.constraints.items():
            constraint_values: pd.Series = constraint(
                self.population.loc[:, self.variables],
            )
            self.population.loc[:, name] = self.penalties[name][t] * constraint_values

    def _divide_by_penalties(self, t: int) -> None:
        """This function divides the constraints by the penalty value to keep the real
        constraint value in the history
        """
        for name in self.constraints:
            self.population.loc[:, name] = (
                self.population.loc[:, name] / self.penalties[name][t]
            )

    def calculate_metric(self, t: int) -> None:
        self._calculate_fn_obj()
        self._calculate_constraints(t=t)
        metric = self.population.loc[:, ["fn_obj"] + list(self.constraints.keys())].sum(
            axis=1,
        )
        self.population.loc[:, "metric"] = metric
        self._divide_by_penalties(t=t)
