from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ...types import ArrayLike
from ...functions.converters import convert_bounds_to_series


class BaseOptimizer(ABC):
    fn_obj: callable
    variables: list[str]
    upper_bounds: pd.Series
    lower_bounds: pd.Series
    iterations: int

    def __init__(
        self,
        variables: list[str],
        upper_bounds: ArrayLike,
        lower_bounds: ArrayLike,
        iterations: int,
        fn_obj: callable,
    ) -> None:
        self.variables = variables
        self.columns = self.variables + ['fn_obj', 'metric']

        self.upper_bounds = convert_bounds_to_series(upper_bounds, self.variables)
        self.lower_bounds = convert_bounds_to_series(lower_bounds, self.variables)
        self.iterations = iterations
        self.fn_obj = fn_obj

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Updates the algorithm's position on each iteration
        """
        raise NotImplementedError

    @abstractmethod
    def optimize(self) -> None:
        """
        Optimization main loop
        """
        raise NotImplementedError


class BaseMetaheuristicOptimizer(BaseOptimizer):
    population: pd.DataFrame = pd.DataFrame()
    seed: int

    def __init__(
        self,
        variables: list[str],
        upper_bounds: ArrayLike,
        lower_bounds: ArrayLike,
        iterations: int,
        fn_obj: callable,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        np.random.seed(self.seed)
        super().__init__(variables, upper_bounds, lower_bounds, iterations, fn_obj)

    def populate(self, size: int) -> None:
        """
        Initialize the population
        """
        self.size = size
        upp = self.upper_bounds.to_numpy()
        low = self.lower_bounds.to_numpy()
        population: np.ndarray = (upp - low) * np.random.uniform(
            size=(size, len(self.variables))
        ) + low

        fn_obj = np.zeros(shape=(size, 1))
        population = np.hstack([population, fn_obj, fn_obj])

        self.population = pd.DataFrame(population, columns=self.columns)

    def _calculate_fn_obj(self) -> None:
        fn_values: pd.Series = self.fn_obj(self.population.loc[:, self.variables])
        self.population.loc[:, 'fn_obj'] = fn_values

    def calculate_metric(self) -> None:
        self._calculate_fn_obj()
        self.population.loc[:, 'metric'] = self.population.loc[:, 'fn_obj']
