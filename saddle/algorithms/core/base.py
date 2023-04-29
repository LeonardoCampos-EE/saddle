from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import pandas as pd

from ...types import ArrayLike
from ...functions.converters import convert_bounds_to_dataframe


class BaseOptimizer(ABC):
    fn_obj: Callable
    variables: list[str]
    upper_bounds: pd.DataFrame
    lower_bounds: pd.DataFrame
    iterations: int

    def __init__(
        self,
        variables: list[str],
        upper_bounds: ArrayLike,
        lower_bounds: ArrayLike,
        iterations: int,
    ) -> None:
        self.variables = variables
        self.columns = self.variables + ['fn_obj']

        self.upper_bounds = convert_bounds_to_dataframe(upper_bounds, self.variables)
        self.lower_bounds = convert_bounds_to_dataframe(lower_bounds, self.variables)
        self.iterations = iterations

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
        seed: int = 42,
    ) -> None:
        self.seed = seed
        super().__init__(variables, upper_bounds, lower_bounds, iterations=iterations)

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
        population = np.hstack([population, fn_obj])

        self.population = pd.DataFrame(population)

    def calc_fn_obj(self) -> None:
        fn_values: pd.Series = self.fn_obj(self.population.loc[:, self.variables])
        self.population.loc[:, -1] = fn_values
