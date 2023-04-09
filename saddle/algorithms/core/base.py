from abc import ABC, abstractmethod
from typing import Callable, Any
import numpy as np
import polars as pl

from ...types import ArrayLike
from ...functions.converters import convert_bounds_to_dataframe


class BaseOptimizer(ABC):
    fn_obj: Callable | None = None
    fn_res: dict[str, Callable] | list[Callable] | None = None
    columns: list[str]
    schema: dict[str, Any]
    upper_bounds: pl.DataFrame
    lower_bounds: pl.DataFrame

    def __init__(
        self, columns: list[str], upper_bounds: ArrayLike, lower_bounds: ArrayLike
    ) -> None:
        self.columns = columns
        self.schema = {col: pl.Float32 for col in self.columns}
        self.upper_bounds = convert_bounds_to_dataframe(upper_bounds, self.columns)
        self.lower_bounds = convert_bounds_to_dataframe(lower_bounds, self.columns)

    @abstractmethod
    def update(self) -> None:
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
    population: pl.DataFrame | None = None

    def populate(self, size: int) -> None:
        """
        Initialize the population
        """

        upp = self.upper_bounds.to_numpy()
        low = self.lower_bounds.to_numpy()
        population = pl.DataFrame(
            (upp - low) * np.random.uniform(size=(size, len(self.columns))) + low,
            schema=self.schema,
        )

        self.population = population
