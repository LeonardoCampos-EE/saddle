from abc import ABC, abstractmethod
from typing import Callable, Any
import polars as pl

from ...types import ArrayLike
from ...functions.converters import convert_to_series


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
        self.upper_bounds = pl.DataFrame(
            convert_to_series(upper_bounds), schema=self.schema
        )
        self.lower_bounds = pl.DataFrame(
            convert_to_series(lower_bounds), schema=self.schema
        )

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

    def populate(self) -> None:
        """
        Initialize the population
        """
        population = pl.DataFrame(schema=self.schema)

        self.population = population
