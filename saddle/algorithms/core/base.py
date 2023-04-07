from abc import ABC, abstractmethod
from typing import Callable

class BaseOptimizer(ABC):
    fn_obj: Callable
    fn_res: dict[str, Callable] | list[Callable]

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
    @abstractmethod
    def populate(self) -> None:
        """
        Initialize the population
        """
        raise NotImplementedError