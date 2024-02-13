"""PSO Primal Dual Algorithm"""

from dataclasses import dataclass

import numpy as np

from saddle.core import (
    NamedArray,
    ParametricFunction,
    init_random_array,
)


@dataclass
class PSOPrimalDualParameters:
    """Parameters for PSO Primal Dual Algorithm"""

    objective: ParametricFunction
    constraints: dict[str, ParametricFunction]

    upper_bounds: NamedArray
    lower_bounds: NamedArray

    variables: list[str]
    metrics: list[str]
    iterations: int
    size: int

    w: float = 0.8
    c1: float = 0.1
    c2: float = 0.1
    seed: int = 42


@dataclass
class Agents:
    """Agents for PSO Primal Dual Algorithm"""

    position: NamedArray
    velocity: NamedArray
    metrics: NamedArray
    history: NamedArray

    global_best: NamedArray
    population_best: NamedArray
    global_best_history: NamedArray

    def __init__(
        self,
        size: int,
        iterations: int,
        variables: list[str],
        metrics: list[str],
        upper_bounds: NamedArray,
        lower_bounds: NamedArray,
    ) -> None:
        """Initialize agents"""
        self.position = NamedArray(
            names=variables,
            arr=init_random_array((size, len(variables)), lower_bounds, upper_bounds),
        )
        self.velocity = NamedArray(
            names=variables,
            arr=np.zeros((size, len(variables))),
        )

        self.metrics = NamedArray(
            names=metrics,
            arr=np.ones((size, len(metrics))) * np.inf,
        )
        self.history = NamedArray(
            names=metrics,
            arr=np.zeros((iterations, size, len(variables))),
        )
        self.global_best = NamedArray(
            names=variables,
            arr=np.zeros((1, len(variables))),
        )
        self.population_best = NamedArray(
            names=variables,
            arr=np.zeros((size, len(variables))),
        )
        self.global_best_history = NamedArray(
            names=metrics,
            arr=np.zeros((iterations, len(variables))),
        )
