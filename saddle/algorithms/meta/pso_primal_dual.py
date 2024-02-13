"""PSO Primal Dual Algorithm"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

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

    r1: npt.NDArray[np.float64] = None  # type: ignore
    r2: npt.NDArray[np.float64] = None  # type: ignore

    def __post_init__(self) -> None:
        """Post initialization"""
        self.r1 = np.random.rand(self.iterations, 1)
        self.r2 = np.random.rand(self.iterations, 1)

        if "objective" not in self.metrics:
            self.metrics += ["objective"]

        if "metric" not in self.metrics:
            self.metrics += ["metric"]


@dataclass
class Agents:
    """Agents for PSO Primal Dual Algorithm"""

    position: NamedArray
    velocity: NamedArray
    metrics: NamedArray
    history: NamedArray

    global_best: NamedArray
    global_best_metrics: NamedArray

    population_best: NamedArray
    population_best_metrics: NamedArray

    global_best_history: NamedArray
    global_best_metrics_history: NamedArray

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
        self.global_best_metrics = NamedArray(
            names=metrics,
            arr=np.ones((1, len(metrics))) * np.inf,
        )
        self.population_best = NamedArray(
            names=variables,
            arr=np.zeros((size, len(variables))),
        )
        self.population_best_metrics = NamedArray(
            names=metrics,
            arr=np.ones((1, len(metrics))) * np.inf,
        )
        self.global_best_history = NamedArray(
            names=metrics,
            arr=np.zeros((iterations, len(variables))),
        )
        self.global_best_metrics_history = NamedArray(
            names=metrics,
            arr=np.zeros((iterations, len(metrics))),
        )


def update_velocity(
    agents: Agents,
    parameters: PSOPrimalDualParameters,
    iteration: int,
) -> NamedArray:
    """Update velocity"""
    velocity = agents.velocity
    velocity.arr = (
        parameters.w * velocity.arr
        + (
            parameters.c1
            * parameters.r1[iteration]
            * (agents.population_best.arr - agents.position.arr)
        )
        + (
            parameters.c2
            * parameters.r2[iteration]
            * (agents.global_best.arr - agents.position.arr)
        )
    )
    agents.velocity = velocity
    return velocity


def update_population(
    agents: Agents,
) -> NamedArray:
    """Update population"""
    position = agents.position
    position.arr = position.arr + agents.velocity.arr
    agents.position = position
    return position


def update_history(agents: Agents, iteration: int) -> Agents:
    """Update history"""
    agents.history.arr[iteration] = agents.position.arr
    agents.global_best_history.arr[iteration] = agents.global_best.arr
    agents.global_best_metrics_history.arr[iteration] = agents.global_best_metrics.arr
    return agents


def update_global_best(
    agents: Agents,
) -> NamedArray:
    """Update global best"""
    best_agent = np.argmin(agents.metrics["metric"]).item()
    # If the metric of the current agent is worse than the global best, do nothing
    if agents.metrics["metric", best_agent] >= agents.global_best_metrics["metric"]:
        return agents.global_best

    agents.global_best.arr = agents.position.arr[best_agent]
    agents.global_best_metrics.arr = agents.metrics.arr[best_agent]
    return agents.global_best


def update_population_best(agents: Agents) -> NamedArray:
    """Update population best"""
    pop_best_metric = agents.population_best_metrics["metric"]
    current_metric = agents.metrics["metric"]

    mask = current_metric < pop_best_metric

    agents.population_best.arr[mask] = agents.position.arr[mask]
    agents.population_best_metrics.arr[mask] = agents.metrics.arr[mask]
    return agents.population_best


def optimize(
    agents: Agents,
    parameters: PSOPrimalDualParameters,
) -> Agents:
    for it in range(parameters.iterations):
        agents.metrics = parameters.objective(agents.position)
        update_population_best(agents)
        update_global_best(agents)
        update_velocity(agents, parameters, it)
        update_history(agents, it)
        update_population(agents)

    return agents
