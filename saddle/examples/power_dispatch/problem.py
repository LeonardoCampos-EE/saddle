import pandas as pd
import numpy as np
from saddle.types import ParametricFunction
from dataclasses import dataclass
from .functions import (
    cost,
    demand_constraint,
    min_power_constraint,
    max_power_constraint,
)


@dataclass
class DispatchProblem:
    params: pd.DataFrame
    demand: float
    fn_obj: ParametricFunction
    demand_constraint: ParametricFunction
    min_power_constraint: ParametricFunction
    max_power_constraint: ParametricFunction

    def __post_init__(self) -> None:
        self.fn_obj = ParametricFunction(func=cost, params=self.params)
        self.demand_constraint = ParametricFunction(
            func=demand_constraint, demand=self.demand
        )
        self.min_power_constraint = ParametricFunction(
            func=min_power_constraint, params=self.params
        )
        self.max_power_constraint = ParametricFunction(
            func=max_power_constraint, params=self.params
        )


@dataclass
class ThreeGenerators(DispatchProblem):
    def __init__(self) -> None:
        self.params = pd.read_csv(
            'systems/3_gen.csv',
            dtype=np.float32,
        )
        self.variables = [f'p{i}' for i in range(1, 4)]
        self.demand = 850.0


@dataclass
class ThirteenGenerators(DispatchProblem):
    def __init__(self) -> None:
        self.params = pd.read_csv(
            'systems/13_gen.csv',
            dtype=np.float32,
        )
        self.variables = [f'p{i}' for i in range(1, 14)]
        self.demand = 2520.0


@dataclass
class NineteenGenerators(DispatchProblem):
    def __init__(self) -> None:
        self.params = pd.read_csv(
            'systems/19_gen.csv',
            dtype=np.float32,
        )
        self.variables = [f'p{i}' for i in range(1, 20)]
        self.demand = 2908.0


@dataclass
class FortyGenerators(DispatchProblem):
    def __init__(self) -> None:
        self.params = pd.read_csv(
            'systems/40_gen.csv',
            dtype=np.float32,
        )
        self.variables = [f'p{i}' for i in range(1, 41)]
        self.demand = 10500.0
