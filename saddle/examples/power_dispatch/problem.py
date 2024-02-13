import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from saddle.types import ParametricFunction

from .functions import (
    cost,
    demand_constraint,
    max_power_constraint,
    min_power_constraint,
)

PATH = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DispatchProblem:
    params: pd.DataFrame
    demand: float
    fn_obj: ParametricFunction
    demand_constraint: ParametricFunction
    min_power_constraint: ParametricFunction
    max_power_constraint: ParametricFunction
    constraints: dict[str, ParametricFunction]

    def __init__(self) -> None:
        self.fn_obj = ParametricFunction(func=cost, params=self.params)
        self.demand_constraint = ParametricFunction(
            func=demand_constraint,
            demand=self.demand,
        )
        self.min_power_constraint = ParametricFunction(
            func=min_power_constraint,
            params=self.params,
        )
        self.max_power_constraint = ParametricFunction(
            func=max_power_constraint,
            params=self.params,
        )
        self.constraints = {
            "demand_constraint": self.demand_constraint,
            "min_power_constraint": self.min_power_constraint,
            "max_power_constraint": self.max_power_constraint,
        }


@dataclass
class ThreeGenerators(DispatchProblem):
    def __init__(self) -> None:
        self.params = pd.read_csv(
            os.path.join(PATH, "systems", "3_gen.csv"),
            dtype=np.float32,
        )
        self.variables = [f"p{i}" for i in range(1, 4)]
        self.demand = 850.0
        super().__init__()


@dataclass
class ThirteenGenerators(DispatchProblem):
    def __init__(self) -> None:
        self.params = pd.read_csv(
            os.path.join(PATH, "systems", "13_gen.csv"),
            dtype=np.float32,
        )
        self.variables = [f"p{i}" for i in range(1, 14)]
        self.demand = 2520.0
        super().__init__()


@dataclass
class NineteenGenerators(DispatchProblem):
    def __init__(self) -> None:
        self.params = pd.read_csv(
            os.path.join(PATH, "systems", "19_gen.csv"),
            dtype=np.float32,
        )
        self.variables = [f"p{i}" for i in range(1, 20)]
        self.demand = 2908.0
        super().__init__()


@dataclass
class FortyGenerators(DispatchProblem):
    def __init__(self) -> None:
        self.params = pd.read_csv(
            os.path.join(PATH, "systems", "40_gen.csv"),
            dtype=np.float32,
        )
        self.variables = [f"p{i}" for i in range(1, 41)]
        self.demand = 10500.0
        super().__init__()
