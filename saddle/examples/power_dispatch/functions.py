import pandas as pd
import numpy as np


def cost(pop: pd.DataFrame, params: pd.DataFrame) -> pd.Series:
    p = pop.values.T
    _cost = (
        params['a'].values * p**2
        + params['b'].values * p
        + params['c'].values
        + np.abs(
            params['e'].values * np.sin(params['f'].values * (params['min'].values - p))
        )
    )
    return pd.Series(
        np.sum(
            _cost,
            axis=0,
        )
    )


def demand_constraint(pop: pd.DataFrame, demand: float) -> pd.Series:
    p = pop.values
    return pd.Series(p.sum(axis=1) - demand)


def min_power_constraint(pop: pd.DataFrame, params: pd.DataFrame) -> pd.Series:
    p = pop.values.T
    violation = params['min'].values - p
    return pd.Series(np.where(violation < 0, 0, violation))


def max_power_constraint(pop: pd.DataFrame, params: pd.DataFrame) -> pd.Series:
    p = pop.values.T
    violation = p - params['max'].values
    return pd.Series(np.where(violation < 0, 0, violation))
