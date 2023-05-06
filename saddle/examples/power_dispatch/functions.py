import pandas as pd
import numpy as np


def cost(pop: pd.DataFrame, params: pd.DataFrame) -> pd.Series:
    p = pop.values.T
    a = np.expand_dims(params['a'].values, axis=1)
    b = np.expand_dims(params['b'].values, axis=1)
    c = np.expand_dims(params['c'].values, axis=1)
    e = np.expand_dims(params['e'].values, axis=1)
    f = np.expand_dims(params['f'].values, axis=1)
    _min = np.expand_dims(params['min'].values, axis=1)
    _cost = a * p**2 + b * p + c + np.abs(e * np.sin(f * (_min - p)))
    return pd.Series(
        np.sum(
            _cost,
            axis=0,
        )
    )


def demand_constraint(pop: pd.DataFrame, demand: float) -> pd.Series:
    p = pop.values
    return pd.Series((p.sum(axis=1) - demand) ** 2)


def min_power_constraint(pop: pd.DataFrame, params: pd.DataFrame) -> pd.Series:
    p = pop.values.T
    _min = np.expand_dims(params['min'].values, axis=1)
    violation = _min - p
    violation = np.sum(np.abs(np.where(violation < 0, 0, violation)), axis=0)
    return pd.Series(violation)


def max_power_constraint(pop: pd.DataFrame, params: pd.DataFrame) -> pd.Series:
    p = pop.values.T
    _max = np.expand_dims(params['max'].values, axis=1)
    violation = p - _max
    violation = np.sum(np.abs(np.where(violation < 0, 0, violation)), axis=0)
    return pd.Series(violation)
