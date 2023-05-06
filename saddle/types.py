from typing import TypeVar, Callable
import pandas as pd
import numpy as np

ArrayLike = TypeVar('ArrayLike', pd.Series, np.ndarray, list)


class ParametricFunction:
    def __init__(self, func: Callable, **kwargs):
        self.func = func
        self.params = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*args, **{**self.params, **kwargs})
