from typing import TypeVar
import polars as pl
import numpy as np

ArrayLike = TypeVar("ArrayLike", pl.Series, np.ndarray, list)
