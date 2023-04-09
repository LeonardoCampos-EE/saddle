from ..types import ArrayLike
import numpy as np
import polars as pl


def convert_to_series(obj: ArrayLike) -> pl.Series:
    match obj:
        case pl.Series():
            return obj
        case np.ndarray():
            return pl.Series(obj)
        case _:
            return pl.Series(obj)
