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


def convert_bounds_to_dataframe(obj: ArrayLike, columns: list[str]) -> pl.DataFrame:
    assert len(obj) == len(columns)
    return pl.DataFrame(dict(zip(columns, obj)))
