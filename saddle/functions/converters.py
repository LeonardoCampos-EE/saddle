import numpy as np
import pandas as pd

from saddle.core.types import ArrayLike


def convert_to_series(obj: ArrayLike) -> pd.Series:
    match obj:
        case pd.Series():
            return obj
        case np.ndarray():
            return pd.Series(obj)
        case _:
            return pd.Series(obj)


def convert_bounds_to_series(obj: ArrayLike, columns: list[str]) -> pd.Series:
    assert len(obj) == len(columns)
    return pd.Series(obj, index=columns)
