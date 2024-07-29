from typing import TypeVar
import pandas as pd
import numpy as np

ArrayLike = TypeVar("ArrayLike", pd.Series, np.ndarray, list)
