from typing import TypeVar

import numpy as np
import pandas as pd

ArrayLike = TypeVar("ArrayLike", pd.Series, np.ndarray, list)
