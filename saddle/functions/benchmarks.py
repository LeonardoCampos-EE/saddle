import numpy as np
import pandas as pd


def parabola(pop: pd.DataFrame) -> pd.Series:
    return pop["x"] ** 2


def f1(pop: pd.DataFrame) -> pd.Series:
    return (
        (pop["x1"] - 3.14) ** 2
        + (pop["x2"] - 2.72) ** 2
        + np.sin(3 * pop["x1"] + 1.41)
        + np.sin(4 * pop["x2"] - 1.73)
    )
