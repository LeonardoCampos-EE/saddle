"""Utilities"""

import numpy as np
import numpy.typing as npt

from saddle.core.named_array import NamedArray


def set_seed(seed: int) -> None:
    """Seed random number generator"""
    np.random.seed(seed)


def init_random_array(
    shape: tuple[int, int],
    lower_bounds: NamedArray,
    upper_bounds: NamedArray,
) -> npt.NDArray[np.float64]:
    """Initialize random array"""
    if len(lower_bounds) != len(upper_bounds):
        raise ValueError("Lower and upper bounds must have the same length")

    if shape[1] != len(lower_bounds) or shape[1] != len(upper_bounds):
        raise ValueError("Shape must have the same length as bounds")

    return (upper_bounds.arr - lower_bounds.arr) * np.random.uniform(
        size=shape,
    ) + lower_bounds.arr
