"""Named array tests"""

# Replace with the actual name of your module
import numpy as np
import pytest
from saddle.core.named_array import (
    NamedArray,
)


def test_column_by_name():
    na = NamedArray(
        ["a", "b", "c"], np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64),
    )
    np.testing.assert_array_equal(na["a"], [1, 4])
    np.testing.assert_array_equal(na["b"], [2, 5])
    np.testing.assert_array_equal(na["c"], [3, 6])


def test_column_by_index():
    na = NamedArray(
        ["a", "b", "c"], np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64),
    )
    np.testing.assert_array_equal(na[0], [1, 4])
    np.testing.assert_array_equal(na[1], [2, 5])
    np.testing.assert_array_equal(na[2], [3, 6])


def test_cell_by_name_and_index():
    na = NamedArray(
        ["a", "b", "c"], np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64),
    )
    assert na["a", 0] == 1
    assert na["b", 1] == 5
    assert na["c", 1] == 6


def test_cell_by_index():
    na = NamedArray(
        ["a", "b", "c"], np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64),
    )
    assert na[0, 0] == 1
    assert na[1, 1] == 5
    assert na[2, 1] == 6


def test_invalid_key_string():
    na = NamedArray(
        ["a", "b", "c"], np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64),
    )
    with pytest.raises(KeyError):
        na["d"]


def test_invalid_key_type():
    na = NamedArray(
        ["a", "b", "c"], np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64),
    )
    with pytest.raises(TypeError):
        na[1.5]  # type: ignore


def test_invalid_key_tuple():
    na = NamedArray(
        ["a", "b", "c"], np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64),
    )
    with pytest.raises(KeyError):
        na["a", "b"]  # type: ignore
