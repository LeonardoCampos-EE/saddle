"""Named array object"""

import numpy as np
import numpy.typing as npt


class NamedArray:
    names: list[str]
    arr: npt.NDArray[np.float64]

    def __init__(self, names: list[str], arr: npt.NDArray[np.float64]):
        self.names = names
        self.arr = arr

    def __getitem__(
        self, key: str | int | tuple[str, int] | tuple[int, int]
    ) -> npt.NDArray[np.float64]:
        if isinstance(key, str):
            if key not in self.names:
                raise KeyError(f"{key} is not in names")
            col_index = self.names.index(key)
            return self.arr[:, col_index]
        elif isinstance(key, int):
            # If key is an int, return the corresponding column
            return self.arr[:, key]
        elif isinstance(key, tuple):
            # If key is a tuple, it can be (str, int) or (int, int)
            if isinstance(key[0], str) and isinstance(key[1], int):
                # If first element is a string and second is an int, find the column and return the specific element
                col_index = self.names.index(key[0])
                return self.arr[key[1], col_index]
            elif isinstance(key[0], int) and isinstance(key[1], int):
                # If both elements are int, return the specific element
                return self.arr[key[1], key[0]]
            else:
                raise KeyError("Invalid key format. Must be (str, int) or (int, int).")
        else:
            raise TypeError(
                "Key must be a str, int, or a tuple of (str, int) or (int, int)."
            )
