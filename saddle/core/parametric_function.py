from typing import Callable


class ParametricFunction:
    def __init__(self, func: Callable, **kwargs):
        self.func = func
        self.params = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*args, **{**self.params, **kwargs})
