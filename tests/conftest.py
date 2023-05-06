from typing import Callable
import pytest
from saddle.functions.benchmarks import parabola


@pytest.fixture
def fn_obj() -> Callable:
    return parabola


@pytest.fixture
def lower_bounds() -> list[float]:
    return [-2.0]


@pytest.fixture
def upper_bounds() -> list[float]:
    return [2.0]


@pytest.fixture
def pop_size() -> int:
    return 5


@pytest.fixture
def iterations() -> int:
    return 10


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def variables() -> list[str]:
    return ['x']
