import numpy as np
import pytest
from saddle.algorithms.meta.pso_primal_dual import (
    Agents,
    ParametricFunction,
    PSOPrimalDualParameters,
)
from saddle.core import NamedArray


@pytest.fixture()
def size() -> int:
    return 4


@pytest.fixture()
def iterations() -> int:
    return 5


@pytest.fixture()
def variables() -> list[str]:
    return ["x", "y", "z"]


@pytest.fixture()
def metrics() -> list[str]:
    return ["objective", "constraint"]


@pytest.fixture()
def upper_bounds() -> NamedArray:
    return NamedArray(names=["x", "y", "z"], arr=np.array([10, 10, 10]))


@pytest.fixture()
def lower_bounds() -> NamedArray:
    return NamedArray(names=["x", "y", "z"], arr=np.array([-10, -10, -10]))


@pytest.fixture()
def objective() -> ParametricFunction:
    return ParametricFunction(lambda arr: arr["x"] ** 2 + arr["y"] ** 2 + arr["z"] ** 2)


@pytest.fixture()
def constraints() -> dict[str, ParametricFunction]:
    return {
        "constraint": ParametricFunction(
            lambda arr: arr["x"] ** 2 + arr["y"] ** 2 + arr["z"] ** 2 - 100,
        ),
    }


def test_agents_init(
    size: int,
    iterations: int,
    variables: list[str],
    metrics: list[str],
    upper_bounds: NamedArray,
    lower_bounds: NamedArray,
):
    pop = Agents(size, iterations, variables, metrics, upper_bounds, lower_bounds)

    assert pop.position.shape == (size, len(variables))
    assert pop.velocity.shape == (size, len(variables))
    assert pop.metrics.shape == (size, len(metrics))
    assert pop.history.shape == (iterations, size, len(variables))
    assert pop.global_best.shape == (1, len(variables))
    assert pop.global_best_history.shape == (iterations, len(variables))
    assert pop.population_best.shape == (size, len(variables))
    assert pop.global_best_metrics_history.shape == (iterations, len(metrics))


def test_params_init(
    size: int,
    iterations: int,
    variables: list[str],
    metrics: list[str],
    objective: ParametricFunction,
    constraints: dict[str, ParametricFunction],
    upper_bounds: NamedArray,
    lower_bounds: NamedArray,
):
    params = PSOPrimalDualParameters(
        objective=objective,
        constraints=constraints,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds,
        variables=variables,
        metrics=metrics,
        iterations=iterations,
        size=size,
    )
    arr = NamedArray(names=variables, arr=np.array([[10, 10, 10]]))
    np.testing.assert_almost_equal(params.objective(arr), 300)
    np.testing.assert_almost_equal(params.constraints["constraint"](arr), 200)
