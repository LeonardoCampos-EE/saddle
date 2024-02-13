import numpy as np
from saddle.algorithms.meta.pso_primal_dual import (
    Agents,
    PSOPrimalDualParameters,
    ParametricFunction,
)
from saddle.core import NamedArray


def test_agents_init():
    size = 4
    iterations = 5
    variables = ["x", "y", "z"]
    metrics = ["metric1", "metric2"]
    upper_bounds = NamedArray(names=variables, arr=np.array([10, 10, 10]))
    lower_bounds = NamedArray(names=variables, arr=np.array([-10, -10, -10]))

    pop = Agents(size, iterations, variables, metrics, upper_bounds, lower_bounds)

    assert pop.position.shape == (size, len(variables))
    assert pop.velocity.shape == (size, len(variables))
    assert pop.metrics.shape == (size, len(metrics))
    assert pop.history.shape == (iterations, size, len(variables))
    assert pop.global_best.shape == (1, len(variables))
    assert pop.global_best_history.shape == (iterations, len(variables))
    assert pop.population_best.shape == (size, len(variables))


def test_params_init():
    size = 4
    iterations = 5
    variables = ["x", "y", "z"]
    metrics = ["metric1", "constraint1"]

    objective = ParametricFunction(
        lambda arr: arr["x"] ** 2 + arr["y"] ** 2 + arr["z"] ** 2
    )
    constraints = {
        "constraint1": ParametricFunction(
            lambda arr: arr["x"] ** 2 + arr["y"] ** 2 + arr["z"] ** 2 - 100
        )
    }
    upper_bounds = NamedArray(names=variables, arr=np.array([10, 10, 10]))
    lower_bounds = NamedArray(names=variables, arr=np.array([-10, -10, -10]))

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
    np.testing.assert_almost_equal(params.constraints["constraint1"](arr), 200)
