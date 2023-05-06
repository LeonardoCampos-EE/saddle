from saddle.algorithms.meta.gwo import GreyWolfOptimizer
from numpy.testing import assert_allclose


def test_gwo_convergence(
    fn_obj, lower_bounds, upper_bounds, pop_size, iterations, seed, variables
):
    gwo = GreyWolfOptimizer(
        variables=variables,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds,
        iterations=iterations,
        fn_obj=fn_obj,
        seed=seed,
    )
    gwo.populate(size=pop_size)
    gwo.optimize()
    assert_allclose(gwo.population["metric"].min(), 0.0, rtol=1e-2, atol=1e-2)
    return
