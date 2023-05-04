from saddle.algorithms.meta.pso import ParticleSwarmOptimizer
from numpy.testing import assert_allclose


def test_gwo_convergence(
    fn_obj, lower_bounds, upper_bounds, pop_size, iterations, seed, variables
):
    pso = ParticleSwarmOptimizer(
        variables=variables,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds,
        iterations=iterations,
        fn_obj=fn_obj,
        seed=seed,
    )
    pso.populate(size=pop_size)
    pso.optimize()
    assert_allclose(pso.population['metric'].min(), 0.0, rtol=1e-2, atol=1e-2)
    return
