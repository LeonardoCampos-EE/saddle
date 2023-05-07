from saddle.functions.benchmarks import rosenbrock
import matplotlib.pyplot as plt
from saddle.algorithms.meta.pso import ParticleSwarmOptimizer
from saddle.functions.visualization.visualization import plot_mesh, plot_contour
from saddle.functions.visualization.algorithms import plot_pso_search

if __name__ == "__main__":
    low = [-1.2, -1.2]
    u = [1.2, 1.2]
    optima = [1.0, 1.0]

    plot_mesh(rosenbrock, low, u)
    plot_contour(rosenbrock, low, u, optima)
    plt.legend()
    plt.show()

    alg = ParticleSwarmOptimizer(
        variables=["x1", "x2"],
        upper_bounds=u,
        lower_bounds=low,
        iterations=20,
        fn_obj=rosenbrock,
        size=20,
        seed=None,
    )
    alg.optimize()
    plot_pso_search(alg, optima=optima)
    plt.show()
