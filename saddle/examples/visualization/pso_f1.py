import matplotlib.pyplot as plt
from saddle.algorithms.meta.pso import ParticleSwarmOptimizer
from saddle.functions.benchmarks import f1
from saddle.functions.visualization.algorithms import plot_pso_search
from saddle.functions.visualization.visualization import plot_contour, plot_mesh

if __name__ == "__main__":
    low = [0.0, 0.0]
    u = [5.0, 5.0]
    optima = [3.18, 3.13]

    plot_mesh(f1, low, u)
    plot_contour(f1, low, u, optima)
    plt.legend()
    # plt.show()

    alg = ParticleSwarmOptimizer(
        variables=["x1", "x2"],
        upper_bounds=u,
        lower_bounds=low,
        iterations=20,
        fn_obj=f1,
        size=20,
    )
    alg.optimize()
    plot_pso_search(alg, optima=optima)
    plt.show()
