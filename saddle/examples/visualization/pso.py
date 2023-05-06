from saddle.functions.benchmarks import f1
import matplotlib.pyplot as plt
from saddle.algorithms.meta.pso import ParticleSwarmOptimizer

if __name__ == '__main__':
    low = [0.0, 0.0]
    u = [5.0, 5.0]
    optima = [3.18, 3.13]

    alg = ParticleSwarmOptimizer(
        variables=['x1', 'x2'],
        upper_bounds=u,
        lower_bounds=low,
        iterations=20,
        fn_obj=f1,
        size=10,
    )
    alg.optimize()
    alg.plot_contours(optima=optima)
    plt.show()
