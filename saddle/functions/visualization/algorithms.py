from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .visualization import plot_contour
from ...algorithms.meta.pso import ParticleSwarmOptimizer


def plot_pso_search(alg: ParticleSwarmOptimizer, optima: list[float]) -> Figure:
    fig, ax = plot_contour(
        fun=alg.fn_obj,
        lower=alg.lower_bounds.to_list(),
        upper=alg.upper_bounds.to_list(),
        optima=optima,
    )

    pop = alg.population_history.loc[alg.population_history["iteration"] == 0][
        alg.variables
    ].to_numpy()
    pop_plot = ax.scatter(
        pop[:, 0],
        pop[:, 1],
        c="blue",
        marker="o",
        alpha=0.5,
        label="Population",
    )
    vel = alg.velocity_history[0]
    vel_plot = ax.quiver(
        pop[:, 0],
        pop[:, 1],
        vel[:, 0],
        vel[:, 1],
        color="blue",
        width=0.005,
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    gbest_plot = ax.scatter(
        x=alg.g_best_history[alg.variables[0]].iloc[0],
        y=alg.g_best_history[alg.variables[1]].iloc[0],
        c="red",
        marker="x",
        s=12,
        label="G Best",
    )

    def animate(i):
        ax.set_title(f"Iteration {i}")
        pop = alg.population_history.loc[alg.population_history["iteration"] == i][
            alg.variables
        ].to_numpy()
        vel = alg.velocity_history[i]
        pop_plot.set_offsets(pop)
        vel_plot.set_offsets(pop)
        vel_plot.set_UVC(vel[:, 0], vel[:, 1])
        gbest_plot.set_offsets(alg.g_best_history.iloc[i][alg.variables])
        plt.legend()
        return ax, pop_plot, vel_plot, gbest_plot

    anim = FuncAnimation(
        fig,
        animate,
        frames=list(range(1, len(alg.velocity_history))),
        interval=500,
        blit=False,
        repeat=True,
    )
    anim.save("pso.gif", dpi=120, writer="pillow")

    return fig
