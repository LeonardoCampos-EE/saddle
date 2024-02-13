# type: ignore

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from saddle.core.parametric_function import ParametricFunction


def mesh(lower: list[float], upper: list[float]) -> np.ndarray:
    return np.array(
        np.meshgrid(
            np.linspace(lower[0], upper[0], 100), np.linspace(lower[1], upper[1], 100),
        ),
    )


def plot_mesh(
    fun: Callable | ParametricFunction,
    lower: list[float],
    upper: list[float],
):
    assert len(lower) == 2
    assert len(upper) == 2

    x1, x2 = mesh(lower, upper)
    df = {"x1": x1, "x2": x2}
    f = fun(df)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surface = ax.plot_surface(
        x1, x2, f, cmap=cm.coolwarm, linewidth=0, antialiased=False,
    )
    ax.set_zlim(f.min() - 0.01, f.max() + 0.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter("{x:.02f}")
    fig.colorbar(surface, shrink=0.5, aspect=5)

    return fig, ax


def plot_contour(
    fun: Callable | ParametricFunction,
    lower: list[float],
    upper: list[float],
    optima: list[float],
):
    assert len(lower) == 2
    assert len(upper) == 2
    assert len(optima) == 2
    x1, x2 = mesh(lower, upper)
    df = {"x1": x1, "x2": x2}
    f = fun(df)

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    img = ax.imshow(
        f,
        cmap=cm.coolwarm,
        extent=[lower[0], upper[0], lower[1], upper[1]],
        origin="lower",
        alpha=0.5,
    )
    fig.colorbar(img, ax=ax)
    ax.scatter(
        [optima[0]],
        [optima[1]],
        marker="x",
        s=20,
        c="gold",
        label="optima",
    )
    contours = ax.contour(
        x1,
        x2,
        f,
        20,
        colors="black",
        alpha=0.4,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    return fig, ax


def plot_metric(metric: np.ndarray):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    x = np.arange(0, len(metric))
    ax.plot(
        x,
        metric,
        marker="o",
        markersize=2,
        c="blue",
        label="metric",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.show()
    return fig, ax


def plot_objective(objective: np.ndarray):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    x = np.arange(0, len(objective))
    ax.plot(
        x,
        objective,
        marker="o",
        markersize=2,
        c="blue",
        label="objective",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective function")
    plt.legend()
    plt.grid(True)
    plt.show()
    return fig, ax


def plot_constraint(constraint: np.ndarray, name: str):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    x = np.arange(0, len(constraint))
    ax.plot(
        x,
        constraint,
        marker="o",
        markersize=2,
        c="blue",
        label=name,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(name)
    plt.legend()
    plt.grid(True)
    plt.show()
    return fig, ax
