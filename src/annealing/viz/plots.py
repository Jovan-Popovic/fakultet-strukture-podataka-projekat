from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from annealing.solver import SolverResult
from annealing.tsp.problem import Tour, TSPProblem

_TOUR_COLOR = "#2563eb"
_TOUR_DOT_COLOR = "#1e3a8a"
_TEMP_COLOR = "#dc2626"


def plot_tour(
    problem: TSPProblem,
    tour: Tour,
    title: str = "",
    annotate: bool = True,
    ax: Axes | None = None,
) -> Axes:
    coords = problem.coordinates()
    closed = np.asarray([*tour, tour[0]])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.plot(coords[closed, 0], coords[closed, 1], "-", color=_TOUR_COLOR, linewidth=1.5)
    ax.plot(coords[:, 0], coords[:, 1], "o", color=_TOUR_DOT_COLOR, markersize=5)
    if annotate:
        for city in problem.cities:
            ax.annotate(
                city.name,
                (city.x, city.y),
                fontsize=7,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    return ax


def plot_convergence(
    result: SolverResult[Tour], title: str = "Convergence", ax: Axes | None = None
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.energy_history, color=_TOUR_COLOR, linewidth=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tour length")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_temperature(
    result: SolverResult[Tour], title: str = "Temperature", ax: Axes | None = None
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.temperature_history, color=_TEMP_COLOR, linewidth=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Temperature")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    return ax


def plot_summary(
    problem: TSPProblem,
    result: SolverResult[Tour],
    output_path: str | Path,
    title: str = "",
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    plot_tour(
        problem,
        result.best_state,
        title=f"Best tour (length={result.best_energy:.2f})",
        ax=axes[0, 0],
    )
    plot_convergence(result, ax=axes[0, 1])
    plot_temperature(result, ax=axes[1, 0])

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05,
        0.95,
        _format_stats(result, title),
        transform=axes[1, 1].transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
    )

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _format_stats(result: SolverResult[Tour], title: str) -> str:
    initial = result.energy_history[0]
    improvement = 1 - result.best_energy / initial if initial else 0.0
    return (
        f"{title}\n\n"
        f"Iterations:        {result.accepted_moves + result.rejected_moves}\n"
        f"Accepted:          {result.accepted_moves}\n"
        f"Rejected:          {result.rejected_moves}\n"
        f"Acceptance rate:   {result.acceptance_rate:.2%}\n"
        f"Initial length:    {initial:.2f}\n"
        f"Best length:       {result.best_energy:.2f}\n"
        f"Improvement:       {improvement:.2%}"
    )
