from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy.typing import NDArray

from annealing.exceptions import AnnealingError
from annealing.solver import SolverResult
from annealing.tsp.problem import Tour, TSPProblem

_BOUND_MARGIN = 0.05


def animate_tour_evolution(
    problem: TSPProblem,
    result: SolverResult[Tour],
    output_path: str | Path,
    fps: int = 12,
    annotate: bool = True,
) -> Path:
    if not result.snapshots:
        raise AnnealingError(
            "Result has no snapshots. Pass `snapshot_every` to the solver."
        )

    coords = problem.coordinates()
    fig, ax = plt.subplots(figsize=(9, 6))

    (line,) = ax.plot([], [], "-", color="#2563eb", linewidth=1.5)
    ax.plot(coords[:, 0], coords[:, 1], "o", color="#1e3a8a", markersize=5)
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
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    _set_padded_limits(ax, coords)

    title = ax.set_title("")

    def update(frame_idx: int) -> list[Artist]:
        iteration, tour, energy = result.snapshots[frame_idx]
        closed = np.asarray([*tour, tour[0]])
        line.set_data(coords[closed, 0], coords[closed, 1])
        title.set_text(f"Iteration {iteration}, tour length: {energy:.2f}")
        return [line, title]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(result.snapshots),
        interval=1000 / fps,
        blit=False,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gif":
        anim.save(out, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out, fps=fps)
    plt.close(fig)
    return out


def _set_padded_limits(ax: Axes, coords: NDArray[np.float64]) -> None:
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    dx, dy = x_max - x_min, y_max - y_min
    ax.set_xlim(x_min - _BOUND_MARGIN * dx, x_max + _BOUND_MARGIN * dx)
    ax.set_ylim(y_min - _BOUND_MARGIN * dy, y_max + _BOUND_MARGIN * dy)
