from __future__ import annotations

import logging
import math
import random
from enum import Enum
from pathlib import Path
from typing import TypeVar

import typer
from rich.console import Console
from rich.table import Table

from annealing.calibration import estimate_alpha, estimate_initial_temperature
from annealing.cli.benchmark import run_benchmark
from annealing.cooling import CoolingScheduleName, build_cooling_schedule
from annealing.exceptions import AnnealingError
from annealing.logging_setup import configure_logging
from annealing.solver import SimulatedAnnealing, SolverResult
from annealing.tsp import DatasetName, NeighborKind, Tour, TSPProblem, load_dataset
from annealing.viz import animate_tour_evolution, plot_summary

E = TypeVar("E", bound=Enum)

app = typer.Typer(
    add_completion=False,
    help="Simulated Annealing for the Travelling Salesman Problem.",
)
console = Console()
logger = logging.getLogger(__name__)

_SNAPSHOT_FRAMES = 120
_FINAL_TEMPERATURE_RATIO = 1e-4
_INITIAL_TOUR_CHOICES = ("nearest", "random")


@app.callback()
def _set_log_level(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    configure_logging(level=logging.DEBUG if verbose else logging.INFO)


@app.command()
def run(
    dataset: str = typer.Option(
        DatasetName.MONTENEGRO.value,
        metavar="[montenegro|random|tsplib]",
        help="Which dataset to solve.",
    ),
    tsplib_path: Path | None = typer.Option(
        None, help="Path to a .tsp file (required for --dataset tsplib)."
    ),
    random_count: int = typer.Option(30, help="Number of cities for --dataset random."),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
    iterations: int = typer.Option(20_000, help="Maximum number of iterations."),
    t0: float | None = typer.Option(
        None, help="Starting temperature (auto-calibrated if omitted)."
    ),
    t_min: float = typer.Option(1e-3, help="Floor temperature for linear cooling."),
    alpha: float | None = typer.Option(
        None, help="Decay factor for exponential cooling (auto if omitted)."
    ),
    cooling: str = typer.Option(
        CoolingScheduleName.EXPONENTIAL.value,
        metavar="[exponential|linear|logarithmic]",
        help="Cooling schedule.",
    ),
    neighbor: str = typer.Option(
        NeighborKind.TWO_OPT.value,
        metavar="[2-opt|swap]",
        help="Neighbour move type.",
    ),
    initial_tour: str = typer.Option(
        "nearest",
        metavar="[nearest|random]",
        help="Initial tour selection.",
    ),
    plot: bool = typer.Option(True, help="Save a summary figure."),
    animate: bool = typer.Option(False, help="Save a GIF of the tour evolution."),
    output_dir: Path = typer.Option(Path("results"), help="Output directory."),
) -> None:
    dataset_enum = _parse_enum(DatasetName, dataset, "--dataset")
    cooling_enum = _parse_enum(CoolingScheduleName, cooling, "--cooling")
    neighbor_enum = _parse_enum(NeighborKind, neighbor, "--neighbor")
    if initial_tour not in _INITIAL_TOUR_CHOICES:
        raise typer.BadParameter(
            f"--initial-tour must be one of {_INITIAL_TOUR_CHOICES}, got {initial_tour!r}."
        )

    try:
        cities = load_dataset(
            dataset_enum, path=tsplib_path, count=random_count, seed=seed
        )
        problem = TSPProblem(cities)
    except AnnealingError as exc:
        raise typer.BadParameter(str(exc)) from exc

    rng = random.Random(seed)
    start_tour = (
        problem.nearest_neighbor_tour()
        if initial_tour == "nearest"
        else problem.random_tour(rng)
    )

    resolved_t0 = t0 or estimate_initial_temperature(
        initial_state=start_tour,
        energy_fn=problem.tour_length,
        neighbor_fn=lambda s: problem.neighbor(s, rng, kind=neighbor_enum),
        rng=rng,
    )
    # Logarithmic cooling decays much slower than exponential. Scale T0 down
    # by log(N+2) so T at the final iteration lands in a comparable range.
    if t0 is None and cooling_enum is CoolingScheduleName.LOGARITHMIC:
        resolved_t0 /= math.log(iterations + 2)
    resolved_alpha = alpha or estimate_alpha(
        initial_temperature=resolved_t0,
        final_temperature=resolved_t0 * _FINAL_TEMPERATURE_RATIO,
        max_iterations=iterations,
    )

    schedule = build_cooling_schedule(
        cooling_enum,
        initial_temperature=resolved_t0,
        minimum_temperature=t_min,
        max_iterations=iterations,
        alpha=resolved_alpha,
    )

    solver = SimulatedAnnealing[Tour](
        initial_state=start_tour,
        energy_fn=problem.tour_length,
        neighbor_fn=lambda state: problem.neighbor(state, rng, kind=neighbor_enum),
        cooling=schedule,
        max_iterations=iterations,
        snapshot_every=max(iterations // _SNAPSHOT_FRAMES, 1) if animate else None,
        rng=rng,
    )

    console.rule(f"[bold]Simulated Annealing - {dataset_enum.value}")
    console.print(
        f"Cities: {problem.n} | iterations: {iterations} | cooling: {cooling_enum.value}"
        f" | neighbor: {neighbor_enum.value} | start: {initial_tour}"
        f" | T0={resolved_t0:.4g} | alpha={resolved_alpha:.6g}"
    )

    with console.status("Optimising..."):
        result = solver.run()

    _print_result_table(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    if plot:
        path = plot_summary(
            problem,
            result,
            output_dir / f"{dataset_enum.value}-summary.png",
            title=f"Simulated Annealing - {dataset_enum.value} ({cooling_enum.value})",
        )
        console.print(f"Saved: {path}")
    if animate:
        gif = animate_tour_evolution(
            problem, result, output_dir / f"{dataset_enum.value}-animation.gif"
        )
        console.print(f"Saved: {gif}")


@app.command()
def benchmark(
    output_dir: Path = typer.Option(Path("results"), help="Output directory."),
    seed: int = typer.Option(42, help="Base random seed."),
    runs: int = typer.Option(5, help="Repeats per configuration."),
) -> None:
    run_benchmark(output_dir=output_dir, seed=seed, runs=runs, console=console)


def _parse_enum(enum_class: type[E], value: str, option_name: str) -> E:
    try:
        return enum_class(value)
    except ValueError as exc:
        valid = ", ".join(repr(e.value) for e in enum_class)
        raise typer.BadParameter(
            f"{option_name}: {value!r} is not one of {valid}."
        ) from exc


def _print_result_table(result: SolverResult[Tour]) -> None:
    initial_length = result.energy_history[0]
    improvement = 1 - result.best_energy / initial_length if initial_length else 0.0

    table = Table(title="Result")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Initial length", f"{initial_length:.2f}")
    table.add_row("Best length", f"{result.best_energy:.2f}")
    table.add_row("Improvement", f"{improvement:.2%}")
    table.add_row("Accepted", str(result.accepted_moves))
    table.add_row("Rejected", str(result.rejected_moves))
    console.print(table)


if __name__ == "__main__":
    app()
