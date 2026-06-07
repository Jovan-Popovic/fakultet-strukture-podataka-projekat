from __future__ import annotations

import csv
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from annealing.cooling import (
    CoolingSchedule,
    CoolingScheduleName,
    build_cooling_schedule,
)
from annealing.solver import SimulatedAnnealing, SolverResult
from annealing.tsp import Tour, TSPProblem, load_montenegro, load_random, load_tsplib
from annealing.tsp.problem import NeighborKind

logger = logging.getLogger(__name__)

TSPLIB_OPTIMA: dict[str, float] = {"berlin52": 7542.0}


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    iterations: int
    initial_temperature: float
    alpha: float


def run_benchmark(
    output_dir: Path,
    seed: int = 42,
    runs: int = 5,
    console: Console | None = None,
) -> Path:
    console = console or Console()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = _resolve_data_dir()

    rows: list[dict[str, object]] = []

    rows += _benchmark_dataset(
        BenchmarkConfig("montenegro", 20_000, 2.0, 0.9995),
        TSPProblem(load_montenegro()),
        runs=runs,
        seed=seed,
        output_dir=output_dir,
        console=console,
    )

    tsplib_file = data_dir / "berlin52.tsp"
    if tsplib_file.exists():
        rows += _benchmark_dataset(
            BenchmarkConfig("berlin52", 80_000, 300.0, 0.99995),
            TSPProblem(load_tsplib(tsplib_file)),
            runs=runs,
            seed=seed,
            output_dir=output_dir,
            console=console,
            optimum=TSPLIB_OPTIMA["berlin52"],
        )
    else:
        logger.warning("Skipping berlin52: file %s is missing.", tsplib_file)

    rows += _benchmark_dataset(
        BenchmarkConfig("random30", 20_000, 100.0, 0.9995),
        TSPProblem(load_random(count=30, seed=seed)),
        runs=runs,
        seed=seed,
        output_dir=output_dir,
        console=console,
    )

    csv_path = output_dir / "benchmark-results.csv"
    _write_csv(csv_path, rows)
    console.print(f"Saved: {csv_path}")
    return csv_path


def _benchmark_dataset(
    config: BenchmarkConfig,
    problem: TSPProblem,
    runs: int,
    seed: int,
    output_dir: Path,
    console: Console,
    optimum: float | None = None,
) -> list[dict[str, object]]:
    console.rule(
        f"[bold]{config.name} (n={problem.n}, T0={config.initial_temperature}, "
        f"alpha={config.alpha})"
    )
    schedules = _build_schedules(config)
    rows: list[dict[str, object]] = []
    histories: dict[str, list[list[float]]] = {name: [] for name in schedules}

    for schedule_name, schedule in schedules.items():
        lengths: list[float] = []
        times: list[float] = []
        for run_idx in range(runs):
            result, elapsed = _run_single(
                problem, schedule, config.iterations, seed + run_idx
            )
            lengths.append(result.best_energy)
            times.append(elapsed)
            histories[schedule_name].append(result.energy_history)

        gap = (mean(lengths) - optimum) / optimum * 100 if optimum else None
        row = {
            "dataset": config.name,
            "cooling": schedule_name,
            "runs": runs,
            "best": round(min(lengths), 2),
            "avg": round(mean(lengths), 2),
            "stdev": round(stdev(lengths) if len(lengths) > 1 else 0.0, 2),
            "avg_time_s": round(mean(times), 3),
            "optimum": optimum if optimum is not None else "",
            "gap_percent": f"{gap:.2f}" if gap is not None else "",
        }
        rows.append(row)

        gap_str = f", gap={gap:.2f}%" if gap is not None else ""
        console.print(
            f"  {schedule_name:>12}: best={row['best']}, "
            f"avg={row['avg']}+-{row['stdev']}, t={row['avg_time_s']}s{gap_str}"
        )

    _plot_convergence(config.name, histories, runs, output_dir)
    return rows


def _build_schedules(config: BenchmarkConfig) -> dict[str, CoolingSchedule]:
    return {
        CoolingScheduleName.EXPONENTIAL.value: build_cooling_schedule(
            CoolingScheduleName.EXPONENTIAL,
            initial_temperature=config.initial_temperature,
            alpha=config.alpha,
        ),
        CoolingScheduleName.LINEAR.value: build_cooling_schedule(
            CoolingScheduleName.LINEAR,
            initial_temperature=config.initial_temperature,
            minimum_temperature=config.initial_temperature * 1e-4,
            max_iterations=config.iterations,
        ),
        CoolingScheduleName.LOGARITHMIC.value: build_cooling_schedule(
            CoolingScheduleName.LOGARITHMIC,
            initial_temperature=config.initial_temperature,
        ),
    }


def _run_single(
    problem: TSPProblem,
    schedule: CoolingSchedule,
    iterations: int,
    seed: int,
) -> tuple[SolverResult[Tour], float]:
    rng = random.Random(seed)
    start_tour = problem.nearest_neighbor_tour()
    solver = SimulatedAnnealing[Tour](
        initial_state=start_tour,
        energy_fn=problem.tour_length,
        neighbor_fn=lambda state: problem.neighbor(
            state, rng, kind=NeighborKind.TWO_OPT
        ),
        cooling=schedule,
        max_iterations=iterations,
        rng=rng,
    )
    started = time.perf_counter()
    result = solver.run()
    return result, time.perf_counter() - started


def _plot_convergence(
    dataset_name: str,
    histories: dict[str, list[list[float]]],
    runs: int,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for schedule_name, run_histories in histories.items():
        max_length = max(len(h) for h in run_histories)
        padded = np.array([h + [h[-1]] * (max_length - len(h)) for h in run_histories])
        ax.plot(padded.mean(axis=0), label=schedule_name, linewidth=1.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Tour length (mean over {runs} runs)")
    ax.set_title(f"Convergence by cooling schedule - {dataset_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"benchmark-convergence-{dataset_name}.png", dpi=140)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"
