# Simulated Annealing for TSP

A Python implementation of Simulated Annealing applied to the Travelling
Salesman Problem.

The solver core is generic and independent of TSP; the TSP layer wires up a
distance matrix, 2-opt / swap neighbour operators, three dataset loaders
(bundled Montenegrin cities, TSPLIB `EUC_2D` files, random points), and a
Typer CLI.

## Install

Python 3.11+ and Poetry:

```bash
poetry install
```

## Run

The CLI auto-calibrates the starting temperature and decay factor from the
problem, so a bare command produces sensible results:

```bash
# Default dataset (21 Montenegrin cities), with a GIF of the tour evolution
poetry run annealing run --dataset montenegro --animate

# TSPLIB benchmark instance; bump iterations because n is larger
poetry run annealing run --dataset tsplib --tsplib-path data/berlin52.tsp \
    --iterations 80000

# Arbitrary random instance
poetry run annealing run --dataset random --random-count 50

# Compare all cooling schedules across all bundled datasets
poetry run annealing benchmark
```

`T0` and `alpha` can be overridden explicitly via `--t0` and `--alpha`.
Figures, animations and CSV land in `results/`.

## Develop

```bash
poetry run pytest
poetry run ruff check .
poetry run black --check .
poetry run mypy src
```

## Layout

```
src/annealing/
├── solver.py        # generic SA solver
├── cooling.py       # cooling schedules + factory
├── calibration.py   # T0 / alpha auto-calibration heuristics
├── exceptions.py    # domain exceptions
├── logging_setup.py # rich logging
├── tsp/             # TSP problem, datasets, neighbour operators
├── viz/             # matplotlib plots and GIF animations
└── cli/             # Typer entrypoints
tests/unit/
data/berlin52.tsp    # bundled TSPLIB instance
```

## Algorithm

Each iteration draws a neighbour of the current tour, evaluates the change in
energy `delta`, and accepts it either when it improves the energy or with
probability `exp(-delta / T)` (Metropolis). The temperature `T` follows one of
three schedules: exponential `T0 * alpha^t`, linear between `T0` and `T_min`,
or logarithmic `T0 / log(t + 2)`.

When `T0` is not given, it is set so that ~70% of uphill moves are accepted
at the start (sampled from the initial state). When `alpha` is not given, it
is set so the temperature reaches `1e-4 * T0` after `iterations` steps.

Reference: Kirkpatrick, Gelatt, Vecchi, *Optimization by Simulated Annealing*,
Science 220 (1983).
