from __future__ import annotations

import random

from annealing.cooling import ExponentialCooling
from annealing.solver import SimulatedAnnealing
from annealing.tsp import TSPProblem


def test_sa_improves_random_tour(small_random_problem: TSPProblem) -> None:
    """SA must beat a random starting tour by at least 20%."""
    rng = random.Random(0)
    start_tour = small_random_problem.random_tour(rng)
    start_length = small_random_problem.tour_length(start_tour)

    solver = SimulatedAnnealing(
        initial_state=start_tour,
        energy_fn=small_random_problem.tour_length,
        neighbor_fn=lambda s: small_random_problem.random_two_opt(s, rng),
        cooling=ExponentialCooling(initial_temperature=100, alpha=0.999),
        max_iterations=5_000,
        rng=rng,
    )
    result = solver.run()
    assert result.best_energy < start_length * 0.8


def test_sa_records_history_and_snapshots(small_random_problem: TSPProblem) -> None:
    rng = random.Random(0)
    start_tour = small_random_problem.nearest_neighbor_tour()

    solver = SimulatedAnnealing(
        initial_state=start_tour,
        energy_fn=small_random_problem.tour_length,
        neighbor_fn=lambda s: small_random_problem.random_two_opt(s, rng),
        cooling=ExponentialCooling(initial_temperature=50, alpha=0.99),
        max_iterations=500,
        snapshot_every=50,
        rng=rng,
    )
    result = solver.run()

    assert len(result.energy_history) == 501  # initial + N iterations
    assert len(result.temperature_history) == 500
    assert len(result.snapshots) > 0
    assert 0.0 <= result.acceptance_rate <= 1.0


def test_solver_rejects_zero_iterations(small_random_problem: TSPProblem) -> None:
    import pytest

    rng = random.Random(0)
    with pytest.raises(ValueError):
        SimulatedAnnealing(
            initial_state=small_random_problem.nearest_neighbor_tour(),
            energy_fn=small_random_problem.tour_length,
            neighbor_fn=lambda s: small_random_problem.random_two_opt(s, rng),
            cooling=ExponentialCooling(initial_temperature=10),
            max_iterations=0,
            rng=rng,
        )
