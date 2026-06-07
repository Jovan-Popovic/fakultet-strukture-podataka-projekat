from __future__ import annotations

import math
import random

import pytest

from annealing.calibration import estimate_alpha, estimate_initial_temperature
from annealing.tsp import TSPProblem


def test_estimate_initial_temperature_is_positive(
    small_random_problem: TSPProblem, rng: random.Random
) -> None:
    start = small_random_problem.nearest_neighbor_tour()
    t0 = estimate_initial_temperature(
        initial_state=start,
        energy_fn=small_random_problem.tour_length,
        neighbor_fn=lambda s: small_random_problem.random_two_opt(s, rng),
        rng=rng,
    )
    assert t0 > 0


def test_estimate_initial_temperature_target_acceptance(
    small_random_problem: TSPProblem, rng: random.Random
) -> None:
    """With target acceptance P, T should equal mean_delta / -ln(P)."""
    start = small_random_problem.nearest_neighbor_tour()
    low = estimate_initial_temperature(
        initial_state=start,
        energy_fn=small_random_problem.tour_length,
        neighbor_fn=lambda s: small_random_problem.random_two_opt(s, rng),
        target_acceptance=0.5,
        rng=random.Random(0),
    )
    high = estimate_initial_temperature(
        initial_state=start,
        energy_fn=small_random_problem.tour_length,
        neighbor_fn=lambda s: small_random_problem.random_two_opt(s, rng),
        target_acceptance=0.9,
        rng=random.Random(0),
    )
    # Higher target acceptance => higher T0.
    assert high > low


def test_estimate_alpha_matches_temperature_drop() -> None:
    alpha = estimate_alpha(
        initial_temperature=100.0, final_temperature=0.01, max_iterations=1000
    )
    final = 100.0 * alpha**1000
    assert math.isclose(final, 0.01, rel_tol=1e-9)


def test_estimate_alpha_rejects_non_positive_temperatures() -> None:
    with pytest.raises(ValueError):
        estimate_alpha(initial_temperature=0, final_temperature=1, max_iterations=100)
    with pytest.raises(ValueError):
        estimate_alpha(initial_temperature=1, final_temperature=-1, max_iterations=100)
