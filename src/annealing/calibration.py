from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import TypeVar

State = TypeVar("State")


def estimate_initial_temperature(
    initial_state: State,
    energy_fn: Callable[[State], float],
    neighbor_fn: Callable[[State], State],
    sample_size: int = 100,
    target_acceptance: float = 0.7,
    rng: random.Random | None = None,
) -> float:
    """Pick T0 so that ~`target_acceptance` of uphill moves are accepted at start.

    Samples random neighbours of the initial state and uses the mean positive
    energy delta. With acceptance probability P = exp(-delta / T), this gives
    T = -delta / ln(P).
    """
    rng = rng or random.Random()
    base_energy = energy_fn(initial_state)
    positive_deltas: list[float] = []

    for _ in range(sample_size):
        delta = energy_fn(neighbor_fn(initial_state)) - base_energy
        if delta > 0:
            positive_deltas.append(delta)

    if not positive_deltas:
        return 1.0

    mean_delta = sum(positive_deltas) / len(positive_deltas)
    return mean_delta / -math.log(target_acceptance)


def estimate_alpha(
    initial_temperature: float,
    final_temperature: float,
    max_iterations: int,
) -> float:
    """Pick alpha so that T0 * alpha**N == T_final."""
    if initial_temperature <= 0 or final_temperature <= 0:
        raise ValueError("Temperatures must be positive.")
    return float((final_temperature / initial_temperature) ** (1 / max_iterations))
