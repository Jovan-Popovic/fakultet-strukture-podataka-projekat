from __future__ import annotations

import math
from itertools import pairwise

import pytest

from annealing.cooling import (
    CoolingScheduleName,
    ExponentialCooling,
    LinearCooling,
    LogarithmicCooling,
    build_cooling_schedule,
)
from annealing.exceptions import InvalidScheduleError


def test_exponential_decay_is_monotonic() -> None:
    schedule = ExponentialCooling(initial_temperature=100.0, alpha=0.95)
    values = [schedule(i) for i in range(20)]
    assert values[0] == 100.0
    assert all(a > b for a, b in pairwise(values))


def test_linear_reaches_minimum_temperature() -> None:
    schedule = LinearCooling(
        initial_temperature=10.0, minimum_temperature=0.1, max_iterations=100
    )
    assert schedule(0) == 10.0
    assert math.isclose(schedule(100), 0.1, rel_tol=1e-9)
    assert schedule(200) == 0.1


def test_logarithmic_is_positive_and_decreasing() -> None:
    schedule = LogarithmicCooling(initial_temperature=100.0)
    assert schedule(10) > schedule(1000) > 0


def test_build_cooling_schedule_returns_expected_type() -> None:
    schedule = build_cooling_schedule(
        CoolingScheduleName.EXPONENTIAL,
        initial_temperature=50.0,
        alpha=0.99,
    )
    assert isinstance(schedule, ExponentialCooling)


def test_build_cooling_schedule_raises_for_unknown_name() -> None:
    with pytest.raises(InvalidScheduleError):
        build_cooling_schedule("does-not-exist", initial_temperature=1.0)
