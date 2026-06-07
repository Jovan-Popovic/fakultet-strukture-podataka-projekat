from __future__ import annotations

import random

import pytest

from annealing.tsp import TSPProblem, load_random


@pytest.fixture
def small_random_problem() -> TSPProblem:
    return TSPProblem(load_random(count=15, seed=1))


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0)
