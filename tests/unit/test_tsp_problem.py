from __future__ import annotations

import random

import pytest

from annealing.tsp import City, NeighborKind, TSPProblem


@pytest.fixture
def unit_square() -> TSPProblem:
    # Square with sides of length 1; the optimal tour has length 4.
    return TSPProblem(
        [
            City("A", 0, 0),
            City("B", 1, 0),
            City("C", 1, 1),
            City("D", 0, 1),
        ]
    )


def test_tour_length_of_unit_square_is_four(unit_square: TSPProblem) -> None:
    assert unit_square.tour_length([0, 1, 2, 3]) == 4.0


def test_two_opt_swap_preserves_city_set(unit_square: TSPProblem) -> None:
    tour = [0, 1, 2, 3]
    new_tour = unit_square.two_opt_swap(tour, 1, 2)
    assert sorted(new_tour) == sorted(tour)
    assert new_tour != tour


def test_nearest_neighbor_returns_full_permutation(unit_square: TSPProblem) -> None:
    tour = unit_square.nearest_neighbor_tour(start=0)
    assert sorted(tour) == list(range(unit_square.n))


def test_random_swap_is_permutation(unit_square: TSPProblem) -> None:
    rng = random.Random(0)
    tour = [0, 1, 2, 3]
    new_tour = unit_square.random_swap(tour, rng)
    assert sorted(new_tour) == sorted(tour)


def test_neighbor_dispatches_to_correct_operator(unit_square: TSPProblem) -> None:
    rng = random.Random(0)
    tour = [0, 1, 2, 3]
    swap_result = unit_square.neighbor(tour, rng, kind=NeighborKind.SWAP)
    assert sorted(swap_result) == sorted(tour)
