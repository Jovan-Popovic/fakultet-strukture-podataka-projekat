from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from annealing._compat import StrEnum
from annealing.tsp.city import City

Tour = list[int]


class NeighborKind(StrEnum):
    TWO_OPT = "2-opt"
    SWAP = "swap"


class TSPProblem:
    """Symmetric Euclidean TSP with a precomputed distance matrix."""

    def __init__(self, cities: Sequence[City]) -> None:
        if len(cities) < 3:
            raise ValueError("TSP requires at least 3 cities.")
        self._cities: tuple[City, ...] = tuple(cities)
        self._distance_matrix = self._compute_distance_matrix()

    @property
    def cities(self) -> tuple[City, ...]:
        return self._cities

    @property
    def n(self) -> int:
        return len(self._cities)

    @property
    def distance_matrix(self) -> NDArray[np.float64]:
        return self._distance_matrix

    def distance(self, i: int, j: int) -> float:
        return float(self._distance_matrix[i, j])

    def tour_length(self, tour: Tour) -> float:
        idx = np.asarray(tour)
        return float(self._distance_matrix[idx, np.roll(idx, -1)].sum())

    def random_tour(self, rng: random.Random) -> Tour:
        tour = list(range(self.n))
        rng.shuffle(tour)
        return tour

    def nearest_neighbor_tour(self, start: int = 0) -> Tour:
        unvisited = set(range(self.n))
        unvisited.remove(start)
        tour: Tour = [start]
        current = start
        while unvisited:
            nxt = min(unvisited, key=lambda j: self._distance_matrix[current, j])
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return tour

    @staticmethod
    def two_opt_swap(tour: Tour, i: int, k: int) -> Tour:
        return tour[:i] + tour[i : k + 1][::-1] + tour[k + 1 :]

    def random_two_opt(self, tour: Tour, rng: random.Random) -> Tour:
        i = rng.randint(1, self.n - 2)
        k = rng.randint(i + 1, self.n - 1)
        return self.two_opt_swap(tour, i, k)

    def random_swap(self, tour: Tour, rng: random.Random) -> Tour:
        a, b = rng.sample(range(self.n), 2)
        new_tour = tour.copy()
        new_tour[a], new_tour[b] = new_tour[b], new_tour[a]
        return new_tour

    def neighbor(
        self,
        tour: Tour,
        rng: random.Random,
        kind: NeighborKind = NeighborKind.TWO_OPT,
    ) -> Tour:
        if kind is NeighborKind.TWO_OPT:
            return self.random_two_opt(tour, rng)
        return self.random_swap(tour, rng)

    def coordinates(self) -> NDArray[np.float64]:
        return np.array([(c.x, c.y) for c in self._cities], dtype=float)

    def city_names(self) -> list[str]:
        return [c.name for c in self._cities]

    def _compute_distance_matrix(self) -> NDArray[np.float64]:
        coords = np.array([(c.x, c.y) for c in self._cities], dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]
        result: NDArray[np.float64] = np.sqrt((diff**2).sum(axis=-1))
        return result
