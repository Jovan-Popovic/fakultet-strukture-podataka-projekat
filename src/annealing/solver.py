from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from annealing.cooling import CoolingSchedule

State = TypeVar("State")

logger = logging.getLogger(__name__)

_MIN_TEMPERATURE = 1e-12


@dataclass
class SolverResult(Generic[State]):
    best_state: State
    best_energy: float
    energy_history: list[float] = field(default_factory=list)
    temperature_history: list[float] = field(default_factory=list)
    accepted_moves: int = 0
    rejected_moves: int = 0
    snapshots: list[tuple[int, State, float]] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_moves + self.rejected_moves
        return self.accepted_moves / total if total else 0.0


class SimulatedAnnealing(Generic[State]):
    """Generic SA optimizer; the caller supplies energy and neighbour functions."""

    def __init__(
        self,
        initial_state: State,
        energy_fn: Callable[[State], float],
        neighbor_fn: Callable[[State], State],
        cooling: CoolingSchedule,
        max_iterations: int = 10_000,
        snapshot_every: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self._initial_state = initial_state
        self._energy_fn = energy_fn
        self._neighbor_fn = neighbor_fn
        self._cooling = cooling
        self._max_iterations = max_iterations
        self._snapshot_every = snapshot_every
        self._rng = rng or random.Random()

    def run(self) -> SolverResult[State]:
        current = self._initial_state
        current_energy = self._energy_fn(current)
        best_state, best_energy = current, current_energy

        result: SolverResult[State] = SolverResult(
            best_state=best_state, best_energy=best_energy
        )
        result.energy_history.append(current_energy)

        logger.debug(
            "SA started: initial_energy=%.4f max_iterations=%d",
            current_energy,
            self._max_iterations,
        )

        for iteration in range(self._max_iterations):
            temperature = max(self._cooling(iteration), _MIN_TEMPERATURE)

            candidate = self._neighbor_fn(current)
            candidate_energy = self._energy_fn(candidate)
            delta_energy = candidate_energy - current_energy

            if self._accept(delta_energy, temperature):
                current, current_energy = candidate, candidate_energy
                result.accepted_moves += 1
                if current_energy < best_energy:
                    best_state, best_energy = current, current_energy
            else:
                result.rejected_moves += 1

            result.energy_history.append(current_energy)
            result.temperature_history.append(temperature)

            if (
                self._snapshot_every is not None
                and iteration % self._snapshot_every == 0
            ):
                result.snapshots.append((iteration, current, current_energy))

        result.best_state = best_state
        result.best_energy = best_energy
        if self._snapshot_every is not None:
            result.snapshots.append((self._max_iterations, best_state, best_energy))

        logger.debug(
            "SA finished: best_energy=%.4f acceptance_rate=%.2f%%",
            best_energy,
            100 * result.acceptance_rate,
        )
        return result

    def _accept(self, delta_energy: float, temperature: float) -> bool:
        if delta_energy < 0:
            return True
        return self._rng.random() < math.exp(-delta_energy / temperature)
