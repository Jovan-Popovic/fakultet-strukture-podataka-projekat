from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from annealing._compat import StrEnum
from annealing.exceptions import InvalidScheduleError


class CoolingScheduleName(StrEnum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"


class CoolingSchedule(Protocol):
    def __call__(self, iteration: int) -> float: ...


@dataclass(frozen=True)
class ExponentialCooling:
    """T(t) = T0 * alpha**t."""

    initial_temperature: float
    alpha: float = 0.995

    def __call__(self, iteration: int) -> float:
        return self.initial_temperature * (self.alpha**iteration)


@dataclass(frozen=True)
class LinearCooling:
    """Linear ramp from T0 to T_min over max_iterations steps."""

    initial_temperature: float
    minimum_temperature: float
    max_iterations: int

    def __call__(self, iteration: int) -> float:
        fraction = iteration / max(self.max_iterations, 1)
        value = self.initial_temperature - fraction * (
            self.initial_temperature - self.minimum_temperature
        )
        return max(value, self.minimum_temperature)


@dataclass(frozen=True)
class LogarithmicCooling:
    """T(t) = T0 / log(t + 2). Hajek (1988); rarely useful in practice."""

    initial_temperature: float

    def __call__(self, iteration: int) -> float:
        return self.initial_temperature / math.log(iteration + 2)


_SCHEDULE_BUILDERS: dict[CoolingScheduleName, Callable[..., CoolingSchedule]] = {
    CoolingScheduleName.EXPONENTIAL: lambda **kw: ExponentialCooling(
        initial_temperature=kw["initial_temperature"],
        alpha=kw.get("alpha", 0.995),
    ),
    CoolingScheduleName.LINEAR: lambda **kw: LinearCooling(
        initial_temperature=kw["initial_temperature"],
        minimum_temperature=kw.get("minimum_temperature", 1e-3),
        max_iterations=kw["max_iterations"],
    ),
    CoolingScheduleName.LOGARITHMIC: lambda **kw: LogarithmicCooling(
        initial_temperature=kw["initial_temperature"],
    ),
}


def build_cooling_schedule(
    name: str | CoolingScheduleName, **kwargs: float | int
) -> CoolingSchedule:
    try:
        key = CoolingScheduleName(name)
    except ValueError as exc:
        raise InvalidScheduleError(str(name)) from exc
    return _SCHEDULE_BUILDERS[key](**kwargs)
