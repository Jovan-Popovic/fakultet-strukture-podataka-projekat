"""Simulated Annealing solver and TSP application."""

from __future__ import annotations

from annealing.cooling import (
    CoolingSchedule,
    ExponentialCooling,
    LinearCooling,
    LogarithmicCooling,
    build_cooling_schedule,
)
from annealing.exceptions import (
    AnnealingError,
    InvalidDatasetError,
    InvalidScheduleError,
)
from annealing.solver import SimulatedAnnealing, SolverResult
from annealing.tsp.problem import TSPProblem

__version__ = "0.1.0"

__all__ = [
    "AnnealingError",
    "CoolingSchedule",
    "ExponentialCooling",
    "InvalidDatasetError",
    "InvalidScheduleError",
    "LinearCooling",
    "LogarithmicCooling",
    "SimulatedAnnealing",
    "SolverResult",
    "TSPProblem",
    "build_cooling_schedule",
]
