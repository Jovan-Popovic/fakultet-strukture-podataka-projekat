from __future__ import annotations

from pathlib import Path

import pytest

from annealing.exceptions import InvalidDatasetError, UnsupportedTSPFormatError
from annealing.tsp import DatasetName, load_dataset, load_montenegro, load_random


def test_load_montenegro_has_21_cities() -> None:
    cities = load_montenegro()
    assert len(cities) == 21
    assert all(c.name for c in cities)


def test_load_random_is_deterministic_with_seed() -> None:
    a = load_random(count=5, seed=42)
    b = load_random(count=5, seed=42)
    assert [(c.x, c.y) for c in a] == [(c.x, c.y) for c in b]


def test_load_random_rejects_too_few_cities() -> None:
    with pytest.raises(InvalidDatasetError):
        load_random(count=2)


def test_load_dataset_dispatches_by_name() -> None:
    via_enum = load_dataset(DatasetName.MONTENEGRO)
    via_string = load_dataset("montenegro")
    assert len(via_enum) == len(via_string) == 21


def test_load_tsplib_rejects_unknown_format(tmp_path: Path) -> None:
    fake = tmp_path / "fake.tsp"
    fake.write_text(
        "NAME: fake\nEDGE_WEIGHT_TYPE: GEO\nNODE_COORD_SECTION\n1 1 2\nEOF\n"
    )
    with pytest.raises(UnsupportedTSPFormatError):
        load_dataset(DatasetName.TSPLIB, path=fake)


def test_load_tsplib_parses_minimal_instance(tmp_path: Path) -> None:
    fake = tmp_path / "tiny.tsp"
    fake.write_text(
        "NAME: tiny\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n"
        "1 0.0 0.0\n2 1.0 0.0\n3 1.0 1.0\nEOF\n"
    )
    cities = load_dataset(DatasetName.TSPLIB, path=fake)
    assert [c.name for c in cities] == ["1", "2", "3"]
