from __future__ import annotations

import random
from pathlib import Path

from annealing._compat import StrEnum
from annealing.exceptions import InvalidDatasetError, UnsupportedTSPFormatError
from annealing.tsp.city import City


class DatasetName(StrEnum):
    MONTENEGRO = "montenegro"
    RANDOM = "random"
    TSPLIB = "tsplib"


# Longitude/latitude treated as planar (x, y).
MONTENEGRO_CITIES: list[City] = [
    City("Podgorica", 19.2594, 42.4304),
    City("Niksic", 18.9483, 42.7731),
    City("Pljevlja", 19.3583, 43.3567),
    City("Bijelo Polje", 19.7475, 43.0353),
    City("Berane", 19.8736, 42.8425),
    City("Mojkovac", 19.5828, 42.9605),
    City("Kolasin", 19.5158, 42.8225),
    City("Cetinje", 18.9242, 42.3911),
    City("Budva", 18.8403, 42.2911),
    City("Bar", 19.0944, 42.0931),
    City("Ulcinj", 19.2247, 41.9292),
    City("Tivat", 18.6958, 42.4347),
    City("Kotor", 18.7711, 42.4247),
    City("Herceg Novi", 18.5375, 42.4531),
    City("Plav", 19.9408, 42.5961),
    City("Rozaje", 20.1664, 42.8403),
    City("Andrijevica", 19.7847, 42.7361),
    City("Danilovgrad", 19.1075, 42.5536),
    City("Savnik", 19.0917, 42.9583),
    City("Zabljak", 19.1228, 43.1547),
    City("Pluzine", 18.8419, 43.1539),
]

_SUPPORTED_EDGE_WEIGHT_TYPES = frozenset({"EUC_2D", "ATT"})


def load_montenegro() -> list[City]:
    return list(MONTENEGRO_CITIES)


def load_random(count: int, seed: int | None = None, box: float = 100.0) -> list[City]:
    if count < 3:
        raise InvalidDatasetError("Random dataset requires at least 3 cities.")
    rng = random.Random(seed)
    return [
        City(name=f"C{i}", x=rng.uniform(0, box), y=rng.uniform(0, box))
        for i in range(count)
    ]


def load_tsplib(path: str | Path) -> list[City]:
    """Parse a TSPLIB EUC_2D instance (berlin52, eil51, eil76, ch130, ...)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise InvalidDatasetError(f"TSPLIB file not found: {file_path}")

    edge_weight_type, coord_lines = _split_tsplib_header(
        file_path.read_text(encoding="utf-8", errors="ignore")
    )
    if edge_weight_type and edge_weight_type not in _SUPPORTED_EDGE_WEIGHT_TYPES:
        raise UnsupportedTSPFormatError(edge_weight_type)

    parsed: list[City | None] = [_parse_coord_line(line) for line in coord_lines]
    cities: list[City] = [c for c in parsed if c is not None]
    if not cities:
        raise InvalidDatasetError(f"No NODE_COORD_SECTION found in {file_path}")
    return cities


def load_dataset(
    name: str | DatasetName,
    *,
    path: str | Path | None = None,
    count: int = 30,
    seed: int | None = None,
) -> list[City]:
    try:
        dataset = DatasetName(name)
    except ValueError as exc:
        raise InvalidDatasetError(f"Unknown dataset: {name!r}") from exc

    if dataset is DatasetName.MONTENEGRO:
        return load_montenegro()
    if dataset is DatasetName.RANDOM:
        return load_random(count=count, seed=seed)
    if path is None:
        raise InvalidDatasetError("TSPLIB dataset requires a 'path' argument.")
    return load_tsplib(path)


def _split_tsplib_header(text: str) -> tuple[str | None, list[str]]:
    edge_weight_type: str | None = None
    coord_lines: list[str] = []
    in_coords = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        upper = line.upper()
        if upper.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip().upper()
            continue
        if upper.startswith("NODE_COORD_SECTION"):
            in_coords = True
            continue
        if upper == "EOF":
            break
        if in_coords and line:
            coord_lines.append(line)
    return edge_weight_type, coord_lines


def _parse_coord_line(line: str) -> City | None:
    parts = line.split()
    if len(parts) < 3:
        return None
    return City(name=parts[0], x=float(parts[1]), y=float(parts[2]))
