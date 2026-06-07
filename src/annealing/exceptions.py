from __future__ import annotations


class AnnealingError(Exception):
    pass


class InvalidScheduleError(AnnealingError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Unknown cooling schedule: {name!r}")
        self.name = name


class InvalidDatasetError(AnnealingError):
    pass


class UnsupportedTSPFormatError(InvalidDatasetError):
    def __init__(self, edge_weight_type: str) -> None:
        super().__init__(f"Only EUC_2D format is supported, got: {edge_weight_type}")
        self.edge_weight_type = edge_weight_type
