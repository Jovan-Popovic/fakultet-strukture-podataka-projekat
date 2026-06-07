from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # pragma: no cover
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


__all__ = ["StrEnum"]
