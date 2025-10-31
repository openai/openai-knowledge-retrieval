from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


def _normalise(value: str) -> str:
    return value.strip().lower()


def _slugify(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    id: str
    filename: str
    title: str
    description: str | None = None

    @property
    def stem(self) -> str:
        return Path(self.filename).stem


"""
This module defines the `DocumentMetadata` dataclass and helpers. The actual
document list is now discovered dynamically from the configured data paths in
`app.main` and is no longer hard-coded here.
"""


def as_dicts(documents: Iterable[DocumentMetadata]) -> list[dict[str, str | None]]:
    return [asdict(document) for document in documents]


__all__ = [
    "DocumentMetadata",
    "as_dicts",
]
