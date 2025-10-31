from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Page:
    number: int
    text: str


@dataclass
class RawDocument:
    doc_id: str
    title: str
    path: str
    pages: List[Page]
    text: str
    metadata: Dict[str, Any]


@dataclass
class ChunkRecord:
    id: str
    doc_id: str
    text: str
    page: Optional[int]
    start_offset: Optional[int]
    end_offset: Optional[int]
    metadata: Dict[str, Any]
