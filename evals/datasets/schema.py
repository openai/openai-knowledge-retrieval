from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, ValidationError
import json


class EvalItem(BaseModel):
    id: Optional[str] = None
    question: str
    citation_text: str
    correct_answer: str
    source_id: Optional[str] = None
    page: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    difficulty: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class EvalRow(BaseModel):
    item: EvalItem


def load_and_validate_jsonl(path: str | Path) -> List[EvalRow]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    rows: List[EvalRow] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}")

            # Allow legacy/flat rows by coercing into {"item": {...}} shape.
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Invalid JSON object on line {i}: expected dict, got {type(obj)}"
                )

            try:
                rows.append(EvalRow(**obj))
            except ValidationError as e:
                raise ValueError(f"Schema error on line {i}: {e}")
    if not rows:
        raise ValueError("Dataset is empty after parsing")
    return rows


def write_jsonl(rows: Iterable[EvalRow | Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, EvalRow):
                data = r.model_dump()
            else:
                data = r
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
