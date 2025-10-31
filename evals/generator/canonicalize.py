from __future__ import annotations

import re
from typing import Any, Dict


UNIT_MAP: Dict[str, str] = {
    "months": "mo",
    "month": "mo",
    "years": "yr",
    "year": "yr",
    "hours": "h",
    "hour": "h",
    "amp": "a",
    "amps": "a",
    "ampere": "a",
    "amperes": "a",
    "volts": "v",
    "volt": "v",
    "watts": "w",
    "watt": "w",
    "newton meters": "nm",
    "newton-meters": "nm",
}


def canonicalize_text(s: Any) -> str:
    # Coerce non-strings (e.g., numbers) to string before normalization
    if s is None:
        s = ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .;:,")
    return s


def canonicalize_answer(ans: str) -> str:
    s = canonicalize_text(ans)
    # Normalize common units
    # e.g., "24 months" -> "24 mo", "4.5 A" -> "4.5 a"
    for k, v in UNIT_MAP.items():
        s = re.sub(rf"\b{k}\b", v, s)
    # Normalize spacing in numbers + units (e.g., "4.5a" -> "4.5 a")
    s = re.sub(r"(\d)([a-zA-Z]+)\b", r"\1 \2", s)
    return s
