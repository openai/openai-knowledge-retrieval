from __future__ import annotations

import re
from typing import Tuple


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    p = set(_normalize(pred).split())
    g = set(_normalize(gold).split())
    if not p or not g:
        return 0.0
    inter = len(p & g)
    prec = inter / len(p)
    rec = inter / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def em_f1(pred: str, gold: str) -> Tuple[float, float]:
    return exact_match(pred, gold), f1_score(pred, gold)
