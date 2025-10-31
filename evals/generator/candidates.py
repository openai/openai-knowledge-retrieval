from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, DefaultDict
from collections import defaultdict


def mine_candidate_spans(chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract simple candidate fact spans from chunks.

    This v1 stub uses regexes for numbers + units and definition patterns.
    Each candidate is a dict with keys: text, source_id, page, char_start, char_end, tags
    """
    candidates: List[Dict[str, Any]] = []
    qty_pat = re.compile(
        r"\b(?:\d+[\d\.\,]*\s*(?:%|v|a|w|nm|mo|yr|kg|g|lb|hz|mhz|ghz))\b", re.I
    )
    def_pat = re.compile(
        r"\b([A-Z][A-Za-z0-9_\- ]{2,})\s+is\s+(?:defined as|the)\b", re.I
    )
    for ch in chunks:
        text = ch.get("text", "")
        source_id = ch.get("doc_id") or ch.get("source_id")
        page = ch.get("page")
        for m in qty_pat.finditer(text):
            span = m.group(0)
            candidates.append(
                {
                    "text": span,
                    "source_id": source_id,
                    "page": page,
                    "char_start": m.start(),
                    "char_end": m.end(),
                    "tags": ["quantity"],
                }
            )
        for m in def_pat.finditer(text):
            span = m.group(0)
            candidates.append(
                {
                    "text": span,
                    "source_id": source_id,
                    "page": page,
                    "char_start": m.start(),
                    "char_end": m.end(),
                    "tags": ["definition"],
                }
            )
    return candidates


def make_context_candidates(
    chunks: Iterable[Dict[str, Any]],
    *,
    min_chars: int = 1200,
    max_chars: int = 2400,
    per_doc_cap: int = 10,
) -> List[Dict[str, Any]]:
    """Build larger, context-rich candidates from chunks with per-doc balancing.

    Heuristics:
    - Group by source/doc id and sample up to per_doc_cap contexts per doc.
    - Use the full chunk text up to max_chars. If text is shorter than min_chars,
      still accept it (we don't attempt ordering/merging since scroll order may be arbitrary).
    - Tag context type if patterns suggest tables/code/numbers to aid diversification.
    """
    by_doc: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ch in chunks:
        doc_id = (
            ch.get("doc_id")
            or ch.get("source_id")
            or ch.get("metadata", {}).get("file_id")
        )
        if not doc_id:
            # fallback bucket
            doc_id = "_unknown"
        by_doc[doc_id].append(ch)

    def infer_tags(text: str) -> List[str]:
        tags: List[str] = []
        if "|" in text and "---" in text:
            tags.append("table")
        if "```" in text or "def " in text or "class " in text:
            tags.append("code")
        if any(sym in text for sym in ["%", "+/-", "+/−", "±"]):
            tags.append("numeric")
        if "TODO" in text or "FIXME" in text:
            tags.append("notes")
        return tags

    contexts: List[Dict[str, Any]] = []
    for doc_id, lst in by_doc.items():
        taken = 0
        for ch in lst:
            if taken >= per_doc_cap:
                break
            text = (ch.get("text") or "").strip()
            if not text:
                continue
            snippet = text[:max_chars]
            tags = infer_tags(snippet)
            # include heading-derived tags if available
            meta = ch.get("metadata") or {}
            for h in (
                (meta.get("heading") or [])
                if isinstance(meta.get("heading"), list)
                else []
            ):
                if isinstance(h, str):
                    tags.append("heading:" + h.strip()[:50])
            if isinstance(meta.get("headings"), list):
                for h in meta.get("headings"):
                    if isinstance(h, str):
                        tags.append("heading:" + h.strip()[:50])
            contexts.append(
                {
                    "text": snippet,
                    "source_id": doc_id,
                    "page": ch.get("page"),
                    "char_start": 0,
                    "char_end": len(snippet),
                    "tags": tags,
                }
            )
            taken += 1
    return contexts
