from __future__ import annotations

from typing import Any, Dict, List

from rapidfuzz import fuzz


def extract_spans(
    answer: str, chunks: List[Dict[str, Any]], max_per_source: int = 3
) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for c in chunks[:max_per_source]:
        text = c["text"]
        # fuzzy match a sentence from chunk to the answer
        score = fuzz.partial_ratio(answer, text)
        if score > 50:
            citations.append(
                {
                    "doc_id": c["doc_id"],
                    "title": c.get("title") or c.get("metadata", {}).get("title"),
                    "page": c.get("page"),
                    "span_start": -1,
                    "span_end": -1,
                }
            )
    return citations
