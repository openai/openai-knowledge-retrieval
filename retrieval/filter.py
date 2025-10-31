from __future__ import annotations

from typing import Any, Dict, List
from rich import print as rprint

def apply_similarity_filter(
    hits: List[Dict[str, Any]],
    config: Any,
) -> List[Dict[str, Any]]:
    if not config.enabled or config.threshold <= 0.0:
        return hits

    kept: List[Dict[str, Any]] = []
    dropped = 0
    for hit in hits:
        score = float(hit.get("similarity_score", 0.0))
        if score >= config.threshold:
            kept.append(hit)
        else:
            dropped += 1
    rprint(
        f"[dim]Similarity threshold {config.threshold:.3f} removed {dropped} result(s).[/dim]"
    )
    return kept
