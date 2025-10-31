from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def sample_by_mix(
    items: List[Dict[str, Any]],
    mix: Dict[str, float],
    per_tag_cap: int = 30,
    total: int = 150,
    per_source_cap: int | None = None,
) -> List[Dict[str, Any]]:
    """Sample items by difficulty mix and tag cap.

    Simple round-robin sampling honoring per-tag caps.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        buckets[(it.get("difficulty") or "easy").lower()].append(it)
    out: List[Dict[str, Any]] = []
    tag_counts: Dict[str, int] = defaultdict(int)
    source_counts: Dict[str, int] = defaultdict(int)
    # target per difficulty
    targets = {k: int(total * mix.get(k, 0.0)) for k in {"easy", "medium", "hard"}}
    for diff, target in targets.items():
        pool = buckets.get(diff, [])
        i = 0
        while (
            i < len(pool)
            and len(out) < total
            and sum(1 for x in out if x.get("difficulty") == diff) < target
        ):
            cand = pool[i]
            tags = cand.get("tags") or []
            src = cand.get("source_id") or ""
            if all(tag_counts[t] < per_tag_cap for t in tags) and (
                per_source_cap is None or source_counts[src] < per_source_cap
            ):
                out.append(cand)
                for t in tags:
                    tag_counts[t] += 1
                if src:
                    source_counts[src] += 1
            i += 1
    # Fill any remaining slots
    if len(out) < total:
        rest = [it for lst in buckets.values() for it in lst]
        for cand in rest:
            if len(out) >= total:
                break
            tags = cand.get("tags") or []
            src = cand.get("source_id") or ""
            if all(tag_counts[t] < per_tag_cap for t in tags) and (
                per_source_cap is None or source_counts[src] < per_source_cap
            ):
                out.append(cand)
                for t in tags:
                    tag_counts[t] += 1
                if src:
                    source_counts[src] += 1
    return out[:total]
