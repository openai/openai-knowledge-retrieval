from __future__ import annotations

from typing import Any, Dict, List


def hit_at_k(retrieved_doc_ids: List[str], gold_doc_id: str, k: int = 8) -> float:
    return 1.0 if gold_doc_id and gold_doc_id in retrieved_doc_ids[:k] else 0.0


def recall_at_k(
    retrieved_doc_ids: List[str], gold_doc_ids: List[str], k: int = 8
) -> float:
    if not gold_doc_ids:
        return 0.0
    got = sum(1 for g in set(gold_doc_ids) if g in set(retrieved_doc_ids[:k]))
    return got / len(set(gold_doc_ids))


def mrr(retrieved_doc_ids: List[str], gold_doc_id: str) -> float:
    if not gold_doc_id:
        return 0.0
    for i, d in enumerate(retrieved_doc_ids):
        if d == gold_doc_id:
            return 1.0 / float(i + 1)
    return 0.0


def support_coverage(concatenated_context: str, citation_text: str) -> float:
    if not citation_text:
        return 0.0
    ctx = (concatenated_context or "").lower()
    cit = (citation_text or "").lower()
    return 1.0 if cit in ctx else 0.0


def context_precision(concatenated_context: str, citation_text: str) -> float:
    ctx = concatenated_context or ""
    cit = citation_text or ""
    if not ctx.strip():
        return 0.0
    # naive token overlap precision proxy
    ctx_tokens = set(ctx.lower().split())
    cit_tokens = set(cit.lower().split())
    common = len(ctx_tokens & cit_tokens)
    return common / max(1, len(ctx_tokens))


def compute_retrieval_metrics(
    retrieved: List[Dict[str, Any]],
    gold_doc_id: str | None,
    citation_text: str,
    top_k: int = 8,
) -> Dict[str, float]:
    doc_ids = [r.get("doc_id") for r in retrieved if r.get("doc_id")]
    concatenated = "\n".join(r.get("text", "") for r in retrieved[:top_k])
    return {
        "hit@k": hit_at_k(doc_ids, gold_doc_id or "", k=top_k),
        "recall@k": recall_at_k(doc_ids, [gold_doc_id] if gold_doc_id else [], k=top_k),
        "MRR": mrr(doc_ids, gold_doc_id or ""),
        "context_precision": context_precision(concatenated, citation_text),
        "support_coverage": support_coverage(concatenated, citation_text),
    }
