from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from prompts.loader import load_prompt
from pydantic import BaseModel
from typing import Literal
from cli.config import RerankConfig

try:
    from rich import print as rprint
except ImportError:  # pragma: no cover - optional dependency

    def rprint(*args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)


class RelevanceLabel(BaseModel):
    label: Literal["Yes", "No"]


def _normalize(token: str) -> str:
    return token.strip().strip('"').lower()


def _collect_logprobs(response: Any) -> List[Any]:
    entries: List[Any] = []
    for message in response.output or []:
        for block in getattr(message, "content", []) or []:
            block_logprobs = getattr(block, "logprobs", None)
            if block_logprobs:
                entries.extend(block_logprobs)
    return entries


def _find_logprob(entries: List[Any], target: str) -> float:
    target_norm = target.lower()
    for item in entries:
        token = _normalize(str(getattr(item, "token", "")))
        lp = getattr(item, "logprob", None)
        if token == target_norm and lp is not None:
            return float(lp)
        for top in getattr(item, "top_logprobs", []) or []:
            top_token = _normalize(str(getattr(top, "token", "")))
            top_lp = getattr(top, "logprob", None)
            if top_token == target_norm and top_lp is not None:
                return float(top_lp)
    raise ValueError(f"Logprob for '{target}' not found")


def _softmax_score(yes_lp: float, no_lp: float) -> float:
    max_lp = max(yes_lp, no_lp)
    yes_exp = math.exp(yes_lp - max_lp)
    no_exp = math.exp(no_lp - max_lp)
    return yes_exp / (yes_exp + no_exp)


def _evaluate_candidate(
    client: Any,
    *,
    model: str,
    developer_prompt: str,
    query: str,
    passage: str,
) -> Dict[str, Any]:
    messages = [
        {"role": "developer", "content": developer_prompt},
        {
            "role": "user",
            "content": f"Query:\n{query}\n\nPassage:\n{passage}",
        },
    ]
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=RelevanceLabel,
        include=["message.output_text.logprobs"],
        top_logprobs=5,
    )
    logprobs = _collect_logprobs(response)
    yes_lp = _find_logprob(logprobs, "yes")
    no_lp = _find_logprob(logprobs, "no")
    rerank_score = _softmax_score(yes_lp, no_lp)
    rprint(
        f"[dim]Rerank logprob[/dim] label={response.output_parsed.label} "
        f"yes_lp={yes_lp:.3f} no_lp={no_lp:.3f} rerank_score={rerank_score:.3f}"
    )
    return {
        "label": response.output_parsed.label,
        "logprob_yes": yes_lp,
        "logprob_no": no_lp,
        "rerank_score": rerank_score,
    }


def rerank_with_cross_encoder(
    client: Any,
    *,
    config: RerankConfig,
    query: str,
    hits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not config.enabled or not hits:
        return hits

    prompt = load_prompt(config.prompt_path)
    limit = min(len(hits), config.max_candidates)

    def _worker(idx: int) -> Dict[str, Any]:
        return _evaluate_candidate(
            client=client,
            model=config.model,
            developer_prompt=prompt,
            query=query,
            passage=hits[idx]["text"],
        )

    with ThreadPoolExecutor(max_workers=min(8, max(1, limit))) as executor:
        future_map = {executor.submit(_worker, idx): idx for idx in range(limit)}
        for future in as_completed(future_map):
            idx = future_map[future]
            data = future.result()
            hits[idx]["rerank_score"] = data["rerank_score"]
            hits[idx]["rerank_label"] = data["label"]
            hits[idx]["rerank_logprob_yes"] = data["logprob_yes"]
            hits[idx]["rerank_logprob_no"] = data["logprob_no"]

    for entry in hits:
        entry.setdefault("rerank_score", entry.get("similarity_score", 0.0))

    threshold = config.score_threshold
    if threshold > 0.0:
        before = len(hits)
        hits = [h for h in hits if h["rerank_score"] >= threshold]
        rprint(
            f"[dim]Rerank threshold {threshold:.3f} removed {before - len(hits)} result(s).[/dim]"
        )

    hits.sort(key=lambda x: x.get("rerank_score", x.get("score", 0.0)), reverse=True)

    top_cap = min(len(hits), config.max_candidates, 5)
    if top_cap:
        rprint("[dim]Top reranked results:[/dim]")
        for i, hit in enumerate(hits[:top_cap], start=1):
            snippet = " ".join(hit["text"].strip().split())
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            label = hit.get("rerank_label") or ""
            summary = (
                f"  • #{i} Rerank score={hit['rerank_score']:.3f} Similarity score={hit['similarity_score']:.3f} doc={hit.get('doc_id') or '-'}"
            )
            if label:
                summary += f" ({label})"
            if snippet:
                summary += f" — {snippet}"
            rprint(summary)

    return hits
