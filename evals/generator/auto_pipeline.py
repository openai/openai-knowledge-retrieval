from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

from rich import print as rprint

from cli.config import RootConfig
from evals.datasets.schema import EvalItem, EvalRow, write_jsonl
from evals.generator.canonicalize import canonicalize_answer
from evals.generator.candidates import make_context_candidates
from evals.generator.corpus_iter import iterate_chunks
from evals.generator.qgen import generate_one, generate_many_from_context
from evals.generator.sampler import sample_by_mix


@dataclass
class AutoGenSummary:
    total_chunks: int
    total_candidates: int
    total_generated: int
    kept_after_quality: int
    sampled: int
    dataset_path: str


def _quality_gate(rows: List[Dict[str, Any]], cfg: RootConfig) -> List[Dict[str, Any]]:
    """Apply lightweight quality gates: self-consistency and answerability.

    For v1 we do:
    - self-consistency: re-run QGen once for same span and check same canonical answer
    - answerability: ensure correct_answer substr appears in citation_text (proxy)
    """
    out: List[Dict[str, Any]] = []
    # group by citation_text to re-ask
    by_cit: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cit.setdefault(r["citation_text"], []).append(r)

    # Self-consistency: parallel limited regen on small spans only.
    limit = int(getattr(cfg.evals.auto, "quality_regen_limit", 100))
    todo: List[Tuple[str, str]] = []  # (cit_text, difficulty)
    for i, (cit_text, group) in enumerate(by_cit.items()):
        if i >= limit:
            break
        # Only attempt regen if citation is a short span; for large contexts the model may choose
        # different questions, making answer-only consistency unreliable.
        if len(cit_text or "") > 400:
            continue
        diff = group[0].get("difficulty", "easy")
        todo.append((cit_text, diff))

    regen_map: Dict[str, str] = {}
    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        workers = min(
            len(todo), max(1, int(getattr(cfg.evals.auto, "qgen_workers", 4)))
        )
        rprint(
            f"[cyan]Quality gate[/cyan]: regen {len(todo)} citations with workers={workers}"
        )

        def _task(args: Tuple[str, str]) -> Tuple[str, str]:
            cit, diff = args
            row = generate_one(cfg, {"text": cit}, diff)
            ans = canonicalize_answer(row["correct_answer"]) if row else ""
            return (cit, ans)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_task, t): t for t in todo}
            done = 0
            for fut in as_completed(futs):
                cit, ans = fut.result()
                regen_map[cit] = ans
                done += 1
                if done % 10 == 0 or done == len(todo):
                    rprint(f"[dim]quality regen progress[/dim]: {done}/{len(todo)}")

    for cit_text, group in by_cit.items():
        regen_ans = regen_map.get(cit_text)
        for r in group:
            can = canonicalize_answer(r.get("correct_answer", ""))
            # If we performed regen and answer disagrees, drop.
            if regen_ans and regen_ans and regen_ans != can:
                continue
            # answerability proxy
            cit_l = (cit_text or "").lower()
            if can and (can not in cit_text) and (can not in cit_l):
                continue
            out.append(r)
    # dedupe
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for r in out:
        key = (
            r.get("question", "").strip().lower(),
            canonicalize_answer(r.get("correct_answer", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def auto_generate_dataset(cfg: RootConfig) -> Tuple[AutoGenSummary, List[EvalRow]]:
    # 1) Iterate corpus
    scan_docs = getattr(cfg.evals.auto, "scan_docs", None)
    max_docs = (
        scan_docs if scan_docs is not None else max(50, cfg.evals.auto.items_target)
    )
    chunks = []
    for ch in iterate_chunks(cfg, max_docs=max_docs):
        chunks.append(ch)
    rprint(f"[cyan]Auto-evals[/cyan]: iterated chunks={len(chunks)}")

    # 2) Build larger, balanced context candidates (improves question quality/diversity)
    cands = make_context_candidates(
        chunks,
        min_chars=cfg.evals.auto.context_chars_min,
        max_chars=cfg.evals.auto.context_chars_max,
        per_doc_cap=cfg.evals.auto.per_doc_context_cap,
    )
    rprint(f"[cyan]Context candidates[/cyan]: {len(cands)}")

    # 3) QGen
    # QGen: parallel tasks
    from concurrent.futures import ThreadPoolExecutor, as_completed

    gen_rows: List[Dict[str, Any]] = []
    # target: slightly oversample to allow quality gating and sampling
    q_per_ctx = max(1, int(getattr(cfg.evals.auto, "questions_per_context", 3)))
    max_contexts = max(
        1,
        int(max(1, (cfg.evals.auto.items_target * 2) // q_per_ctx)),
    )
    tasks: List[Dict[str, Any]] = cands[:max_contexts]
    workers = max(1, int(getattr(cfg.evals.auto, "qgen_workers", 8)))
    rprint(
        f"[cyan]QGen[/cyan]: contexts={len(tasks)} workers={workers} q_per_ctx={q_per_ctx}"
    )

    def _qtask(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        return generate_many_from_context(cfg, ctx, n_questions=q_per_ctx)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_qtask, ctx): ctx for ctx in tasks}
        done = 0
        for fut in as_completed(futs):
            rows = fut.result() or []
            if rows:
                gen_rows.extend(rows)
            done += 1
            if done % 5 == 0 or done == len(tasks):
                rprint(
                    f"[dim]QGen progress[/dim]: {done}/{len(tasks)} → total_items={len(gen_rows)}"
                )
    rprint(f"[cyan]Generated[/cyan]: {len(gen_rows)} items from {len(tasks)} contexts")

    # 4) Quality gates
    filtered = _quality_gate(gen_rows, cfg)
    rprint(f"[cyan]After quality gates[/cyan]: {len(filtered)}")

    # 5) Sampling
    sampled = sample_by_mix(
        filtered,
        mix=cast(Dict[str, float], dict(cfg.evals.auto.mix)),
        per_tag_cap=cfg.evals.auto.per_tag_cap,
        total=cfg.evals.auto.items_target,
        per_source_cap=cfg.evals.auto.per_source_cap,
    )
    rprint(f"[cyan]Sampled[/cyan]: {len(sampled)}")

    # 6) Canonicalize + wrap rows
    eval_rows: List[EvalRow] = []
    for idx, it in enumerate(sampled, start=1):
        item = EvalItem(
            id=f"ex-{idx:04d}",
            question=it.get("question", ""),
            citation_text=it.get("citation_text", ""),
            correct_answer=canonicalize_answer(it.get("correct_answer", "")),
            source_id=it.get("source_id"),
            page=it.get("page"),
            char_start=it.get("char_start"),
            char_end=it.get("char_end"),
            difficulty=it.get("difficulty"),
            tags=it.get("tags") or [],
        )
        eval_rows.append(EvalRow(item=item))

    # 7) Write JSONL
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path("evals") / "datasets" / f"auto_{ts}.jsonl"
    write_jsonl(eval_rows, out_path)

    summary = AutoGenSummary(
        total_chunks=len(chunks),
        total_candidates=len(cands),
        total_generated=len(gen_rows),
        kept_after_quality=len(filtered),
        sampled=len(sampled),
        dataset_path=str(out_path),
    )
    return summary, eval_rows
