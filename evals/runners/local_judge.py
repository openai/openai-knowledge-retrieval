from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from rich import print as rprint

from cli.config import RootConfig
from evals.datasets.schema import load_and_validate_jsonl
from evals.generator.canonicalize import canonicalize_answer
from evals.graders.schema import JudgeDecision
from evals.graders.judge_prompts import SINGLE_RUBRIC_QA_GROUNDED
from evals.graders.runner import GraderResult, run_config_graders, summarize_graders
from retrieval.pipeline import answer_query
from concurrent.futures import ThreadPoolExecutor, as_completed


def _heuristic_judge(
    model_answer: str, correct_answer: str, citation_text: str
) -> JudgeDecision:
    ma = canonicalize_answer(model_answer or "")
    ca = canonicalize_answer(correct_answer or "")
    ct = (citation_text or "").lower()
    correctness = 1.0 if ma == ca or ca in ma or ma in ca else 0.0
    grounding = 1.0 if ca and ca in ct else 0.0
    decision = "pass" if correctness >= 1.0 and grounding >= 0.8 else "fail"
    return JudgeDecision(
        decision=decision,
        correctness=correctness,
        grounding=grounding,
        rationale="heuristic",
    )


def _llm_judge(
    cfg: RootConfig, model_answer: str, correct_answer: str, citation_text: str
) -> JudgeDecision:
    from cli.env_utils import build_openai_client

    client = build_openai_client(cfg)
    prompt = SINGLE_RUBRIC_QA_GROUNDED
    payload = {
        "question": "",
        "model_answer": model_answer,
        "correct_answer": correct_answer,
        "citation_text": citation_text,
    }
    resp = client.responses.create(
        model=cfg.evals.judge.model,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )
    text = getattr(resp, "output_text", "") or ""
    try:
        data = json.loads(text)
        return JudgeDecision(**data)
    except Exception:
        # Fallback heuristic if model output is malformed
        return _heuristic_judge(model_answer, correct_answer, citation_text)


def _normalize_score(gr: GraderResult) -> float | None:
    if gr.score is None:
        return None
    rng = list(gr.score_range) if gr.score_range is not None else None
    if rng and len(rng) >= 2:
        lo = float(rng[0])
        hi = float(rng[-1])
        if hi > lo:
            norm = (float(gr.score) - lo) / (hi - lo)
            return max(0.0, min(1.0, norm))
    return 1.0 if gr.passed else 0.0


def _apply_config_graders(
  cfg: RootConfig, ex_item, answer_text: str, base_decision: JudgeDecision
) -> tuple[JudgeDecision, List[Dict[str, Any]]]:
    graders = run_config_graders(cfg, item=ex_item, answer_text=answer_text)
    if not graders:
        return base_decision, []

    graded_with_scores = [g for g in graders if g.score is not None]
    overall_pass = bool(graded_with_scores) and all(g.passed for g in graded_with_scores)

    decision = base_decision
    decision.decision = "pass" if overall_pass else "fail"
    summary = summarize_graders(graders)
    decision.rationale = summary

    for gr in graders:
        norm = _normalize_score(gr)
        if norm is None:
            continue
        name_l = gr.name.lower()
        if "ground" in name_l:
            decision.grounding = norm
        elif any(key in name_l for key in ("relev", "correct", "accuracy")):
            decision.correctness = norm

    return decision, [g.to_dict() for g in graders]


def run_local_judge(cfg: RootConfig, dataset_path: str | Path, config_path: str | None = None) -> Dict[str, Any]:
    rows = load_and_validate_jsonl(dataset_path)
    # Main loop: use app pipeline to answer then judge locally
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    from stores.base import make_store_from_config

    store = make_store_from_config(cfg)

    def process_example(ex):
        q = ex.item.question
        rprint(f"[cyan]Evaluating[/cyan]: {q[:60]}...")
        t_q = time.time()
        out = answer_query(store=store, cfg=cfg, query=q)
        latency_ms = int((time.time() - t_q) * 1000)
        answer_text = ""
        if isinstance(out, dict):
            answer_text = out.get("answer_text") or out.get("answer") or ""
        # Judge
        decision = _heuristic_judge(answer_text, ex.item.correct_answer, ex.item.citation_text)
        grader_payload = {}
        if cfg.evals.judge.model and not cfg.evals.openai_evals.graders:
            decision = _llm_judge(
                cfg, answer_text, ex.item.correct_answer, ex.item.citation_text
            )
        enriched_decision, graders = _apply_config_graders(
            cfg=cfg,
            ex_item=ex.item.model_dump(),
            answer_text=answer_text,
            base_decision=decision,
        )
        grader_payload = {"graders": graders}
        return {
            "id": ex.item.id,
            "question": ex.item.question,
            "answer": answer_text,
            "judge": {**enriched_decision.model_dump(), **grader_payload},
            "latency_ms": latency_ms,
            "grader_payload": grader_payload,
        }

    with ThreadPoolExecutor() as executor:
        future_to_ex = {executor.submit(process_example, ex): ex for ex in rows}
        for future in as_completed(future_to_ex):
            result = future.result()
            results.append(result)

    total_ms = int((time.time() - t0) * 1000)
    pass_rate = sum(1 for r in results if r["judge"]["decision"] == "pass") / max(
        1, len(results)
    )
    summary = {
        "project": cfg.project,
        "num_items": len(rows),
        "total_ms": total_ms,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / max(1, len(results)),
        "pass_rate": pass_rate,
        "threshold_pass": pass_rate >= cfg.evals.thresholds.pass_rate,
        "rows": results,
    }
    if config_path:
        summary["config_path"] = config_path
    return summary
