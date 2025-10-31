from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from rich import print as rprint

from cli.config import RootConfig
from evals.datasets.schema import load_and_validate_jsonl
from evals.metrics.qa import em_f1
from evals.reporters import write_markdown_report
from evals.reporters.html import write_html_report
from evals.runners.local_judge import run_local_judge
from evals.runners.openai_evals import run_openai_evals
from retrieval.pipeline import answer_query
from stores.base import make_store_from_config


def _write_openai_report_link(project: str, report_url: str) -> str:
    out_dir = "evals/reports"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "openai_report_link.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# OpenAI Evals Report\n\nProject: {project}\n\nReport: {report_url}\n"
        )
    return path


def run_evals(cfg: RootConfig, run_ablations: bool = False, config_path: str | None = None) -> None:
    mode = getattr(cfg.evals, "mode", "user")

    if mode == "user":
        if not cfg.evals.dataset_path:
            raise SystemExit("evals.dataset_path not configured for user mode")

        # 1) Local run with judge (for pass rate + HTML)
        local_summary = run_local_judge(cfg, cfg.evals.dataset_path, config_path=config_path)
        rprint(
            f"[green]Local judge pass rate[/green]: {local_summary.get('pass_rate'):.3f}"
        )
        # Also compute simple EM/F1 to keep existing MD reporter compatible
        rows = load_and_validate_jsonl(cfg.evals.dataset_path)
        store = make_store_from_config(cfg)
        md_rows: List[Dict[str, Any]] = []
        t0 = time.time()
        for ex in rows:
            t_q = time.time()
            out = answer_query(store, cfg, ex.item.question)
            t_ms = int((time.time() - t_q) * 1000)
            ans = out.get("answer_text") if isinstance(out, dict) else str(out)
            em, f1 = em_f1(ans or "", ex.item.correct_answer or "")
            md_rows.append(
                {
                    "id": ex.item.id or "",
                    "latency_ms": t_ms,
                    "em": em,
                    "f1": f1,
                    "answer": ans,
                }
            )
        total_ms = int((time.time() - t0) * 1000)
        md_report = {
            "project": cfg.project,
            "num_examples": len(rows),
            "total_ms": total_ms,
            "avg_latency_ms": sum(r["latency_ms"] for r in md_rows)
            / max(1, len(md_rows)),
            "avg_em": sum(r["em"] for r in md_rows) / max(1, len(md_rows)),
            "avg_f1": sum(r["f1"] for r in md_rows) / max(1, len(md_rows)),
            "rows": md_rows,
        }
        if config_path:
            md_report["config_path"] = config_path
        md_path = write_markdown_report(md_report)
        html_path = os.path.join("evals", "reports", "local_summary.html")
        write_html_report(local_summary, html_path)
        rprint(f"[cyan]Wrote reports[/cyan]: {md_path} and {html_path}")

        #   2) OpenAI Evals (optional)
        if getattr(cfg.evals.openai_evals, "enabled", True):
            # OpenAI Evals
            link = run_openai_evals(cfg, cfg.evals.dataset_path)
            if link.get("skipped"):
                rprint(
                    "[yellow]Skipped OpenAI Evals mirror[/yellow]: "
                    "backend does not expose an OpenAI vector store."
                )
            else:
                link_url = link.get("portal_url") or link.get("report_url", "")
                link_path = _write_openai_report_link(cfg.project, link_url)
                rprint(
                    f"[cyan]OpenAI Evals link[/cyan]: {link_url} (saved at {link_path})"
                )
            return

    elif mode == "auto":
        from evals.generator.auto_pipeline import auto_generate_dataset

        rprint("[cyan]Starting auto-generation pipeline...[/cyan]")
        summary, rows = auto_generate_dataset(cfg)
        # Guard: if no rows were generated, stop with actionable guidance instead of failing later
        if not rows:
            raise SystemExit(
                "Auto-evals produced 0 items. Check that your corpus is ingested and accessible.\n"
                "Tips:\n"
                "- Run 'rag ingest --config <cfg>' first.\n"
                "- If using openai_file_search, set OPENAI_API_KEY and vector_store_id.\n"
                "- For offline/local runs, switch to custom.qdrant backend and ensure Qdrant is up.\n"
                "- Tweak evals.auto parameters (scan_docs, items_target) to broaden sampling."
            )
        rprint(
            f"[green]Auto-generated dataset[/green]: {summary.sampled} items → {summary.dataset_path}"
        )
        # Run local judge on the generated dataset
        local_summary = run_local_judge(cfg, summary.dataset_path, config_path=config_path)
        html_path = os.path.join("evals", "reports", "local_summary.html")
        write_html_report(local_summary, html_path)
        rprint(
            f"[cyan]Local judge completed[/cyan]: pass_rate={local_summary.get('pass_rate'):.3f}"
        )
        # Optionally run OpenAI Evals
        if getattr(cfg.evals.openai_evals, "enabled", True):
            link = run_openai_evals(cfg, summary.dataset_path)
            link_url = link.get("portal_url") or link.get("report_url", "")
            link_path = _write_openai_report_link(cfg.project, link_url)
            rprint(f"[cyan]OpenAI Evals link[/cyan]: {link_url} (saved at {link_path})")
        return
    else:
        raise SystemExit(f"Unknown evals.mode: {mode}")
