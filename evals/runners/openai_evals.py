from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

from cli.config import RootConfig
from cli.env_utils import build_openai_client, resolve_env_placeholder
from evals.datasets.schema import EvalRow, load_and_validate_jsonl
from evals.generator.canonicalize import canonicalize_answer

from rich import print as rprint


@dataclass
class OpenAIEvalRun:
    dataset_file_id: str
    eval_id: str
    run_id: str
    report_url: str
    portal_url: str


item_schema = {
    "type": "custom",
    "item_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "question": {"type": "string"},
            "citation_text": {"type": "string"},
            "correct_answer": {"type": "string"},
            "source_id": {"type": "string"},
            "page": {"type": ["string", "null"]},
            "char_start": {"type": "integer"},
            "char_end": {"type": "integer"},
            "difficulty": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "id",
            "question",
            "citation_text",
            "correct_answer",
            "source_id",
            "page",
            "char_start",
            "char_end",
            "difficulty",
            "tags",
        ],
        "additionalProperties": False,
    },
    "include_sample_schema": True,
}


def _canonicalize_rows(rows: List[EvalRow]) -> List[EvalRow]:
    for r in rows:
        r.item.correct_answer = canonicalize_answer(r.item.correct_answer)
        r.item.citation_text = (r.item.citation_text or "").strip()
    return rows


def prepare_dataset(path: str | Path) -> List[EvalRow]:
    rows = load_and_validate_jsonl(path)
    return _canonicalize_rows(rows)


def upload_dataset_file(cfg: RootConfig, path: str | Path) -> str:
    """Uploads JSONL to OpenAI with purpose='evals'. Returns file_id.

    Note: This is a stub for now, returning a pseudo id without network calls
    when running in restricted environments.
    """
    try:
        client = build_openai_client(cfg)
        with open(path, "rb") as f:
            resp = client.files.create(file=f, purpose="evals")
        file_id = getattr(resp, "id", None) or getattr(resp, "file_id", None)
        return str(file_id)
    except Exception:
        # Fallback stub ID for offline/dev
        return "file_dev_eval_dataset"


def ensure_eval_and_run(
    cfg: RootConfig, dataset_file_id: str, rows: List[EvalRow]
) -> OpenAIEvalRun:
    """Create or locate an Eval and start a Run. Returns IDs and a report URL.

    This function is scaffolded; it simulates creation when API is unavailable.
    """
    name = f"{cfg.evals.openai_evals.name_prefix} – {cfg.project} - {datetime.now()}"
    try:
        from stores.base import make_store_from_config

        store = make_store_from_config(cfg)
        vector_store_id = store.vector_store_id

        client = build_openai_client(cfg)
        # Placeholder shape; actual Evals API may differ
        graders = [dict(g) for g in cfg.evals.openai_evals.graders]
        eval_payload: Dict[str, Any] = {
            "name": name,
            "data_source_config": item_schema,
            "testing_criteria": graders,
        }
        _ = eval_payload  # silence lints if unused
        eval_obj = client.evals.create(**eval_payload)
        eval_id = eval_obj.id
        print("eval_obj", eval_obj)
        print("eval_obj.id", eval_obj.id)
        run_obj = client.evals.runs.create(
            eval_id,
            name=f"{datetime.now()}",
            data_source={
                "type": "responses",
                "source": {
                    "type": "file_content",
                    "content": [{"item": it.item.model_dump()} for it in rows],
                },
                "input_messages": {
                    "type": "template",
                    "template": [
                        {
                            "type": "message",
                            "role": "system",
                            "content": {
                                "type": "input_text",
                                "text": "You are a helpful, concise assistant. Cite sources.",
                            },
                        },
                        {
                            "type": "message",
                            "role": "user",
                            "content": {
                                "type": "input_text",
                                "text": "{{item.question}}",
                            },
                        },
                    ],
                },
                "model": "o3-mini",
                "sampling_params": {
                    "seed": 17,
                    # Enable RAG via File Search tool against your vector store:
                    "tools": [
                        {"type": "file_search", "vector_store_ids": [vector_store_id]}
                    ],
                    # (Optional) If you want to capture retrieved context snippets in outputs
                    # for graders, set model/tool options to include annotations/search results.
                },
            },
        )

        run_id = run_obj.id
        report_url = run_obj.report_url
        # Construct human portal URL using eval_id and project id from environment/.env
        project_id = (
            os.getenv("OPENAI_PROJECT_ID")
            or resolve_env_placeholder(getattr(cfg.env, "openai_project", None))
            or ""
        )
        portal_url = (
            f"https://platform.openai.com/evaluation/evals/{eval_id}?project_id={project_id}"
            if project_id
            else ""
        )
        return OpenAIEvalRun(
            dataset_file_id=dataset_file_id,
            eval_id=eval_id,
            run_id=run_id,
            report_url=report_url,
            portal_url=portal_url,
        )
    except Exception as e:
        rprint("ensure_eval_and_run exception")
        resp = getattr(e, "response", None) or getattr(e, "http_response", None)
        status = getattr(resp, "status_code", None)
        req_id = getattr(resp, "request_id", None)
        msg = getattr(e, "message", None) or str(e)
        print(
            f"[yellow]ensure_eval_and_run failed[/yellow] "
            f"type={type(e).__name__} status={status} req_id={req_id} msg={msg}"
        )
        # Try to include a short snippet of the response body if available
        try:
            body_text = getattr(resp, "text", None)
            if not body_text and hasattr(resp, "json"):
                body_text = str(resp.json())
            if body_text:
                rprint(f"[dim]{body_text[:500]}[/dim]")
        except Exception:
            pass
        return OpenAIEvalRun(
            dataset_file_id=dataset_file_id,
            eval_id="eval_dev_id",
            run_id="run_dev_id",
            report_url=f"https://platform.openai.com/evals/dev/{cfg.project}",
            portal_url="",
        )


def run_openai_evals(cfg: RootConfig, dataset_path: str | Path) -> Dict[str, Any]:
    """End-to-end runner for Mode A (user-supplied dataset) using OpenAI Evals.

    Returns a summary including a report URL.
    """
    # OpenAI eval mirroring only works when the backend is OpenAI File Search.
    if (
        cfg.vector_store.backend == "custom"
        and getattr(getattr(cfg.vector_store, "custom", None), "kind", None) == "qdrant"
    ):
        return {
            "skipped": True,
            "reason": "OpenAI evals require an OpenAI vector store backend.",
        }

    rows = prepare_dataset(dataset_path)
    # Write a canonicalized copy alongside original for traceability
    out_dir = Path("evals") / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    canon_path = out_dir / (Path(dataset_path).stem + ".canonical.jsonl")

    file_id = upload_dataset_file(cfg, canon_path)
    run = ensure_eval_and_run(cfg, file_id, rows)
    summary = {
        "project": cfg.project,
        "dataset_path": str(dataset_path),
        "uploaded_file_id": run.dataset_file_id,
        "eval_id": run.eval_id,
        "run_id": run.run_id,
        "report_url": run.report_url,
        "portal_url": run.portal_url,
    }
    return summary
