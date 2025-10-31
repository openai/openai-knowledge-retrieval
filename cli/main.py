from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from cli.config import load_config, lint_config
from ingestion.pipeline import run_ingestion
from retrieval.pipeline import answer_query
from evals.harness import run_evals
from stores.base import make_store_from_config

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="RAG Starter Kit CLI"
)


@app.command()
def init(
    template_arg: Optional[str] = typer.Argument(
        None, help="Template name (e.g., openai, custom-qdrant)"
    ),
    template: Optional[str] = typer.Option(
        None, "--template", "-t", help="Template name (e.g., openai, custom-qdrant)"
    ),
    chunking: Optional[str] = typer.Option(
        None,
        "--chunking",
        "-c",
        help="Chunking strategy for qdrant templates: recursive|heading|hybrid|xml_aware|custom",
    ),
    list_templates: bool = typer.Option(
        False, "--list", help="List available templates"
    ),
):
    """Print a starter YAML config to stdout."""
    repo_root = Path(__file__).resolve().parents[1]
    templates_dir = repo_root / "templates"
    if list_templates:
        names = [p.stem for p in templates_dir.glob("*.yaml")]
        if not names:
            raise SystemExit("No templates found in templates/")
        print("\n".join(sorted(names)))
        return

    # If no template is provided, default to configs/default.openai.yaml
    if not template and not template_arg:
        default_cfg = repo_root / "configs" / "default.openai.yaml"
        if not default_cfg.exists():
            raise SystemExit("Default config not found at configs/default.openai.yaml")
        print(default_cfg.read_text())
        return

    tpl = (template or template_arg or "openai").lower()
    # Map synonyms
    alias = {
        "openai": "openai",
        "custom": "custom-qdrant",
        "qdrant": "custom-qdrant",
        "custom-qdrant": "custom-qdrant",
    }.get(tpl, tpl)
    # Resolve path possibly using chunking strategy for qdrant
    path = templates_dir / f"{alias}.yaml"
    if alias == "custom-qdrant" and chunking:
        ch = chunking.lower()
        allowed = {"recursive", "heading", "hybrid", "xml_aware", "custom"}
        if ch not in allowed:
            raise SystemExit(
                f"Unknown chunking strategy '{chunking}'. Allowed: {', '.join(sorted(allowed))}"
            )
        # Prefer qdrant-*.yaml naming
        alt = templates_dir / f"qdrant-{ch}.yaml"
        if alt.exists():
            path = alt
        else:
            # Fallback to custom-qdrant.yaml
            path = templates_dir / "custom-qdrant.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in templates_dir.glob("*.yaml")))
        raise SystemExit(f"Unknown template '{tpl}'. Available: {available}")
    print(path.read_text())


@app.command()
def config_lint(
    config: Path = typer.Option(..., exists=True, readable=True, help="Path to YAML")
):
    """Validate and print normalized config with warnings."""
    result = lint_config(str(config))
    print(json.dumps(result, indent=2))


@app.command()
def ingest(config: Path = typer.Option(..., exists=True, readable=True)):
    cfg_path = str(config)
    cfg = load_config(cfg_path)
    vs_id = run_ingestion(cfg)

    if vs_id and cfg.vector_store.backend == "openai_file_search":
        rprint("\n[cyan]Vector Store is ready. Save this to your .env:[/cyan]")
        rprint(f"  [bold]VECTOR_STORE_ID={vs_id}[/bold]")
        rprint("[cyan]Or export it for this shell:[/cyan]")
        rprint(f"  [bold]export VECTOR_STORE_ID={vs_id}[/bold]\n")

        typer.prompt(
            "Copy the above line into your .env and press Enter to continue", default=""
        )


@app.command()
def chat(
    config: Path = typer.Option(..., exists=True, readable=True),
    query: Optional[str] = typer.Option(
        None, help="Query text; if omitted, read from stdin"
    ),
):
    cfg = load_config(str(config))
    # Determine mode: single-turn vs interactive loop
    interactive = False
    if query is None:
        if sys.stdin is not None and not sys.stdin.isatty():
            rprint("[cyan]Reading prompt from stdin...[/cyan]")
            question = sys.stdin.read()
        else:
            interactive = True
            question = None  # type: ignore
    else:
        # Query was provided via option; run single-turn mode with that question
        question = query

    rprint("[cyan]Initializing vector store and chat pipeline...[/cyan]")
    store = make_store_from_config(cfg)
    # If using OpenAI File Search, prefer configured ID; else prompt user to reuse an existing one
    try:
        from stores.openai_file_search import OpenAIFileSearchStore
    except Exception:
        OpenAIFileSearchStore = None  # type: ignore
    if OpenAIFileSearchStore and isinstance(store, OpenAIFileSearchStore):
        cfg_vs_id = getattr(
            cfg.vector_store.openai_file_search, "vector_store_id", None
        )
        env_primary = os.getenv("VECTOR_STORE_ID")
        env_legacy = os.getenv("VECTOR_STORE_ID")
        chosen = cfg_vs_id or env_primary or env_legacy
        if not chosen:
            chosen = typer.prompt(
                "Enter existing OpenAI Vector Store ID (e.g., vs_...)", default=""
            )
            if not chosen:
                raise SystemExit(
                    "Vector Store ID is required for chat with OpenAI File Search.\n"
                    "Set vector_store.openai_file_search.vector_store_id in your config or provide VECTOR_STORE_ID."
                )
            else:
                rprint(f"[cyan]Will use provided Vector Store ID[/cyan]: {chosen}")
        if chosen:
            try:
                store.set_vector_store_id(chosen)  # type: ignore[attr-defined]
            except Exception:
                pass
    # Chat history (stateless API, managed client-side)
    history: list[dict[str, str]] = []

    def _run_once(q: str) -> dict:
        rprint("[cyan]Submitting query to retrieval/synthesis...[/cyan]")
        try:
            result = answer_query(store=store, cfg=cfg, query=q, history=history)
            # Prefer unified answer_text key
            answer_text = result.get("answer_text") or result.get("answer") or ""
            if not answer_text:
                rprint(
                    "[yellow]No assistant text returned. Printing raw response debug info.[/yellow]"
                )
                try:
                    raw = result.get("raw_response")
                    if raw:
                        import json as _json

                        keys = list(raw.keys())
                        snippet = {k: raw.get(k) for k in keys[:8]}
                        rprint(_json.dumps(snippet, indent=2)[:1200])
                except Exception:
                    pass
            rprint(f"[bold blue]assistant:[/bold blue] {answer_text}")
            return result
        except Exception as e:
            rprint(f"[red]Chat failed[/red]: {e}")
            raise

    if interactive:
        rprint("[magenta]Interactive chat. Type 'exit' or 'quit' to leave.[/magenta]")
        while True:
            try:
                user_msg = typer.prompt("you")
            except Exception:
                break
            if user_msg.strip().lower() in {"exit", "quit"} or not user_msg.strip():
                break
            history.append({"role": "user", "content": user_msg})
            result = _run_once(user_msg)
            answer_text = result.get("answer_text") or result.get("answer") or ""
            history.append({"role": "assistant", "content": answer_text})
        rprint("[cyan]Goodbye![/cyan]")
    else:
        # Single turn
        q = (question or "").strip()
        if not q:
            raise SystemExit(
                "Empty query. Provide --query, pipe stdin, or use interactive mode."
            )
        history.append({"role": "user", "content": q})
        result = _run_once(q)
        # Do not append assistant in single-turn mode
        print(json.dumps(result.get("answer_text"), indent=2))


@app.command()
def eval(
    config: Path = typer.Option(..., exists=True, readable=True),
    ablations: bool = False,
):
    cfg = load_config(str(config))
    run_evals(cfg, run_ablations=ablations, config_path=str(config))


@app.command()
def clear(config: Path = typer.Option(..., exists=True, readable=True)):
    cfg = load_config(str(config))
    store = make_store_from_config(cfg)
    print("Deleting all documents from store...")
    # For demo simplicity, call delete with empty to imply clear if supported
    try:
        store.delete(doc_ids=["*"])  # convention: clear-all
    except Exception:
        print("Store does not support clear-all via wildcard. Skipping.")


if __name__ == "__main__":
    app()
