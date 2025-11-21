from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.assistant_agent import clear_last_retrieved_chunks
from app.main import AssistantMessageContent, AssistantMessageItem, KnowledgeAssistantServer


def _make_config(tmp_path: Path, data_dir: Path) -> Path:
    config = {
        "project": "test-project",
        "env": {},
        "data": {
            "paths": [str(data_dir)],
            "include_extensions": [".txt"],
            "exclude_globs": [],
        },
        "vector_store": {
            "backend": "openai_file_search",
            "openai_file_search": {"vector_store_name": "vs-test"},
        },
        "synthesis": {"model": "gpt-5"},
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _make_server(monkeypatch, config_path: Path) -> KnowledgeAssistantServer:
    monkeypatch.setenv("RAG_CONFIG", str(config_path))
    return KnowledgeAssistantServer(agent=object())


def test_documents_from_text_matches_bracket_hints(tmp_path, monkeypatch):
    """`_documents_from_text` should resolve a doc referenced via [doc: ...] hint."""
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    doc_path = data_dir / "doc_one.txt"
    doc_path.write_text("Example content", encoding="utf-8")

    cfg_path = _make_config(tmp_path, data_dir)
    server = _make_server(monkeypatch, cfg_path)
    cfg = server._load_cfg(server._cfg_path)

    docs = server._scan_documents(server._cfg_path, cfg)
    assert docs, "Expected at least one document from the configured path"
    expected = docs[0]

    matches = server._documents_from_text(f"See sources [doc: {expected.title}]", cfg)

    assert matches and matches[0].id == expected.id


def test_extract_citations_falls_back_to_text_when_no_annotations(tmp_path, monkeypatch):
    """`_extract_citations` should fall back to text parsing when no annotations exist."""
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    doc_path = data_dir / "doc_two.txt"
    doc_path.write_text("Example content", encoding="utf-8")

    cfg_path = _make_config(tmp_path, data_dir)
    server = _make_server(monkeypatch, cfg_path)
    cfg = server._load_cfg(server._cfg_path)
    documents = server._scan_documents(server._cfg_path, cfg)
    target = documents[0]

    thread_id = "thr_fallback"
    clear_last_retrieved_chunks(thread_id)

    item = AssistantMessageItem(
        id="msg_test",
        thread_id=thread_id,
        created_at=datetime.now(timezone.utc),
        content=[
            AssistantMessageContent(
                text=f"Answer text with citation hint [doc: {target.filename}]",
                annotations=[],
                type="output_text",
            )
        ],
    )

    citations = server._extract_citations(item, cfg, thread_id)

    assert citations, "Expected citation derived from assistant text"
    assert citations[0]["document_id"] == target.id
