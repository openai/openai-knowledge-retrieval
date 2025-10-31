from __future__ import annotations

from typing import Any, Dict, Generator, Optional

from cli.config import RootConfig
from stores.base import make_store_from_config

import os
from pathlib import Path
import fnmatch


def iterate_chunks(
    cfg: RootConfig, max_docs: Optional[int] = None, max_chars_per_doc: int = 8000
) -> Generator[Dict[str, Any], None, None]:
    """Yield chunk-like dicts from the configured vector store.

    - For Qdrant: scroll over points and yield payloads as chunks.
    - For OpenAI File Search: list vector store files and fetch file content via Files API; yield pseudo-chunks.
    """
    store = make_store_from_config(cfg)
    if cfg.vector_store.backend == "custom":
        from stores.custom_qdrant import QdrantStore

        if isinstance(store, QdrantStore):
            client = store.client
            coll = store.collection
            next_page = None
            yielded = 0
            while True:
                # Qdrant client's scroll returns a tuple: (points, next_page_offset)
                points, next_page = client.scroll(
                    collection_name=coll,
                    with_payload=True,
                    with_vectors=False,
                    limit=256,
                    offset=next_page,
                )
                for p in points or []:
                    payload = getattr(p, "payload", {}) or {}
                    text = payload.get("text") or ""
                    if not text:
                        continue
                    doc_id = payload.get("doc_id") or payload.get("file_id")
                    page = payload.get("page")
                    yield {
                        "doc_id": doc_id,
                        "text": text[:max_chars_per_doc],
                        "page": page,
                        "metadata": payload,
                    }
                    yielded += 1
                    if max_docs and yielded >= max_docs:
                        return
                if not next_page:
                    break
            return
    else:
        # Fallback: iterate raw files under data.paths and yield 2k-char chunks
        # This is a convenience path for local auto-evals when a vector store is unavailable.
        data_dirs = [Path(p) for p in getattr(cfg.data, "paths", []) or []]
        # Collect readable files honoring include_extensions and exclude_globs
        include_exts = set((getattr(cfg.data, "include_extensions", None) or []))
        exclude_globs = list(getattr(cfg.data, "exclude_globs", []) or [])

        files: list[Path] = []
        for d in data_dirs:
            if not d.exists() or not d.is_dir():
                continue
            for p in d.rglob("*"):
                if not p.is_file():
                    continue
                if include_exts and p.suffix.lower() not in {
                    e.lower() for e in include_exts
                }:
                    continue
                # Exclusion filters
                rel = str(p)
                if any(fnmatch.fnmatch(rel, pat) for pat in exclude_globs):
                    continue
                files.append(p)

        if not files:
            raise NotImplementedError(
                "No files found under data.paths; provide files or use a supported custom store (e.g., Qdrant)."
            )

        # Stream file contents as ~2000 character chunks
        step = 2000
        max_file_count = max_docs or len(files)
        for i, path in enumerate(files):
            if i >= max_file_count:
                break
            try:
                # Best-effort text read; binary formats will degrade to partial text.
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                # Skip unreadable files silently in this fallback
                continue
            if not text:
                continue
            for j in range(0, min(len(text), max_chars_per_doc), step):
                chunk_text = text[j : j + step]
                if not chunk_text:
                    continue
                fid = str(path)
                filename = os.path.basename(fid)
                yield {
                    "doc_id": fid,
                    "text": chunk_text,
                    "page": None,
                    "metadata": {"file_name": filename},
                }
        return
