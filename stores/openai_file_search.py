from __future__ import annotations

import io
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from cli.config import RootConfig
from cli.env_utils import build_openai_client
from stores.base import Chunk, ScoredChunk, VectorStore
from rich import print


class OpenAIFileSearchStore(VectorStore):
    def __init__(self, cfg: RootConfig) -> None:
        self.client = build_openai_client(cfg)
        self.cfg = cfg
        self._vector_store_id: Optional[str] = None

    def set_vector_store_id(self, vs_id: str) -> None:
        """Manually set an existing Vector Store ID to avoid creating a new one."""
        self._vector_store_id = vs_id
        print(f"[cyan]Using existing Vector Store[/cyan] id='{vs_id}'")

    @property
    def vector_store_id(self) -> str:
        # Precedence: explicit setter > environment override > config field > create new
        if self._vector_store_id:
            return self._vector_store_id
        cfg_vs_id = None
        try:
            cfg_vs_id = getattr(
                self.cfg.vector_store.openai_file_search, "vector_store_id", None
            )
        except Exception:
            cfg_vs_id = None
        # Support new preferred env var name VECTOR_STORE_ID while keeping legacy VECTOR_STORE_ID
        env_primary = os.getenv("VECTOR_STORE_ID")
        env_legacy = os.getenv("VECTOR_STORE_ID")
        if env_primary and env_legacy and env_primary != env_legacy:
            print(
                f"[yellow]Env VECTOR_STORE_ID overrides legacy VECTOR_STORE_ID[/yellow]: VECTOR_STORE_ID='{env_primary}', VECTOR_STORE_ID='{env_legacy}'"
            )
        env_vs_id = env_primary or env_legacy
        if env_vs_id and cfg_vs_id and env_vs_id != cfg_vs_id:
            which = "VECTOR_STORE_ID" if env_vs_id == env_primary else "VECTOR_STORE_ID"
            print(
                f"[yellow]Env {which} overrides config vector_store_id[/yellow]: env='{env_vs_id}', cfg='{cfg_vs_id}'"
            )
        chosen = env_vs_id or cfg_vs_id
        if chosen:
            self._vector_store_id = chosen
            print(f"[cyan]Using Vector Store[/cyan] id='{chosen}'")
            return self._vector_store_id
        vs_name = self.cfg.vector_store.openai_file_search.vector_store_name
        print(f"[cyan]Creating OpenAI Vector Store[/cyan] name='{vs_name}' ...")
        vs = self.client.vector_stores.create(name=vs_name)
        # Apply expiry policy if set
        exp_days = self.cfg.vector_store.openai_file_search.expiry_days
        if exp_days:
            try:
                self.client.vector_stores.update(
                    vector_store_id=vs.id,
                    expires_after={"anchor": "last_active_at", "days": int(exp_days)},
                )
            except Exception:
                pass
        self._vector_store_id = vs.id
        print(f"[green]Vector Store ready[/green] id='{vs.id}', name='{vs_name}'")
        return self._vector_store_id

    def _chunking_strategy_kwargs(self) -> Dict[str, Any]:
        ch = self.cfg.vector_store.openai_file_search.chunking
        if not ch:
            return {}
        return {
            "chunking_strategy": {
                "type": "static",
                "static": {
                    "max_chunk_size_tokens": int(ch.max_chunk_size_tokens),
                    "chunk_overlap_tokens": int(ch.chunk_overlap_tokens),
                },
            }
        }

    def upsert_paths(self, file_paths: List[str]) -> None:
        """Upload local files and ensure they are attached to the Vector Store.

        - Splits errors between file-store upload and vector-store attach/indexing.
        - Skips duplicates in both stores (by filename+size for file store; by file_id for vector store).
        - Runs in parallel; retries the step that failed only.
        """
        file_paths = list(file_paths)
        if not file_paths:
            print("[yellow]No files to upload[/yellow]")
            return

        # Ensure vector store is created up-front to avoid races across threads
        vs_id = self.vector_store_id

        ch = self.cfg.vector_store.openai_file_search.chunking
        ch_desc = (
            f"static(max={int(ch.max_chunk_size_tokens)}, overlap={int(ch.chunk_overlap_tokens)})"
            if ch
            else "platform-default"
        )
        print(
            f"Preparing to process {len(file_paths)} file(s) for Vector Store '{vs_id}' with chunking={ch_desc} ..."
        )

        # Helpers: list existing files and attachments for dedup/skip
        def _retry_sleep_seconds(attempt: int, err: Exception) -> float:
            base = float(os.getenv("RAG_RETRY_BASE_SECONDS", "1.0"))
            max_sleep = float(os.getenv("RAG_RETRY_MAX_SLEEP", "30.0"))
            resp = getattr(err, "response", None) or getattr(err, "http_response", None)
            headers = getattr(resp, "headers", None)
            if headers:
                try:
                    for k, v in headers.items():
                        if str(k).lower() == "retry-after":
                            val = float(v)
                            return min(max_sleep, val) if val > 0 else base
                except Exception:
                    pass
            sleep_s = base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            return min(max_sleep, sleep_s)

        def _list_all_files_by_name() -> Dict[str, List[Any]]:
            out: Dict[str, List[Any]] = {}
            try:
                cursor = None
                while True:
                    kwargs: Dict[str, Any] = {"limit": 100}
                    if cursor:
                        kwargs["after"] = cursor
                    resp = self.client.files.list(**kwargs)
                    data = getattr(resp, "data", []) or []
                    for fi in data:
                        name = getattr(fi, "filename", None) or getattr(
                            fi, "name", None
                        )
                        if name:
                            out.setdefault(name, []).append(fi)
                    has_more = getattr(resp, "has_more", False)
                    if not has_more:
                        break
                    cursor = getattr(data[-1], "id", None)
                    if not cursor:
                        break
            except Exception:
                return {}
            return out

        def _list_vector_store_file_ids() -> set[str]:
            ids: set[str] = set()
            try:
                cursor = None
                while True:
                    kwargs: Dict[str, Any] = {"limit": 100, "vector_store_id": vs_id}
                    if cursor:
                        kwargs["after"] = cursor
                    resp = self.client.vector_stores.files.list(**kwargs)
                    data = getattr(resp, "data", []) or []
                    for rec in data:
                        fid = getattr(rec, "file_id", None) or getattr(rec, "id", None)
                        if fid:
                            ids.add(str(fid))
                    has_more = getattr(resp, "has_more", False)
                    if not has_more:
                        break
                    cursor = getattr(data[-1], "id", None)
                    if not cursor:
                        break
            except Exception:
                return set()
            return ids

        existing_by_name = _list_all_files_by_name()
        existing_vs_ids = _list_vector_store_file_ids()

        def _ensure_file_uploaded(path: str) -> Tuple[str, str]:
            basename = os.path.basename(path)
            size = os.path.getsize(path)
            # Deduplicate by filename and size if available
            existing = existing_by_name.get(basename, [])
            for fi in existing:
                try:
                    fi_bytes = int(getattr(fi, "bytes", 0))
                    if fi_bytes == size:
                        fid = str(getattr(fi, "id", ""))
                        if fid:
                            print(
                                f"[yellow]skip file-store duplicate[/yellow] {basename} (id={fid})"
                            )
                            return fid, "skipped"
                except Exception:
                    continue

            # Upload with retries
            max_attempts = int(os.getenv("RAG_RETRY_MAX_ATTEMPTS", "5"))
            for attempt in range(1, max_attempts + 1):
                try:
                    with open(path, "rb") as f:
                        created = self.client.files.create(file=f, purpose="assistants")
                    fid = str(getattr(created, "id", ""))
                    if not fid:
                        raise RuntimeError("no file id returned")
                    print(
                        f"[green]uploaded to file store[/green] {basename} (id={fid})"
                    )
                    # Update cache so other threads can see it
                    existing_by_name.setdefault(basename, []).append(created)
                    return fid, "uploaded"
                except Exception as e:
                    if attempt < max_attempts:
                        sleep_s = _retry_sleep_seconds(attempt, e)
                        print(
                            f"[yellow]retrying file upload[/yellow] {basename} in {sleep_s:.1f}s "
                            f"({attempt}/{max_attempts}) due to: {e}"
                        )
                        time.sleep(sleep_s)
                        continue
                    raise

        def _ensure_in_vector_store(file_id: str, basename: str) -> str:
            if file_id in existing_vs_ids:
                print(
                    f"[yellow]skip vector-store duplicate[/yellow] {basename} (file_id={file_id})"
                )
                return "skipped"

            # Attach with retries
            max_attempts = int(os.getenv("RAG_RETRY_MAX_ATTEMPTS", "5"))
            last_err: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    rec = self.client.vector_stores.files.create(
                        vector_store_id=vs_id,
                        file_id=file_id,
                        **self._chunking_strategy_kwargs(),
                    )
                    vs_file_id = str(getattr(rec, "id", ""))
                    # Poll processing status
                    for poll_try in range(10):
                        time.sleep(1.0 + 0.25 * poll_try)
                        fetched = self.client.vector_stores.files.retrieve(
                            vector_store_id=vs_id, file_id=file_id
                        )
                        status = (
                            getattr(fetched, "status", None)
                            or getattr(fetched, "state", None)
                            or ""
                        ).lower()
                        if status in ("completed", "processed", "succeeded"):
                            print(
                                f"[green]added to vector store[/green] {basename} (file_id={file_id}, vs_file_id={vs_file_id})"
                            )
                            existing_vs_ids.add(file_id)
                            return "attached"
                        if status in ("failed", "errored", "cancelled"):
                            last_err = RuntimeError(
                                f"vector store indexing status={status}"
                            )
                            raise last_err
                    # If we exit the poll loop, treat as timeout and retry
                    last_err = TimeoutError("vector store processing timed out")
                    raise last_err
                except Exception as e:
                    last_err = e
                    if attempt < max_attempts:
                        sleep_s = _retry_sleep_seconds(attempt, e)
                        print(
                            f"[yellow]retrying vector-store add[/yellow] {basename} in {sleep_s:.1f}s "
                            f"({attempt}/{max_attempts}) due to: {e}"
                        )
                        time.sleep(sleep_s)
                        continue
                    raise

        def _process(path: str) -> Tuple[str, str]:
            basename = os.path.basename(path)
            try:
                file_id, act = _ensure_file_uploaded(path)
            except Exception as e:
                return (
                    "file_store_error",
                    f"[red]file-store failed[/red] {basename}: {e}",
                )
            try:
                act2 = _ensure_in_vector_store(file_id, basename)
                return ("ok", f"processed {basename} ({act}, {act2})")
            except Exception as e:
                return (
                    "vector_store_error",
                    f"[red]vector-store failed[/red] {basename}: {e}",
                )

        # Network-bound; threads are appropriate. Keep concurrency reasonable.
        env_workers = os.getenv("RAG_INGEST_WORKERS")
        max_workers = (
            int(env_workers)
            if env_workers and env_workers.isdigit() and int(env_workers) > 0
            else min(8, max(2, (os.cpu_count() or 4)))
        )
        print(f"[cyan]Parallel processing[/cyan]: workers={max_workers}")
        results: List[str] = []
        counters = {"ok": 0, "file_store_error": 0, "vector_store_error": 0}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_process, p): p for p in file_paths}
            for fut in as_completed(futs):
                status, msg = fut.result()
                results.append(msg)
                print(msg)
                counters[status] = counters.get(status, 0) + 1
        # Summary
        total = len(file_paths)
        print(
            f"[bold]Summary[/bold]: ok={counters['ok']}, file_store_errors={counters['file_store_error']}, "
            f"vector_store_errors={counters['vector_store_error']}, total={total}"
        )
        if counters["file_store_error"] or counters["vector_store_error"]:
            raise RuntimeError(
                f"Ingestion completed with errors: file_store={counters['file_store_error']}, "
                f"vector_store={counters['vector_store_error']}"
            )

    def upsert(self, chunks: List[Chunk]) -> None:
        # Fallback: if chunks are provided, concatenate and upload as a single file
        by_doc: Dict[str, List[Chunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)
        for _, cks in by_doc.items():
            text = "\n\n".join([c.text for c in cks])
            file_content = io.BytesIO(text.encode("utf-8"))
            self.client.vector_stores.files.upload_and_poll(
                vector_store_id=self.vector_store_id,
                file=file_content,
                **self._chunking_strategy_kwargs(),
            )

    def delete(self, doc_ids: List[str]) -> None:
        # Track file IDs per doc to support deletion (not implemented in starter kit)
        return None

    def search(
        self, query: str, k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredChunk]:
        # Retrieval is handled in models.synthesis / retrieval pipeline
        raise NotImplementedError(
            "Use models.synthesis with file_search for OpenAI backend"
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": "openai_file_search",
            "vector_store_id": self._vector_store_id,
        }
