from __future__ import annotations

import hashlib
import re
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
import fnmatch

from cli.config import RootConfig
from cli.env_utils import build_openai_client
from ingestion.types import RawDocument, Page, ChunkRecord
from ingestion.custom_chunker_runner import run_custom_chunker
from stores.base import Chunk
from stores.base import make_store_from_config
from stores.openai_file_search import OpenAIFileSearchStore
from ingestion.chunkers.hybrid import hybrid_chunk
from ingestion.chunkers.heading import heading_chunk
from ingestion.chunkers.recursive import recursive_chunk
from ingestion.chunkers.xml_aware import xml_aware_chunk
from rich import print

# Chunking utilities moved to ingestion/chunkers/utils.py


# --- Loaders (basic) ---


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf_text(path: str) -> List[Page]:
    try:
        from pypdf import PdfReader
    except Exception:
        return [Page(number=1, text=_read_text_file(path))]
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        try:
            pages.append(Page(number=i, text=p.extract_text() or ""))
        except Exception:
            pages.append(Page(number=i, text=""))
    return pages


def load_docs(
    paths: List[str], include_exts: List[str], exclude_globs: List[str]
) -> List[RawDocument]:
    files: List[Path] = []
    normalized_exts = {
        e.lower() if e.startswith(".") else f".{e.lower()}" for e in include_exts
    }
    for base in paths:
        base_path = Path(base)
        if base_path.is_dir():
            for p in base_path.rglob("*"):
                if p.is_file() and p.suffix.lower() in normalized_exts:
                    files.append(p)
        elif base_path.is_file():
            files.append(base_path)
    docs: List[RawDocument] = []
    for path in files:
        path = Path(path)
        # Exclusions (glob patterns applied to posix path)
        pstr = path.as_posix()
        if any(fnmatch.fnmatch(pstr, pat) for pat in exclude_globs):
            continue
        ext = path.suffix.lower()
        text = ""
        pages: List[Page] = []
        title = path.name
        if ext == ".pdf":
            pages = _load_pdf_text(str(path))
            text = "\n\n".join(p.text for p in pages)
        else:
            text = _read_text_file(str(path))
            pages = [Page(number=1, text=text)]
        doc_id = hashlib.sha1(str(path).encode()).hexdigest()
        docs.append(
            RawDocument(
                doc_id=doc_id,
                title=title,
                path=str(path),
                pages=pages,
                text=text,
                metadata={"path": str(path)},
            )
        )
    return docs


# --- Preprocessors ---


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# --- Embeddings ---


def embed_texts_with_client(
    texts: List[str], model: str, batch_size: int, cfg: RootConfig
) -> List[List[float]]:
    """Embed texts sequentially (baseline)."""
    client = build_openai_client(cfg)
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def _retry_sleep_seconds(attempt: int, err: Exception) -> float:
    base = float(os.getenv("RAG_RETRY_BASE_SECONDS", "1.0"))
    max_sleep = float(os.getenv("RAG_RETRY_MAX_SLEEP", "30.0"))
    retry_after = None

    resp = getattr(err, "response", None) or getattr(err, "http_response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        try:
            for k, v in headers.items():
                if str(k).lower() == "retry-after":
                    retry_after = float(v)
                    break
        except Exception:
            retry_after = None
    if retry_after is not None and retry_after > 0:
        return min(max_sleep, retry_after)
    sleep_s = base * (2 ** (attempt - 1)) * (1 + random.random() * 0.25)
    return min(max_sleep, sleep_s)


def embed_texts_concurrent(
    texts: List[str], model: str, batch_size: int, cfg: RootConfig
) -> List[List[float]]:
    """Embed texts using multiple parallel requests while preserving order.

    Controlled by env RAG_EMBED_WORKERS (default min(8, cpu)).
    Includes simple retries with backoff per batch.
    """
    client = build_openai_client(cfg)
    total = len(texts)
    if total == 0:
        return []
    batches: List[Tuple[int, List[str]]] = []
    for i in range(0, total, batch_size):
        batches.append((i, texts[i : i + batch_size]))

    env_workers = os.getenv("RAG_EMBED_WORKERS")
    max_workers = (
        int(env_workers)
        if env_workers and env_workers.isdigit() and int(env_workers) > 0
        else min(8, max(2, (os.cpu_count() or 4)))
    )
    print(
        f"[cyan]Embedding[/cyan]: {total} texts, batch_size={batch_size}, workers={max_workers}"
    )
    start_time = time.time()
    results: List[Tuple[int, List[List[float]]]] = []

    def _embed_batch(offset: int, batch: List[str]) -> Tuple[int, List[List[float]]]:
        max_attempts = int(os.getenv("RAG_RETRY_MAX_ATTEMPTS", "5"))
        for attempt in range(1, max_attempts + 1):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vecs = [d.embedding for d in resp.data]
                return (offset, vecs)
            except Exception as e:
                if attempt < max_attempts:
                    sleep_s = _retry_sleep_seconds(attempt, e)
                    print(
                        f"[yellow]retrying embed batch[/yellow] offset={offset} size={len(batch)} in {sleep_s:.1f}s due to: {e}"
                    )
                    time.sleep(sleep_s)
                    continue
                raise

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_embed_batch, off, b) for off, b in batches]
        completed = 0
        for fut in as_completed(futs):
            off, vecs = fut.result()
            results.append((off, vecs))
            completed += 1
            if completed % max(1, len(futs) // 10) == 0:
                print(f"[dim]Embedded {completed}/{len(futs)} batches[/dim]")

    # Stitch results preserving order
    vectors: List[List[float]] = [None] * total  # type: ignore
    for off, vecs in results:
        vectors[off : off + len(vecs)] = vecs
    dur = time.time() - start_time
    rate = total / dur if dur > 0 else total
    print(
        f"[green]Embeddings ready[/green]: {total} vectors in {dur:.1f}s (~{rate:.1f}/s)"
    )
    return vectors


# --- Orchestrator ---


def run_ingestion(cfg: RootConfig) -> str | None:
    print("[bold green]Ingestion started[/bold green]")
    docs = load_docs(
        cfg.data.paths, cfg.data.include_extensions, cfg.data.exclude_globs
    )

    # For OpenAI File Search, upload raw files and let the platform chunk+embed
    if cfg.vector_store.backend == "openai_file_search":
        print(f"Using OpenAI File Search backend. Found {len(docs)} file(s) to upload.")
        preview = ", ".join([Path(d.path).name for d in docs[:5]])
        if preview:
            print(f"Examples: {preview}{' ...' if len(docs) > 5 else ''}")
        store = make_store_from_config(cfg)
        assert isinstance(store, OpenAIFileSearchStore)
        paths = [d.path for d in docs]
        try:
            store.upsert_paths(paths)
        except Exception as e:
            print(f"[red]Ingestion failed[/red]: {e}")
            raise
        vs_id = store.vector_store_id
        print("[bold green]Ingestion completed (OpenAI File Search)[/bold green]")
        print(f"[cyan]Vector Store ID[/cyan]: {vs_id}")
        print("Add this to your .env to reuse it in chat and appV2:")
        print("  VECTOR_STORE_ID=" + vs_id)
        print("Or export it for the current shell session:")
        print("  export VECTOR_STORE_ID=" + vs_id)
        return vs_id

    # preprocess for custom backends
    print(
        f"Using custom vector store backend. Found {len(docs)} file(s) for ingestion."
    )
    preview = ", ".join([Path(d.path).name for d in docs[:5]])
    if preview:
        print(f"Examples: {preview}{' ...' if len(docs) > 5 else ''}")
    print("Cleaning documents...")
    for d in docs:
        d.text = clean_text(d.text)
        d.pages = [Page(number=p.number, text=clean_text(p.text)) for p in d.pages]

    # chunk
    print("Chunking documents...")
    if cfg.chunking.strategy == "custom":
        chunks: List[ChunkRecord] = list(
            run_custom_chunker(
                module_path=cfg.custom_chunker.module_path,
                class_name=cfg.custom_chunker.class_name,
                init_args=cfg.custom_chunker.init_args,
                docs=docs,
            )
        )
    elif cfg.chunking.strategy == "recursive":
        chunks = []
        for d in docs:
            chunks.extend(
                recursive_chunk(
                    d, cfg.chunking.target_token_range, cfg.chunking.overlap_tokens
                )
            )
    elif cfg.chunking.strategy == "heading":
        chunks = []
        for d in docs:
            chunks.extend(
                heading_chunk(
                    d, cfg.chunking.target_token_range, cfg.chunking.overlap_tokens
                )
            )
    elif cfg.chunking.strategy == "xml_aware":
        chunks = []
        for d in docs:
            chunks.extend(
                xml_aware_chunk(
                    d, cfg.chunking.target_token_range, cfg.chunking.overlap_tokens
                )
            )
    else:  # hybrid
        chunks = []
        for d in docs:
            chunks.extend(
                hybrid_chunk(
                    d, cfg.chunking.target_token_range, cfg.chunking.overlap_tokens
                )
            )

    print(f"Generated {len(chunks)} chunks from {len(docs)} documents")

    # embed if needed
    do_embed = cfg.vector_store.backend == "custom"
    embeddings: List[List[float]] = []
    if do_embed:
        texts = [c.text for c in chunks]
        # Choose concurrent embedding by default for speed
        embeddings = embed_texts_concurrent(
            texts,
            model=cfg.embeddings.model,
            batch_size=cfg.embeddings.batch_size,
            cfg=cfg,
        )

    # upsert
    store = make_store_from_config(cfg)
    store_chunks: List[Chunk] = []
    for idx, c in enumerate(chunks):
        emb = embeddings[idx] if do_embed else None
        store_chunks.append(
            Chunk(
                id=c.id,
                doc_id=c.doc_id,
                text=c.text,
                embedding=emb,
                metadata=c.metadata,
            )
        )

    # Batch + parallel upserts for custom stores
    if cfg.vector_store.backend == "custom":
        batch_size = int(os.getenv("RAG_QDRANT_UPSERT_BATCH", "500"))
        env_workers = os.getenv("RAG_QDRANT_UPSERT_WORKERS")
        max_workers = (
            int(env_workers)
            if env_workers and env_workers.isdigit() and int(env_workers) > 0
            else min(8, max(2, (os.cpu_count() or 4)))
        )
        batches = [
            store_chunks[i : i + batch_size]
            for i in range(0, len(store_chunks), batch_size)
        ]
        print(
            f"[cyan]Upserting to vector store[/cyan]: {len(store_chunks)} chunks in {len(batches)} batches, workers={max_workers}"
        )
        start = time.time()

        def _upsert_batch(bi: int, batch: List[Chunk]) -> Tuple[int, int]:
            max_attempts = int(os.getenv("RAG_RETRY_MAX_ATTEMPTS", "5"))
            print(
                f"[dim]Starting upsert batch {bi}[/dim]: size={len(batch)} first_id={batch[0].id if batch else 'n/a'}"
            )
            batch_start = time.time()
            for attempt in range(1, max_attempts + 1):
                try:
                    store.upsert(batch)
                    took = time.time() - batch_start
                    print(
                        f"[green]Batch {bi} upserted[/green]: size={len(batch)} in {took:.2f}s"
                    )
                    return (bi, len(batch))
                except Exception as e:
                    if attempt < max_attempts:
                        sleep_s = _retry_sleep_seconds(attempt, e)
                        print(
                            f"[yellow]retrying upsert[/yellow] batch={bi} size={len(batch)} attempt={attempt}/{max_attempts} "
                            f"in {sleep_s:.1f}s due to: {type(e).__name__}: {e}"
                        )
                        time.sleep(sleep_s)
                        continue
                    print(
                        f"[red]Batch {bi} failed[/red]: size={len(batch)} error={type(e).__name__}: {e}"
                    )
                    raise

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_upsert_batch, i, b) for i, b in enumerate(batches)]
            done = 0
            for fut in as_completed(futs):
                bi, n = fut.result()
                done += n
                print(
                    f"[dim]Upserted batch {bi} ({n} pts) — total {done}/{len(store_chunks)}[/dim]"
                )
        dur = time.time() - start
        rate = len(store_chunks) / dur if dur > 0 else len(store_chunks)
        print(
            f"[green]Upsert complete[/green]: {len(store_chunks)} chunks in {dur:.1f}s (~{rate:.1f}/s)"
        )
    else:
        store.upsert(store_chunks)

    print("[bold green]Ingestion completed[/bold green]")
    return None
