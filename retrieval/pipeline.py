from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from rich import print as rprint

from cli.config import RootConfig
from cli.env_utils import build_openai_client
from models.synthesis import synthesize_openai_file_search, chat_with_tools
from prompts.loader import load_prompt
from retrieval.citations import extract_spans
from retrieval.expansion import expand_queries, generate_hyde_documents
from retrieval.filter import apply_similarity_filter
from retrieval.reranker import rerank_with_cross_encoder
from stores.custom_qdrant import QdrantStore


RetrieveLogger = Callable[[str], None]


def _embed(
    client, texts: List[str], model: str = "text-embedding-3-large"
) -> List[List[float]]:
    vectors: List[List[float]] = []
    for i in range(0, len(texts), 128):
        batch = texts[i : i + 128]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-8)


def _dedupe_by_text(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates by exact text match, preserving first occurrence order."""
    seen_texts: set[str] = set()
    deduped_hits: List[Dict[str, Any]] = []
    for hit in hits:
        text_value = hit.get("text", "")
        if text_value in seen_texts:
            continue
        seen_texts.add(text_value)
        deduped_hits.append(hit)
    return deduped_hits


def _pack_context(cfg: RootConfig, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    max_ctx = getattr(getattr(cfg, "query", None), "max_context_tokens", 4000)
    packed: List[Dict[str, Any]] = []
    token_budget = 0
    for result in results:
        text_value = result.get("text", "")
        token_len = len(text_value) // 4
        if token_budget + token_len > max_ctx:
            break
        packed.append(result)
        token_budget += token_len
    return packed


def retrieve_custom_chunks(
    *,
    cfg: RootConfig,
    store: Any,
    client: Any,
    query: str,
    logger: Optional[RetrieveLogger] = None,
    on_chunks: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> List[Dict[str, Any]]:
    log = logger or (lambda *_args, **_kwargs: None)

    expansion_cfg = cfg.query.expansion
    hyde_cfg = cfg.query.hyde
    similarity_cfg = cfg.query.similarity_filter
    rerank_cfg = cfg.query.rerank
    top_k = int(cfg.query.top_k)

    queries = expand_queries(
        base_query=query,
        config=expansion_cfg,
        openai_client=client,
    )
    hyde_docs = generate_hyde_documents(
        base_query=query,
        config=hyde_cfg,
        openai_client=client,
    )

    search_texts = list(queries) + list(hyde_docs)
    if not search_texts:
        search_texts = [query]

    log(f"[dim]Embedding {len(search_texts)} search text(s).[/dim]")
    query_vectors = _embed(client, search_texts, model=cfg.embeddings.model)

    hits_all: List[Dict[str, Any]] = []
    if isinstance(store, QdrantStore):
        collection_name = store.collection
        for vector in query_vectors:
            raw_hits = store.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            for entry in raw_hits:
                payload = entry.payload or {}
                title: Optional[str] = payload.get("title")
                if not title:
                    path_value = payload.get("path")
                    if isinstance(path_value, str):
                        title = Path(path_value).name
                metadata = {
                    k: v
                    for k, v in payload.items()
                    if k not in {"doc_id", "text"}
                }
                metadata.setdefault("path", payload.get("path"))
                hits_all.append(
                    {
                        "doc_id": payload.get("doc_id"),
                        "title": title,
                        "text": payload.get("text", ""),
                        "score": float(entry.score) if entry.score is not None else None,
                        "similarity_score": float(entry.score)
                        if entry.score is not None
                        else None,
                        "metadata": metadata,
                        "page": payload.get("page"),
                    }
                )
    else:
        raise NotImplementedError(
            "Custom backend search only implemented for Qdrant in starter kit"
        )

    if hits_all:
        log(f"[dim]Collected {len(hits_all)} raw hit(s).[/dim]")

    hits_all = apply_similarity_filter(hits_all, similarity_cfg)

    hits_all = rerank_with_cross_encoder(
        client=client,
        config=rerank_cfg,
        query=query,
        hits=hits_all,
    )

    hits_all = _dedupe_by_text(hits_all)
    packed = _pack_context(cfg, hits_all)

    if on_chunks is not None:
        on_chunks(packed)

    return [dict(chunk) for chunk in packed]


def build_retrieval_handler(
    *,
    cfg: RootConfig,
    store: Any,
    client: Any,
    logger: Optional[RetrieveLogger] = None,
    on_chunks: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    default_query: Optional[str] = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def _handler(args: Dict[str, Any]) -> Dict[str, Any]:
        search_query = args.get("query") or default_query
        if not search_query:
            raise ValueError("retrieve tool requires a non-empty 'query' argument")
        if logger:
            logger(f"[dim]Tool: retrieve[/dim] query='{search_query}'")
        chunks = retrieve_custom_chunks(
            cfg=cfg,
            store=store,
            client=client,
            query=search_query,
            logger=logger,
            on_chunks=on_chunks,
        )
        return {
            "chunks": [
                {
                    "doc_id": chunk.get("doc_id"),
                    "title": chunk.get("title"),
                    "text": chunk.get("text"),
                    "score": chunk.get("score"),
                    "similarity_score": chunk.get("similarity_score"),
                    "metadata": chunk.get("metadata", {}),
                    "page": chunk.get("page"),
                }
                for chunk in chunks
            ]
        }

    return _handler


_TOOL_SPEC_PATH = Path(__file__).resolve().parents[1] / "tools" / "retrieve_tool_spec.json"
with _TOOL_SPEC_PATH.open("r", encoding="utf-8") as _tool_file:
    RETRIEVE_TOOL_SPEC = json.load(_tool_file)


def answer_query(
    store, cfg: RootConfig, query: str, history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    rprint("[bold]Chat[/bold]: starting pipeline")
    if cfg.vector_store.backend == "openai_file_search":
        vs_id = store.vector_store_id
        rprint(
            f"[cyan]Using OpenAI File Search[/cyan] vector_store_id={vs_id} model={cfg.synthesis.model}"
        )
        out = synthesize_openai_file_search(
            cfg, query, vector_store_id=vs_id, history=history
        )
        rprint("[green]Synthesis complete[/green]")
        return out

    # custom backend: provide a retrieval tool the model can call
    client = build_openai_client(cfg)
    rprint("[cyan]Preparing retrieval tool...[/cyan]")

    last_chunks: List[Dict[str, Any]] = []

    def _on_chunks(chunks: List[Dict[str, Any]]) -> None:
        nonlocal last_chunks
        last_chunks = [dict(chunk) for chunk in chunks]

    retrieve_handler = build_retrieval_handler(
        cfg=cfg,
        store=store,
        client=client,
        logger=rprint,
        on_chunks=_on_chunks,
        default_query=query,
    )

    rprint("[cyan]Calling model with retrieval tool...[/cyan]")
    try:
        tool_use_rules = load_prompt("retrieval/tool_use_rules.md")
    except Exception:
        tool_use_rules = (
            "You have access to a function tool named 'retrieve' that searches a vector store and returns "
            "relevant chunks as JSON. If you need additional context to answer, call 'retrieve' with a "
            "clear search query derived from the latest user message (you may rephrase it). After you "
            "receive the chunks, use them to write a grounded answer. If no useful chunks are returned, "
            "ask a clarifying question. Do not fabricate citations."
        )
    output = chat_with_tools(
        cfg=cfg,
        user_query=query,
        history=history,
        tools=RETRIEVE_TOOL_SPEC,
        tool_handlers={"retrieve": retrieve_handler},
        developer_instructions=f"{cfg.synthesis.system_prompt}\n\n{tool_use_rules}",
    )
    rprint("[green]Synthesis complete[/green]")

    if isinstance(output, dict) and "answer_text" in output:
        cits = extract_spans(
            output.get("answer_text", ""),
            last_chunks,
            max_per_source=cfg.query.citations.max_per_source,
        )
        output["citations"] = cits
    return output
