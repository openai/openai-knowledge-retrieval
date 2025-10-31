from __future__ import annotations

import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Agent, ModelSettings, FileSearchTool, function_tool
from chatkit.agents import AgentContext
from cli.config import load_config
from openai import OpenAI
from retrieval.pipeline import retrieve_custom_chunks
from stores.base import make_store_from_config

client = OpenAI()

# Cache the last chunks retrieved for each thread so the backend can derive
# citations once the assistant responds.
_RETRIEVAL_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_THREAD_ID_CTX: ContextVar[Optional[str]] = ContextVar(
    "assistant_thread_id", default=None
)


with open(
    os.path.join(os.path.dirname(__file__), "../../../prompts/system/assistant.md"),
    "r",
    encoding="utf-8",
) as f:
    KNOWLEDGE_ASSISTANT_INSTRUCTIONS = f.read().strip()


def _resolve_cfg_path() -> str:
    cfg_path = os.getenv("RAG_CONFIG") or "configs/default.openai.yaml"
    if os.path.isabs(cfg_path):
        return cfg_path
    project_root = Path(__file__).resolve().parents[3]
    return str((project_root / cfg_path).resolve())


def record_retrieved_chunks(
    thread_id: Optional[str], chunks: List[Dict[str, Any]]
) -> None:
    if not thread_id or not chunks:
        return
    # Store a shallow copy so downstream consumers can mutate safely.
    _RETRIEVAL_CACHE[thread_id] = [dict(c) for c in chunks]


def get_last_retrieved_chunks(thread_id: str) -> List[Dict[str, Any]]:
    chunks = _RETRIEVAL_CACHE.get(thread_id, [])
    return [dict(c) for c in chunks]


def clear_last_retrieved_chunks(thread_id: str) -> None:
    _RETRIEVAL_CACHE.pop(thread_id, None)


def set_current_thread(thread_id: Optional[str]) -> Token[Optional[str]]:
    return _THREAD_ID_CTX.set(thread_id)


def reset_current_thread(token: Token[Optional[str]]) -> None:
    _THREAD_ID_CTX.reset(token)


def _build_tools() -> List[Any]:
    cfg_path = _resolve_cfg_path()
    cfg = load_config(cfg_path)
    backend = getattr(getattr(cfg, "vector_store", None), "backend", None)
    if backend == "openai_file_search":
        vector_store_id = os.getenv("VECTOR_STORE_ID")
        if not vector_store_id:
            # Fall back to configured ID if present
            vs_cfg = getattr(cfg.vector_store, "openai_file_search", None)
            vector_store_id = getattr(vs_cfg, "vector_store_id", None)

        return [
            FileSearchTool(
                vector_store_ids=[vector_store_id],
                max_num_results=getattr(cfg.query, "top_k", 8),
            )
        ]
    if backend == "custom":
        store = make_store_from_config(cfg)

        @function_tool(name_override="retrieve_context")
        def _retrieve_tool(query: str) -> Dict[str, Any]:
            resolved_query = (query or "").strip()
            if not resolved_query:
                raise ValueError("retrieve_context tool requires a non-empty query")
            chunks = retrieve_custom_chunks(
                cfg=cfg,
                store=store,
                client=client,
                query=resolved_query,
            )
            payload_chunks = [
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
            record_retrieved_chunks(_THREAD_ID_CTX.get(), payload_chunks)
            return {"chunks": payload_chunks}

        return [_retrieve_tool]

    return []

model_settings = ModelSettings(
    reasoning_effort="minimal",
)

assistant_agent = Agent[AgentContext](
    model="gpt-5-mini",
    model_settings=model_settings,
    name="Knowledge Retrieval",
    instructions=KNOWLEDGE_ASSISTANT_INSTRUCTIONS,
    tools=_build_tools(),
)
