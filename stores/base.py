from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

# Factory
from cli.config import RootConfig


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def upsert(self, chunks: List[Chunk]) -> None: ...
    def delete(self, doc_ids: List[str]) -> None: ...
    def search(
        self, query: str, k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredChunk]: ...
    def stats(self) -> Dict[str, Any]: ...


def make_store_from_config(cfg: RootConfig) -> VectorStore:
    if cfg.vector_store.backend == "openai_file_search":
        from stores.openai_file_search import OpenAIFileSearchStore

        return OpenAIFileSearchStore(cfg)
    elif cfg.vector_store.backend == "custom":
        kind = cfg.vector_store.custom.kind
        if kind == "qdrant":
            from stores.custom_qdrant import QdrantStore

            return QdrantStore(cfg)
        elif kind == "plugin":
            import importlib.util

            mod_path = cfg.vector_store.custom.plugin.module_path
            class_name = cfg.vector_store.custom.plugin.class_name
            spec = importlib.util.spec_from_file_location("plugin_store", mod_path)
            if not spec or not spec.loader:
                raise RuntimeError(f"Cannot load plugin module at {mod_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            store_cls = getattr(module, class_name)
            return store_cls(**cfg.vector_store.custom.plugin.init_args)
        else:
            raise NotImplementedError(
                f"Custom store kind '{kind}' not implemented in starter kit"
            )
    else:
        raise NotImplementedError(f"Backend {cfg.vector_store.backend} not supported")
