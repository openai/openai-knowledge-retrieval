from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from rich import print as rprint

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
)

from cli.config import RootConfig
from stores.base import Chunk, ScoredChunk, VectorStore


_DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclid": Distance.EUCLID,
}

EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class QdrantStore(VectorStore):
    def __init__(self, cfg: RootConfig) -> None:
        q = cfg.vector_store.custom.qdrant
        timeout_s = float(os.getenv("RAG_QDRANT_TIMEOUT", "60"))
        prefer_grpc_env = os.getenv("RAG_QDRANT_PREFER_GRPC", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        model = getattr(cfg.embeddings, "model", "") or ""
        self._vector_dim = EMBEDDING_DIMS.get(model)
        if not self._vector_dim:
            self._vector_dim = 1536
            rprint(
                "[yellow]Unknown embedding dimension for model[/yellow] "
                f"{model!r}; defaulting to 1536"
            )
        # Reduce noisy version warnings while we log config explicitly
        self.client = QdrantClient(
            url=q.url, api_key=q.api_key, timeout=timeout_s, prefer_grpc=prefer_grpc_env
        )
        self.collection = q.collection
        self.distance = _DISTANCE_MAP[q.distance]
        self._ensure_collection(cfg)
        rprint(
            f"[cyan]Qdrant client[/cyan] url={q.url} collection={self.collection} "
            f"distance={q.distance} timeout={timeout_s}s prefer_grpc={prefer_grpc_env}"
        )

    def _ensure_collection(self, cfg: RootConfig) -> None:
        exists = False
        try:
            info = self.client.get_collection(self.collection)
            exists = info is not None
        except Exception:
            exists = False
        if not exists:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self._vector_dim, distance=self.distance),
            )
            rprint(
                "[green]Created Qdrant collection[/green] "
                f"name={self.collection} size={self._vector_dim} distance={self.distance}"
            )
        try:
            info = self.client.get_collection(self.collection)
            pc = getattr(info, "points_count", None)
            if pc is not None:
                rprint(f"[dim]Qdrant collection points[/dim]: {pc}")
        except Exception:
            pass

    def upsert(self, chunks: List[Chunk]) -> None:
        points = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError("QdrantStore.upsert requires embeddings on chunks")
            points.append(
                PointStruct(
                    id=c.id,
                    vector=c.embedding,
                    payload={"doc_id": c.doc_id, "text": c.text, **(c.metadata or {})},
                )
            )
        if points:
            # Use uploader which auto-splits to respect server payload limits and leverages gRPC if enabled
            self.client.upload_points(collection_name=self.collection, points=points)
            rprint(f"[dim]Uploaded {len(points)} point(s) to Qdrant[/dim]")

    def delete(self, doc_ids: List[str]) -> None:
        if not doc_ids:
            return
        if len(doc_ids) == 1 and doc_ids[0] == "*":
            self.client.delete(
                collection_name=self.collection, points_selector={"filter": {}}
            )
            return
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))])
        self.client.delete(
            collection_name=self.collection, points_selector={"filter": flt.dict()}
        )

    def search(
        self, query: str, k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredChunk]:
        # Note: We expect caller to pass an embedding for query or use client-side embedding
        # For simplicity, we embed in retrieval pipeline and call client.search with vector
        raise NotImplementedError(
            "Use retrieval.pipeline.search_qdrant with embedded query vector"
        )

    def stats(self) -> Dict[str, Any]:
        info = self.client.get_collection(self.collection)
        return {
            "backend": "qdrant",
            "collection": self.collection,
            "points_count": getattr(info, "points_count", None),
        }
