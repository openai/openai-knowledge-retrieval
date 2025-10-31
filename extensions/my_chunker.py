from __future__ import annotations

import uuid
from typing import Iterable

from ingestion.types import ChunkRecord, RawDocument


class MyChunker:
    """Simple custom chunker.

    Splits each document's text into overlapping windows by approximate token count
    (4 chars ≈ 1 token heuristic). Produces `ChunkRecord` items compatible with the
    ingestion pipeline.
    """

    def __init__(self, max_tokens: int = 700, overlap: int = 80) -> None:
        self.max_tokens = max(1, int(max_tokens))
        self.overlap = max(0, int(overlap))

        # Convert token settings into character counts using a rough heuristic.
        self.window_chars = self.max_tokens * 4
        self.overlap_chars = self.overlap * 4

    def chunk(self, docs: Iterable[RawDocument]) -> Iterable[ChunkRecord]:
        for doc in docs:
            # Prefer page-aware splitting if pages are present; otherwise fall back to raw text.
            if doc.pages:
                for page in doc.pages:
                    yield from self._chunk_text(
                        doc_id=doc.doc_id,
                        full_text=page.text or "",
                        page_number=page.number,
                        base_metadata=doc.metadata,
                    )
            else:
                yield from self._chunk_text(
                    doc_id=doc.doc_id,
                    full_text=doc.text or "",
                    page_number=None,
                    base_metadata=doc.metadata,
                )

    def _chunk_text(
        self,
        *,
        doc_id: str,
        full_text: str,
        page_number: int | None,
        base_metadata: dict,
    ) -> Iterable[ChunkRecord]:
        if not full_text:
            return

        start: int = 0
        end: int
        length: int = len(full_text)
        step: int = max(1, self.window_chars - self.overlap_chars)

        while start < length:
            end = min(length, start + self.window_chars)
            chunk_text = full_text[start:end]

            # Derive a stable UUIDv5 for the chunk (Qdrant requires int or UUID IDs)
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{start}:{end}"))

            metadata: dict = dict(base_metadata or {})
            metadata.update({"chunk_start": start, "chunk_end": end})

            yield ChunkRecord(
                id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                page=page_number,
                start_offset=start,
                end_offset=end,
                metadata=metadata,
            )

            if end >= length:
                break
            start = start + step
