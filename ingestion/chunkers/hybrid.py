from __future__ import annotations

import uuid
from typing import Iterable

from ingestion.types import RawDocument, ChunkRecord
from .utils import detect_heading_sections, tokenize_len, pack_tokens


def hybrid_chunk(
    d: RawDocument, target: tuple[int, int], overlap: int
) -> Iterable[ChunkRecord]:
    low, high = target
    sections = detect_heading_sections(d.text)
    for head, body in sections:
        block = (head + "\n\n" + body).strip() if head else body
        size = tokenize_len(block)
        max_tokens = min(max(size // 2, low), high)
        for span in pack_tokens(block, max_tokens, overlap):
            yield ChunkRecord(
                id=str(uuid.uuid4()),
                doc_id=d.doc_id,
                text=span,
                page=None,
                start_offset=None,
                end_offset=None,
                metadata=(
                    {"title": d.title, "path": d.path, "heading": head}
                    if head
                    else {"title": d.title, "path": d.path}
                ),
            )
