from __future__ import annotations

import uuid
from typing import Iterable, List

from ingestion.types import RawDocument, ChunkRecord
from .utils import detect_heading_sections, split_sentences, tokenize_len, pack_tokens


def recursive_chunk(
    d: RawDocument, target: tuple[int, int], overlap: int
) -> Iterable[ChunkRecord]:
    low, high = target

    def recurse(text: str, level: int = 0) -> List[str]:
        if tokenize_len(text) <= high:
            return [text]
        if level == 0:
            parts = [b for _, b in detect_heading_sections(text)]
        elif level == 1:
            import re

            parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        elif level == 2:
            parts = split_sentences(text)
        else:
            toks = text.split()
            win = 80
            parts = [" ".join(toks[i : i + win]) for i in range(0, len(toks), win)]
        if not parts or len(parts) == 1:
            size = tokenize_len(text)
            max_local = min(max(size // 2, low), high)
            return pack_tokens(text, max_local, overlap)
        out: List[str] = []
        for p in parts:
            out.extend(recurse(p, level + 1))
        return out

    for span in recurse(d.text, 0):
        yield ChunkRecord(
            id=str(uuid.uuid4()),
            doc_id=d.doc_id,
            text=span,
            page=None,
            start_offset=None,
            end_offset=None,
            metadata={"title": d.title, "path": d.path},
        )
