from __future__ import annotations

import uuid
from typing import Iterable

from ingestion.types import RawDocument, ChunkRecord
from .utils import detect_heading_sections, split_sentences, tokenize_len, pack_tokens


def heading_chunk(
    d: RawDocument, target: tuple[int, int], overlap: int
) -> Iterable[ChunkRecord]:
    low, high = target
    for head, body in detect_heading_sections(d.text):
        buf = ""
        for s in split_sentences(body):
            buf = f"{buf} {s}".strip() if buf else s
            if tokenize_len(buf) >= high:
                for span in pack_tokens(buf, high, overlap):
                    yield ChunkRecord(
                        id=str(uuid.uuid4()),
                        doc_id=d.doc_id,
                        text=(head + "\n\n" + span).strip() if head else span,
                        page=None,
                        start_offset=None,
                        end_offset=None,
                        metadata=(
                            {"title": d.title, "path": d.path, "heading": head}
                            if head
                            else {"title": d.title, "path": d.path}
                        ),
                    )
                buf = ""
        if buf:
            for span in pack_tokens(buf, max(tokenize_len(buf), low), overlap):
                yield ChunkRecord(
                    id=str(uuid.uuid4()),
                    doc_id=d.doc_id,
                    text=(head + "\n\n" + span).strip() if head else span,
                    page=None,
                    start_offset=None,
                    end_offset=None,
                    metadata=(
                        {"title": d.title, "path": d.path, "heading": head}
                        if head
                        else {"title": d.title, "path": d.path}
                    ),
                )
