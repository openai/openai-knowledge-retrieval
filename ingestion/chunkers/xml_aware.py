from __future__ import annotations

import uuid
from typing import Iterable, List

from ingestion.types import RawDocument, ChunkRecord
from .utils import pack_tokens


def xml_aware_chunk(
    d: RawDocument, target: tuple[int, int], overlap: int
) -> Iterable[ChunkRecord]:
    low, high = target
    try:
        from lxml import etree

        root = etree.fromstring(d.text.encode("utf-8"))
    except Exception:
        # Not XML; fall back to single-pack
        for span in pack_tokens(d.text, high, overlap):
            yield ChunkRecord(
                id=str(uuid.uuid4()),
                doc_id=d.doc_id,
                text=span,
                page=None,
                start_offset=None,
                end_offset=None,
                metadata={"title": d.title, "path": d.path, "format": "xml-fallback"},
            )
        return

    def text_of(elem) -> str:
        return "".join(elem.itertext())

    def table_to_md(elem) -> str:
        rows: List[str] = []
        for row in elem.findall(".//row"):
            cells = [text_of(c).strip() for c in row.findall(".//entry")]
            rows.append("| " + " | ".join(cells) + " |")
        if not rows:
            return text_of(elem)
        hdr = rows[0]
        sep = "|" + " --- |" * max(1, hdr.count("|") - 1)
        return "\n".join([rows[0], sep, *rows[1:]])

    blocks: List[str] = []
    for blk in root.findall(".//block"):
        parts: List[str] = []
        head = blk.find(".//head")
        if head is not None:
            parts.append(text_of(head).strip())
        for tbl in list(blk.findall(".//table")) + list(blk.findall(".//tgroup")):
            parts.append(table_to_md(tbl))
        for step in blk.findall(".//tsmalf"):
            parts.append("- " + text_of(step).strip())
        remaining = text_of(blk).strip()
        if remaining:
            parts.append(remaining)
        txt = "\n\n".join([p for p in parts if p])
        if txt:
            blocks.append(txt)
    if not blocks:
        blocks = [text_of(root)]

    for b in blocks:
        for span in pack_tokens(b, high, overlap):
            yield ChunkRecord(
                id=str(uuid.uuid4()),
                doc_id=d.doc_id,
                text=span,
                page=None,
                start_offset=None,
                end_offset=None,
                metadata={"title": d.title, "path": d.path, "format": "xml"},
            )
