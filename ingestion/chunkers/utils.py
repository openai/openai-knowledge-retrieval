from __future__ import annotations

import re
from typing import List, Tuple

import tiktoken


def tokenize_len(text: str, model: str = "gpt-5-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def pack_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(enc.decode(chunk_tokens))
            if end == len(tokens):
                break
            start = max(0, end - overlap)
        return chunks
    except Exception:
        approx = max_tokens * 4
        step = approx - (overlap * 4)
        return [text[i : i + approx] for i in range(0, len(text), max(step, 1))]


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=\n)\n+")


def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]
    return parts if parts else [text]


def is_heading_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("#") and len(s) > 1:
        return True
    if len(s) <= 80 and s.upper() == s and re.search(r"[A-Z]", s):
        return True
    if re.match(r"^\d+[\d\.]*\s+\S+", s):
        return True
    return False


def detect_heading_sections(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    sections: List[Tuple[str, str]] = []
    cur_head: str | None = None
    cur_body: List[str] = []
    for ln in lines:
        if is_heading_line(ln):
            if cur_head is not None:
                sections.append((cur_head, "\n".join(cur_body).strip()))
                cur_body = []
            cur_head = ln.strip("# ").strip()
        else:
            cur_body.append(ln)
    if cur_head is None:
        return [("", text)]
    sections.append((cur_head or "", "\n".join(cur_body).strip()))
    return sections
