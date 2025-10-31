from __future__ import annotations

import importlib.util
from typing import Iterable

from ingestion.types import RawDocument, ChunkRecord


class CustomChunkerProtocol:
    def __init__(self, **kwargs): ...
    def chunk(self, docs: Iterable[RawDocument]) -> Iterable[ChunkRecord]: ...


def run_custom_chunker(
    module_path: str, class_name: str, init_args: dict, docs: Iterable[RawDocument]
):
    spec = importlib.util.spec_from_file_location("user_chunker", module_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load custom chunker module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ChunkerCls = getattr(module, class_name)
    chunker: CustomChunkerProtocol = ChunkerCls(**init_args)
    for rec in chunker.chunk(docs):
        yield rec
