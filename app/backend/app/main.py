from __future__ import annotations

import mimetypes
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List

from agents import Agent, RunConfig, Runner
from agents.model_settings import ModelSettings
from chatkit.agents import AgentContext, stream_agent_response
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.types import (
    Annotation,
    AssistantMessageContent,
    AssistantMessageItem,
    Attachment,
    ClientToolCallItem,
    FileSource,
    ThreadItem,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
)
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from cli.config import RootConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
except Exception:
    pass

from cli.config import load_config
from retrieval.citations import extract_spans

from .assistant_agent import (
    assistant_agent,
    clear_last_retrieved_chunks,
    get_last_retrieved_chunks,
    reset_current_thread,
    set_current_thread,
)
from .documents import (
    DocumentMetadata,
    as_dicts,
)
from .memory_store import MemoryStore

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # legacy fallback


def _sha1_of_path(path: Path) -> str:
    import hashlib

    return hashlib.sha1(str(path).encode()).hexdigest()


def _humanize_title(filename: str) -> str:
    stem = Path(filename).stem
    stem = stem.replace("_", " ").replace("-", " ")
    return " ".join(part.capitalize() for part in stem.split())


def _user_message_text(item: UserMessageItem) -> str:
    parts: list[str] = []
    for part in item.content:
        text = getattr(part, "text", None)
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _assistant_message_text(item: AssistantMessageItem) -> str:
    parts: list[str] = []
    for content in item.content:
        if isinstance(content, AssistantMessageContent):
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return " ".join(parts).strip()


def _normalise(value: str) -> str:
    return value.strip().lower()


def _resolve_document(
    annotation: Annotation, document_path_cache: Dict[str, Dict[str, Path]]
) -> DocumentMetadata | None:
    source = getattr(annotation, "source", None)
    if not source or getattr(source, "type", None) != "file":
        return None

    filename = getattr(source, "filename", None)
    if not filename:
        return None

    target_name = _normalise_filename(filename)

    # Search the cached mapping of document ids to file paths for a filename match
    for _cfg_path, id_to_path in document_path_cache.items():
        for doc_id, file_path in id_to_path.items():
            if _normalise_filename(file_path.name) == target_name:
                title = _humanize_title(file_path.name)
                return DocumentMetadata(
                    id=doc_id, filename=file_path.name, title=title, description=None
                )

    return None


def _normalise_filename(value: str) -> str:
    return Path(value).name.strip().lower()


def _is_tool_completion_item(item: Any) -> bool:
    return isinstance(item, ClientToolCallItem)


class KnowledgeAssistantServer(ChatKitServer[dict[str, Any]]):
    def __init__(self, agent: Agent[AgentContext]) -> None:
        self.store = MemoryStore()
        super().__init__(self.store)
        self.assistant = agent
        self._cfg_cache: Dict[str, RootConfig] = {}
        self._cfg_path = os.getenv("RAG_CONFIG") or "configs/default.openai.yaml"
        self._documents_cache: Dict[str, list[DocumentMetadata]] = {}
        self._document_path_cache: Dict[str, Dict[str, Path]] = {}
        cfg = self._load_cfg(self._cfg_path)
        vector_store = getattr(cfg, "vector_store", None)
        self.vector_store_backend: str | None = getattr(vector_store, "backend", None)

    def _load_cfg(self, path: str) -> RootConfig:
        # Resolve path relative to repository root if not absolute
        resolved = (
            str((PROJECT_ROOT / path).resolve()) if not os.path.isabs(path) else path
        )
        if resolved not in self._cfg_cache:
            self._cfg_cache[resolved] = load_config(resolved)
        return self._cfg_cache[resolved]

    def _scan_documents(self, cfg_path: str, cfg: RootConfig) -> list[DocumentMetadata]:
        # Cached
        cached = self._documents_cache.get(cfg_path)
        if cached is not None:
            return cached

        include_exts = set(
            getattr(getattr(cfg, "data", None), "include_extensions", []) or []
        )
        include_exts = {ext.lower() for ext in include_exts}
        exclude_globs = list(
            getattr(getattr(cfg, "data", None), "exclude_globs", []) or []
        )

        roots: list[Path] = []
        data_paths = getattr(getattr(cfg, "data", None), "paths", None)
        if data_paths:
            for p in data_paths:
                # Expand env vars and user home in configured paths
                expanded = os.path.expandvars(os.path.expanduser(p))
                base = Path(expanded)
                if not base.is_absolute():
                    base = (PROJECT_ROOT / base).resolve()
                if base.exists() and base.is_dir():
                    roots.append(base)
        if not roots:
            roots = [(_DATA_DIR).resolve()]

        documents: list[DocumentMetadata] = []
        id_to_path: Dict[str, Path] = {}

        import fnmatch

        for root in roots:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                if include_exts and ext not in include_exts:
                    continue
                # Apply exclude globs relative to root and absolute
                rel = str(path.relative_to(root))
                abs_path_str = str(path)
                if any(
                    fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(abs_path_str, pat)
                    for pat in exclude_globs
                ):
                    continue
                doc_id = _sha1_of_path(path)
                filename = path.name
                title = _humanize_title(filename)
                documents.append(
                    DocumentMetadata(
                        id=doc_id, filename=filename, title=title, description=None
                    )
                )
                id_to_path.setdefault(doc_id, path)

        # Deduplicate preserving first occurrence
        seen: set[str] = set()
        deduped: list[DocumentMetadata] = []
        for d in documents:
            if d.id in seen:
                continue
            seen.add(d.id)
            deduped.append(d)

        self._documents_cache[cfg_path] = deduped
        self._document_path_cache[cfg_path] = id_to_path
        return deduped

    def _lookup_document(
        self, cfg_path: str, cfg: RootConfig, identifier: str | None
    ) -> DocumentMetadata | None:
        if not identifier:
            return None
        docs = self._scan_documents(cfg_path, cfg)
        # exact id
        for d in docs:
            if d.id == identifier:
                return d
        norm = _normalise(identifier)
        for d in docs:
            if d.filename.lower() == norm or Path(d.filename).stem.lower() == norm:
                return d
        slug = "".join(ch for ch in identifier.lower() if ch.isalnum())
        for d in docs:
            if "".join(ch for ch in d.title.lower() if ch.isalnum()) == slug:
                return d
        return None

    def _resolve_document_path(self, cfg_path: str, doc_id: str) -> Path | None:
        return (self._document_path_cache.get(cfg_path) or {}).get(doc_id)

    async def respond(
        self,
        thread: ThreadMetadata,
        item: ThreadItem | None,
        context: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:
        if item is None:
            return

        if _is_tool_completion_item(item):
            return

        if not isinstance(item, UserMessageItem):
            return

        message_text = _user_message_text(item)
        if not message_text:
            return

        enriched_text = await self._build_history_prompt(
            thread_id=thread.id,
            base_text=message_text,
            context=context,
        )

        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=context,
        )
        cfg = self._load_cfg(self._cfg_path)
        token = set_current_thread(thread.id)
        try:
            result = Runner.run_streamed(
                self.assistant,
                enriched_text,
                context=agent_context,
                run_config=RunConfig(model_settings=ModelSettings()),
            )

            async for event in stream_agent_response(agent_context, result):
                if isinstance(event, ThreadItemDoneEvent) and isinstance(
                    event.item, AssistantMessageItem
                ):
                    self._apply_cached_citations_to_item(thread.id, event.item, cfg)
                yield event
        finally:
            reset_current_thread(token)

    async def to_message_content(self, input: Attachment) -> ResponseInputContentParam:
        raise RuntimeError("File attachments are not supported in this demo.")

    async def latest_citations(
        self, thread_id: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        items = await self.store.load_thread_items(
            thread_id,
            after=None,
            limit=50,
            order="desc",
            context=context,
        )

        cfg = self._load_cfg(self._cfg_path)
        for item in items.data:
            if isinstance(item, AssistantMessageItem):
                citations = self._extract_citations(item, cfg, thread_id)
                if citations:
                    return citations
        return []

    def _extract_citations(
        self,
        item: AssistantMessageItem,
        cfg: RootConfig,
        thread_id: str,
    ) -> List[dict[str, Any]]:
        citations = self._citations_from_annotations(item, cfg)
        if citations:
            return citations

        derived = self._apply_cached_citations_to_item(thread_id, item, cfg)
        if derived:
            return derived

        texts = chain.from_iterable(
            content.text.splitlines()
            for content in item.content
            if isinstance(content, AssistantMessageContent)
        )
        for line in texts:
            for document in _documents_from_text(line):
                citations.append(
                    {
                        "document_id": document.id,
                        "filename": document.filename,
                        "title": document.title,
                        "description": document.description,
                        "annotation_index": None,
                    }
                )
        return citations

    def _citations_from_annotations(
        self,
        item: AssistantMessageItem,
        cfg: RootConfig,
    ) -> List[dict[str, Any]]:
        citations: List[dict[str, Any]] = []
        for content in item.content:
            if not isinstance(content, AssistantMessageContent):
                continue
            for annotation in content.annotations:
                document = _resolve_document(annotation, self._document_path_cache)
                if not document:
                    source_doc_id = getattr(annotation.source, "group", None)
                    if source_doc_id:
                        document = self._lookup_document(
                            self._cfg_path, cfg, source_doc_id
                        )
                doc_id = (
                    document.id
                    if document
                    else getattr(annotation.source, "group", None)
                )
                filename = (
                    document.filename
                    if document
                    else getattr(getattr(annotation, "source", None), "filename", "")
                )
                title = (
                    document.title
                    if document
                    else getattr(annotation.source, "title", "")
                )
                description = (
                    document.description
                    if document
                    else getattr(annotation.source, "description", None)
                )
                citations.append(
                    {
                        "document_id": doc_id or "",
                        "filename": filename or "",
                        "title": title,
                        "description": description,
                        "annotation_index": annotation.index,
                    }
                )
        return citations

    def _apply_cached_citations_to_item(
        self,
        thread_id: str,
        item: AssistantMessageItem,
        cfg: RootConfig,
    ) -> List[dict[str, Any]]:
        results, annotations = self._build_citations_from_chunks(thread_id, item, cfg)
        if annotations:
            for content in item.content:
                if isinstance(content, AssistantMessageContent):
                    content.annotations = [
                        ann.model_copy(deep=True) for ann in annotations
                    ]
        return results

    def _build_citations_from_chunks(
        self,
        thread_id: str,
        item: AssistantMessageItem,
        cfg: RootConfig,
    ) -> tuple[List[dict[str, Any]], List[Annotation]]:
        chunks = get_last_retrieved_chunks(thread_id)
        if not chunks:
            return ([], [])

        answer_text_parts = [
            getattr(content, "text", "") or ""
            for content in item.content
            if isinstance(content, AssistantMessageContent)
        ]
        answer_text = " ".join(
            part.strip() for part in answer_text_parts if part
        ).strip()
        if not answer_text:
            return ([], [])

        spans = extract_spans(
            answer_text,
            chunks,
            max_per_source=getattr(
                getattr(cfg.query, "citations", None), "max_per_source", 3
            ),
        )
        if not spans:
            clear_last_retrieved_chunks(thread_id)
            return ([], [])

        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            chunk_id = chunk.get("doc_id")
            if isinstance(chunk_id, str):
                chunk_lookup[chunk_id] = chunk

        results: List[dict[str, Any]] = []
        annotations: List[Annotation] = []
        for idx, span in enumerate(spans):
            doc_id = span.get("doc_id")
            if not doc_id:
                continue
            document = (
                self._lookup_document(self._cfg_path, cfg, doc_id) if doc_id else None
            )
            chunk_info = chunk_lookup.get(doc_id)
            chunk_title = None
            chunk_filename = None
            if chunk_info:
                chunk_title = chunk_info.get("title")
                path_value = (
                    (chunk_info.get("metadata") or {}).get("path")
                    if isinstance(chunk_info.get("metadata"), dict)
                    else None
                )
                if isinstance(path_value, str):
                    chunk_filename = Path(path_value).name

            filename = document.filename if document else chunk_filename or ""
            title = (
                (document.title if document else None)
                or span.get("title")
                or chunk_title
                or (filename and _humanize_title(filename))
                or (doc_id or "")
            )
            description = document.description if document else chunk_title

            source = FileSource(
                filename=filename or doc_id,
                title=title,
                description=description,
                group=(document.id if document else doc_id) or None,
            )
            annotations.append(Annotation(source=source, index=idx))

            results.append(
                {
                    "document_id": (document.id if document else doc_id) or "",
                    "filename": filename or doc_id,
                    "title": title,
                    "description": description,
                    "annotation_index": idx,
                }
            )

        clear_last_retrieved_chunks(thread_id)
        return results, annotations

    async def _build_history_prompt(
        self,
        thread_id: str,
        base_text: str,
        context: dict[str, Any],
        max_turns: int = 8,
    ) -> str:
        if not thread_id:
            return base_text

        try:
            history_page = await self.store.load_thread_items(
                thread_id,
                after=None,
                limit=max_turns * 2,
                order="asc",
                context=context,
            )
        except Exception:
            return base_text

        lines: List[str] = []
        for entry in history_page.data[-(max_turns * 2) :]:
            if isinstance(entry, UserMessageItem):
                text = _user_message_text(entry)
                if text:
                    lines.append(f"User: {text}")
            elif isinstance(entry, AssistantMessageItem):
                text = _assistant_message_text(entry)
                if text:
                    lines.append(f"Assistant: {text}")

        if not lines:
            return base_text

        history_block = "\n".join(lines[-(max_turns * 2) :])
        return (
            "Conversation history so far (chronological):\n"
            f"{history_block}\n\n"
            f"Latest user message:\n{base_text}"
        )


knowledge_server = KnowledgeAssistantServer(agent=assistant_agent)

app = FastAPI(title="ChatKit Knowledge Retrieval API")

def _parse_cors_allow_origins(value: str | None) -> list[str]:
    if not value:
        return []
    return [origin.strip() for origin in value.split(",") if origin.strip()]


cors_allow_origins = _parse_cors_allow_origins(
    os.getenv("KNOWLEDGE_CORS_ALLOW_ORIGINS")
) or [
    "http://localhost:5172",
    "http://127.0.0.1:5172",
]

if cors_allow_origins:
    # Restrict cross-origin access to the known frontend origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "Authorization"],
        expose_headers=["Content-Disposition"],
    )


def get_server() -> KnowledgeAssistantServer:
    return knowledge_server


@app.post("/knowledge/chatkit")
async def chatkit_endpoint(
    request: Request, server: KnowledgeAssistantServer = Depends(get_server)
) -> Response:
    payload = await request.body()
    result = await server.process(payload, {"request": request})
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    if hasattr(result, "json"):
        return Response(content=result.json, media_type="application/json")
    return JSONResponse(result)


@app.get("/knowledge/documents")
async def list_documents(
    server: KnowledgeAssistantServer = Depends(get_server),
) -> dict[str, Any]:
    cfg_path = server._cfg_path
    cfg = server._load_cfg(cfg_path)
    docs = server._scan_documents(cfg_path, cfg)
    return {"documents": as_dicts(docs)}


@app.get("/knowledge/documents/{document_id}/file")
async def document_file(
    document_id: str, server: KnowledgeAssistantServer = Depends(get_server)
) -> FileResponse:
    cfg_path = server._cfg_path
    cfg = server._load_cfg(cfg_path)
    document = server._lookup_document(cfg_path, cfg, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = server._resolve_document_path(cfg_path, document.id)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not available")

    media_type, _ = mimetypes.guess_type(str(file_path))
    headers = {"Content-Disposition": f'inline; filename="{document.filename}"'}
    return FileResponse(
        file_path,
        media_type=media_type or "application/octet-stream",
        headers=headers,
    )


@app.get("/knowledge/threads/{thread_id}/citations")
async def thread_citations(
    thread_id: str,
    request: Request,
    server: KnowledgeAssistantServer = Depends(get_server),
) -> dict[str, Any]:
    context = {"request": request}
    citations = await server.latest_citations(thread_id, context=context)
    doc_ids = sorted(
        {
            citation.get("document_id") or citation.get("doc_id")
            for citation in citations
            if citation
        }
    )
    return {"documentIds": doc_ids, "citations": citations}


@app.get("/knowledge/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}
