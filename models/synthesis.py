from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
import json
from rich import print as rprint

from cli.config import RootConfig
from cli.env_utils import build_openai_client
from retrieval.citations import extract_spans

Message = Dict[str, str]


def synthesize_openai_file_search(
    cfg: RootConfig,
    query: str,
    vector_store_id: str,
    history: Optional[List[Message]] = None,
) -> Dict[str, Any]:
    client = build_openai_client(cfg)

    # If ranking options are specified, use Retrieval API search first
    ranking = getattr(cfg.vector_store.openai_file_search, "ranking_options", None)
    if ranking is not None:
        params: Dict[str, Any] = {
            "vector_store_id": vector_store_id,
            "query": query,
        }
        ro_dict: Dict[str, Any] = {}
        if ranking.ranker is not None:
            ro_dict["ranker"] = ranking.ranker
        if ranking.score_threshold is not None:
            ro_dict["score_threshold"] = float(ranking.score_threshold)
        if ro_dict:
            params["ranking_options"] = ro_dict

        results = client.vector_stores.search(**params)
        # Resolve filenames for nicer citation linking
        file_ids: List[str] = []
        for item in getattr(results, "data", []) or []:
            fid = getattr(item, "file_id", None)
            if fid and fid not in file_ids:
                file_ids.append(fid)
        id_to_name: Dict[str, str] = {}
        for fid in file_ids:
            try:
                fmeta = client.files.retrieve(file_id=fid)
                # filename may be under 'filename' or 'name' depending on SDK version
                id_to_name[fid] = getattr(fmeta, "filename", None) or getattr(
                    fmeta, "name", fid
                )
            except Exception:
                id_to_name[fid] = fid
        # Format results as context chunks using filename as identifier so app can map to local doc
        ctx_chunks: List[Dict[str, Any]] = []
        for item in getattr(results, "data", []) or []:
            text_parts = [
                c.text for c in item.content if getattr(c, "type", "") == "text"
            ]
            combined = "\n".join(text_parts)
            fid = getattr(item, "file_id", "")
            name = id_to_name.get(fid, fid)
            ctx_chunks.append({"doc_id": name, "text": combined, "score": item.score})
        return synthesize_with_context(cfg, query, ctx_chunks, history=history)

    # Default path: Stateless Responses API with file_search tool and explicit message history
    tools = [{"type": "file_search", "vector_store_ids": [vector_store_id]}]
    messages: List[Message] = [
        {"role": "developer", "content": cfg.synthesis.system_prompt}
    ]
    if history:
        for m in history:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m.get("content", "")})
    messages.append({"role": "user", "content": query})

    kwargs: Dict[str, Any] = {
        "model": cfg.synthesis.model,
        "reasoning": {"effort": cfg.synthesis.reasoning_effort},
        "input": messages,
        "tools": tools,
        "store": False,
    }

    resp = client.responses.create(**kwargs)
    text = getattr(resp, "output_text", None)
    if text is None:
        text = ""
        if hasattr(resp, "output"):
            for item in resp.output:
                if getattr(item, "type", "") == "message":
                    for block in getattr(item, "content", []) or []:
                        btype = getattr(block, "type", "")
                        if btype in ("output_text", "text"):
                            piece = getattr(block, "text", None) or getattr(
                                block, "value", None
                            )
                            if piece:
                                text += piece
    # Derive citations by searching top files and fuzzy-matching spans
    citations: List[Dict[str, Any]] = []
    try:
        sr = client.vector_stores.search(vector_store_id=vector_store_id, query=query)
        # Map file ids to filenames
        file_ids = []
        for item in getattr(sr, "data", []) or []:
            fid = getattr(item, "file_id", None)
            if fid and fid not in file_ids:
                file_ids.append(fid)
        id_to_name: Dict[str, str] = {}
        for fid in file_ids:
            try:
                fmeta = client.files.retrieve(file_id=fid)
                id_to_name[fid] = getattr(fmeta, "filename", None) or getattr(
                    fmeta, "name", fid
                )
            except Exception:
                id_to_name[fid] = fid
        ctx_chunks: List[Dict[str, Any]] = []
        for item in getattr(sr, "data", []) or []:
            text_parts = [
                c.text for c in item.content if getattr(c, "type", "") == "text"
            ]
            combined = "\n".join(text_parts)
            fid = getattr(item, "file_id", "")
            name = id_to_name.get(fid, fid)
            ctx_chunks.append(
                {
                    "doc_id": name,
                    "text": combined,
                    "score": getattr(item, "score", None),
                }
            )
        if text:
            citations = extract_spans(
                text, ctx_chunks, max_per_source=cfg.query.citations.max_per_source
            )
    except Exception:
        citations = []
    return {
        "answer_text": text or "",
        "citations": citations,
        "raw_response": resp.model_dump(mode="json"),
    }


def synthesize_with_context(
    cfg: RootConfig,
    query: str,
    context_chunks: List[Dict[str, Any]],
    history: Optional[List[Message]] = None,
) -> Dict[str, Any]:
    client = build_openai_client(cfg)
    context_text = "\n\n".join(
        [
            f"[{i+1}] (doc:{c['doc_id']}) {c['text']}"
            for i, c in enumerate(context_chunks)
        ]
    )
    developer_content = (
        cfg.synthesis.system_prompt
        + "\n\nContext below. Use it when relevant; if insufficient, ask a clarifying question.\n"
        + f"Context:\n{context_text}"
    )
    messages: List[Message] = [{"role": "developer", "content": developer_content}]
    if history:
        for m in history:
            if m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m.get("content", "")})
    messages.append({"role": "user", "content": query})

    kwargs: Dict[str, Any] = {
        "model": cfg.synthesis.model,
        "reasoning": {"effort": cfg.synthesis.reasoning_effort},
        "input": messages,
        "store": False,
    }

    resp = client.responses.create(**kwargs)
    text = getattr(resp, "output_text", None)
    if text is None:
        text = ""
        if hasattr(resp, "output"):
            for item in resp.output:
                if getattr(item, "type", "") == "message":
                    for block in getattr(item, "content", []) or []:
                        btype = getattr(block, "type", "")
                        if btype in ("output_text", "text"):
                            piece = getattr(block, "text", None) or getattr(
                                block, "value", None
                            )
                            if piece:
                                text += piece
    citations: List[Dict[str, Any]] = []
    try:
        if text:
            citations = extract_spans(
                text, context_chunks, max_per_source=cfg.query.citations.max_per_source
            )
    except Exception:
        citations = []
    return {
        "answer_text": text or "",
        "citations": citations,
        "notes": {},
        "raw_response": resp.model_dump(mode="json"),
    }


def chat_with_tools(
    cfg: RootConfig,
    user_query: str,
    history: Optional[List[Message]],
    tools: List[Dict[str, Any]],
    tool_handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
    developer_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Stateless Responses API tool loop per OpenAI docs.

    Accumulates response.output into the input list, detects function_call events,
    executes handlers, appends function_call_output items, and re-calls create
    until the model returns a normal assistant message.
    """
    client = build_openai_client(cfg)
    dev_text = developer_instructions or cfg.synthesis.system_prompt
    input_list: List[Any] = [{"role": "developer", "content": dev_text}]
    if history:
        for m in history:
            if m.get("role") in ("user", "assistant"):
                input_list.append({"role": m["role"], "content": m.get("content", "")})

    if not history or (
        history
        and (
            not history[-1].get("role") == "user"
            or history[-1].get("content") != user_query
        )
    ):
        input_list.append({"role": "user", "content": user_query})

    last_tool_outputs: List[Dict[str, Any]] = []
    max_iters = 6
    for _ in range(max_iters):
        resp = client.responses.create(
            model=cfg.synthesis.model,
            reasoning={"effort": cfg.synthesis.reasoning_effort},
            input=input_list,
            tools=tools,
            store=False,
        )

        out_items = getattr(resp, "output", []) or []

        sanitized_calls: List[Dict[str, Any]] = []
        for it in out_items:
            if getattr(it, "type", "") == "function_call":
                call_id = getattr(it, "call_id", None) or getattr(it, "id", None)
                sanitized_calls.append(
                    {
                        "type": "function_call",
                        "name": getattr(it, "name", ""),
                        "arguments": getattr(it, "arguments", "{}") or "{}",
                        "call_id": call_id,
                    }
                )
        input_list.extend(sanitized_calls)

        # Detect function calls
        function_calls = [
            it for it in out_items if getattr(it, "type", "") == "function_call"
        ]
        if function_calls:
            rprint(
                f"[cyan]Model requested function calls[/cyan]: {len(function_calls)}"
            )
            last_tool_outputs = []
            for fc in function_calls:
                name = getattr(fc, "name", "")
                args_text = getattr(fc, "arguments", "{}") or "{}"
                try:
                    args_parsed = json.loads(args_text)
                except Exception:
                    args_parsed = {"raw": args_text}
                rprint(f"[dim]Executing tool[/dim]: {name} args={args_parsed}")
                handler = tool_handlers.get(name)
                result = (
                    handler(args_parsed)
                    if handler
                    else {"error": f"no handler for tool '{name}'"}
                )
                last_tool_outputs.append(
                    {"name": name, "args": args_parsed, "result": result}
                )
                fc_call_id = getattr(fc, "call_id", None) or getattr(fc, "id", None)
                input_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": fc_call_id,
                        "output": json.dumps(result),
                    }
                )
            rprint("[cyan]Tool outputs appended; continuing model run...[/cyan]")
            continue

        # No function calls; extract assistant text
        text = getattr(resp, "output_text", None)
        if text is None:
            text = ""
            for item in out_items:
                if getattr(item, "type", "") == "message":
                    for block in getattr(item, "content", []) or []:
                        btype = getattr(block, "type", "")
                        if btype in ("output_text", "text"):
                            piece = getattr(block, "text", None) or getattr(
                                block, "value", None
                            )
                            if piece:
                                text += piece
        return {
            "answer_text": text or "",
            "tool_run": last_tool_outputs,
            "raw_response": resp.model_dump(mode="json"),
        }

    # If we exhausted iterations, return with whatever we have
    return {
        "answer_text": "",
        "tool_run": last_tool_outputs,
        "raw_response": resp.model_dump(mode="json"),
    }
