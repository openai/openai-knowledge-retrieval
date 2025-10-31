from __future__ import annotations

from typing import Any, Dict, List, Optional

from cli.config import RootConfig
from prompts.loader import load_prompt
from cli.env_utils import build_openai_client


def _extract_output_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text is not None:
        return text
    # Fallback: assemble from response.output message blocks
    text_out = ""
    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for block in getattr(item, "content", []) or []:
                    btype = getattr(block, "type", "")
                    if btype in ("output_text", "text"):
                        piece = getattr(block, "text", None) or getattr(
                            block, "value", None
                        )
                        if piece:
                            text_out += piece
    except Exception:
        pass
    return text_out


def _json_loads_safe(s: str) -> Any:
    import json
    import re

    t = (s or "").strip()
    # strip common code fences
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", t, flags=re.IGNORECASE)
    # remove trailing junk after last } or ]
    last_brace = max(t.rfind("}"), t.rfind("]"))
    if last_brace != -1:
        t = t[: last_brace + 1]
    return json.loads(t)


try:
    SYS_PROMPT = load_prompt("evals/qgen_single.md")
except Exception:
    SYS_PROMPT = (
        "You are generating evaluation questions about the provided source span. "
        "Create ONE question that a user might ask. The answer must be directly recoverable "
        "from the span. Output JSON with key 'questions' containing an array with one object having keys question, correct_answer, difficulty, tags."
    )

# Richer prompt for multi-question generation from a larger context
try:
    SYS_PROMPT_CONTEXT = load_prompt("evals/qgen_many.md")
except Exception:
    SYS_PROMPT_CONTEXT = (
        "You are an expert assessment writer for knowledge-assistant RAG systems.\n"
        "Task: Given the context, generate multiple high-quality, diverse questions.\n"
        "Requirements:\n"
        "- Each question must be answerable strictly from the provided context.\n"
        "- Prefer specific, detailed questions (e.g., values, conditions, constraints, procedures, edge cases).\n"
        "- Vary types: definitions, procedures/how-to, comparisons, why/causal, numeric lookups, exceptions, troubleshooting.\n"
        "- Avoid duplicates, trivial restatements, or vague questions.\n"
        "- For tables, code, or configurations, ask precise extraction or interpretation questions.\n"
        "- Calibrate difficulty (easy/medium/hard) based on reasoning and specificity required.\n"
        "Output: JSON with key 'questions' containing an array of objects, each with keys: question, correct_answer, difficulty, tags."
    )


def generate_one(
    cfg: RootConfig,
    span: Dict[str, Any],
    difficulty: str = "easy",
    client: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    try:
        client = client or build_openai_client(cfg)
        payload = {
            "span": span.get("text", ""),
            "hint_tags": span.get("tags", []),
            "difficulty": difficulty,
        }

        kwargs = {
            "model": cfg.synthesis.model,
            "input": [
                {"role": "developer", "content": SYS_PROMPT},
                {"role": "user", "content": str(payload)},
            ],
            "reasoning": {"effort": "minimal"},
            "store": False,
        }

        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": "qgen_many",
                "schema": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "correct_answer": {"type": "string"},
                                    "difficulty": {
                                        "type": "string",
                                        "enum": ["easy", "medium", "hard"],
                                    },
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "question",
                                    "correct_answer",
                                    "difficulty",
                                    "tags",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["questions"],
                    "additionalProperties": False,
                    "strict": True,
                },
            },
        }
        resp = client.responses.create(**kwargs)

        text = _extract_output_text(resp)
        data = _json_loads_safe(text)
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            items = data["questions"]
            data = items[0] if items else {}
        elif isinstance(data, list):
            data = data[0] if data else {}
        return {
            "question": data.get("question", ""),
            "correct_answer": data.get("correct_answer", ""),
            "citation_text": span.get("text", ""),
            "source_id": span.get("source_id"),
            "page": span.get("page"),
            "char_start": span.get("char_start"),
            "char_end": span.get("char_end"),
            "difficulty": data.get("difficulty", difficulty),
            "tags": span.get("tags", []),
        }
    except Exception:
        return None


def generate_many_from_context(
    cfg: RootConfig,
    context: Dict[str, Any],
    n_questions: int = 3,
    style_hints: Optional[List[str]] = None,
    client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Generate multiple diverse questions from a larger context block.

    Returns a list of rows compatible with downstream schema, using the full
    context as the citation_text to ensure the answer is recoverable.
    """
    try:
        client = client or build_openai_client(cfg)
        payload = {
            "context": context.get("text", ""),
            "request": {
                "count": int(max(1, n_questions)),
                "style_hints": style_hints
                or [
                    "definition",
                    "procedure",
                    "numeric",
                    "comparison",
                    "why",
                    "edge_case",
                ],
            },
        }
        kwargs = {
            "model": cfg.synthesis.model,
            "input": [
                {"role": "developer", "content": SYS_PROMPT_CONTEXT},
                {"role": "user", "content": __import__("json").dumps(payload)},
            ],
            "store": False,
        }
        use_structured = bool(getattr(cfg.synthesis, "structured_outputs", False))
        if use_structured:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "qgen_many",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "correct_answer": {"type": "string"},
                                        "difficulty": {
                                            "type": "string",
                                            "enum": ["easy", "medium", "hard"],
                                        },
                                        "tags": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": [
                                        "question",
                                        "correct_answer",
                                        "difficulty",
                                        "tags",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["questions"],
                        "additionalProperties": False,
                        "strict": True,
                    },
                },
            }
        resp = client.responses.create(**kwargs)

        text = _extract_output_text(resp)
        data = _json_loads_safe(text)
        if isinstance(data, dict):
            if isinstance(data.get("items"), list):
                data = data["items"]
            elif isinstance(data.get("questions"), list):
                data = data["questions"]
        if not isinstance(data, list):
            return []
        rows: List[Dict[str, Any]] = []
        for obj in data:
            if not isinstance(obj, dict):
                continue
            q = (obj.get("question") or "").strip()
            a = (obj.get("correct_answer") or "").strip()
            if not q or not a:
                continue
            rows.append(
                {
                    "question": q,
                    "correct_answer": a,
                    "citation_text": context.get("text", ""),
                    "source_id": context.get("source_id") or context.get("doc_id"),
                    "page": context.get("page"),
                    "char_start": context.get("char_start"),
                    "char_end": context.get("char_end"),
                    "difficulty": (obj.get("difficulty") or "").lower() or None,
                    "tags": obj.get("tags") or context.get("tags") or [],
                }
            )
        return rows
    except Exception:
        return []


def generate_questions(
    cfg: RootConfig,
    spans: List[Dict[str, Any]],
    difficulty: str = "easy",
) -> List[Dict[str, Any]]:
    """LLM-based question generation for spans -> (question, answer, citation_text).

    Returns list of dicts matching the QGen schema.
    """
    out: List[Dict[str, Any]] = []
    for sp in spans:
        row = generate_one(cfg, sp, difficulty)
        if row:
            out.append(row)
    return out
