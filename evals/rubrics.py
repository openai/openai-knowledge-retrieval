from __future__ import annotations

from typing import Any, Dict, List

RUBRIC_GROUNDEDNESS = """
You are grading if the assistant's answer is grounded in the provided context. Return a JSON with fields: {"grounded": true|false, "notes": "..."}
"""

OPENAI_GROUNDEDNESS_SYSTEM_PROMPT = """You grade **Groundedness** on 1-5.
Only reward claims supported by the provided Context (retrieved snippets).
Penalize any unsupported or contradictory statement.
Return JSON: {"result":"float"} (stringified float 1..5)."""

OPENAI_GROUNDEDNESS_USER_PROMPT = """Context (retrieved snippets):
{{item.citation_text}}
Candidate Answer:
{{sample.output_text}}"""

OPENAI_RELEVANCE_SYSTEM_PROMPT = """You grade **Answer Relevance** on 1-7.
Score how directly and completely the answer addresses the user query.
Return JSON: {"result":"float"} (stringified float 1..7)."""

OPENAI_RELEVANCE_USER_PROMPT = """Query:
{{item.question}}
Answer:
{{sample.output_text}}"""


def default_openai_evals_graders() -> List[Dict[str, Any]]:
    """Default graders for OpenAI eval runs.

    Returns the groundedness and relevance graders mirroring the previous hard-coded
    configuration. Users can override or extend these in the YAML config.
    """
    return [
        {
            "type": "score_model",
            "name": "Groundedness",
            "model": "gpt-5-mini",
            "input": [
                {"role": "system", "content": OPENAI_GROUNDEDNESS_SYSTEM_PROMPT},
                {"role": "user", "content": OPENAI_GROUNDEDNESS_USER_PROMPT},
            ],
            "range": [1, 5],
            "pass_threshold": 4.3,
        },
        {
            "type": "score_model",
            "name": "Answer Relevance",
            "model": "gpt-5-mini",
            "input": [
                {"role": "system", "content": OPENAI_RELEVANCE_SYSTEM_PROMPT},
                {"role": "user", "content": OPENAI_RELEVANCE_USER_PROMPT},
            ],
            "range": [1, 7],
            "pass_threshold": 5.5,
        },
    ]
