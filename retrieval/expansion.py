from __future__ import annotations

from typing import Callable, List, Literal, Sequence

from pydantic import BaseModel
from openai import OpenAI
from prompts.loader import load_prompt
from cli.config import ExpansionConfig, HydeConfig

from rich import print as rprint

ReasoningEffort = Literal["minimal", "low", "medium", "high"]

class QuerySchema(BaseModel):
    items: List[str]

def _call_llm(
    client: OpenAI,
    *,
    model: str,
    reasoning_effort: ReasoningEffort,
    user_prompt: str,
    developer_prompt: str | None = None,
    expected_items: int,
) -> List[str]:
    messages = []
    messages.append({"role": "developer", "content": developer_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    output = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        input=messages,
        text_format=QuerySchema,
    )
    items = output.output_parsed.items[:expected_items]
    return items

def expand_queries(
    base_query: str,
    *,
    config: ExpansionConfig,
    openai_client: OpenAI | None = None,
) -> List[str]:
    queries = [base_query]
    if not config.enabled:
        return queries

    styles: Sequence[Literal["keywords", "paraphrase", "antonyms"]] = config.style
    styles_text = ", ".join(styles) if styles else "none"

    developer_template = load_prompt(config.prompt_path)
    developer_prompt = developer_template.format(
        variant_count=config.variants,
        styles=styles_text,
    )
    
    user_prompt = base_query
    llm_items = _call_llm(
        openai_client,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        user_prompt=user_prompt,
        developer_prompt=developer_prompt,
        expected_items=config.variants,
    )

    rprint("[dim]Expansion variants:[/dim]")
    for variant in llm_items:
        rprint(f"  • {variant}")
    
    return llm_items


def generate_hyde_documents(
    base_query: str,
    *,
    config: HydeConfig,
    openai_client: OpenAI | None = None,
) -> List[str]:
    if not config.enabled or not openai_client:
        return []
    developer_prompt = load_prompt(config.prompt_path)
    
    user_prompt = base_query
    hyde_items = _call_llm(
        openai_client,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        user_prompt=user_prompt,
        developer_prompt=developer_prompt,
        expected_items=1,
    )

    rprint("[dim]HyDE passage:[/dim]")
    for doc in hyde_items:
        preview = " ".join(doc.strip().split())
        if len(preview) > 240:
            preview = preview[:237] + "..."
        rprint(f"  • {preview}")
    
    return hyde_items
