from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, model_validator
from prompts.loader import load_prompt
import yaml

from evals.rubrics import default_openai_evals_graders


def _resolve_prompt_references(data: Any) -> Any:
    if isinstance(data, dict):
        if "prompt_file" in data:
            path = data.pop("prompt_file")
            if path:
                data["content"] = load_prompt(str(path))
        for key, value in list(data.items()):
            data[key] = _resolve_prompt_references(value)
        return data
    if isinstance(data, list):
        return [_resolve_prompt_references(item) for item in data]
    return data


class EnvConfig(BaseModel):
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    openai_project: Optional[str] = None


class DataConfig(BaseModel):
    paths: List[str]
    include_extensions: List[str] = Field(
        default_factory=lambda: [
            ".pdf",
            ".md",
            ".html",
            ".txt",
            ".docx",
            ".csv",
            ".xml",
        ]
    )
    exclude_globs: List[str] = Field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/.git/**",
            "**/data/example_data/**",
        ]
    )


class ChunkingRules(BaseModel):
    tables: Optional[Literal["extract_markdown", "ignore", "as_text"]] = (
        "extract_markdown"
    )
    code: Optional[Literal["block_preserve", "inline"]] = "block_preserve"
    xml: Optional[Dict[str, Any]] = None


class ChunkingMetadata(BaseModel):
    add: Dict[str, Any] = Field(default_factory=dict)
    derive: Dict[str, Any] = Field(
        default_factory=lambda: {
            "headings": True,
            "page_numbers": True,
            "language": True,
        }
    )


class ChunkingConfig(BaseModel):
    strategy: Literal["recursive", "heading", "hybrid", "xml_aware", "custom"] = (
        "hybrid"
    )
    target_token_range: Tuple[int, int] = (300, 700)
    overlap_tokens: int = 60
    rules: ChunkingRules = Field(default_factory=ChunkingRules)
    metadata: ChunkingMetadata = Field(default_factory=ChunkingMetadata)


class CustomChunkerConfig(BaseModel):
    module_path: str
    class_name: str
    init_args: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingsConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    model: str = "text-embedding-3-large"
    batch_size: int = 128


class OpenAIChunkingOptions(BaseModel):
    max_chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 400


class OpenAIFileSearchConfig(BaseModel):
    vector_store_name: str
    # Optional existing vector store id to reuse (e.g., "vs_...")
    vector_store_id: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    expiry_days: Optional[int] = 30
    chunking: Optional[OpenAIChunkingOptions] = None
    ranking_options: Optional["RankingOptions"] = None


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    collection: str
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    ef: int = 128
    m: int = 64


class PluginStoreConfig(BaseModel):
    module_path: str
    class_name: str
    init_args: Dict[str, Any] = Field(default_factory=dict)


class CustomVectorStoreConfig(BaseModel):
    kind: Literal["qdrant", "pgvector", "pinecone", "weaviate", "plugin"] = "qdrant"
    qdrant: Optional[QdrantConfig] = None
    plugin: Optional[PluginStoreConfig] = None


class VectorStoreConfig(BaseModel):
    backend: Literal["openai_file_search", "custom"] = "openai_file_search"
    openai_file_search: Optional[OpenAIFileSearchConfig] = None
    custom: Optional[CustomVectorStoreConfig] = None

    @model_validator(mode="after")
    def _validate_backend(self) -> "VectorStoreConfig":
        if self.backend == "openai_file_search":
            if not self.openai_file_search:
                raise ValueError(
                    "vector_store.openai_file_search is required when backend == openai_file_search"
                )
        elif self.backend == "custom":
            if not self.custom:
                raise ValueError(
                    "vector_store.custom is required when backend == custom"
                )
            if self.custom.kind == "qdrant" and not self.custom.qdrant:
                raise ValueError(
                    "vector_store.custom.qdrant is required when kind == qdrant"
                )
            if self.custom.kind == "plugin" and not self.custom.plugin:
                raise ValueError(
                    "vector_store.custom.plugin is required when kind == plugin"
                )
        return self


class RerankConfig(BaseModel):
    enabled: bool = True
    prompt_path: str = "rerank/cross_encoder.md"
    model: str = "gpt-5-nano"
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal"
    max_candidates: int = 24
    score_threshold: float = 0.0


class SimilarityFilterConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.0


class ExpansionConfig(BaseModel):
    enabled: bool = False
    prompt_path: str = "expansion/query_expansion.md"
    model: str = "gpt-5-mini"
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal"
    variants: int = 3
    style: List[Literal["keywords", "paraphrase", "antonyms"]] = Field(
        default_factory=lambda: ["keywords", "paraphrase"]
    )


class HydeConfig(BaseModel):
    enabled: bool = False
    prompt_path: str = "expansion/hyde.md"
    model: str = "gpt-5-mini"
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal"


class CitationsConfig(BaseModel):
    include_spans: bool = True
    max_per_source: int = 3


class RankingOptions(BaseModel):
    ranker: Literal["auto", "default-2024-08-21"] | None = None
    score_threshold: float | None = None


class QueryConfig(BaseModel):
    # Generic retrieval options (used for custom backends)
    top_k: int = 8
    similarity_filter: SimilarityFilterConfig = Field(
        default_factory=SimilarityFilterConfig
    )
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    expansion: ExpansionConfig = Field(default_factory=ExpansionConfig)
    hyde: HydeConfig = Field(default_factory=HydeConfig)
    filters: Dict[str, Any] = Field(default_factory=dict)
    max_context_tokens: int = 4000
    citations: CitationsConfig = Field(default_factory=CitationsConfig)


def _default_system_prompt() -> str:
    try:
        return load_prompt("system/assistant.md")
    except Exception:
        return (
            "You are a grounded enterprise assistant. Answer strictly from the provided context.\n"
            "If insufficient, ask a clarifying question. Always cite sources with file and page."
        )


class SynthesisConfig(BaseModel):
    model: str = "gpt-5"
    system_prompt: str = Field(default_factory=_default_system_prompt)
    structured_outputs: bool = True
    reasoning_effort: Literal["low", "medium", "high"] = "low"


class AppConfig(BaseModel):
    ui: Literal["streamlit", "nextjs"] = "streamlit"
    auth: Literal["none", "basic"] = "none"


class JudgeConfig(BaseModel):
    model: str = "gpt-5-mini"
    style: Literal["single", "pairwise"] = "single"
    rubric: Literal["qa_grounded"] = "qa_grounded"
    structured_outputs: bool = True


class ThresholdsConfig(BaseModel):
    pass_rate: float = 0.75
    grounding_min: float = 0.8


class OpenAIEvalsConfig(BaseModel):
    enabled: bool = True
    name_prefix: str = "RAG QA"
    graders: List[Dict[str, Any]] = Field(default_factory=default_openai_evals_graders)


class RetrievalMetricsConfig(BaseModel):
    top_k: int = 8
    compute: List[
        Literal["hit@k", "recall@k", "MRR", "context_precision", "support_coverage"]
    ] = Field(
        default_factory=lambda: [
            "hit@k",
            "recall@k",
            "MRR",
            "context_precision",
            "support_coverage",
        ]
    )


class AutoEvalsConfig(BaseModel):
    items_target: int = 150
    mix: Dict[Literal["easy", "medium", "hard"], float] = Field(
        default_factory=lambda: {"easy": 0.5, "medium": 0.35, "hard": 0.15}
    )
    per_tag_cap: int = 30
    per_source_cap: Optional[int] = None
    exclude_patterns: List[str] = Field(
        default_factory=lambda: ["confidential", "legal_hold"]
    )
    # Auto-mode controls
    scan_docs: Optional[int] = (
        None  # number of docs/chunks to scan; default derives from items_target
    )
    qgen_workers: int = 8  # parallel threads for question generation
    quality_regen_limit: int = 100  # cap on LLM self-consistency regen checks
    # Context/question controls
    questions_per_context: int = 3
    context_chars_min: int = 1200
    context_chars_max: int = 2400
    per_doc_context_cap: int = 10


class EvalsConfig(BaseModel):
    mode: Literal["user", "auto"] = "user"
    dataset_path: Optional[str] = None
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    openai_evals: OpenAIEvalsConfig = Field(default_factory=OpenAIEvalsConfig)
    retrieval_metrics: RetrievalMetricsConfig = Field(
        default_factory=RetrievalMetricsConfig
    )
    auto: AutoEvalsConfig = Field(default_factory=AutoEvalsConfig)
    # Back-compat fields (deprecated):
    judge_model: Optional[str] = None
    metrics: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "retrieval": [
                "hit@k",
                "recall@k",
                "MRR",
                "context_precision",
                "support_coverage",
            ],
            "qa": ["exact_match", "f1", "llm_accuracy", "grounding_score"],
            "latency": [
                "ingest_ms",
                "retrieve_ms",
                "rerank_ms",
                "synthesize_ms",
                "e2e_ms",
            ],
        }
    )
    ablations: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coerce_legacy(self) -> "EvalsConfig":
        # If legacy judge_model is provided, map to judge.model
        if self.judge_model and (
            not self.judge or self.judge.model == JudgeConfig().model
        ):
            self.judge.model = self.judge_model
        return self


class RootConfig(BaseModel):
    project: str
    env: EnvConfig = Field(default_factory=EnvConfig)
    data: DataConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    custom_chunker: Optional[CustomChunkerConfig] = None
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    evals: EvalsConfig = Field(default_factory=EvalsConfig)

    @model_validator(mode="after")
    def _validate_dependencies(self) -> "RootConfig":
        if self.chunking.strategy == "custom" and not self.custom_chunker:
            raise ValueError(
                "custom_chunker must be provided when chunking.strategy == custom"
            )
        return self


def load_config(path: str) -> RootConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_prompt_references(cfg)
    try:
        return RootConfig(**cfg)
    except ValidationError as e:
        raise SystemExit(f"Invalid config: {e}")


def lint_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    resolved = _resolve_prompt_references(raw)
    normalized = RootConfig(**resolved).model_dump()
    # Warn about unused keys by shallow diff
    unused = set(raw.keys()) - set(normalized.keys())
    return {"normalized": normalized, "unused_top_level_keys": sorted(list(unused))}
