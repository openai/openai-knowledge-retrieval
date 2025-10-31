import os
from typing import Optional, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from cli.config import RootConfig


def _ensure_dotenv_precedence() -> None:
    """Load .env with precedence over exported env vars.

    We want values defined in a .env file to override already-exported environment
    variables for OPENAI_API_KEY, OPENAI_ORG_ID, and OPENAI_PROJECT_ID. Using
    load_dotenv with override=True achieves this behavior. If no .env is present,
    this is a no-op.
    """
    try:
        # Locate nearest .env starting from cwd; if not found, load_dotenv will no-op
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=True)
        else:
            load_dotenv(override=True)
    except Exception:
        # Best-effort loading; ignore if dotenv is unavailable or any error occurs
        pass


def resolve_env_placeholder(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return (
            None
            if value
            in (
                None,
                "",
            )
            else value
        )
    if value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.getenv(env_name)
    return value


def build_openai_client(cfg: RootConfig) -> OpenAI:
    # Ensure .env values take precedence over exported environment variables
    _ensure_dotenv_precedence()

    api_key = resolve_env_placeholder(cfg.env.openai_api_key) or os.getenv(
        "OPENAI_API_KEY"
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI usage")
    organization = resolve_env_placeholder(
        getattr(cfg.env, "openai_organization", None)
    ) or os.getenv("OPENAI_ORG_ID")
    project = resolve_env_placeholder(
        getattr(cfg.env, "openai_project", None)
    ) or os.getenv("OPENAI_PROJECT_ID")

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if organization:
        kwargs["organization"] = organization
    if project:
        kwargs["project"] = project
    return OpenAI(**kwargs)
