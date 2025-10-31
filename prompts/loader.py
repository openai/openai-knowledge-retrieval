from __future__ import annotations

import os
from functools import lru_cache


_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_PROMPTS_DIR = os.path.join(_ROOT, "prompts")


@lru_cache(maxsize=128)
def load_prompt(rel_path: str) -> str:
    """Load a prompt file from the prompts directory.

    rel_path: path relative to the prompts/ directory, e.g., "system/assistant.md".
    Returns the file content as a UTF-8 string. Raises FileNotFoundError if missing.
    """
    path = os.path.join(_PROMPTS_DIR, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
