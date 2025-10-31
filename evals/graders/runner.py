from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel

from cli.config import RootConfig
from cli.env_utils import build_openai_client


_TEMPLATE_PATTERN = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


def _lookup_path(scope: Any, path: str) -> Any:
  """Resolve dotted paths like `item.question` against nested dict-like scopes."""
  value: Any = scope
  for part in path.split("."):
    if isinstance(value, dict):
      value = value.get(part)
    else:
      value = getattr(value, part, None)
    if value is None:
      return ""
  return value


def _render_template(value: Any, context: Dict[str, Any]) -> Any:
  """Recursively interpolate {{ paths }} placeholders inside strings."""
  if isinstance(value, str):
    def _replace(match: re.Match[str]) -> str:
      expr = match.group(1).strip()
      return str(_lookup_path(context, expr) or "")

    return _TEMPLATE_PATTERN.sub(_replace, value)
  if isinstance(value, dict):
    return {k: _render_template(v, context) for k, v in value.items()}
  if isinstance(value, list):
    return [_render_template(v, context) for v in value]
  return value


class GraderStructuredOutput(BaseModel):
  """Schema enforced via OpenAI structured outputs for grader responses."""

  result: float
  reasoning: Optional[str] = None


@dataclass
class GraderResult:
  name: str
  score: Optional[float]
  passed: bool
  threshold: Optional[float] = None
  score_range: Optional[Iterable[float]] = None
  raw_text: str | None = None
  error: str | None = None

  def to_dict(self) -> Dict[str, Any]:
    return {
      "name": self.name,
      "score": self.score,
      "passed": self.passed,
      "threshold": self.threshold,
      "range": list(self.score_range) if self.score_range is not None else None,
      "raw_text": self.raw_text,
      "error": self.error,
    }


def run_config_graders(cfg: RootConfig, *, item: Dict[str, Any], answer_text: str) -> List[GraderResult]:
  """Execute graders configured under cfg.evals.openai_evals.graders."""
  graders_cfg = getattr(cfg.evals.openai_evals, "graders", None) or []
  if not graders_cfg:
    return []

  try:
    client = build_openai_client(cfg)
  except Exception as exc:  # pragma: no cover - environmental failure
    return [
      GraderResult(
        name=str(gr_cfg.get("name", "grader")),
        score=None,
        passed=False,
        threshold=gr_cfg.get("pass_threshold"),
        score_range=gr_cfg.get("range"),
        raw_text=None,
        error=f"Failed to create client: {exc}",
      )
      for gr_cfg in graders_cfg
    ]

  scope = {
    "item": item,
    "sample": {
      "output_text": answer_text,
      "answer_text": answer_text,
    },
  }

  results: List[GraderResult] = []
  for gr_cfg in graders_cfg:
    name = str(gr_cfg.get("name", "grader"))
    model = gr_cfg.get("model")
    messages = gr_cfg.get("input", [])
    rendered_messages = _render_template(messages, scope)
    try:
      response = client.responses.parse(
        model=model,
        input=rendered_messages,
        text_format=GraderStructuredOutput,
      )
      raw_text = getattr(response, "output_text", "") or ""
      score_value: Optional[float] = None
      parsed = response.output_parsed
      if parsed is not None:
        if not raw_text:
          raw_text = parsed.model_dump_json()
        try:
          score_value = float(parsed.result)
        except (TypeError, ValueError):
          score_value = None

      threshold = gr_cfg.get("pass_threshold")
      score_range = gr_cfg.get("range")
      passed = False
      if score_value is not None and threshold is not None:
        passed = score_value >= float(threshold)
      elif score_value is not None:
        passed = True

      results.append(
        GraderResult(
          name=name,
          score=score_value,
          passed=passed,
          threshold=threshold,
          score_range=score_range,
          raw_text=raw_text,
          error=None if score_value is not None else "Missing numeric result",
        )
      )
    except Exception as exc:  # pragma: no cover - API/runtime failure
      results.append(
        GraderResult(
          name=name,
          score=None,
          passed=False,
          threshold=gr_cfg.get("pass_threshold"),
          score_range=gr_cfg.get("range"),
          raw_text=None,
          error=str(exc),
        )
      )
  return results


def summarize_graders(graders: List[GraderResult]) -> str:
  if not graders:
    return "No graders configured."
  parts: List[str] = []
  for g in graders:
    if g.score is None:
      parts.append(f"{g.name}: error ({g.error or 'no score'})")
      continue
    max_range = ""
    if g.score_range:
      try:
        rng = list(g.score_range)
        if rng:
          max_range = f"/{rng[-1]}"
      except TypeError:
        max_range = ""
    verdict = "pass" if g.passed else "fail"
    threshold = f" ≥{g.threshold}" if g.threshold is not None else ""
    parts.append(f"{g.name}: {g.score:.2f}{max_range} {verdict}{threshold}")
  return "; ".join(parts)