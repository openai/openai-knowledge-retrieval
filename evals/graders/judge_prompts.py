from prompts.loader import load_prompt


def _fallback_single() -> str:
    return (
        "You are a strict evaluation judge.\n"
        "Decide if model_answer semantically matches correct_answer and is fully supported by citation_text.\n"
        "Penalize any unsupported claims or hallucinations. Ignore minor formatting or phrasing differences.\n"
        "Return a compact JSON object with keys: decision (pass|fail), correctness (0..1), grounding (0..1), rationale."
    )


def _fallback_pairwise() -> str:
    return (
        "You are a strict evaluation judge comparing two candidate answers (A and B).\n"
        "Assess correctness and grounding against correct_answer and citation_text; prefer concise grounded answers.\n"
        "Return JSON with winner (A|B|tie), correctness_a, correctness_b, grounding_a, grounding_b, rationale."
    )


try:
    SINGLE_RUBRIC_QA_GROUNDED = load_prompt("evals/judge_single.md")
except Exception:
    SINGLE_RUBRIC_QA_GROUNDED = _fallback_single()

try:
    PAIRWISE_RUBRIC_QA_GROUNDED = load_prompt("evals/judge_pairwise.md")
except Exception:
    PAIRWISE_RUBRIC_QA_GROUNDED = _fallback_pairwise()
