from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Dict
from datetime import datetime, timezone


def write_html_report(summary: Dict[str, Any], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    head = "<html><head><meta charset='utf-8'><title>RAG Evals</title></head><body>"
    tail = "</body></html>"
    body = ["<h1>RAG Evals Summary</h1>"]
    ts = datetime.now(timezone.utc).isoformat()
    body.append(f"<div><em>Generated: {escape(ts)}</em></div>")
    body.append("<ul>")
    for k in ["project", "config_path", "num_items", "pass_rate", "report_url"]:
        if k in summary:
            body.append(f"<li><b>{escape(str(k))}</b>: {escape(str(summary[k]))}</li>")
    body.append("</ul>")
    if "rows" in summary:
        body.append("<h2>Rows</h2><ol>")
        for r in summary["rows"][:100]:
            body.append("<li>")
            body.append(f"<div><b>Q:</b> {escape(str(r.get('question','')[:200]))}</div>")
            body.append(f"<div><b>A:</b> {escape(str(r.get('answer','')[:200]))}</div>")
            j = r.get("judge", {})
            body.append(
                f"<div><b>Judge:</b> {escape(str(j.get('decision','?')))} "
                f"(corr={escape(str(j.get('correctness','?')))} ground={escape(str(j.get('grounding','?')))})</div>"
            )
            graders = j.get("graders") or []
            if graders:
                body.append("<ul>")
                for g in graders:
                    score = g.get("score")
                    raw = g.get("raw_text") or ""
                    verdict = "pass" if g.get("passed") else "fail"
                    threshold = g.get("threshold")
                    threshold_txt = f" (threshold {threshold})" if threshold is not None else ""
                    score_txt = f"{score:.2f}" if isinstance(score, (int, float)) else str(score or "—")
                    body.append(
                        "<li>"
                        f"<b>{escape(str(g.get('name','grader')))}:</b> {escape(score_txt)} "
                        f"{escape(verdict)}{escape(threshold_txt)}"
                        + (f"<details><summary>raw</summary><pre>{escape(raw[:2000])}</pre></details>" if raw else "")
                        + "</li>"
                    )
                body.append("</ul>")
            body.append("</li>")
        body.append("</ol>")
    html = head + "\n" + "\n".join(body) + "\n" + tail
    p.write_text(html, encoding="utf-8")