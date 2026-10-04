"""Structured Markdown meeting-minutes assembly (Phase 9).

Output contract (README / spec):
- Meeting Information (title, date, participants)
- Executive Summary
- Key Discussion Points
- Decisions Made
- Action Items table (Action | Responsible Person | Deadline | Confidence)
- Important Follow-ups
- Source Verification (speaker, timestamp, source sentence for auditability)

Any field that cannot be reliably determined is rendered as "Not specified".
The output is a DRAFT for human review — never an authoritative record.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .utils import save_json

NOT_SPECIFIED = "Not specified"


def _md_escape(text: str) -> str:
    """Escape characters that break Markdown tables."""
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def _as_list(values) -> list:
    if not values:
        return []
    if isinstance(values, str):
        return [values]
    return list(values)


def _fmt_participants(meeting: dict) -> str:
    people = _as_list(meeting.get("participants"))
    people = [p for p in people if p]
    return ", ".join(people) if people else NOT_SPECIFIED


def render_minutes(meeting: dict,
                   summary: dict | str,
                   decisions: Sequence[dict] = (),
                   actions: Sequence[dict] = (),
                   followups: Sequence[dict] = (),
                   include_source: bool = True,
                   reviewed_by: str = NOT_SPECIFIED) -> str:
    """Render the full Markdown minutes document (pure function, no I/O)."""
    if isinstance(summary, str):
        summary = {"final_summary": summary, "key_points": []}
    exec_summary = (summary.get("final_summary") or "").strip() or NOT_SPECIFIED
    key_points = [p for p in _as_list(summary.get("key_points")) if p]
    title = (meeting.get("title") or "").strip() or NOT_SPECIFIED
    date = (meeting.get("date") or "").strip() or NOT_SPECIFIED

    lines: list[str] = ["# Meeting Minutes", "", "## Meeting Information", "",
                        f"**Meeting Title:** {title}",
                        f"**Date:** {date}",
                        f"**Participants:** {_fmt_participants(meeting)}", "",
                        "## Executive Summary", "", exec_summary, "",
                        "## Key Discussion Points", ""]
    if key_points:
        lines += [f"{i}. {p}" for i, p in enumerate(key_points, 1)]
    else:
        lines.append(f"1. {NOT_SPECIFIED}")
    lines += ["", "## Decisions Made", ""]
    if decisions:
        for i, d in enumerate(decisions, 1):
            conf = d.get("confidence")
            conf_s = f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else ""
            lines.append(f"{i}. {_md_escape(d.get('text', NOT_SPECIFIED))}{conf_s}")
    else:
        lines.append(f"1. {NOT_SPECIFIED}")
    lines += ["", "## Action Items", "",
              "| Action | Responsible Person | Deadline | Confidence |",
              "|---|---|---|---|"]
    if actions:
        for a in actions:
            conf = a.get("confidence")
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else NOT_SPECIFIED
            lines.append(f"| {_md_escape(a.get('action', NOT_SPECIFIED))} "
                         f"| {_md_escape(a.get('owner') or NOT_SPECIFIED)} "
                         f"| {_md_escape(a.get('deadline') or NOT_SPECIFIED)} "
                         f"| {conf_s} |")
    else:
        lines.append(f"| {NOT_SPECIFIED} | {NOT_SPECIFIED} | {NOT_SPECIFIED} | {NOT_SPECIFIED} |")
    lines += ["", "## Important Follow-ups", ""]
    if followups:
        for i, f in enumerate(followups, 1):
            lines.append(f"{i}. {_md_escape(f.get('text', NOT_SPECIFIED))}")
    else:
        lines.append(f"1. {NOT_SPECIFIED}")
    if include_source:
        lines += ["", "## Source Verification", "",
                  "_Every decision and action item below quotes its source sentence so a "
                  "human reviewer can audit it in the original transcript._", ""]
        if decisions:
            lines.append("### Decisions — source")
            lines.append("")
            for i, d in enumerate(decisions, 1):
                lines.append(f"{i}. speaker: {_md_escape(d.get('speaker') or NOT_SPECIFIED)} · "
                             f"timestamp: {_md_escape(d.get('timestamp') or NOT_SPECIFIED)} · "
                             f"signals: {', '.join(d.get('signals') or []) or NOT_SPECIFIED}")
                lines.append(f"   > {_md_escape(d.get('text', NOT_SPECIFIED))}")
            lines.append("")
        if actions:
            lines.append("### Actions — source")
            lines.append("")
            for i, a in enumerate(actions, 1):
                lines.append(f"{i}. speaker: {_md_escape(a.get('speaker') or NOT_SPECIFIED)} · "
                             f"timestamp: {_md_escape(a.get('timestamp') or NOT_SPECIFIED)} · "
                             f"signals: {', '.join(a.get('signals') or []) or NOT_SPECIFIED}")
                lines.append(f"   > {_md_escape(a.get('action', NOT_SPECIFIED))}")
            lines.append("")
    lines += ["---", "",
              "**Review status:** DRAFT — generated automatically, requires human review.",
              f"**Reviewed by:** {reviewed_by}",
              "**Model provenance:** see `outputs/metrics/` and the config snapshot "
              "saved with this run."]
    return "\n".join(lines) + "\n"


def generate_minutes(meeting: dict,
                     summary: dict | str,
                     decisions: Sequence[dict] = (),
                     actions: Sequence[dict] = (),
                     followups: Sequence[dict] = (),
                     out_path: str | Path | None = None,
                     include_source: bool = True) -> str:
    """Render minutes and optionally save to .md (plus a .json audit sidecar)."""
    md = render_minutes(meeting, summary, decisions, actions, followups,
                        include_source=include_source)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        save_json({
            "meeting": {k: meeting.get(k) for k in
                        ("meeting_id", "title", "date", "participants", "source")},
            "summary": summary.get("final_summary") if isinstance(summary, dict) else summary,
            "key_points": summary.get("key_points", []) if isinstance(summary, dict) else [],
            "decisions": list(decisions),
            "actions": list(actions),
            "followups": list(followups),
            "review_status": "DRAFT — requires human review",
        }, out_path.with_suffix(".json"))
    return md
