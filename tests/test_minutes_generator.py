import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.minutes_generator import generate_minutes, render_minutes

MEETING = {
    "meeting_id": "m1",
    "title": "Weekly Sync",
    "date": "2026-10-01",
    "participants": ["Alice", "Bob"],
    "turns": [],
    "reference_summary": "",
    "source": "user_input",
}
SUMMARY = {"final_summary": "The team discussed the launch plan.",
           "key_points": ["Launch date confirmed", "Budget reviewed"]}
DECISIONS = [{"text": "We decided to launch on 15 October.", "speaker": "Alice",
              "timestamp": "10.0s-12.0s", "confidence": 0.87,
              "signals": ["rule:decision"]}]
ACTIONS = [{"action": "I will send the deck by Friday.", "owner": "Bob",
            "deadline": "by Friday", "speaker": "Bob", "timestamp": "20.0s-22.0s",
            "confidence": 0.91, "signals": ["rule:action"]}]
FOLLOWUPS = [{"text": "Carry over hiring discussion.", "speaker": "Alice",
              "timestamp": "Not specified", "confidence": 0.5,
              "signals": ["rule:followup"]}]


def test_render_contains_all_sections():
    md = render_minutes(MEETING, SUMMARY, DECISIONS, ACTIONS, FOLLOWUPS)
    for heading in ("# Meeting Minutes", "## Meeting Information", "## Executive Summary",
                    "## Key Discussion Points", "## Decisions Made", "## Action Items",
                    "## Important Follow-ups", "## Source Verification"):
        assert heading in md
    assert "**Meeting Title:** Weekly Sync" in md
    assert "**Date:** 2026-10-01" in md
    assert "**Participants:** Alice, Bob" in md
    assert "DRAFT" in md


def test_action_table_row():
    md = render_minutes(MEETING, SUMMARY, DECISIONS, ACTIONS)
    assert "| Action | Responsible Person | Deadline | Confidence |" in md
    assert "| I will send the deck by Friday. | Bob | by Friday | 0.91 |" in md


def test_missing_fields_render_not_specified():
    bare = {"meeting_id": "x", "title": "", "date": "", "participants": [],
            "turns": []}
    md = render_minutes(bare, {"final_summary": "", "key_points": []})
    assert "**Meeting Title:** Not specified" in md
    assert "**Date:** Not specified" in md
    assert "**Participants:** Not specified" in md
    assert "\nNot specified\n" in md                 # empty executive summary
    assert "| Not specified | Not specified | Not specified | Not specified |" in md


def test_pipe_escaped_in_table():
    md = render_minutes(MEETING, SUMMARY, [],
                        [{"action": "A | B task", "owner": "Not specified",
                          "deadline": "Not specified", "confidence": 0.5}])
    assert "A \\| B task" in md


def test_generate_minutes_writes_md_and_json(tmp_path):
    out = tmp_path / "demo.md"
    md = generate_minutes(MEETING, SUMMARY, DECISIONS, ACTIONS, FOLLOWUPS, out)
    assert out.exists() and out.read_text(encoding="utf-8") == md
    side = out.with_suffix(".json")
    assert side.exists()
    import json
    data = json.loads(side.read_text(encoding="utf-8"))
    assert data["review_status"].startswith("DRAFT")
    assert data["decisions"][0]["text"] == DECISIONS[0]["text"]


def test_source_verification_quotes_sentences():
    md = render_minutes(MEETING, SUMMARY, DECISIONS, ACTIONS)
    assert "> We decided to launch on 15 October." in md
    assert "speaker: Alice" in md
    assert "timestamp: 10.0s-12.0s" in md
