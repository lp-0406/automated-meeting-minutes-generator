import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import Chunk
from src.data_loader import load_qmsum, load_transcript_file, qmsum_row_to_meeting
from src.preprocessing import Turn
from src.summarizer import HierarchicalSummarizer

QMSUM_ROW = {
    "topic_list": [{"topic": "Project planning", "relevant_text_span": [["0", "3"]]}],
    "general_query_list": [{"query": "Summarize the whole meeting.",
                            "answer": "The team planned the project."}],
    "specific_query_list": [],
    "meeting_transcripts": [
        {"speaker": "Alice", "content": "Let us start."},
        {"speaker": "Bob", "content": "We decided on the plan."},
        {"speaker": "Alice", "content": ""},
    ],
}


def test_qmsum_row_to_meeting_schema():
    m = qmsum_row_to_meeting(QMSUM_ROW, "val", 0)
    assert m["meeting_id"] == "qmsum_val_000"
    assert m["title"] == "Project planning"
    assert m["date"] == "Not specified"
    assert m["participants"] == ["Alice", "Bob"]
    assert len(m["turns"]) == 2                      # empty turn dropped
    assert m["reference_summary"] == "The team planned the project."
    assert m["source"] == "qmsum"


def test_load_qmsum_from_local_file(tmp_path):
    import json
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(QMSUM_ROW) for _ in range(3)), encoding="utf-8")
    ms = load_qmsum(p, "val", limit=2)
    assert len(ms) == 2 and ms[1]["meeting_id"] == "qmsum_val_001"


def test_load_transcript_txt(tmp_path):
    p = tmp_path / "meeting.txt"
    p.write_text("Alice: We decided to proceed.\nBob: I will handle it.\n",
                 encoding="utf-8")
    m = load_transcript_file(p)
    assert m["meeting_id"] == "meeting"
    assert m["participants"] == ["Alice", "Bob"]
    assert len(m["turns"]) == 2
    assert m["reference_summary"] == ""


def test_load_transcript_json_segments(tmp_path):
    import json
    p = tmp_path / "asr.json"
    p.write_text(json.dumps([{"start": 0.0, "end": 1.5, "text": "Hello team."},
                             {"start": 1.5, "end": 2.0, "text": ""}]), encoding="utf-8")
    m = load_transcript_file(p)
    assert len(m["turns"]) == 1 and m["turns"][0]["speaker"] == "Unknown"


def test_empty_chunk_skipped_not_crashing():
    s = HierarchicalSummarizer()      # no model download happens yet
    results = s.summarize_chunks([Chunk(chunk_id="m::c000", meeting_id="m", text="")])
    assert results[0]["skipped"] and results[0]["reason"] == "empty"


def test_summarize_text_empty_returns_without_model():
    s = HierarchicalSummarizer()
    assert s.summarize_text("") == ""
    assert s.summarize_text("   ") == ""


def test_key_points_from_text():
    s = HierarchicalSummarizer()
    pts = s.extract_key_points("The team decided to launch. Budget was reviewed. "
                               "Short. Hiring is pending.", [])
    assert len(pts) == 3                              # 'Short.' (<15 chars) filtered
