import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_turns, count_tokens
from src.preprocessing import Turn


def _turns(n=40):
    return [Turn(speaker=f"S{i % 3}",
                 text=f"This is sentence number {i} about roadmap and deadlines.",
                 start=float(i) * 5.0, end=float(i) * 5.0 + 4.0)
            for i in range(n)]


def test_empty_input_returns_no_chunks():
    assert chunk_turns([], "m1") == []


def test_budget_respected_with_whitespace_tokenizer():
    chunks = chunk_turns(_turns(60), "m1", max_tokens=100, overlap_turns=1)
    assert len(chunks) > 1
    assert max(c.token_count for c in chunks) <= 100


def test_full_coverage_no_content_lost():
    chunks = chunk_turns(_turns(60), "m1", max_tokens=150, overlap_turns=1)
    joined = " ".join(c.text for c in chunks)
    for i in range(60):
        assert f"number {i}" in joined
    assert chunks[0].start_time == 0.0
    assert chunks[-1].end_time == 59 * 5.0 + 4.0


def test_metadata_present():
    chunks = chunk_turns(_turns(30), "meetingX", max_tokens=120, overlap_turns=0)
    for c in chunks:
        assert c.chunk_id.startswith("meetingX::c")
        assert c.meeting_id == "meetingX"
        assert c.n_turns >= 1 and c.speakers
        assert c.start_time is not None and c.end_time is not None


def test_never_splits_words_on_oversized_turn():
    chunks = chunk_turns([Turn("A", "supercalifragilistic " * 100)], "m1", max_tokens=50)
    assert len(chunks) > 1
    # every word must survive intact in at least one chunk
    joined = " ".join(c.text for c in chunks)
    assert joined.split().count("supercalifragilistic") == 100


def test_overlap_shares_boundary_turns():
    chunks = chunk_turns(_turns(40), "m1", max_tokens=100, overlap_turns=2)
    if len(chunks) >= 2:
        first_words = chunks[0].text.split()
        assert any(w in chunks[1].text for w in first_words[-6:])


def test_single_turn_progress_no_infinite_loop():
    chunks = chunk_turns(_turns(5), "m1", max_tokens=10_000, overlap_turns=3)
    assert len(chunks) == 1                     # whole meeting fits one chunk


def test_count_tokens_fallback():
    assert count_tokens("hello world", None) >= 1
