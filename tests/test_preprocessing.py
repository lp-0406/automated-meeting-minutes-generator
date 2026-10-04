import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (Turn, clean_text, collapse_repeated_words,
                               merge_short_turns, normalize_whitespace,
                               parse_transcript_text, sentencize,
                               turn_sentences, turns_to_text)


def test_normalize_whitespace():
    assert normalize_whitespace("  a \t b \r\n c  ") == "a b\nc"


def test_collapse_repeated_words():
    assert collapse_repeated_words("very very very good") == "very good"
    assert collapse_repeated_words("the the final") == "the final"
    assert collapse_repeated_words("no repetition here") == "no repetition here"


def test_clean_text_preserves_cue_words():
    text = "We um decided to ship by Friday and we really really need to confirm."
    out = clean_text(text)
    assert "um" not in out.split()
    for cue in ("decided", "by Friday", "need to", "confirm"):
        assert cue.lower() in out.lower()


def test_parse_transcript_with_timestamps_and_speakers():
    raw = ("[00:10 - 00:25] Alice: We decided to go with option B.\n"
           "Bob: I will send the deck.\n"
           "continuation without speaker.\n"
           "Unknown: ok")
    turns = parse_transcript_text(raw, "m1")
    assert len(turns) == 3
    assert turns[0].speaker == "Alice" and turns[0].start == 10.0 and turns[0].end == 25.0
    assert turns[1].speaker == "Bob"
    assert "continuation without speaker" in turns[1].text      # merged into prev turn
    assert turns[2].speaker == "Unknown"


def test_turn_sentences_keeps_provenance():
    turns = parse_transcript_text("Alice: First sentence. Second one here.\nBob: Hi.")
    sents = turn_sentences(turns, "m1")
    assert [s.speaker for s in sents] == ["Alice", "Alice", "Bob"]
    assert all(s.meeting_id == "m1" for s in sents)
    assert sents[0].start == turns[0].start


def test_sentencize_fallback_or_spacy():
    out = sentencize("One. Two! Three?")
    assert len(out) == 3


def test_merge_short_turns_same_speaker_only():
    turns = [Turn("A", "Hello everyone and welcome."), Turn("A", "um"),
             Turn("B", "Hi")]
    merged = merge_short_turns(turns, min_words=3)
    assert len(merged) == 2
    assert merged[0].speaker == "A" and merged[1].speaker == "B"


def test_turns_to_text_roundtrip():
    turns = [Turn("Alice", "We agreed.", 1.0, 2.0)]
    plain = turns_to_text(turns, with_speaker=True, with_timestamps=True)
    assert plain.startswith("[1.0-2.0] Alice: ")
    back = parse_transcript_text(plain)
    assert back[0].speaker == "Alice" and back[0].start == 1.0
