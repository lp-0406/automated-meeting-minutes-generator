import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.asr import (WhisperASR, _ensure_ffmpeg_on_path, transcript_to_turns,
                     word_error_rate)


def test_ffmpeg_resolution():
    exe = _ensure_ffmpeg_on_path()
    assert exe is not None and Path(exe).exists()      # system or imageio-ffmpeg binary


def test_wer_values():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("a b c", "a b x") == pytest.approx(1 / 3, abs=1e-6)
    assert word_error_rate("a b", "") == 1.0


def test_unsupported_format_raises(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hi", encoding="utf-8")
    asr = WhisperASR(model_size="tiny")
    with pytest.raises(ValueError):
        asr.transcribe(bad)                              # fails before model load


def test_missing_file_raises(tmp_path):
    asr = WhisperASR(model_size="tiny")
    with pytest.raises(FileNotFoundError):
        asr.transcribe(tmp_path / "nope.wav")


def test_transcript_to_turns():
    result = {"segments": [
        {"start": 0.0, "end": 2.0, "text": " Hello "},
        {"start": 2.0, "end": 3.0, "text": ""},
        {"start": 3.0, "end": 5.0, "text": "world"},
    ]}
    turns = transcript_to_turns(result)
    assert len(turns) == 2
    assert turns[0].speaker == "Unknown" and turns[0].text == "Hello"
    assert turns[1].start == 3.0


def test_save_transcript(tmp_path):
    asr = WhisperASR(model_size="tiny")
    result = {"meeting_id": "demo", "text": "Hello world",
              "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world"}]}
    jp, tp = asr.save_transcript(result, tmp_path)
    assert jp.exists() and tp.exists()
    assert "Unknown: Hello world" in tp.read_text(encoding="utf-8")
    import json
    assert json.loads(jp.read_text(encoding="utf-8"))["meeting_id"] == "demo"
