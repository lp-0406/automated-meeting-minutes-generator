"""Whisper ASR (Phase 4).

- Lazy, process-wide cached model loading (one load per model size).
- Supports wav / mp3 / m4a / flac / ogg via ffmpeg (uses the imageio-ffmpeg binary
  when no system ffmpeg is installed — common on Colab-lite and Windows).
- Emits timestamped segments, saves JSON + plain-text transcripts.
- CPU fallback is automatic (Whisper runs on CPU when CUDA is unavailable).
- Optional WER against a reference transcript via jiwer.

Speaker diarization is intentionally OUT of scope: speaker = "Unknown" for raw audio.
Diarization is documented as future work (see README / methodology).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .utils import save_json

SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}

_ASR_CACHE: dict = {}


def _ensure_ffmpeg_on_path() -> str | None:
    """Return an ffmpeg executable path, preferring the system one and falling back
    to the static binary shipped with imageio-ffmpeg. Returns None if none found."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        bindir = str(Path(exe).parent)
        if bindir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        return exe
    except Exception:
        return None


def resolve_device(requested: str = "auto") -> str:
    """Local re-export so asr users don't need utils import order."""
    from .utils import resolve_device as _rd
    return _rd(requested)


class WhisperASR:
    """Thin wrapper around openai-whisper with lazy loading and caching."""

    def __init__(self, model_size: str = "small", device: str = "auto",
                 language: str = "en", condition_on_previous_text: bool = False):
        self.model_size = model_size
        self.device = resolve_device(device)
        self.language = language
        self.condition_on_previous_text = condition_on_previous_text
        self._model = None

    # ------------------------------------------------------------------ loading
    @property
    def model(self):
        if self._model is None:
            key = (self.model_size, self.device)
            if key not in _ASR_CACHE:
                import whisper
                _ensure_ffmpeg_on_path()
                _ASR_CACHE[key] = whisper.load_model(self.model_size, device=self.device)
            self._model = _ASR_CACHE[key]
        return self._model

    # ------------------------------------------------------------------ inference
    def transcribe(self, audio_path: str | Path) -> dict:
        """Transcribe an audio file.

        Returns dict: {meeting_id, audio_path, language, text, segments:[{start,end,text}]}
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO:
            raise ValueError(f"Unsupported audio format: {audio_path.suffix} "
                             f"(supported: {sorted(SUPPORTED_AUDIO)})")
        _ensure_ffmpeg_on_path()
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            condition_on_previous_text=self.condition_on_previous_text,
            verbose=False,
        )
        segments = [{
            "id": i,
            "start": round(float(s["start"]), 3),
            "end": round(float(s["end"]), 3),
            "text": s["text"].strip(),
        } for i, s in enumerate(result.get("segments", []))]
        return {
            "meeting_id": audio_path.stem,
            "audio_path": str(audio_path),
            "language": result.get("language", self.language),
            "model": self.model_size,
            "device": self.device,
            "text": result.get("text", "").strip(),
            "segments": segments,
        }

    def save_transcript(self, result: dict, out_dir: str | Path) -> tuple[Path, Path]:
        """Save outputs/transcripts/<id>.json and <id>.txt. Returns (json, txt)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mid = result.get("meeting_id", "transcript")
        jpath = out_dir / f"{mid}.json"
        tpath = out_dir / f"{mid}.txt"
        save_json(result, jpath)
        # Plain text uses one timestamped line per segment (parseable by preprocessing).
        lines = [f"[{s['start']:.1f}-{s['end']:.1f}] Unknown: {s['text']}"
                 for s in result.get("segments", [])]
        tpath.write_text("\n".join(lines) or result.get("text", ""), encoding="utf-8")
        return jpath, tpath


def transcribe_file(audio_path: str | Path, model_size: str = "small",
                    device: str = "auto", language: str = "en") -> dict:
    """Convenience functional API (model instance is cached across calls)."""
    asr = WhisperASR(model_size=model_size, device=device, language=language)
    return asr.transcribe(audio_path)


def transcript_to_turns(result: dict):
    """Convert a saved ASR result into preprocessing Turns (speaker = Unknown)."""
    from .preprocessing import Turn
    return [Turn(speaker="Unknown", text=s["text"].strip(),
                 start=s["start"], end=s["end"])
            for s in result.get("segments", []) if s.get("text", "").strip()]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER via jiwer. Returns 0.0 for an empty reference only when hyp is empty too."""
    try:
        import jiwer
    except ImportError as e:
        raise ImportError("jiwer is required for WER: pip install jiwer") from e
    ref, hyp = reference.strip(), hypothesis.strip()
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    return float(jiwer.wer(ref, hyp))
