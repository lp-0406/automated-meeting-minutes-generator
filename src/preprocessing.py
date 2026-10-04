"""Transcript preprocessing (Phase 3).

Design rules (see README / methodology):
- Light cleaning only: whitespace, filler words, repeated-word collapse.
- Cue words ("decided", "agreed", "by Friday", "will", "need to", ...) are NEVER removed.
- Speaker labels and timestamps are preserved whenever present.
- Sentence segmentation uses spaCy when the English model is installed, with a
  deterministic regex fallback so tests and cold environments still work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- structures
@dataclass
class Turn:
    """One speaker turn. start/end are seconds (None when unknown)."""
    speaker: str = "Unknown"
    text: str = ""
    start: float | None = None
    end: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def span(self) -> str:
        if self.start is None:
            return ""
        return f"[{self.start:.1f}-{self.end if self.end is not None else '?'}]"


@dataclass
class Sentence:
    """One sentence with provenance, used by classifier + extraction."""
    text: str
    speaker: str = "Unknown"
    meeting_id: str = ""
    start: float | None = None
    end: float | None = None
    turn_index: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- cleaning
# Standalone fillers only; multi-word phrases containing them are untouched.
FILLERS = {"um", "uh", "er", "erm", "hmm", "mmm", "hm", "uhm", "ah", "eh", "mm"}

# Words that must never be dropped by any cleaning step.
CUE_WORDS = {
    "decided", "decision", "agreed", "agree", "approved", "approve", "will", "should",
    "must", "need", "needs", "following", "follow", "action", "deadline", "due",
    "friday", "monday", "tuesday", "wednesday", "thursday", "week", "month", "today",
    "tomorrow", "asap", "let's", "confirm", "review", "send", "schedule", "assign",
}

_WS = re.compile(r"[ \t]+")
_REPEAT_WORD = re.compile(r"\b(\w+)(?:\s+\1\b){1,}", re.IGNORECASE)
_FILLER_TOKEN = re.compile(r"\b(?:%s)\b" % "|".join(sorted(FILLERS)), re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def remove_fillers(text: str) -> str:
    """Remove standalone fillers only; never touches cue phrases like 'we need to'."""
    return _FILLER_TOKEN.sub("", text)


def collapse_repeated_words(text: str) -> str:
    """'very very very good' -> 'very good' (one pass, then repeat to convergence)."""
    prev = None
    while prev != text:
        prev = text
        text = _REPEAT_WORD.sub(r"\1", text)
    return text


def clean_text(text: str, *, drop_fillers: bool = True) -> str:
    text = normalize_whitespace(text)
    if drop_fillers:
        text = remove_fillers(text)
    text = collapse_repeated_words(text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)   # no space before punctuation
    text = re.sub(r"([,.;:?!])(\S)", r"\1 \2", text)
    return normalize_whitespace(text)


# --------------------------------------------------------------------------- parsing
_SPEAKER_LINE = re.compile(
    r"^\s*(?:\[([\d:.]+)\s*[-–]\s*([\d:.]+)\]\s*)?"      # optional [mm:ss - mm:ss]
    r"([A-Za-z][\w .'\-()]{0,60}?|Speaker\s*\d+|Unknown)\s*:\s+(.*)$"
)


def _to_seconds(t: str) -> float | None:
    """'12:03' -> 723.0 ; '01:02:03' -> 3723.0 ; '12.5' -> 12.5"""
    try:
        parts = [float(p) for p in t.split(":")]
    except ValueError:
        return None
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def parse_transcript_text(text: str, meeting_id: str = "") -> list[Turn]:
    """Parse 'Speaker: utterance' lines (optionally timestamped) into Turns.

    Lines without a speaker prefix continue the previous turn; unknown speakers
    become 'Unknown'. This is the format used by our samples and Whisper output.
    """
    turns: list[Turn] = []
    for raw in normalize_whitespace(text).split("\n"):
        if not raw.strip():
            continue
        m = _SPEAKER_LINE.match(raw)
        if m:
            start, end, speaker, content = m.groups()
            turns.append(Turn(
                speaker=speaker.strip() or "Unknown",
                text=content.strip(),
                start=_to_seconds(start) if start else None,
                end=_to_seconds(end) if end else None,
            ))
        elif turns:
            turns[-1].text = f"{turns[-1].text} {raw.strip()}".strip()
        else:
            turns.append(Turn(speaker="Unknown", text=raw.strip()))
    return turns


# --------------------------------------------------------------------------- sentences
_SPACY_NLP = None
_SPACY_TRIED = False


def _get_spacy():
    global _SPACY_NLP, _SPACY_TRIED
    if not _SPACY_TRIED:
        _SPACY_TRIED = True
        try:
            import spacy
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
        except Exception:
            _SPACY_NLP = None
    return _SPACY_NLP


def sentencize(text: str) -> list[str]:
    """Sentence split. spaCy when available, deterministic regex fallback otherwise."""
    text = normalize_whitespace(text)
    if not text:
        return []
    nlp = _get_spacy()
    if nlp is not None:
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    # Regex fallback: split after . ? ! followed by capital/quote or EOS.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", text)
    return [p.strip() for p in parts if p.strip()]


def turn_sentences(turns: Sequence[Turn], meeting_id: str = "") -> list[Sentence]:
    """Segment turns into Sentence objects keeping speaker/timestamp provenance."""
    out: list[Sentence] = []
    for i, turn in enumerate(turns):
        for s in sentencize(turn.text):
            out.append(Sentence(text=clean_text(s), speaker=turn.speaker,
                                meeting_id=meeting_id, start=turn.start,
                                end=turn.end, turn_index=i))
    return [s for s in out if s.text]


def merge_short_turns(turns: Sequence[Turn], min_words: int = 3,
                      same_speaker_only: bool = True) -> list[Turn]:
    """Merge very short turns into their neighbour to reduce fragmentation."""
    merged: list[Turn] = []
    for turn in turns:
        if (merged and len(turn.text.split()) < min_words
                and (not same_speaker_only or merged[-1].speaker == turn.speaker)):
            prev = merged[-1]
            prev.text = f"{prev.text} {turn.text}".strip()
            if turn.end is not None:
                prev.end = turn.end
        else:
            merged.append(Turn(**turn.to_dict()))
    return merged


def turns_to_text(turns: Iterable[Turn], *, with_speaker: bool = True,
                  with_timestamps: bool = False) -> str:
    """Render turns back to a plain transcript (used for chunk text + demo output)."""
    lines = []
    for t in turns:
        prefix = ""
        if with_timestamps and t.start is not None:
            prefix += f"[{t.start:.1f}-{t.end if t.end is not None else '?'}] "
        if with_speaker:
            prefix += f"{t.speaker}: "
        lines.append(prefix + t.text)
    return "\n".join(lines)
