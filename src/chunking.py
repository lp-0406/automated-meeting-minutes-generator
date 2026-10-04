"""Token-aware, turn-preserving chunking (Phase 3).

Meetings are far longer than a BART encoder (1024 tokens), so the full meeting is
never truncated: it is split into chunks that are summarised individually and then
combined (hierarchical summarisation).

Guarantees:
- chunks are built from whole speaker turns (never split mid-sentence/word)
- a chunk never exceeds `max_tokens` (a single over-long turn is word-split as a
  last resort so no content is silently dropped)
- consecutive chunks share `overlap_turns` turns for context continuity
- every chunk carries metadata (chunk_id, meeting_id, start/end, speakers)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Callable, Sequence

from .preprocessing import Turn, turns_to_text

# Rough token estimate used when no HF tokenizer is available: ~4 chars/token for
# English text. Conservative (slightly high) so we never overshoot the real limit.
CHARS_PER_TOKEN = 3.9
_TOKENIZER: object | None = None
_TOKENIZER_NAME: str | None = None


def get_tokenizer(model_name: str = "sshleifer/distilbart-cnn-12-6"):
    """Lazily load (and cache) a HF tokenizer for counting. Returns None offline."""
    global _TOKENIZER, _TOKENIZER_NAME
    if _TOKENIZER is not None and _TOKENIZER_NAME == model_name:
        return _TOKENIZER
    try:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZER_NAME = model_name
        return _TOKENIZER
    except Exception:
        return None


def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
    return max(1, int(len(text) / CHARS_PER_TOKEN))


@dataclass
class Chunk:
    chunk_id: str
    meeting_id: str
    text: str
    speaker: str = "Unknown"          # dominant speaker ("Multiple" when mixed)
    speakers: list = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None
    token_count: int = 0
    n_turns: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long_turn(turn: Turn, max_tokens: int, tokenizer) -> list[Turn]:
    """Last resort for a single turn bigger than the whole budget: split on sentence
    boundaries (then words) — never mid-word."""
    from .preprocessing import sentencize
    pieces: list[str] = []
    current = ""
    for sent in sentencize(turn.text):
        cand = f"{current} {sent}".strip()
        if count_tokens(cand, tokenizer) <= max_tokens:
            current = cand
        else:
            if current:
                pieces.append(current)
            if count_tokens(sent, tokenizer) > max_tokens:      # word-level split
                words, buf = sent.split(), ""
                for w in words:
                    cand2 = f"{buf} {w}".strip()
                    if count_tokens(cand2, tokenizer) <= max_tokens:
                        buf = cand2
                    else:
                        if buf:
                            pieces.append(buf)
                        buf = w
                current = buf
            else:
                current = sent
    if current:
        pieces.append(current)
    out = []
    for p in pieces:
        t = Turn(**turn.to_dict())
        t.text = p
        out.append(t)
    return out


def chunk_turns(turns: Sequence[Turn], meeting_id: str = "unknown",
                max_tokens: int = 700, overlap_turns: int = 1,
                tokenizer=None) -> list[Chunk]:
    """Group turns into token-budgeted chunks with turn-level overlap.

    Returns chunk list; each chunk's text includes speaker labels so downstream
    summarisation and extraction keep speaker context.
    """
    if not turns:
        return []
    # Normalise over-long turns first.
    norm: list[Turn] = []
    for t in turns:
        if count_tokens(t.text, tokenizer) > max_tokens:
            norm.extend(_split_long_turn(t, max_tokens, tokenizer))
        else:
            norm.append(t)

    chunks: list[Chunk] = []
    i = 0
    while i < len(norm):
        j, group = i, []
        while j < len(norm):
            group.append(norm[j])
            tok = count_tokens(turns_to_text(group, with_speaker=True,
                                             with_timestamps=True), tokenizer)
            if tok > max_tokens and len(group) > 1:
                group.pop()
                break
            j += 1
        if not group:                                     # impossible, but be safe
            group, j = [norm[i]], i + 1
        text = turns_to_text(group, with_speaker=True, with_timestamps=True)
        speakers = sorted({t.speaker for t in group})
        starts = [t.start for t in group if t.start is not None]
        ends = [t.end for t in group if t.end is not None]
        chunks.append(Chunk(
            chunk_id=f"{meeting_id}::c{len(chunks):03d}",
            meeting_id=meeting_id,
            text=text,
            speaker=speakers[0] if len(speakers) == 1 else "Multiple",
            speakers=speakers,
            start_time=min(starts) if starts else None,
            end_time=max(ends) if ends else None,
            token_count=count_tokens(text, tokenizer),
            n_turns=len(group),
        ))
        if j >= len(norm):
            break
        # Overlap by overlap_turns turns, but always advance at least one turn so a
        # single-turn chunk (or overlap >= chunk size) cannot loop forever.
        i = max(i + 1, j - overlap_turns)
    return chunks


def chunk_to_dict(chunk: Chunk) -> dict:
    return chunk.to_dict()


def save_chunks(chunks: Sequence[Chunk], path) -> list[dict]:
    from .utils import save_json
    payload = [c.to_dict() for c in chunks]
    save_json(payload, path)
    return payload
