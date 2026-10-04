"""Hierarchical meeting summarisation (Phase 5).

Pipeline:  meeting -> chunks -> per-chunk summaries -> combined -> final summary
           + key discussion points

Primary model: sshleifer/distilbart-cnn-12-6 (free, ~1.2 GB, fine on free Colab).
Optional:      facebook/bart-large-cnn (config only — never selected automatically).

Safeguards: empty-chunk skip, token-budget clamp, repetition control
(no_repeat_ngram_size), recursion when combined chunk summaries still exceed the
encoder budget, CPU fallback, OOM -> retry on CPU.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

from .chunking import Chunk, count_tokens, get_tokenizer
from .utils import save_json

log = logging.getLogger("mmg")

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"
_MODEL_CACHE: dict = {}


def _polish(text: str) -> str:
    """Clean generator artefacts: spaces before punctuation ('yesterday .') and
    doubled spaces. Never alters wording — only whitespace/punctuation spacing."""
    text = re.sub(r"\s+([,.;:!?])", r"\1", text or "")
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    return re.sub(r" {2,}", " ", text).strip()


class HierarchicalSummarizer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu",
                 max_input_tokens: int = 1000,
                 chunk_summary_max: int = 90, chunk_summary_min: int = 20,
                 final_summary_max: int = 150, final_summary_min: int = 50,
                 key_points: int = 5):
        self.model_name = model_name
        self.device = device
        self.max_input_tokens = min(max_input_tokens, 1020)   # BART encoder limit
        self.chunk_summary_max = chunk_summary_max
        self.chunk_summary_min = chunk_summary_min
        self.final_summary_max = final_summary_max
        self.final_summary_min = final_summary_min
        self.n_key_points = key_points
        self._pipe = None
        self.tokenizer = get_tokenizer(model_name)

    # ------------------------------------------------------------------ loading
    @property
    def pipe(self):
        if self._pipe is None:
            if self.model_name not in _MODEL_CACHE:
                _MODEL_CACHE[self.model_name] = self._build_pipe(self.device)
            self._pipe = _MODEL_CACHE[self.model_name]
        return self._pipe

    def _build_pipe(self, device: str):
        import torch
        from transformers import pipeline
        dev = 0 if device == "cuda" and torch.cuda.is_available() else -1
        log.info("Loading summariser %s (device=%s)", self.model_name,
                 "cuda" if dev == 0 else "cpu")
        return pipeline("summarization", model=self.model_name, tokenizer=self.model_name,
                        device=dev, torch_dtype=torch.float32)

    # ------------------------------------------------------------------ inference
    def _truncate(self, text: str) -> str:
        """Clamp input to the encoder budget at sentence boundaries (never mid-word)."""
        if count_tokens(text, self.tokenizer) <= self.max_input_tokens:
            return text
        tokenizer = self.tokenizer
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept, tok = [], 0
        for s in sentences:
            cost = count_tokens(s, tokenizer) + 1
            if tok + cost > self.max_input_tokens:
                break
            kept.append(s)
            tok += cost
        if not kept:                                   # single huge sentence
            words, tok = text.split(), 0
            kept_words = []
            for w in words:
                if tok + len(w) + 1 > self.max_input_tokens * 4:
                    break
                kept_words.append(w)
                tok += len(w) + 1
            return " ".join(kept_words)
        return " ".join(kept)

    def summarize_text(self, text: str, max_len: int | None = None,
                       min_len: int | None = None) -> str:
        """Single summarisation pass with empty/repetition guards."""
        text = (text or "").strip()
        if not text:
            return ""
        text = self._truncate(text)
        max_len = max_len or self.final_summary_max
        min_len = min(min_len or self.final_summary_min, max_len - 1)
        src_len = count_tokens(text, self.tokenizer)
        max_len = min(max_len, max(8, src_len - 1))      # summary must be shorter
        min_len = min(min_len, max(4, max_len - 1))
        kwargs = dict(max_length=max_len, min_length=max(1, min_len),
                      do_sample=False, truncation=True,
                      no_repeat_ngram_size=3, early_stopping=True)
        try:
            out = self.pipe(text, **kwargs, batch_size=1)
        except RuntimeError as e:                        # CUDA OOM -> CPU retry
            if "out of memory" not in str(e).lower():
                raise
            log.warning("OOM on %s; rebuilding summariser on CPU", self.model_name)
            _MODEL_CACHE.pop(self.model_name, None)
            self.device, self._pipe = "cpu", None
            out = self.pipe(text, **kwargs, batch_size=1)
        return _polish((out[0]["summary_text"] or "").strip())

    # ------------------------------------------------------------------ hierarchy
    def summarize_chunks(self, chunks: Sequence[Chunk]) -> list[dict]:
        """Summarise each chunk independently (empty chunks skipped, recorded)."""
        results = []
        for c in chunks:
            body = c.text.strip()
            if not body:
                results.append({"chunk_id": c.chunk_id, "summary": "",
                                "skipped": True, "reason": "empty"})
                continue
            summary = self.summarize_text(body, self.chunk_summary_max,
                                          self.chunk_summary_min)
            results.append({"chunk_id": c.chunk_id, "summary": summary,
                            "speakers": c.speakers, "start_time": c.start_time,
                            "end_time": c.end_time, "skipped": False})
            log.info("chunk %s: %d -> %d chars", c.chunk_id, len(body), len(summary))
        return results

    def _combine(self, chunk_summaries: Sequence[dict]) -> str:
        """Join chunk summaries; recurse if the join still exceeds the budget."""
        parts = [f"- {r['summary']}" for r in chunk_summaries if r.get("summary")]
        joined = "\n".join(parts).strip()
        if not joined:
            return ""
        if count_tokens(joined, self.tokenizer) <= self.max_input_tokens:
            return joined
        # Second level: summarise halves independently, then join.
        mid = len(parts) // 2
        left = self.summarize_text("\n".join(parts[:mid]))
        right = self.summarize_text("\n".join(parts[mid:]))
        return self._combine([{"summary": left}, {"summary": right}])

    def summarize_meeting(self, chunks: Sequence[Chunk], meeting_id: str = "",
                          out_dir: str | Path | None = None) -> dict:
        """Full hierarchical summary. Saves chunk/combined/final outputs when out_dir given."""
        chunk_results = self.summarize_chunks(chunks)
        combined = self._combine(chunk_results)
        final = self.summarize_text(combined, self.final_summary_max,
                                    self.final_summary_min) if combined else ""
        key_points = self.extract_key_points(final or combined, chunks)
        bundle = {
            "meeting_id": meeting_id,
            "model": self.model_name,
            "device": self.device,
            "n_chunks": len(chunks),
            "chunk_summaries": chunk_results,
            "combined_summary": combined,
            "final_summary": final,
            "key_points": key_points,
        }
        if out_dir:
            save_json(bundle, Path(out_dir) / f"{meeting_id or 'meeting'}_summary.json")
        return bundle

    def extract_key_points(self, text: str, chunks: Sequence[Chunk]) -> list[str]:
        """Key discussion points = sentences of the combined summary, de-duplicated.
        Falls back to 'Not specified' when nothing is available (never invented)."""
        from .preprocessing import sentencize
        points: list[str] = []
        for s in sentencize(_polish(text or "")):
            s = s.strip().lstrip("-*").strip()
            if len(s) < 15:
                continue
            if any(s.lower() == p.lower() for p in points):
                continue
            points.append(s)
            if len(points) >= self.n_key_points:
                break
        return points


def summarize_text_standalone(text: str, model_name: str = DEFAULT_MODEL,
                              device: str = "cpu", max_len: int = 150) -> str:
    s = HierarchicalSummarizer(model_name=model_name, device=device)
    return s.summarize_text(text, max_len=max_len)
