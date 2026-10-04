"""Preprocess raw meetings into clean, chunked, processed meetings (Phase 3).

For each meeting: light-clean turns -> merge short turns -> sentence records ->
token-aware chunks with metadata. Outputs land in data/processed/. Resumable:
existing processed files are skipped (use --force to redo).

Usage:
    python scripts/preprocess_data.py                      # config subset
    python scripts/preprocess_data.py --split val --limit 30
    python scripts/preprocess_data.py --split test --limit 5 --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_turns, get_tokenizer, save_chunks
from src.config import load_config, resolve_path
from src.data_loader import load_qmsum
from src.preprocessing import Turn, clean_text, merge_short_turns, turn_sentences
from src.utils import load_json, save_json, set_seed, setup_logging, timed


def process_meeting(meeting: dict, tokenizer, max_tokens: int,
                    overlap_turns: int) -> tuple[dict, list]:
    """Clean + chunk one unified-schema meeting; returns processed record."""
    turns = [Turn(speaker=t.get("speaker") or "Unknown",
                  text=clean_text(t.get("text") or ""),
                  start=t.get("start"), end=t.get("end"))
             for t in meeting.get("turns", [])]
    turns = [t for t in turns if t.text]
    turns = merge_short_turns(turns, min_words=3, same_speaker_only=True)
    sentences = turn_sentences(turns, meeting_id=meeting["meeting_id"])
    chunks = chunk_turns(turns, meeting_id=meeting["meeting_id"],
                         max_tokens=max_tokens, overlap_turns=overlap_turns,
                         tokenizer=tokenizer)
    out = dict(meeting)
    out["turns"] = [t.to_dict() for t in turns]
    out["n_sentences"] = len(sentences)
    out["n_chunks"] = len(chunks)
    out["preprocessing"] = {"clean": True, "fillers_removed": True,
                            "merge_short_turns": True, "max_tokens": max_tokens,
                            "overlap_turns": overlap_turns}
    return out, chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None,
                    help="max meetings (default: config data.*_meetings)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    log = setup_logging(cfg, "preprocess")
    raw = resolve_path(cfg, "data_raw")
    out_dir = resolve_path(cfg, "data_processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    limit = args.limit
    if limit is None:
        limit = (cfg.data.smoke_test_meetings if args.split != "train"
                 else cfg.data.initial_eval_meetings)
    qmsum_path = raw / f"qmsum_{args.split}.jsonl"
    if not qmsum_path.exists():
        log.error("Missing %s — run: python scripts/download_data.py", qmsum_path)
        return 1

    with timed(f"preprocess {args.split}", log):
        meetings = load_qmsum(qmsum_path, args.split, limit=limit)
        tokenizer = get_tokenizer(cfg.summarization.model)
        done = skipped = 0
        for m in meetings:
            dest = out_dir / f"{m['meeting_id']}.json"
            if dest.exists() and not args.force and load_json(dest).get("chunks"):
                skipped += 1
                continue
            processed, chunks = process_meeting(
                m, tokenizer, cfg.summarization.chunk_max_tokens,
                cfg.summarization.chunk_overlap_turns)
            save_json(processed, dest)
            save_chunks(chunks, out_dir / f"{m['meeting_id']}_chunks.json")
            done += 1
            log.info("%s: %d turns -> %d sentences, %d chunks (%.1f KB)",
                     m["meeting_id"], len(processed["turns"]), processed["n_sentences"],
                     processed["n_chunks"], dest.stat().st_size / 1024)
        save_json({"split": args.split, "processed": done, "skipped_existing": skipped,
                   "total": len(meetings),
                   "max_tokens": cfg.summarization.chunk_max_tokens},
                  out_dir / f"preprocess_{args.split}_manifest.json")
        log.info("done: %d processed, %d skipped (already existed)", done, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
