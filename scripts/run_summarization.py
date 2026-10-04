"""Hierarchical summarisation over processed meetings (Phase 5). Resumable per meeting.

Usage:
    python scripts/run_summarization.py                    # config subset (val)
    python scripts/run_summarization.py --split test --limit 5
    python scripts/run_summarization.py --model sshleifer/distilbart-cnn-12-6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import Chunk
from src.config import load_config, resolve_path
from src.summarizer import HierarchicalSummarizer
from src.utils import load_json, select_models, set_seed, setup_logging, snapshot_config


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**c) for c in load_json(path)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None, help="override summarisation model")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    log = setup_logging(cfg, "run_summarization")
    proc = resolve_path(cfg, "data_processed")
    out_dir = resolve_path(cfg, "summaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(cfg, out_dir, tag=f"summarization_{args.split}")

    chosen = select_models(cfg)
    model_name = args.model or chosen["summarization"]
    limit = args.limit or cfg.data.smoke_test_meetings
    meeting_files = sorted(proc.glob("qmsum_*.json"))[:limit]
    if not meeting_files:
        log.error("No processed meetings in %s — run scripts/preprocess_data.py first", proc)
        return 1

    summarizer = HierarchicalSummarizer(
        model_name=model_name, device=chosen["device"],
        max_input_tokens=1020,
        chunk_summary_max=cfg.summarization.chunk_summary_max_tokens,
        chunk_summary_min=cfg.summarization.chunk_summary_min_tokens,
        final_summary_max=cfg.summarization.final_summary_max_tokens,
        final_summary_min=cfg.summarization.final_summary_min_tokens)

    done = skipped = 0
    for mf in meeting_files:
        mid = mf.stem
        dest = out_dir / f"{mid}_summary.json"
        if dest.exists() and not args.force:
            skipped += 1
            log.info("skip %s (summary exists)", mid)
            continue
        chunks_path = proc / f"{mid}_chunks.json"
        if not chunks_path.exists():
            log.warning("no chunks for %s, skipping", mid)
            continue
        chunks = load_chunks(chunks_path)
        t0 = time.perf_counter()
        bundle = summarizer.summarize_meeting(chunks, meeting_id=mid, out_dir=out_dir)
        dt = time.perf_counter() - t0
        bundle["runtime_seconds"] = round(dt, 2)
        from src.utils import save_json
        save_json(bundle, dest)
        done += 1
        log.info("%s: %d chunks -> summary in %.1fs | final: %s", mid,
                 bundle["n_chunks"], dt, (bundle["final_summary"] or "")[:110])
    log.info("summarisation done: %d written, %d skipped, model=%s device=%s",
             done, skipped, model_name, chosen["device"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
