"""Download datasets (Phase 2). Resumable: existing files are skipped.

Usage:
    python scripts/download_data.py                      # QMSum only (~17 MB)
    python scripts/download_data.py --with-dialogsum     # + DialogSum warm-up
    python scripts/download_data.py --with-ami --ami-meetings 2   # stream 2 AMI text meetings
    python scripts/download_data.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path
from src.data_loader import (download_dialogsum, download_qmsum, load_ami_streaming,
                             save_meeting)
from src.utils import save_json, set_seed, setup_logging, timed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-dialogsum", action="store_true")
    ap.add_argument("--with-ami", action="store_true")
    ap.add_argument("--with-meetingbank", action="store_true",
                    help="download MeetingBank transcripts (~large; NOT default)")
    ap.add_argument("--ami-meetings", type=int, default=2)
    ap.add_argument("--all", action="store_true", help="QMSum + DialogSum + AMI subset")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.with_dialogsum = args.with_ami = True

    cfg = load_config()
    set_seed(cfg.seed)
    log = setup_logging(cfg, "download_data")
    raw = resolve_path(cfg, "data_raw")
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {"stages": {}}

    with timed("QMSum download", log):
        paths = download_qmsum(raw, force=args.force)
        manifest["stages"]["qmsum"] = {
            "url": "https://github.com/Yale-LILY/QMSum",
            "license": "MIT",
            "files": {k: {"path": v.name, "bytes": v.stat().st_size}
                      for k, v in paths.items()},
        }
        log.info("QMSum ready: %s", {k: v.name for k, v in paths.items()})

    if args.with_dialogsum:
        with timed("DialogSum download", log):
            res = download_dialogsum(raw, force=args.force)
            manifest["stages"]["dialogsum"] = {
                "id": "knkarthick/dialogsum", "license": "CC-BY-NC-SA-4.0", **res}
            log.info("DialogSum: %s", res)

    if args.with_ami:
        with timed("AMI text streaming", log):
            meetings = load_ami_streaming(n_meetings=args.ami_meetings)
            out_dir = raw / "ami"
            for m in meetings:
                save_meeting(m, out_dir / f"{m['meeting_id']}.json")
            manifest["stages"]["ami"] = {
                "id": "edinburghcstr/ami", "license": "CC-BY-4.0",
                "mode": "streaming text-only (audio never downloaded)",
                "n_meetings": len(meetings),
                "meeting_ids": [m["meeting_id"] for m in meetings]}
            log.info("AMI: %d meetings streamed to %s", len(meetings), out_dir)

    if args.with_meetingbank:
        log.warning("MeetingBank is optional and large; downloading transcripts via HF")
        from datasets import load_dataset
        ds = load_dataset("lytang/MeetingBank-transcript", split="train")
        keep = ds.select(range(min(20, len(ds))))      # NEVER the whole corpus by default
        keep.to_json(str(raw / "meetingbank_sample.jsonl"))
        manifest["stages"]["meetingbank"] = {"id": "lytang/MeetingBank-transcript",
                                             "license": "CC-BY-NC-SA-4.0",
                                             "downloaded_examples": len(keep)}
        log.info("MeetingBank sample: %d examples", len(keep))

    save_json(manifest, raw / "download_manifest.json")
    log.info("manifest -> %s", raw / "download_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
