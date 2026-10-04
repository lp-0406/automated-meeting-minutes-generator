"""Transcribe meeting audio with Whisper (Phase 4). Resumable per file.

Usage:
    python scripts/run_asr.py data/samples/sample_meeting.wav
    python scripts/run_asr.py audio_dir/ --model-size base
    python scripts/run_asr.py file.wav --reference ref.txt   # also computes WER
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import SUPPORTED_AUDIO, WhisperASR, word_error_rate
from src.config import load_config, resolve_path
from src.utils import set_seed, setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="audio file or directory of audio files")
    ap.add_argument("--model-size", default=None, help="tiny|base|small (default: config)")
    ap.add_argument("--device", default=None, help="auto|cuda|cpu (default: config)")
    ap.add_argument("--reference", default=None,
                    help="reference transcript (.txt) for WER (single file only)")
    ap.add_argument("--force", action="store_true", help="redo existing transcripts")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    log = setup_logging(cfg, "run_asr")
    out_dir = resolve_path(cfg, "transcripts")
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)
    files = ([inp] if inp.is_file()
             else sorted(p for p in inp.iterdir() if p.suffix.lower() in SUPPORTED_AUDIO))
    if not files:
        log.error("No supported audio found at %s (%s)", inp, sorted(SUPPORTED_AUDIO))
        return 1

    # Model size policy: config -> CLI, still downgraded on CPU by select_models.
    from src.utils import select_models
    chosen = select_models(cfg)
    size = args.model_size or chosen["whisper"]
    device = args.device or chosen["device"]
    if args.model_size is None and chosen["downgraded"]:
        log.info("resource policy: using %s on %s", size, device)
    asr = WhisperASR(model_size=size, device=device,
                     language=cfg.whisper.language,
                     condition_on_previous_text=cfg.whisper.condition_on_previous_text)

    failures = 0
    for f in files:
        dest = out_dir / f"{f.stem}.json"
        if dest.exists() and not args.force:
            log.info("skip %s (transcript exists: %s)", f.name, dest)
            continue
        t0 = time.perf_counter()
        try:
            result = asr.transcribe(f)
        except Exception as e:
            failures += 1
            log.error("FAILED %s: %s", f, e)
            continue
        jp, tp = asr.save_transcript(result, out_dir)
        log.info("%s -> %s (%d segments, %.1fs)", f.name, jp.name,
                 len(result["segments"]), time.perf_counter() - t0)
        if args.reference:
            ref = Path(args.reference).read_text(encoding="utf-8")
            wer = word_error_rate(ref, result["text"])
            log.info("WER vs reference: %.4f", wer)
            from src.utils import save_json
            save_json({"wer": wer, "reference": str(args.reference),
                       "hypothesis": result["text"]},
                      out_dir / f"{f.stem}_wer.json")
    log.info("ASR complete: %d file(s), %d failure(s)", len(files), failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
