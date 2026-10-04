"""End-to-end meeting-minutes generation (Phase 9). Resumable at the ASR stage.

Usage:
    python scripts/generate_minutes.py --input meeting.wav \
        --output outputs/minutes/demo.md
    python scripts/generate_minutes.py --input transcript.txt \
        --output outputs/minutes/demo.md --input-type transcript
    python scripts/generate_minutes.py --input data/samples/sample_meeting.txt \
        --title "Sprint Planning" --date 2026-10-01

Pipeline: input -> (Whisper) -> parse turns -> clean -> chunk -> hierarchical
summary -> DistilBERT+rules decisions/actions/follow-ups -> Markdown minutes.
Missing fields always render as "Not specified" — nothing is invented.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import SUPPORTED_AUDIO, WhisperASR
from src.chunking import chunk_turns, get_tokenizer
from src.classifier import SentenceClassifier
from src.config import load_config, resolve_path
from src.data_loader import load_transcript_file
from src.extraction import extract_actions, extract_decisions, extract_followups
from src.minutes_generator import generate_minutes
from src.preprocessing import Turn, clean_text, merge_short_turns, turn_sentences
from src.summarizer import HierarchicalSummarizer
from src.utils import select_models, set_seed, setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="audio file or transcript (.txt/.json)")
    ap.add_argument("--output", default="outputs/minutes/demo.md")
    ap.add_argument("--input-type", choices=["auto", "audio", "transcript"], default="auto")
    ap.add_argument("--title", default=None, help="meeting title (else: Not specified)")
    ap.add_argument("--date", default=None, help="meeting date (else: Not specified)")
    ap.add_argument("--participants", default=None, help="comma-separated participants")
    ap.add_argument("--model-size", default=None, help="whisper size for audio input")
    ap.add_argument("--no-summary", action="store_true", help="skip the summariser")
    ap.add_argument("--force", action="store_true", help="redo ASR + summarisation")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    log = setup_logging(cfg, "generate_minutes")
    inp = Path(args.input)
    if not inp.exists():
        log.error("Input not found: %s", inp)
        return 1
    chosen = select_models(cfg)
    t_start = time.perf_counter()

    # ---------------------------------------------------------------- stage 1: input
    is_audio = args.input_type == "audio" or (
        args.input_type == "auto" and inp.suffix.lower() in SUPPORTED_AUDIO)
    if is_audio:
        transcript_json = resolve_path(cfg, "transcripts") / f"{inp.stem}.json"
        if transcript_json.exists() and not args.force:
            log.info("[1/7] ASR: cached transcript found -> %s", transcript_json)
            from src.utils import load_json
            meeting = load_transcript_file(transcript_json)
        else:
            log.info("[1/7] ASR: transcribing %s with whisper-%s (this can take minutes "
                     "on CPU)", inp.name, args.model_size or chosen["whisper"])
            asr = WhisperASR(model_size=args.model_size or chosen["whisper"],
                             device=chosen["device"], language=cfg.whisper.language)
            result = asr.transcribe(inp)
            asr.save_transcript(result, resolve_path(cfg, "transcripts"))
            meeting = load_transcript_file(transcript_json)
        log.info("    %d turns, participants=%s", len(meeting["turns"]),
                 meeting["participants"])
    else:
        log.info("[1/7] transcript input: %s", inp)
        meeting = load_transcript_file(inp)
        log.info("    %d turns, participants=%s", len(meeting["turns"]),
                 meeting["participants"])

    if args.title:
        meeting["title"] = args.title
    if args.date:
        meeting["date"] = args.date
    if args.participants:
        meeting["participants"] = [p.strip() for p in args.participants.split(",")
                                   if p.strip()]

    # ---------------------------------------------------------------- stage 2: clean
    turns = [Turn(speaker=t.get("speaker") or "Unknown",
                  text=clean_text(t.get("text") or ""),
                  start=t.get("start"), end=t.get("end"))
             for t in meeting["turns"]]
    turns = merge_short_turns([t for t in turns if t.text])
    log.info("[2/7] cleaned: %d turns", len(turns))

    # ---------------------------------------------------------------- stage 3: chunk
    tokenizer = get_tokenizer(cfg.summarization.model)
    chunks = chunk_turns(turns, meeting_id=meeting["meeting_id"],
                         max_tokens=cfg.summarization.chunk_max_tokens,
                         overlap_turns=cfg.summarization.chunk_overlap_turns,
                         tokenizer=tokenizer)
    log.info("[3/7] chunked: %d chunks (max %d tokens)",
             len(chunks), cfg.summarization.chunk_max_tokens)

    # ---------------------------------------------------------------- stage 4: summary
    summary: dict
    if args.no_summary:
        summary = {"final_summary": "", "key_points": []}
        log.info("[4/7] summarisation skipped (--no-summary)")
    else:
        log.info("[4/7] summarising with %s ...", chosen["summarization"])
        summarizer = HierarchicalSummarizer(
            model_name=chosen["summarization"], device=chosen["device"],
            chunk_summary_max=cfg.summarization.chunk_summary_max_tokens,
            chunk_summary_min=cfg.summarization.chunk_summary_min_tokens,
            final_summary_max=cfg.summarization.final_summary_max_tokens,
            final_summary_min=cfg.summarization.final_summary_min_tokens)
        summary = summarizer.summarize_meeting(chunks, meeting_id=meeting["meeting_id"])
        log.info("    summary: %s", (summary["final_summary"] or "")[:100])

    # ---------------------------------------------------------------- stage 5: classify
    classifier = SentenceClassifier(
        model_name=cfg.classifier.model,
        checkpoint_dir=resolve_path(cfg, "models") / "classifier",
        device=chosen["device"], decision_threshold=cfg.classifier.decision_threshold,
        action_threshold=cfg.classifier.action_threshold)
    sents = turn_sentences(turns, meeting_id=meeting["meeting_id"])
    analyzed = classifier.analyze(sents)
    log.info("[5/7] classified %d sentences (model=%s)", len(analyzed),
             "distilbert checkpoint" if classifier.available else "rules only")

    # ---------------------------------------------------------------- stage 6: extract
    decisions = extract_decisions(analyzed, cfg.classifier.decision_threshold)
    actions = extract_actions(analyzed, meeting["participants"],
                              cfg.classifier.action_threshold)
    followups = extract_followups(analyzed)
    log.info("[6/7] extracted: %d decisions, %d actions, %d follow-ups",
             len(decisions), len(actions), len(followups))

    # ---------------------------------------------------------------- stage 7: minutes
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parents[1] / out_path
    generate_minutes(meeting, summary, decisions, actions, followups, out_path)
    log.info("[7/7] minutes written -> %s (total %.1fs)", out_path,
             time.perf_counter() - t_start)
    print("\n" + "=" * 60)
    print(out_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
