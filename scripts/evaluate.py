"""Controlled experiment: compare all systems on the SAME meetings (Phase 8).

Systems:
  tfidf      TF-IDF extractive baseline
  textrank   TextRank extractive baseline
  lstm       small LSTM seq2seq baseline (needs scripts/train_lstm_baseline.py)
  distilbart hierarchical DistilBART zero-shot (the abstractive core)
  hybrid     proposed system: importance-ranked chunks (rule or classifier signals)
             -> hierarchical DistilBART

Outputs:
  outputs/metrics/results.csv + results.json   (merged across runs)
  outputs/metrics/predictions_<split>.json     (per-meeting predictions, for error analysis)

Every number in results.* comes from this script actually running. Optional
--with-bertscore adds BERTScore (slow on CPU; model downloads once).

Usage:
    python scripts/evaluate.py --limit 5                  # smoke-test size
    python scripts/evaluate.py --limit 30 --with-bertscore
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import LSTMBaseline, text_rank_summarize, tfidf_summarize
from src.chunking import Chunk
from src.classifier import SentenceClassifier, sentences_from_meetings, weak_label_sentences
from src.config import load_config, resolve_path
from src.data_loader import load_meetings
from src.evaluation import classification_metrics, rouge_scores, write_results
from src.summarizer import HierarchicalSummarizer
from src.utils import load_json, select_models, set_seed, setup_logging, snapshot_config


def _meeting_text(meeting: dict) -> str:
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in meeting.get("turns", []))


def _load_chunks(proc_dir: Path, meeting_id: str) -> list[Chunk]:
    p = proc_dir / f"{meeting_id}_chunks.json"
    if not p.exists():
        return []
    return [Chunk(**c) for c in load_json(p)]


def rank_chunks_by_importance(chunks: list[Chunk], classifier: SentenceClassifier | None,
                              keep_fraction: float = 0.6) -> list[Chunk]:
    """Hybrid chunk selection: chunks with the most decision/action content are kept
    (in original order) before abstractive summarisation. Uses the fine-tuned
    DistilBERT classifier when a checkpoint exists, otherwise rule-based weak
    signals (recorded in the run metadata)."""
    if not chunks:
        return []
    scored = []
    for c in chunks:
        if classifier is not None and classifier.available:
            from src.extraction import analyze_sentences
            from src.preprocessing import sentencize
            sents = [s for s in sentencize(c.text)]
            from src.preprocessing import Sentence
            objs = [Sentence(text=s, speaker=c.speaker, meeting_id=c.meeting_id)
                    for s in sents]
            probs = classifier.predict_proba([o.text for o in objs])
            an = analyze_sentences(objs, probs)
            score = sum(1 for a in an if a["is_decision"] or a["is_action"])
        else:
            from src.extraction import rule_action_hits, rule_decision_hits
            score = len(rule_decision_hits(c.text)) + len(rule_action_hits(c.text))
        scored.append((score, c))
    scores = [s for s, _ in scored]
    threshold = 1 if max(scores, default=0) > 0 else 1
    if sum(1 for s in scores if s >= threshold) >= max(1, int(len(chunks) * keep_fraction)):
        chosen = [c for s, c in scored if s >= threshold]
    else:
        top_k = max(1, int(round(len(chunks) * keep_fraction)))
        chosen = [c for _, c in sorted(scored, key=lambda x: -x[0])[:top_k]]
    order = {c.chunk_id: i for i, c in enumerate(chunks)}
    return sorted(chosen, key=lambda c: order[c.chunk_id])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--with-bertscore", action="store_true")
    ap.add_argument("--with-lstm", action="store_true", help="include LSTM baseline")
    ap.add_argument("--with-hybrid", action="store_true", default=True)
    ap.add_argument("--no-hybrid", dest="with_hybrid", action="store_false")
    ap.add_argument("--skip-abstractive", action="store_true",
                    help="run only the fast extractive systems")
    ap.add_argument("--tag", default="", help="label appended to system names")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    log = setup_logging(cfg, "evaluate")
    proc = resolve_path(cfg, "data_processed")
    out_dir = resolve_path(cfg, "metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(cfg, out_dir, tag=f"evaluate_{args.split}")

    limit = args.limit or cfg.data.initial_eval_meetings
    meeting_files = sorted(proc.glob("qmsum_*.json"))[:limit]
    if not meeting_files:
        log.error("No processed meetings in %s — run preprocess_data.py first", proc)
        return 1
    meetings = [load_json(p) for p in meeting_files]
    meetings = [m for m in meetings if (m.get("reference_summary") or "").strip()]
    log.info("evaluating on %d meetings (split=%s)", len(meetings), args.split)

    tag = f" {args.tag}" if args.tag else ""
    records: list[dict] = []       # per-meeting predictions (error analysis)
    results: list[dict] = []
    predictions: dict[str, dict] = {m["meeting_id"]: {} for m in meetings}

    # ---------------------------------------------------------------- extractive
    for name, fn in (("tfidf", tfidf_summarize), ("textrank", text_rank_summarize)):
        t0, preds = time.perf_counter(), []
        for m in meetings:
            pred = fn(_meeting_text(m), n_sentences=5)
            preds.append(pred)
            predictions[m["meeting_id"]][name] = pred
        refs = [m["reference_summary"] for m in meetings]
        r = rouge_scores(preds, refs)
        row = {"system": f"{name}{tag}", "dataset": "QMSum", "split": args.split,
               "n_meetings": len(meetings), "rouge1": r["rouge1"], "rouge2": r["rouge2"],
               "rougeL": r["rougeL"],
               "runtime_seconds": round(time.perf_counter() - t0, 2),
               "device": "cpu"}
        results.append(row)
        log.info("%s: R1=%.4f R2=%.4f RL=%.4f (%.1fs)", name, r["rouge1"], r["rouge2"],
                 r["rougeL"], row["runtime_seconds"])

    # ---------------------------------------------------------------- LSTM
    lstm_ckpt = resolve_path(cfg, "models") / "lstm_baseline.pt"
    if args.with_lstm:
        if lstm_ckpt.exists():
            t0 = time.perf_counter()
            base = LSTMBaseline.load(lstm_ckpt)
            preds = [base.summarize(_meeting_text(m)) for m in meetings]
            refs = [m["reference_summary"] for m in meetings]
            r = rouge_scores(preds, refs)
            results.append({"system": f"lstm{tag}", "dataset": "QMSum",
                            "split": args.split, "n_meetings": len(meetings),
                            "rouge1": r["rouge1"], "rouge2": r["rouge2"],
                            "rougeL": r["rougeL"],
                            "runtime_seconds": round(time.perf_counter() - t0, 2),
                            "device": select_models(cfg)["device"]})
            for m, p in zip(meetings, preds):
                predictions[m["meeting_id"]]["lstm"] = p
            log.info("lstm: R1=%.4f R2=%.4f RL=%.4f", r["rouge1"], r["rouge2"], r["rougeL"])
        else:
            log.warning("LSTM checkpoint missing (%s) — skipping. "
                        "Run scripts/train_lstm_baseline.py first.", lstm_ckpt)
            results.append({"system": f"lstm{tag}", "status": "not run (no checkpoint)",
                            "rouge1": None, "rouge2": None, "rougeL": None})

    # ---------------------------------------------------------------- abstractive
    if not args.skip_abstractive:
        chosen = select_models(cfg)
        summarizer = HierarchicalSummarizer(
            model_name=chosen["summarization"], device=chosen["device"],
            chunk_summary_max=cfg.summarization.chunk_summary_max_tokens,
            chunk_summary_min=cfg.summarization.chunk_summary_min_tokens,
            final_summary_max=cfg.summarization.final_summary_max_tokens,
            final_summary_min=cfg.summarization.final_summary_min_tokens)
        classifier = SentenceClassifier(
            model_name=cfg.classifier.model,
            checkpoint_dir=resolve_path(cfg, "models") / "classifier",
            device=chosen["device"], decision_threshold=cfg.classifier.decision_threshold,
            action_threshold=cfg.classifier.action_threshold)
        hybrid_uses = ("distilbert checkpoint" if classifier.available else "rules (no checkpoint)")

        # distilbart: all chunks ; hybrid: importance-ranked chunks
        for system in (["distilbart"] + (["hybrid"] if args.with_hybrid else [])):
            t0, preds = time.perf_counter(), []
            for i, m in enumerate(meetings):
                chunks = _load_chunks(proc, m["meeting_id"])
                if not chunks:
                    log.warning("no chunks for %s; skipping", m["meeting_id"])
                    preds.append("")
                    continue
                use = (rank_chunks_by_importance(chunks, classifier)
                       if system == "hybrid" else chunks)
                bundle = summarizer.summarize_meeting(use, meeting_id=m["meeting_id"])
                pred = (bundle["final_summary"] or "").strip()
                preds.append(pred)
                predictions[m["meeting_id"]][system] = pred
                log.info("[%d/%d] %s/%s: %d->%d chunks", i + 1, len(meetings), system,
                         m["meeting_id"], len(chunks), len(use))
            refs = [m["reference_summary"] for m in meetings]
            r = rouge_scores(preds, refs)
            row = {"system": f"{system}{tag}", "dataset": "QMSum", "split": args.split,
                   "n_meetings": len(meetings), "rouge1": r["rouge1"],
                   "rouge2": r["rouge2"], "rougeL": r["rougeL"],
                   "runtime_seconds": round(time.perf_counter() - t0, 2),
                   "device": chosen["device"],
                   "model": chosen["summarization"]}
            if system == "hybrid":
                row["chunk_selection"] = hybrid_uses
            results.append(row)
            log.info("%s: R1=%.4f R2=%.4f RL=%.4f (%.1fs)", system, r["rouge1"],
                     r["rouge2"], r["rougeL"], row["runtime_seconds"])

        # ---------------------------------------------------------------- classifier eval
        if classifier.available:
            sents = sentences_from_meetings(meetings)
            _, y_true = weak_label_sentences(sents)
            texts = [s.text for s in sents]
            probs = classifier.predict_proba(texts)
            if probs is not None:
                y_pred = probs.argmax(-1).tolist()
                cm = classification_metrics(y_true.tolist(), y_pred)
                cm.update({"system": f"distilbert-classifier{tag}",
                           "evaluated_on": "WEAK rule labels (not gold annotations)",
                           "split": args.split, "n_meetings": len(meetings)})
                results.append(cm)
                log.info("classifier (vs weak labels): F1(macro)=%.4f", cm["f1_macro"])
        else:
            log.info("no fine-tuned classifier checkpoint — classifier eval skipped "
                     "(run scripts/train_classifier.py)")

    # ---------------------------------------------------------------- outputs
    if args.with_bertscore:
        from src.evaluation import bertscore
        refs = [m["reference_summary"] for m in meetings]
        for sys_name in sorted({k for p in predictions.values() for k in p}):
            preds = [predictions[m["meeting_id"]].get(sys_name, "") for m in meetings]
            bs = bertscore(preds, refs)
            for row in results:
                if row.get("system", "").startswith(sys_name):
                    row.update(bs)
            log.info("%s BERTScore F1=%.4f", sys_name, bs.get("bertscore_f1") or 0.0)

    csv_p, json_p = write_results(results, out_dir)
    from src.utils import save_json
    save_json({"split": args.split, "n_meetings": len(meetings),
               "seed": cfg.seed, "predictions": predictions},
              out_dir / f"predictions_{args.split}.json")
    log.info("results -> %s, %s", csv_p, json_p)
    log.info("predictions -> %s", out_dir / f"predictions_{args.split}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
