"""Evaluation metrics (Phase 8).

All metrics are computed by ACTUAL EXECUTION — nothing here returns fabricated
numbers. Functions return plain dicts so scripts can dump them straight to
outputs/metrics/results.{csv,json}.

- Summarisation: ROUGE-1/2/L (rouge_score package, local, no network)
- Optional:       BERTScore (downloads roberta-base once; can be slow on CPU)
- Classification: precision / recall / F1 (macro + per class) + confusion matrix
- ASR:            WER (jiwer)
- Action items:   precision / recall / F1 over normalised action strings,
                  plus exact-match accuracy for owner and deadline where a
                  reference pair exists.
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

log = logging.getLogger("mmg")


# --------------------------------------------------------------------------- text
def _norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> dict:
    """Mean ROUGE-1/2/L (F1) with the local rouge_score package."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError("rouge_score required: pip install rouge_score") from e
    if len(predictions) != len(references):
        raise ValueError(f"length mismatch: {len(predictions)} vs {len(references)}")
    pairs = [(p, r) for p, r in zip(predictions, references)
             if (p or "").strip() and (r or "").strip()]
    if not pairs:
        return {"rouge1": None, "rouge2": None, "rougeL": None, "n_evaluated": 0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, r in pairs:
        s = scorer.score(r, p)
        agg["rouge1"] += s["rouge1"].fmeasure
        agg["rouge2"] += s["rouge2"].fmeasure
        agg["rougeL"] += s["rougeL"].fmeasure
    n = len(pairs)
    return {k: round(v / n, 4) for k, v in agg.items()} | {"n_evaluated": n}


def bertscore(predictions: Sequence[str], references: Sequence[str],
              model_type: str = "distilbert-base-uncased") -> dict:
    """BERTScore F1. Downloads the scoring model on first use (free, ~250 MB)."""
    try:
        from bert_score import score as _bs
    except ImportError as e:
        raise ImportError("bert_score required: pip install bert_score") from e
    pairs = [(p, r) for p, r in zip(predictions, references)
             if (p or "").strip() and (r or "").strip()]
    if not pairs:
        return {"bertscore_f1": None, "n_evaluated": 0}
    p, r, f = _bs([x[0] for x in pairs], [x[1] for x in pairs],
                  lang="en", model_type=model_type, verbose=False)
    return {"bertscore_precision": round(float(p.mean()), 4),
            "bertscore_recall": round(float(r.mean()), 4),
            "bertscore_f1": round(float(f.mean()), 4),
            "n_evaluated": len(pairs)}


# --------------------------------------------------------------------------- classification
def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                           labels: Sequence[int] = (0, 1, 2)) -> dict:
    """Per-class + macro/weighted precision/recall/F1 and confusion matrix."""
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true/y_pred length mismatch")
    if not len(y_true):
        return {"accuracy": None, "f1_macro": None, "f1_weighted": None,
                "per_class": {}, "confusion_matrix": []}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(labels),
                                                 average=None, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(labels),
                                                    average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(labels),
                                                    average="weighted", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(fm), 4),
        "f1_weighted": round(float(fw), 4),
        "precision_macro": round(float(pm), 4),
        "recall_macro": round(float(rm), 4),
        "per_class": {
            str(l): {"precision": round(float(pi), 4), "recall": round(float(ri), 4),
                     "f1": round(float(fi), 4), "support": int(si)}
            for l, pi, ri, fi, si in zip(labels, p, r, f, s)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "labels": {str(l): name for l, name in
                   zip(labels, ("other", "decision", "action"))},
        "n_evaluated": len(y_true),
    }


# --------------------------------------------------------------------------- ASR
def word_error_rate(reference: str, hypothesis: str) -> float:
    from .asr import word_error_rate as _wer
    return _wer(reference, hypothesis)


# --------------------------------------------------------------------------- actions
def _action_key(item: dict) -> str:
    return _norm(item.get("action") or item.get("text") or "")


def action_item_metrics(predictions: Sequence[dict], references: Sequence[dict]) -> dict:
    """P/R/F1 for action extraction. Matching = exact match on normalised action
    text (conservative: partial matches are NOT counted as correct)."""
    pred_keys = {_action_key(p) for p in predictions if _action_key(p)}
    ref_keys = {_action_key(r) for r in references if _action_key(r)}
    if not ref_keys and not pred_keys:
        return {"precision": None, "recall": None, "f1": None,
                "n_predicted": 0, "n_reference": 0}
    tp = len(pred_keys & ref_keys)
    precision = tp / len(pred_keys) if pred_keys else 0.0
    recall = tp / len(ref_keys) if ref_keys else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # Owner / deadline exact match on the matched subset (only where a reference exists).
    ref_by_key = {_action_key(r): r for r in references}
    owner_hit = owner_tot = dl_hit = dl_tot = 0
    for p in predictions:
        key = _action_key(p)
        if key in ref_by_key:
            ref = ref_by_key[key]
            if (ref.get("owner") or "Not specified") != "Not specified":
                owner_tot += 1
                owner_hit += _norm(p.get("owner", "")) == _norm(ref.get("owner", ""))
            if (ref.get("deadline") or "Not specified") != "Not specified":
                dl_tot += 1
                dl_hit += _norm(p.get("deadline", "")) == _norm(ref.get("deadline", ""))
    return {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "tp": tp,
        "n_predicted": len(pred_keys), "n_reference": len(ref_keys),
        "owner_exact_match": round(owner_hit / owner_tot, 4) if owner_tot else None,
        "owner_n_evaluated": owner_tot,
        "deadline_exact_match": round(dl_hit / dl_tot, 4) if dl_tot else None,
        "deadline_n_evaluated": dl_tot,
    }


def write_results(rows: Sequence[dict], out_dir) -> tuple:
    """Write outputs/metrics/results.csv + results.json (idempotent, merged)."""
    import csv
    import json
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / "results.csv", out_dir / "results.json"
    flat = []
    for row in rows:
        flat.append({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                     for k, v in row.items()})
    fieldnames = sorted({k for row in flat for k in row})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flat)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)
    return csv_path, json_path
