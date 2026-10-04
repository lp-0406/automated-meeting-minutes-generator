"""Generate result figures from actual metric files (Phase 8).

Reads ONLY outputs/metrics/results.json (plus weak-label counts it recomputes
itself), so every chart reflects real executions. Missing metrics are omitted —
never filled with placeholder numbers.

Figures (outputs/figures/):
  rouge_comparison.png     ROUGE-1/2/L per system
  bertscore_comparison.png BERTScore F1 per system (if computed)
  runtime_comparison.png   runtime per system (if recorded)
  classification_f1.png    per-class F1 of the sentence classifier (if trained)
  class_distribution.png   weak-label class distribution on processed meetings

Usage: python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, resolve_path
from src.utils import set_seed, setup_logging


def _load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _bar(ax, labels, values, title, ylabel=""):
    ax.bar(range(len(labels)), values, color="#4C72B0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)


def main() -> int:
    cfg = load_config()
    set_seed(cfg.seed)
    log = setup_logging(cfg, "make_figures")
    metrics_dir = resolve_path(cfg, "metrics")
    fig_dir = resolve_path(cfg, "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = [r for r in _load_results(metrics_dir / "results.json")
               if r.get("rouge1") is not None]
    if not results:
        log.error("No results with ROUGE in %s — run scripts/evaluate.py first",
                  metrics_dir / "results.json")
        return 1
    names = [r.get("system", "?") for r in results]

    # 1. ROUGE
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(names))
    w = 0.25
    for j, (key, label) in enumerate((("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"),
                                      ("rougeL", "ROUGE-L"))):
        vals = [r.get(key) for r in results]
        if any(v is not None for v in vals):
            ax.bar([i + j * w for i in x], [v or 0 for v in vals], w, label=label)
    ax.set_xticks([i + w for i in x])
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_title("Summarisation: ROUGE comparison (same meetings, same references)")
    ax.set_ylabel("F1 score")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(fig_dir / "rouge_comparison.png", dpi=150)
    plt.close(fig)
    log.info("wrote rouge_comparison.png")

    # 2. BERTScore (only rows that have it)
    bs_rows = [r for r in results if r.get("bertscore_f1") is not None]
    if bs_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        _bar(ax, [r.get("system", "?") for r in bs_rows],
             [r["bertscore_f1"] for r in bs_rows],
             "Summarisation: BERTScore F1", "BERTScore F1")
        fig.tight_layout()
        fig.savefig(fig_dir / "bertscore_comparison.png", dpi=150)
        plt.close(fig)
        log.info("wrote bertscore_comparison.png")

    # 3. Runtime
    rt_rows = [r for r in results if r.get("runtime_seconds") is not None]
    if rt_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        _bar(ax, [r.get("system", "?") for r in rt_rows],
             [r["runtime_seconds"] for r in rt_rows],
             "Runtime per system (same meetings)", "seconds")
        fig.tight_layout()
        fig.savefig(fig_dir / "runtime_comparison.png", dpi=150)
        plt.close(fig)
        log.info("wrote runtime_comparison.png")

    # 4. Classifier per-class F1
    clf = next((r for r in _load_results(metrics_dir / "results.json")
                if r.get("per_class")), None)
    if clf:
        labels, f1s = [], []
        for key, val in sorted(clf["per_class"].items()):
            labels.append(clf.get("labels", {}).get(key, key))
            f1s.append(val["f1"])
        fig, ax = plt.subplots(figsize=(6, 4))
        _bar(ax, labels, f1s, "Sentence classifier: per-class F1\n(evaluated against "
                              "weak rule labels)", "F1")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(fig_dir / "classification_f1.png", dpi=150)
        plt.close(fig)
        log.info("wrote classification_f1.png")

    # 5. Class distribution (recomputed from processed meetings = weak labels)
    from src.classifier import sentences_from_meetings, weak_label_sentences
    proc = resolve_path(cfg, "data_processed")
    files = sorted(proc.glob("qmsum_*.json"))
    if files:
        from src.utils import load_json
        meetings = [load_json(p) for p in files]
        _, y = weak_label_sentences(sentences_from_meetings(meetings))
        counts = [(y == i).sum() for i in range(3)]
        fig, ax = plt.subplots(figsize=(6, 4))
        _bar(ax, ["other (0)", "decision (1)", "action (2)"],
             [float(c) for c in counts],
             f"Weak-label class distribution ({len(files)} processed meetings)",
             "sentences")
        fig.tight_layout()
        fig.savefig(fig_dir / "class_distribution.png", dpi=150)
        plt.close(fig)
        log.info("wrote class_distribution.png (counts=%s)", counts)

    log.info("all figures saved to %s", fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
