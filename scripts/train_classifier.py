"""Fine-tune the DistilBERT decision/action sentence classifier (Phase 6).

Training labels are RULE-BASED WEAK LABELS (see src/extraction.py), not gold
annotations — this is recorded in the metrics file and in the README.
Splits are MEETING-LEVEL: no sentence from a meeting appears in both train and val.

Usage:
    python scripts/train_classifier.py                       # config defaults
    python scripts/train_classifier.py --train-meetings 40 --epochs 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import fine_tune, meeting_level_split
from src.config import load_config, resolve_path
from src.data_loader import load_qmsum
from src.utils import select_models, set_seed, setup_logging, snapshot_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-meetings", type=int, default=40,
                    help="total meetings before the val split (train+val)")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    log = setup_logging(cfg, "train_classifier")
    raw = resolve_path(cfg, "data_raw")
    out_dir = resolve_path(cfg, "models") / "classifier"
    if (out_dir / "config.json").exists() and not args.force:
        log.info("checkpoint exists at %s (use --force to retrain)", out_dir)
        metrics = out_dir / "train_metrics.json"
        if metrics.exists():
            log.info("existing metrics: %s", load_json_text(metrics))
        return 0

    qmsum_train = raw / "qmsum_train.jsonl"
    if not qmsum_train.exists():
        log.error("Missing %s — run scripts/download_data.py first", qmsum_train)
        return 1

    meetings = load_qmsum(qmsum_train, "train", limit=args.train_meetings)
    train_ms, val_ms = meeting_level_split(meetings, args.val_fraction, cfg.seed)
    chosen = select_models(cfg)
    snapshot_config(cfg, out_dir, tag="classifier_training")

    metrics = fine_tune(
        train_ms, val_ms,
        model_name=cfg.classifier.model,
        out_dir=out_dir,
        epochs=args.epochs or cfg.classifier.epochs,
        batch_size=args.batch_size or cfg.classifier.batch_size,
        lr=args.lr or cfg.classifier.learning_rate,
        max_length=cfg.classifier.max_length,
        device=chosen["device"], seed=cfg.seed, logger=log)
    log.info("F1(macro)=%.4f accuracy=%.4f -> %s",
             metrics["f1_macro"], metrics["accuracy"], out_dir)
    return 0


def load_json_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")[:400]


if __name__ == "__main__":
    sys.exit(main())
