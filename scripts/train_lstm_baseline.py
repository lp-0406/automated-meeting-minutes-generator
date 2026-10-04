"""Train the small LSTM seq2seq baseline (Phase 7).

The LSTM is intentionally tiny (64-dim embeddings, 128-dim hidden, 1 layer) so it
trains in minutes on free CPU/GPU. It is an ACADEMIC BASELINE, not the production
model. Training pairs = (meeting transcript, reference summary) from QMSum (or
DialogSum when present). Documented limitations: small vocabulary, no pretraining,
exposure bias, length-limited inputs.

Usage:
    python scripts/train_lstm_baseline.py                     # QMSum train subset
    python scripts/train_lstm_baseline.py --pairs 60 --epochs 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import LSTMBaseline
from src.config import load_config, resolve_path
from src.data_loader import load_qmsum
from src.utils import select_models, set_seed, setup_logging, snapshot_config


def build_pairs(meetings) -> list[tuple[str, str]]:
    pairs = []
    for m in meetings:
        doc = "\n".join(f"{t['speaker']}: {t['text']}" for t in m.get("turns", []))
        ref = (m.get("reference_summary") or "").strip()
        if doc and ref:
            pairs.append((doc, ref))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=60, help="max training pairs")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.seed)
    log = setup_logging(cfg, "train_lstm")
    raw = resolve_path(cfg, "data_raw")
    model_dir = resolve_path(cfg, "models")
    ckpt = model_dir / "lstm_baseline.pt"
    if ckpt.exists() and not args.force:
        log.info("checkpoint exists: %s (use --force to retrain)", ckpt)
        return 0

    src_path = raw / "qmsum_train.jsonl"
    if not src_path.exists():
        log.error("Missing %s — run scripts/download_data.py first", src_path)
        return 1
    meetings = load_qmsum(src_path, "train", limit=args.pairs)
    pairs = build_pairs(meetings)
    if not pairs:
        log.error("No (transcript, summary) pairs built from %d meetings", len(meetings))
        return 1
    log.info("training pairs: %d (from %d meetings)", len(pairs), len(meetings))

    chosen = select_models(cfg)
    snapshot_config(cfg, model_dir, tag="lstm_baseline")
    base = LSTMBaseline()
    t0 = time.perf_counter()
    hist = base.fit(pairs, epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, device=chosen["device"], seed=cfg.seed, logger=log)
    hist["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    hist["dataset"] = "QMSum train (subset)"
    hist["architecture"] = ("LSTM encoder (1 layer) + Bahdanau attention + LSTM decoder; "
                            f"emb={base.emb_dim} hid={base.hid_dim}")
    hist["limitations"] = ("small vocab from training subset; no pretraining; "
                           "short inputs (src_max_len=%d); teacher forcing only" % base.src_max_len)
    base.save(ckpt)
    from src.utils import save_json
    save_json(hist, model_dir / "lstm_baseline_metrics.json")
    log.info("saved %s | %s", ckpt, {k: hist[k] for k in ("n_train_pairs", "vocab_size",
                                                          "runtime_seconds")})
    log.info("history: %s", hist["history"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
