"""DistilBERT sentence classifier (Phase 6): other / decision / action.

- Labels: 0 = other, 1 = decision, 2 = action
- The model is NEVER trained from scratch: a pretrained DistilBERT checkpoint is
  fine-tuned on meeting-level train/val splits.
- Class imbalance is handled with per-class loss weights.
- Weak labels (rule-based) are used when no gold annotation exists; they are
  always reported as WEAK labels — never presented as gold.
- Inference without a fine-tuned checkpoint returns probabilities=None and the
  pipeline degrades gracefully to rules-only extraction (documented in output).
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from .extraction import weak_label
from .preprocessing import Sentence, turn_sentences
from .utils import save_json

log = logging.getLogger("mmg")

LABEL_IDS = {"other": 0, "decision": 1, "action": 2}
ID2LABEL = {0: "other", 1: "decision", 2: "action"}


# --------------------------------------------------------------------------- data prep
def sentences_from_meetings(meetings: Sequence[dict]) -> list[Sentence]:
    out: list[Sentence] = []
    for m in meetings:
        from .preprocessing import Turn
        turns = [Turn(**{k: t.get(k) for k in ("speaker", "text", "start", "end")})
                 for t in m.get("turns", [])]
        out.extend(turn_sentences(turns, meeting_id=m.get("meeting_id", "")))
    return out


def weak_label_sentences(sentences: Sequence[Sentence]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X texts, y weak labels). Label provenance = rule-based weak labelling."""
    y = np.array([weak_label(s.text) for s in sentences], dtype=np.int64)
    return np.array([s.text for s in sentences], dtype=object), y


def meeting_level_split(meetings: Sequence[dict], val_fraction: float = 0.2,
                        seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split by MEETING, never by sentence: sentences from one meeting can never
    appear in both train and val (prevents leakage)."""
    ids = [m.get("meeting_id", str(i)) for i, m in enumerate(meetings)]
    idx = list(range(len(meetings)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(len(meetings) * val_fraction))) if len(meetings) > 1 else 0
    val_idx, train_idx = set(idx[:n_val]), idx[n_val:]
    train = [meetings[i] for i in train_idx]
    val = [meetings[i] for i in val_idx]
    log.info("meeting-level split: train=%d meetings, val=%d meetings", len(train), len(val))
    return train, val


def class_weights(labels: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights (normalised to mean 1) for imbalance."""
    counts = np.bincount(labels.astype(int), minlength=3).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(counts) * counts)
    return w / w.mean()


# --------------------------------------------------------------------------- model
class SentenceClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 checkpoint_dir: str | Path | None = None,
                 device: str = "cpu", max_length: int = 128,
                 batch_size: int = 16, decision_threshold: float = 0.6,
                 action_threshold: float = 0.6):
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.decision_threshold = decision_threshold
        self.action_threshold = action_threshold
        self._model = None
        self._tok = None
        self._finetuned = False

    # ---------------------------------------------------------------- loading
    @property
    def available(self) -> bool:
        """True when a fine-tuned checkpoint exists (a random head on the base
        model would produce meaningless probabilities)."""
        return self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return self._finetuned
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        ckpt = None
        if self.checkpoint_dir and (Path(self.checkpoint_dir) / "config.json").exists():
            ckpt = Path(self.checkpoint_dir)
        try:
            if ckpt:
                self._model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
                self._tok = AutoTokenizer.from_pretrained(str(ckpt))
                self._finetuned = True
            else:
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, num_labels=3, id2label=ID2LABEL,
                    label2id={v: k for k, v in ID2LABEL.items()})
                self._tok = AutoTokenizer.from_pretrained(self.model_name)
                self._finetuned = False
                log.warning("No fine-tuned classifier checkpoint found (%s); "
                            "probabilities will NOT be used — rules only.",
                            self.checkpoint_dir)
        except Exception as e:
            log.error("Classifier could not be loaded: %s", e)
            self._model = None
            return False
        self._model.eval()
        self._model.to("cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu")
        return self._finetuned

    # ---------------------------------------------------------------- inference
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray | None:
        """(n, 3) probabilities, or None when no fine-tuned checkpoint exists."""
        if not texts:
            return None
        if not self._ensure_loaded() or not self._finetuned:
            return None
        import torch
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i:i + self.batch_size])
                enc = self._tok(batch, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self._model.device) for k, v in enc.items()}
                logits = self._model(**enc).logits
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def analyze(self, sentences: Sequence[Sentence]) -> list[dict]:
        """Produce the analyzed-sentence records consumed by extraction."""
        from .extraction import analyze_sentences
        texts = [s.text for s in sentences]
        probs = self.predict_proba(texts)
        return analyze_sentences(sentences, probs,
                                 self.decision_threshold, self.action_threshold)


# --------------------------------------------------------------------------- training
def fine_tune(train_meetings: Sequence[dict], val_meetings: Sequence[dict],
              model_name: str = "distilbert-base-uncased",
              out_dir: str | Path = "models/classifier",
              epochs: int = 3, batch_size: int = 16, lr: float = 5e-5,
              max_length: int = 128, device: str = "cpu",
              seed: int = 42, logger=None) -> dict:
    """Fine-tune DistilBERT on weakly labelled sentences with class-weighted loss.

    Returns a metrics dict (per-class precision/recall/F1 + confusion matrix) that
    is also saved next to the checkpoint. Weak-label provenance is recorded.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    log = logger or log
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    tr_s = sentences_from_meetings(train_meetings)
    va_s = sentences_from_meetings(val_meetings)
    _, tr_y = weak_label_sentences(tr_s)
    _, va_y = weak_label_sentences(va_s)
    log.info("training sentences=%d %s | val sentences=%d %s",
             len(tr_s), dict(zip(*np.unique(tr_y, return_counts=True))),
             len(va_s), dict(zip(*np.unique(va_y, return_counts=True))))

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()})
    dev = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    model.to(dev)

    def encode(sentences):
        return tok([s.text for s in sentences], padding=True, truncation=True,
                   max_length=max_length, return_tensors="pt")

    tr_enc = encode(tr_s)
    tr_enc = {"input_ids": tr_enc["input_ids"], "attention_mask": tr_enc["attention_mask"],
              "labels": torch.tensor(tr_y, dtype=torch.long)}
    va_enc = encode(va_s)
    va_enc = {"input_ids": va_enc["input_ids"], "attention_mask": va_enc["attention_mask"],
              "labels": torch.tensor(va_y, dtype=torch.long)}

    class WeightedTrainer:
        """Minimal training loop (avoids pulling in the HF Trainer API surface)."""

        def __init__(self):
            weights = class_weights(tr_y)
            sample_w = torch.tensor([weights[y] for y in tr_y], dtype=torch.double)
            sampler = WeightedRandomSampler(sample_w, num_samples=len(tr_y), replacement=True)
            self.loader = DataLoader(_TorchDataset(tr_enc), batch_size=batch_size,
                                     sampler=sampler)
            self.opt = torch.optim.AdamW(model.parameters(), lr=lr)
            total = epochs * len(self.loader)
            self.sched = get_linear_schedule_with_warmup(
                self.opt, num_warmup_steps=max(1, total // 10), num_training_steps=total)
            self.w = torch.tensor(weights, dtype=torch.float).to(dev)

        def train_epoch(self):
            model.train()
            tot = 0.0
            for batch in self.loader:
                batch = {k: v.to(dev) for k, v in batch.items()}
                logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
                loss = torch.nn.functional.cross_entropy(
                    logits, batch["labels"], weight=self.w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.opt.step()
                self.sched.step()
                self.opt.zero_grad()
                tot += float(loss.item())
            return tot / max(1, len(self.loader))

    trainer = WeightedTrainer()
    history = []
    for ep in range(epochs):
        loss = trainer.train_epoch()
        history.append({"epoch": ep + 1, "train_loss": round(loss, 4)})
        log.info("epoch %d/%d loss=%.4f", ep + 1, epochs, loss)

    # ---------------------------------------------------------------- evaluation
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(va_s), batch_size):
            batch = [s.text for s in va_s[i:i + batch_size]]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=int)

    from .evaluation import classification_metrics
    metrics = classification_metrics(va_y.tolist(), y_pred.tolist())
    metrics.update({"label_provenance": "rule-based weak labels (NOT gold annotations)",
                    "model": model_name, "epochs": epochs,
                    "n_train_sentences": len(tr_s), "n_val_sentences": len(va_s),
                    "train_meetings": len(train_meetings), "val_meetings": len(val_meetings),
                    "history": history, "device": dev})
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    save_json(metrics, out_dir / "train_metrics.json")
    log.info("classifier saved to %s | val F1(macro)=%.4f",
             out_dir, metrics["f1_macro"])
    return metrics


class _TorchDataset:
    def __init__(self, enc: dict):
        self.enc = enc

    def __len__(self):
        return len(self.enc["labels"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}
