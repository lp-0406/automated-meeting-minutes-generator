"""Baseline systems (Phase 7).

1. TF-IDF extractive summarisation  (centroid similarity ranking)
2. TextRank extractive summarisation (PageRank over a sentence-similarity graph,
   implemented with plain NumPy — no networkx dependency)
3. Small LSTM seq2seq summariser with additive attention — an academic baseline,
   intentionally tiny so it trains on free-CPU/free-GPU budgets in minutes.

The LSTM is NOT the production model; its limitations are documented in
reports/experiment_results.md (short vocab, exposure bias, no pretraining).
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch

from .preprocessing import sentencize


# --------------------------------------------------------------------------- shared
def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in sentencize(text) if len(s.split()) >= 3]


# --------------------------------------------------------------------------- TF-IDF
def tfidf_summarize(text: str, n_sentences: int = 5) -> str:
    """Rank sentences by cosine similarity to the TF-IDF centroid."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    sents = _split_sentences(text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sents).toarray()
    centroid = X.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centroid) or 1.0
    scores = (X @ centroid.T).ravel() / (np.linalg.norm(X, axis=1) + 1e-9) / norm
    top = np.argsort(scores)[::-1][:n_sentences]
    return " ".join(sents[i] for i in sorted(top))


# --------------------------------------------------------------------------- TextRank
def text_rank_summarize(text: str, n_sentences: int = 5, damping: float = 0.85,
                        iters: int = 50, tol: float = 1e-6) -> str:
    """PageRank over cosine-similarity graph of TF-IDF sentence vectors."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    sents = _split_sentences(text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    X = TfidfVectorizer(stop_words="english").fit_transform(sents).toarray()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    S = (X / norms) @ (X / norms).T
    np.fill_diagonal(S, 0.0)
    row_sum = S.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    S = S / row_sum
    n = len(sents)
    score = np.full(n, 1.0 / n)
    for _ in range(iters):
        new = (1 - damping) / n + damping * S.T @ score
        if np.linalg.norm(new - score) < tol:
            score = new
            break
        score = new
    top = np.argsort(score)[::-1][:n_sentences]
    return " ".join(sents[i] for i in sorted(top))


# --------------------------------------------------------------------------- LSTM
PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SPECIALS = [PAD, UNK, SOS, EOS]


class Vocab:
    def __init__(self, itos: Sequence[str]):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: Sequence[str], max_size: int = 20000, min_freq: int = 2) -> "Vocab":
        from collections import Counter
        cnt = Counter(w.lower() for t in texts for w in t.split())
        words = [w for w, c in cnt.most_common() if c >= min_freq][: max_size - len(SPECIALS)]
        return cls(SPECIALS + words)

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        ids = [self.stoi.get(w.lower(), self.stoi[UNK]) for w in text.split()]
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        words = [self.itos[i] for i in ids
                 if i not in (self.stoi[PAD], self.stoi[SOS], self.stoi[EOS], self.stoi[UNK])]
        return " ".join(words)


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hid_dim: int = 128):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        emb = self.embedding(src)
        outputs, (h, c) = self.lstm(emb)
        return outputs, h, c


class Attention(torch.nn.Module):
    """Additive (Bahdanau) attention."""

    def __init__(self, hid_dim: int, enc_dim: int):
        super().__init__()
        self.w = torch.nn.Linear(hid_dim + enc_dim, hid_dim)
        self.v = torch.nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, enc_outputs, mask):
        # hidden: (B, H) ; enc_outputs: (B, S, E)
        energy = torch.tanh(self.w(torch.cat([
            hidden.unsqueeze(1).expand(-1, enc_outputs.size(1), -1), enc_outputs], dim=-1)))
        scores = self.v(energy).squeeze(-1)          # (B, S)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hid_dim: int = 128,
                 enc_dim: int | None = None):
        super().__init__()
        enc_dim = enc_dim or hid_dim
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = Attention(hid_dim, enc_dim)
        self.lstm = torch.nn.LSTM(emb_dim + enc_dim, hid_dim, batch_first=True)
        self.out = torch.nn.Linear(hid_dim, vocab_size)

    def forward(self, token, hidden, cell, enc_outputs, mask):
        context, attn = self.attention(hidden[-1], enc_outputs, mask)
        emb = self.embedding(token).unsqueeze(1)     # (B, 1, E)
        inp = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        out, (hidden, cell) = self.lstm(inp, (hidden, cell))
        return self.out(out.squeeze(1)), hidden, cell, attn


class LSTMSeq2Seq(torch.nn.Module):
    """Tiny seq2seq: 1-layer LSTM encoder + Bahdanau attention + LSTM decoder.
    Defaults (64 emb / 128 hidden) are deliberately small for free compute."""

    def __init__(self, src_vocab: int, tgt_vocab: int, emb_dim: int = 64,
                 hid_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb_dim, hid_dim)
        self.decoder = Decoder(tgt_vocab, emb_dim, hid_dim)

    def forward(self, src, tgt, teacher_forcing: float = 0.5):
        B, S = src.shape
        enc_out, h, c = self.encoder(src)
        mask = (src != 0).float()
        token = tgt[:, 0].clone()                   # SOS assumed at index 0 of tgt
        outputs = []
        for t in range(1, tgt.size(1)):
            logits, h, c, _ = self.decoder(token, h, c, enc_out, mask)
            outputs.append(logits)
            top = logits.argmax(-1)
            token = tgt[:, t] if random.random() < teacher_forcing else top
        return torch.stack(outputs, dim=1)          # (B, T-1, V)

    @torch.no_grad()
    def greedy_decode(self, src, max_len: int = 60):
        self.eval()
        B = src.size(0)
        enc_out, h, c = self.encoder(src)
        mask = (src != 0).float()
        token = torch.full((B,), 2, dtype=torch.long, device=src.device)  # SOS
        result = []
        for _ in range(max_len):
            logits, h, c, _ = self.decoder(token, h, c, enc_out, mask)
            token = logits.argmax(-1)
            result.append(token)
            if bool((token == 3).all()):            # EOS
                break
        return torch.stack(result, dim=1)           # (B, L)


class LSTMBaseline:
    """Train / save / load / summarise wrapper around LSTMSeq2Seq.

    Training data = (document, summary) pairs of plain text (QMSum or DialogSum).
    """

    def __init__(self, emb_dim: int = 64, hid_dim: int = 128,
                 src_max_len: int = 120, tgt_max_len: int = 45,
                 min_freq: int = 2, vocab_size: int = 20000):
        self.emb_dim, self.hid_dim = emb_dim, hid_dim
        self.src_max_len, self.tgt_max_len = src_max_len, tgt_max_len
        self.min_freq, self.vocab_size = min_freq, vocab_size
        self.vocab: Vocab | None = None
        self.model: LSTMSeq2Seq | None = None
        self.device = "cpu"

    # ------------------------------------------------------------------ data
    def _pairs_to_tensors(self, pairs):
        src_ids, tgt_ids = [], []
        for doc, summ in pairs:
            s = self.vocab.encode(doc, self.src_max_len)
            t = [self.vocab.stoi["<sos>"]] + self.vocab.encode(summ, self.tgt_max_len) \
                + [self.vocab.stoi["<eos>"]]
            src_ids.append(s or [self.vocab.stoi[UNK]])
            tgt_ids.append(t)
        pad = self.vocab.stoi[PAD]
        src = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in src_ids], batch_first=True, padding_value=pad)
        tgt = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in tgt_ids], batch_first=True, padding_value=pad)
        return src, tgt

    # ------------------------------------------------------------------ training
    def fit(self, pairs: Sequence[tuple[str, str]], val_pairs: Sequence[tuple[str, str]] = (),
            epochs: int = 3, batch_size: int = 16, lr: float = 1e-3,
            device: str = "cpu", seed: int = 42, logger=None) -> dict:
        """Train on (document, summary) pairs. Returns history dict."""
        import torch.nn.functional as F
        torch.manual_seed(seed)
        random.seed(seed)
        self.device = device
        if not pairs:
            raise ValueError("LSTM baseline needs at least one training pair")
        self.vocab = Vocab.build([d + " " + s for d, s in pairs],
                                 max_size=self.vocab_size, min_freq=self.min_freq)
        self.model = LSTMSeq2Seq(len(self.vocab), len(self.vocab),
                                 self.emb_dim, self.hid_dim).to(device)
        src, tgt = self._pairs_to_tensors(pairs)
        n = src.size(0)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        history = []
        pad_idx = self.vocab.stoi[PAD]
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            tot, nb = 0.0, 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                s, t = src[idx].to(device), tgt[idx].to(device)
                logits = self.model(s, t, teacher_forcing=0.5)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       t[:, 1:].reshape(-1), ignore_index=pad_idx)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += float(loss.item())
                nb += 1
            history.append({"epoch": ep + 1, "train_loss": round(tot / max(nb, 1), 4)})
        result = {"history": history, "n_train_pairs": n,
                  "vocab_size": len(self.vocab), "device": device}
        if logger:
            logger.info("LSTM baseline: %s", result)
        return result

    # ------------------------------------------------------------------ inference
    def summarize(self, text: str) -> str:
        if self.model is None or self.vocab is None:
            raise RuntimeError("LSTM baseline not trained/loaded")
        sents = _split_sentences(text)
        if len(sents) > self.src_max_len // 6:
            # Feed the lead sentences only — the baseline is length-limited by design.
            text = " ".join(sents[: self.src_max_len // 6])
        src, _ = self._pairs_to_tensors([(text, "")])
        ids = self.model.greedy_decode(src.to(self.device), max_len=self.tgt_max_len)[0]
        return self.vocab.decode(ids.tolist()).strip()

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "vocab": self.vocab.itos,
                    "emb_dim": self.emb_dim, "hid_dim": self.hid_dim,
                    "src_max_len": self.src_max_len, "tgt_max_len": self.tgt_max_len},
                   path)
        return path

    @classmethod
    def load(cls, path, device: str = "cpu") -> "LSTMBaseline":
        state = torch.load(path, map_location=device)
        obj = cls(emb_dim=state["emb_dim"], hid_dim=state["hid_dim"],
                  src_max_len=state["src_max_len"], tgt_max_len=state["tgt_max_len"])
        obj.vocab = Vocab(state["vocab"])
        obj.model = LSTMSeq2Seq(len(obj.vocab), len(obj.vocab),
                                obj.emb_dim, obj.hid_dim).to(device)
        obj.model.load_state_dict(state["model"])
        obj.model.eval()
        obj.device = device
        return obj


def run_extractive_baseline(text: str, method: str = "tfidf",
                            n_sentences: int = 5) -> str:
    method = method.lower()
    if method == "tfidf":
        return tfidf_summarize(text, n_sentences)
    if method == "textrank":
        return text_rank_summarize(text, n_sentences)
    raise ValueError(f"unknown extractive method: {method}")
