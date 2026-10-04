import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import (LSTMBaseline, run_extractive_baseline,
                           text_rank_summarize, tfidf_summarize)

DOC = ("The team met on Monday to discuss the product launch. "
       "Decisions were made about the marketing budget. "
       "The engineering group reported that the new API is almost ready. "
       "They agreed that testing should finish this week. "
       "The design team presented the new dashboard mockups. "
       "Budget concerns were raised by the finance representative. "
       "It was decided to postpone the beta until next month. "
       "The sales team asked for better reporting tools. ") * 3


def test_tfidf_returns_requested_sentences():
    out = tfidf_summarize(DOC, n_sentences=3)
    assert 15 <= len(out.split()) <= 200
    assert len(out) > 0


def test_textrank_returns_requested_sentences():
    out = text_rank_summarize(DOC, n_sentences=3)
    assert len(out) > 0


def test_short_text_returns_all():
    s = "One two three. Four five six."
    assert tfidf_summarize(s, 5) == s
    assert text_rank_summarize(s, 5) == s


def test_run_extractive_dispatch():
    assert run_extractive_baseline(DOC, "tfidf", 2)
    assert run_extractive_baseline(DOC, "textrank", 2)


def test_lstm_train_predict_save(tmp_path):
    pairs = [
        ("we decided to launch on friday i will send the deck by monday "
         "please review the budget", "decided to launch friday deck by monday"),
        ("the weather is nice and the team discussed lunch plans today "
         "nothing important happened at all here", "discussed lunch nothing important"),
        ("i will follow up with the vendor and send the contract tomorrow "
         "the vendor agreed to the terms", "follow up with vendor send contract"),
    ]
    base = LSTMBaseline(emb_dim=16, hid_dim=32, src_max_len=40, tgt_max_len=15)
    hist = base.fit(pairs, epochs=1, batch_size=2, lr=1e-3)
    assert hist["n_train_pairs"] == 3 and hist["vocab_size"] > 4
    assert "train_loss" in hist["history"][0]
    out = base.summarize(pairs[0][0])
    assert isinstance(out, str)
    path = base.save(tmp_path / "lstm.pt")
    reloaded = LSTMBaseline.load(path)
    assert reloaded.summarize(pairs[1][0]) is not None
