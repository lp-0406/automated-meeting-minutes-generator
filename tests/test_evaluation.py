import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.evaluation import (action_item_metrics, bertscore, classification_metrics,
                            rouge_scores, word_error_rate, write_results)


def test_rouge_perfect_match():
    r = rouge_scores(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert r["rouge1"] == 1.0 and r["rougeL"] == 1.0 and r["n_evaluated"] == 1


def test_rouge_mismatch_lower():
    a = rouge_scores(["the cat sat"], ["completely different words here"])["rouge1"]
    b = rouge_scores(["the cat sat"], ["the cat sat"])["rouge1"]
    assert a < b


def test_rouge_skips_empty_pairs():
    r = rouge_scores(["", "hello world"], ["ref", "hello world"])
    assert r["n_evaluated"] == 1


def test_rouge_length_mismatch_raises():
    with pytest.raises(ValueError):
        rouge_scores(["a"], ["a", "b"])


def test_classification_metrics_values():
    m = classification_metrics([0, 1, 1, 2, 0], [0, 1, 2, 2, 0])
    assert set(m["per_class"]) == {"0", "1", "2"}
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 1, 1], [0, 0, 1]]
    assert 0.0 <= m["f1_macro"] <= 1.0 and m["accuracy"] == 0.8


def test_classification_metrics_empty():
    m = classification_metrics([], [])
    assert m["f1_macro"] is None


def test_action_item_metrics_exact_match():
    preds = [{"action": "Send the report by Friday"},
             {"action": "Invent an unrelated task"}]
    refs = [{"action": "send the report by Friday", "owner": "Bob",
             "deadline": "by Friday"}]
    m = action_item_metrics(preds, refs)
    assert m["tp"] == 1 and m["precision"] == 0.5 and m["recall"] == 1.0
    assert m["owner_n_evaluated"] == 1
    assert m["owner_exact_match"] in (0.0, 1.0)     # depends on predicted owner


def test_action_owner_deadline_exact_match():
    preds = [{"action": "Send report", "owner": "Bob", "deadline": "by Friday"}]
    refs = [{"action": "Send report", "owner": "Bob", "deadline": "by Friday"}]
    m = action_item_metrics(preds, refs)
    assert m["f1"] == 1.0
    assert m["owner_exact_match"] == 1.0
    assert m["deadline_exact_match"] == 1.0


def test_action_metrics_empty():
    m = action_item_metrics([], [])
    assert m["f1"] is None


def test_wer_perfect_and_broken():
    assert word_error_rate("a b c d", "a b c d") == 0.0
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("a b", "") == 1.0


def test_write_results(tmp_path):
    rows = [{"model": "tfidf", "rouge1": 0.3, "extra": {"a": 1}},
            {"model": "textrank", "rouge1": 0.32, "extra": {"a": 2}}]
    csv_p, json_p = write_results(rows, tmp_path)
    assert csv_p.exists() and json_p.exists()
    import json
    assert len(json.loads(json_p.read_text(encoding="utf-8"))) == 2
    assert "model" in csv_p.read_text(encoding="utf-8").splitlines()[0]


def test_bertscore_import_error_or_run():
    try:
        out = bertscore(["hi there"], ["hi there"])
    except ImportError:
        pytest.skip("bert_score not installed")
    assert out["bertscore_f1"] is None or 0.0 <= out["bertscore_f1"] <= 1.0
