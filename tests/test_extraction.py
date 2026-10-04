import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction import (analyze_sentences, extract_actions, extract_deadline,
                            extract_decisions, extract_followups, extract_owner,
                            rule_decision_hits, weak_label)
from src.preprocessing import Sentence


def S(text, speaker="Unknown", start=None, end=None, i=0):
    return Sentence(text=text, speaker=speaker, meeting_id="m1",
                    start=start, end=end, turn_index=i)


def test_weak_labels():
    assert weak_label("We decided to go with the new vendor.") == 1
    assert weak_label("I will send the report by Friday.") == 2
    assert weak_label("The weather is nice today.") == 0
    assert weak_label("Please review the draft before Monday.") == 2


def test_decision_cues_detected():
    assert rule_decision_hits("We agreed that option A wins.")
    assert not rule_decision_hits("Nothing relevant here.")


def test_analyze_without_model_uses_rules():
    sents = [S("We decided to launch on Monday.", "Alice", 1.0, 3.0),
             S("Coffee tastes good.", "Bob")]
    out = analyze_sentences(sents, probs=None)
    assert out[0]["is_decision"] and not out[0]["is_action"]
    assert out[0]["confidence"] == 0.5            # rule-only neutral score
    assert out[0]["p_decision"] is None           # model absent is recorded honestly
    assert not out[1]["is_decision"] and not out[1]["is_action"]


def test_analyze_with_model_probabilities():
    sents = [S("We settled on option B."), S("We settled on option C.")]
    probs = [[0.9, 0.05, 0.05], [0.05, 0.05, 0.90]]
    out = analyze_sentences(sents, probs, decision_threshold=0.6, action_threshold=0.6)
    assert out[0]["is_decision"]
    assert out[1]["is_decision"]                 # rule cue wins even if model disagrees
    assert "model:decision" not in out[1]["signals"]
    assert "rule:decision" in out[1]["signals"]


def test_extract_decisions_provenance():
    sents = [S("We decided to ship the beta.", "Alice", 12.5, 15.0)]
    dec = extract_decisions(analyze_sentences(sents))
    assert len(dec) == 1
    d = dec[0]
    assert d["speaker"] == "Alice" and d["timestamp"] == "12.5s-15.0s"
    assert d["confidence"] > 0 and d["signals"]


def test_owner_first_person_commitment():
    sents = [S("I will prepare the slides.", "Bob", 1.0, 2.0)]
    acts = extract_actions(analyze_sentences(sents), participants=["Bob", "Alice"])
    assert acts[0]["owner"] == "Bob"


def test_owner_named_participant():
    sents = [S("Alice will circulate the budget.", "Carol")]
    acts = extract_actions(analyze_sentences(sents), participants=["Alice", "Bob"])
    assert acts[0]["owner"] == "Alice"


def test_owner_not_specified_when_unknown():
    sents = [S("Can you please share the notes?", "Unknown")]
    acts = extract_actions(analyze_sentences(sents), participants=["Alice"])
    assert acts[0]["owner"] == "Not specified"


def test_deadline_extraction_and_not_specified():
    sents = [S("Send the report by Friday.", "Bob"),
             S("We should look into it sometime.", "Bob")]
    acts = extract_actions(analyze_sentences(sents))
    assert acts[0]["deadline"].lower() == "by friday"
    assert acts[1]["deadline"] == "Not specified"


def test_extract_deadline_helper():
    assert extract_deadline({"text": "Finish this by next week."}) != "Not specified"
    assert extract_deadline({"text": "Finish this eventually."}) == "Not specified"


def test_extract_followups():
    sents = [S("Let us carry over the pricing discussion to the next meeting.", "Alice"),
             S("It is sunny.", "Alice")]
    out = extract_followups(analyze_sentences(sents))
    assert len(out) == 1 and "carry over" in out[0]["text"]


def test_never_invents_fields():
    sents = [S("We discussed laptops.", "Unknown")]
    an = analyze_sentences(sents)
    assert extract_decisions(an) == []
    assert extract_actions(an) == []
