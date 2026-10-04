"""Decision / action / follow-up extraction (Phase 6).

Extraction combines THREE signals and never invents content:

1. DistilBERT probabilities (when a fine-tuned checkpoint exists)
2. deterministic decision / action / follow-up cue patterns
3. spaCy for PERSON entities and DATE/TIME deadline phrases

Every extracted item keeps provenance: source sentence, speaker, timestamp,
confidence and which signal fired — so a human can audit it.

Hard rule: any field that cannot be reliably determined is "Not specified".
No owner, date, deadline, decision or meeting fact is ever fabricated.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from .preprocessing import Sentence

NOT_SPECIFIED = "Not specified"

# --------------------------------------------------------------------------- cues
# Normalised once (lowercased) before matching. Deliberately conservative:
# patterns require an explicit commitment/decision marker.
DECISION_CUES = [
    r"\bwe\s+(?:have\s+)?decided\b", r"\bdecided\s+to\b", r"\bdecision\s+(?:is|was)\b",
    r"\bwe\s+(?:have\s+)?agreed\b", r"\bagreed\s+to\b", r"\bagreed\s+that\b",
    r"\blet'?s\s+(?:go\s+with|use|stick\s+to|settle)\b", r"\bapproved\b", r"\bwe\s+will\s+use\b",
    r"\bfinal\s+decision\b", r"\bsettled\s+on\b", r"\bwe\s+will\s+go\s+with\b",
    r"\bconfirmed\s+that\b", r"\braised\s+to\s+approved\b", r"\bconsensus\b",
    r"\bit\s+was\s+decided\b", r"\bwe\s+chose\b", r"\bchosen\b",
]
ACTION_CUES = [
    r"\bi\s+will\b", r"\bi'?ll\b", r"\bwe\s+will\b", r"\bwe'?ll\b",
    r"\bcan\s+you\b", r"\bcould\s+you\b", r"\bplease\b", r"\bneeds?\s+to\b",
    r"\bneed\s+to\b", r"\b(?:we|you|i|he|she|they)\s+should\b",
    r"\bshould\s+(?:we|you|i|the)\b", r"\bfollow\s+up\b",
    r"\baction\s+item\b", r"\bby\s+(?:monday|tuesday|wednesday|thursday|friday|"
    r"tomorrow|today|next\s+week|eod|cob|end\s+of\s+(?:day|week|month)|\d{1,2}\s+\w+)\b",
    r"\bsend\s+(?:me|you|us|the)\b", r"\bschedule\b", r"\bassign(?:ed)?\s+(?:to|me|you)\b",
    r"\bI'?ll\s+(?:take|handle|send|draft|prepare|review|check)\b",
    r"\bresponsible\s+for\b", r"\bdeliver\b", r"\bdeadline\b", r"\bdue\s+(?:on|by)\b",
]
FOLLOWUP_CUES = [
    r"\bneed\s+to\s+follow\s+up\b", r"\bcarry\s+(?:over|forward)\b", r"\bdefer(?:red)?\b",
    r"\bpostpone(?:d)?\b", r"\bnext\s+meeting\b", r"\bpending\b", r"\btbd\b",
    r"\bwe\s+still\s+need\b", r"\bfor\s+future\s+(?:discussion|reference)\b",
    r"\bopen\s+(?:question|issue|item)s?\b", r"\bto\s+be\s+discussed\b",
]

DECISION_RE = [re.compile(p, re.IGNORECASE) for p in DECISION_CUES]
ACTION_RE = [re.compile(p, re.IGNORECASE) for p in ACTION_CUES]
FOLLOWUP_RE = [re.compile(p, re.IGNORECASE) for p in FOLLOWUP_CUES]
# Case-sensitive: third-person commitments ("Alice will circulate..."). IGNORECASE
# would match "weather will", so these deliberately require capitalised names.
ACTION_RE_CS = [
    re.compile(r"\b[A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15})?\s+will\s+\w+"),
    re.compile(r"\bThe\s+[a-z]+(?:\s+[a-z]+)?\s+will\s+\w+"),
]


def _hits(patterns: Sequence[re.Pattern], text: str) -> list[str]:
    return [m.group(0).strip().lower() for p in patterns for m in p.finditer(text)]


def rule_decision_hits(text: str) -> list[str]:
    return _hits(DECISION_RE, text)


def rule_action_hits(text: str) -> list[str]:
    return _hits(ACTION_RE, text) + _hits(ACTION_RE_CS, text)


def rule_followup_hits(text: str) -> list[str]:
    return _hits(FOLLOWUP_RE, text)


def weak_label(text: str) -> int | None:
    """Rule-based weak label: 1=decision, 2=action, 0=other.

    Used ONLY for training-data pre-labelling (documented as weak, never gold).
    Decision cues win ties because commitment phrases like "I will" often co-occur
    inside decision sentences ("we decided I will send the deck").
    """
    if rule_decision_hits(text):
        return 1
    if rule_action_hits(text):
        return 2
    return 0


# --------------------------------------------------------------------------- analysis
def analyze_sentences(sentences: Sequence[Sentence],
                      probs: Sequence[Sequence[float]] | None = None,
                      decision_threshold: float = 0.6,
                      action_threshold: float = 0.6) -> list[dict]:
    """Attach rule hits + (optional) model probabilities to each sentence.

    `probs` is an (n, 3) array aligned with `sentences` (columns = other/decision/action).
    Missing probabilities are recorded as None — rules still work.
    """
    out = []
    for i, s in enumerate(sentences):
        p = list(probs[i]) if probs is not None else None
        d_rules, a_rules, f_rules = (rule_decision_hits(s.text),
                                     rule_action_hits(s.text),
                                     rule_followup_hits(s.text))
        p_dec = p[1] if p else None
        p_act = p[2] if p else None
        model_decision = p_dec is not None and p_dec >= decision_threshold
        model_action = p_act is not None and p_act >= action_threshold
        # Combined decision score: rules are evidence, model is evidence; when the
        # model is absent only rules count (recorded in `signals`).
        signals = []
        if d_rules:
            signals.append("rule:decision")
        if a_rules:
            signals.append("rule:action")
        if f_rules:
            signals.append("rule:followup")
        if model_decision:
            signals.append("model:decision")
        if model_action:
            signals.append("model:action")
        is_decision = bool(d_rules) or model_decision
        is_action = (bool(a_rules) or model_action) and not is_decision
        if p is not None:
            # Model probability when available; rule-only evidence gets a fixed
            # neutral score (rules are binary, not calibrated probabilities).
            confidence = round(max(p_dec or 0.0, p_act or 0.0), 4)
        else:
            confidence = 0.5 if (d_rules or a_rules) else 0.0
        out.append({
            "index": i,
            "text": s.text,
            "speaker": s.speaker,
            "start": s.start,
            "end": s.end,
            "meeting_id": s.meeting_id,
            "p_other": p[0] if p else None,
            "p_decision": p_dec,
            "p_action": p_act,
            "decision_cues": d_rules,
            "action_cues": a_rules,
            "followup_cues": f_rules,
            "is_decision": is_decision,
            "is_action": is_action,
            "signals": signals,
            "confidence": confidence,
        })
    return out


# --------------------------------------------------------------------------- owners
_PERSON_RE = re.compile(r"\b([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15})?)\b")
_ROLE_RE = re.compile(
    r"\b(?:the\s+)?(project\s+manager|tech\s+lead|team\s+lead|designer|developer|"
    r"engineer|intern|analyst|secretary|chair|reviewer|owner|admin|marketing|"
    r"sales|frontend|backend)\b", re.IGNORECASE)
_ADDRESSEE_RE = re.compile(r"\b(?:can|could|would)\s+you\b", re.IGNORECASE)

_SPACY_NLP = None
_SPACY_TRIED = False


def _spacy():
    global _SPACY_NLP, _SPACY_TRIED
    if not _SPACY_TRIED:
        _SPACY_TRIED = True
        try:
            import spacy
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception:
            _SPACY_NLP = None
    return _SPACY_NLP


def extract_owner(item: dict, participants: Sequence[str] = ()) -> str:
    """Owner strategy (strict order, never invents):
    1. speaker when the sentence says 'I will ...'  (first-person commitment)
    2. explicit PERSON recognised by spaCy / a participant name
    3. explicit role phrase ('the project manager will ...')
    4. 'X will ...' where X is a capitalised name
    5. otherwise 'Not specified'
    """
    text, speaker = item["text"], item.get("speaker", "Unknown")
    if re.search(r"\bI\s+will\b|\bI'?ll\b", text, re.IGNORECASE):
        if speaker and speaker != "Unknown":
            return speaker
    # Named participants mentioned explicitly.
    for p in participants or ():
        if p and p != NOT_SPECIFIED and re.search(rf"\b{re.escape(p)}\b", text):
            return p
    nlp = _spacy()
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()
    if re.search(r"\byou\b|\bplease\b", text, re.IGNORECASE) and _ADDRESSEE_RE.search(text):
        return NOT_SPECIFIED          # addressee not identifiable from text alone
    m = re.match(r"\s*([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15})?)\s+(?:will|shall|is|to)\b",
                 text)
    if m and m.group(1).lower() not in {"the", "we", "i", "it", "this", "that"}:
        return m.group(1)
    role = _ROLE_RE.search(text)
    if role and re.search(r"\b(?:will|shall|is\s+responsible|to)\b", text, re.IGNORECASE):
        return role.group(1).strip().lower().replace("  ", " ").capitalize()
    return NOT_SPECIFIED


# --------------------------------------------------------------------------- deadlines
_DEADLINE_PATTERNS = [
    re.compile(r"\bby\s+(?:the\s+end\s+of\s+)?"
               r"(?:monday|tuesday|wednesday|thursday|friday|sunday|saturday|"
               r"tomorrow|today|tonight|next\s+week|next\s+month|eod|cob|cop|"
               r"end\s+of\s+(?:day|week|month)|"
               r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
               r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
               r"dec(?:ember)?(?:\s+\d{1,2}(?:st|nd|rd|th)?)?)\b", re.IGNORECASE),
    re.compile(r"\bdue\s+(?:on|by)\s+\S+(?:\s+\S+){0,2}", re.IGNORECASE),
    re.compile(r"\b(?:before|until|prior\s+to)\s+"
               r"(?:the\s+)?(?:end\s+of\s+(?:day|week|month)|monday|tuesday|"
               r"wednesday|thursday|friday|tomorrow|today|next\s+week)\b", re.IGNORECASE),
    re.compile(r"\bwithin\s+\d+\s+(?:business\s+)?days?\b", re.IGNORECASE),
    re.compile(r"\bin\s+\d+\s+(?:business\s+)?days?\b", re.IGNORECASE),
    re.compile(r"\bon\s+(?:monday|tuesday|wednesday|thursday|friday|"
               r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|jun(?:e)?|"
               r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
               r"dec(?:ember)?)\b", re.IGNORECASE),
]


def extract_deadline(item: dict) -> str:
    """Deadline from explicit phrases only: spaCy DATE/TIME first (longest span),
    then deterministic patterns. 'Not specified' when nothing explicit exists."""
    text = item["text"]
    nlp = _spacy()
    if nlp is not None:
        doc = nlp(text)
        cands = [e.text.strip() for e in doc.ents if e.label_ in {"DATE", "TIME"}]
        # Prefer phrases that read like deadlines.
        scored = sorted(cands, key=lambda c: (not bool(re.search(r"\bby\b|\bdue\b", c, re.I)),
                                              -len(c)))
        for c in scored:
            if re.search(r"\bby\b|\bdue\b", c, re.I) or len(c.split()) > 1:
                return c
    for pat in _DEADLINE_PATTERNS:
        m = pat.search(text)
        if m:
            return re.sub(r"\s+", " ", m.group(0)).strip()
    return NOT_SPECIFIED


# --------------------------------------------------------------------------- extractors
def extract_decisions(analyzed: Sequence[dict],
                      threshold: float = 0.6) -> list[dict]:
    """All sentences flagged as decisions, with provenance. Sorted by confidence."""
    items = []
    for a in analyzed:
        if not a["is_decision"]:
            continue
        if a["p_decision"] is not None and a["p_decision"] < threshold and not a["decision_cues"]:
            continue
        items.append({
            "text": a["text"],
            "speaker": a["speaker"] or NOT_SPECIFIED,
            "timestamp": _fmt_time(a["start"], a["end"]),
            "confidence": a["confidence"],
            "signals": a["signals"],
        })
    items.sort(key=lambda d: -d["confidence"])
    return items


def extract_actions(analyzed: Sequence[dict],
                    participants: Sequence[str] = (),
                    threshold: float = 0.6) -> list[dict]:
    """Action items with owner + deadline. Never invents either field."""
    items = []
    for a in analyzed:
        if not a["is_action"]:
            continue
        if a["p_action"] is not None and a["p_action"] < threshold and not a["action_cues"]:
            continue
        item = dict(a)
        items.append({
            "action": a["text"],
            "owner": extract_owner(a, participants),
            "deadline": extract_deadline(a),
            "speaker": a["speaker"] or NOT_SPECIFIED,
            "timestamp": _fmt_time(a["start"], a["end"]),
            "confidence": a["confidence"],
            "signals": a["signals"],
        })
    items.sort(key=lambda d: -d["confidence"])
    return items


def extract_followups(analyzed: Sequence[dict]) -> list[dict]:
    items = []
    for a in analyzed:
        if a["followup_cues"]:
            items.append({
                "text": a["text"],
                "speaker": a["speaker"] or NOT_SPECIFIED,
                "timestamp": _fmt_time(a["start"], a["end"]),
                "confidence": a["confidence"],
                "signals": a["signals"],
            })
    return items


def _fmt_time(start, end) -> str:
    if start is None:
        return NOT_SPECIFIED
    if end is None:
        return f"{start:.1f}s"
    return f"{start:.1f}s-{end:.1f}s"
