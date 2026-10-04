"""One-off dataset-discovery probe (Phase 2). Verifies HF dataset identifiers, configs, splits and
example counts WITHOUT downloading full corpora (uses streaming where possible).

Run: python scripts/probe_datasets.py
Output: printed report; used to fill in data/README.md with verified facts.
"""
from __future__ import annotations


def _probe(name: str, fn) -> None:
    print(f"--- {name}")
    try:
        fn()
    except Exception as e:
        print(f"    FAIL: {type(e).__name__}: {str(e)[:400]}")
    print()


def qmsum_github():
    # QMSum official data lives in the GitHub repo (data/ALL/jsonl).
    import json, urllib.request
    url = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/test"
    with urllib.request.urlopen(url, timeout=60) as r:
        lines = r.read().decode("utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    print(f"    rows={len(rows)} keys={sorted(rows[0].keys())}")
    print(f"    n_general_queries={len(rows[0]['general_query_list'])} "
          f"n_turns={len(rows[0]['meeting_transcripts'])}")


def scrolls_qmsum():
    from datasets import load_dataset
    for split in ("validation", "test"):
        ds = load_dataset("tau/scrolls", "qmsum", split=split, streaming=True)
        first = next(iter(ds))
        print(f"    {split}: first keys={sorted(first.keys())}")


def qmsum_cleaned():
    from datasets import get_dataset_config_names
    print(f"    configs={get_dataset_config_names('pszemraj/qmsum-cleaned')}")


def ami():
    from datasets import get_dataset_config_names, load_dataset
    configs = get_dataset_config_names("edinburghcstr/ami")
    print(f"    configs={configs}")
    cfg = "only_words" if "only_words" in configs else configs[0]
    ds = load_dataset("edinburghcstr/ami", cfg, split="validation", streaming=True)
    first = next(iter(ds))
    print(f"    {cfg} validation first keys={sorted(first.keys())}")


def dialogsum():
    from datasets import load_dataset
    ds = load_dataset("knkarthick/dialogsum", split="test")
    print(f"    test size={len(ds)} cols={ds.column_names}")
    print(f"    sample dialogue[:120]={ds[0]['dialogue'][:120]!r}")


def samsum():
    from datasets import load_dataset
    ds = load_dataset("Samsung/samsum", split="test")
    print(f"    test size={len(ds)} cols={ds.column_names}")


def meetingbank():
    from datasets import get_dataset_config_names
    print(f"    configs={get_dataset_config_names('lytang/MeetingBank-transcript')}")


if __name__ == "__main__":
    _probe("QMSum (official GitHub, ALL/jsonl)", qmsum_github)
    _probe("tau/scrolls qmsum (HF mirror)", scrolls_qmsum)
    _probe("pszemraj/qmsum-cleaned", qmsum_cleaned)
    _probe("edinburghcstr/ami", ami)
    _probe("knkarthick/dialogsum", dialogsum)
    _probe("Samsung/samsum", samsum)
    _probe("lytang/MeetingBank-transcript", meetingbank)
