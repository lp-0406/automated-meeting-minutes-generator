"""Dataset download + loading (Phases 2–3).

Verified dataset sources (2026-10-04, see data/README.md):

- QMSum (primary summarisation benchmark)
  Official repo: https://github.com/Yale-LILY/QMSum (MIT licence)
  Files: data/ALL/jsonl/{train,val,test}.jsonl (~17 MB total, no HF token needed)
- AMI (meeting structure / optional audio demo)
  HF: edinburghcstr/ami (CC-BY-4.0). FULL download is ~15 GB, so we only ever
  stream a few meetings' text fields; audio is never downloaded by default.
- DialogSum (optional warm-up)
  HF: knkarthick/dialogsum (CC-BY-NC-SA-4.0), 12460/500/1500 splits.
- SAMSum: official `Samsung/samsum` was NOT accessible during verification
  (HTTP 401); mirror `knkarthick/samsum` (CC-BY-NC-ND-4.0) exists — optional only.

All meetings are normalised to one schema:

    {
      "meeting_id": str,
      "title": str,
      "date": str,                 # "Not specified" when absent
      "participants": [str],       # unique speakers, or ["Not specified"]
      "turns": [{"speaker","text","start","end"}],
      "reference_summary": str,    # "" when no reference exists
      "source": str
    }
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

from .preprocessing import Turn
from .utils import load_json, save_json

NOT_SPECIFIED = "Not specified"

QMSUM_RAW = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/{}.jsonl"
QMSUM_SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- download
def download_qmsum(raw_dir: str | Path, splits: Sequence[str] = QMSUM_SPLITS,
                   force: bool = False) -> dict:
    """Download QMSum jsonl splits from the official GitHub repo. Resumable: existing
    non-empty files are skipped unless force=True. Returns {split: path}."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    for split in splits:
        dest = raw_dir / f"qmsum_{split}.jsonl"
        if dest.exists() and dest.stat().st_size > 0 and not force:
            out[split] = dest
            continue
        url = QMSUM_RAW.format(split)
        tmp = dest.with_suffix(".jsonl.part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
        tmp.replace(dest)
        out[split] = dest
    return out


def download_dialogsum(raw_dir: str | Path, force: bool = False) -> dict:
    """Download DialogSum (optional warm-up). Uses HF datasets; small (~200 MB csv)."""
    from datasets import load_dataset
    raw_dir = Path(raw_dir)
    marker = raw_dir / "dialogsum.done"
    if marker.exists() and not force:
        return {"status": "cached", "path": str(raw_dir)}
    ds = load_dataset("knkarthick/dialogsum")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split, table in ds.items():
        table.to_csv(raw_dir / f"dialogsum_{split}.csv", index=False)
    n = sum(len(t) for t in ds.values())
    save_json({"dataset": "knkarthick/dialogsum",
               "splits": {k: len(v) for k, v in ds.items()}, "total": n},
              raw_dir / "dialogsum_manifest.json")
    marker.write_text("ok", encoding="utf-8")
    return {"status": "downloaded", "total": n}


# --------------------------------------------------------------------------- QMSum
def _load_jsonl(path: str | Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def qmsum_row_to_meeting(row: dict, split: str, idx: int) -> dict:
    turns = [Turn(speaker=(t.get("speaker") or "Unknown").strip(),
                  text=(t.get("content") or "").strip())
             for t in row.get("meeting_transcripts", [])]
    turns = [t for t in turns if t.text]
    speakers = []
    for t in turns:
        if t.speaker and t.speaker not in speakers:
            speakers.append(t.speaker)
    # Prefer the general "summarize the whole meeting" answer as reference.
    ref = ""
    for q in row.get("general_query_list", []):
        if q.get("answer"):
            ref = q["answer"].strip()
            break
    topic = (row.get("topic_list") or [{}])[0].get("topic", "")
    return {
        "meeting_id": f"qmsum_{split}_{idx:03d}",
        "title": topic or NOT_SPECIFIED,
        "date": NOT_SPECIFIED,
        "participants": speakers or [NOT_SPECIFIED],
        "turns": [t.to_dict() for t in turns],
        "reference_summary": ref,
        "reference_query": "Summarize the whole meeting.",
        "source": "qmsum",
        "split": split,
    }


def load_qmsum(path: str | Path, split: str = "val", limit: int | None = None) -> list[dict]:
    """Load QMSum jsonl into the unified meeting schema (limit = first N meetings)."""
    rows = _load_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return [qmsum_row_to_meeting(r, split, i) for i, r in enumerate(rows)]


# --------------------------------------------------------------------------- AMI
def load_ami_streaming(n_meetings: int = 3, config: str = "ihm",
                       split: str = "validation") -> list[dict]:
    """Stream a few AMI meetings' TEXT only from HF (never the ~15 GB audio).

    Speech segments arrive with speaker_id + begin/end times; consecutive segments
    from one speaker are merged into turns. Requires network, not a full download.
    """
    from datasets import Audio, load_dataset
    ds = load_dataset("edinburghcstr/ami", config, split=split, streaming=True)
    # Never decode/download audio: cast the audio column to raw bytes/paths only.
    # (Without this, datasets>=3 raises "install torchcodec" and would pull GBs.)
    ds = ds.cast_column("audio", Audio(decode=False))
    # Rows are grouped by meeting; collect text fields only (audio is never touched,
    # so no torchcodec/ffmpeg decoding and no multi-GB download happens).
    grouped: list[tuple[str, list]] = []
    current_id, current = None, []
    for row in ds:
        mid = row.get("meeting_id") or "ami_unknown"
        if mid != current_id:
            if current_id is not None:
                grouped.append((current_id, current))
                if len(grouped) >= n_meetings:
                    break
            current_id, current = mid, []
        current.append({
            "speaker": row.get("speaker_id") or "Unknown",
            "text": (row.get("text") or "").strip(),
            "start": float(row["begin_time"]) if row.get("begin_time") is not None else None,
            "end": float(row["end_time"]) if row.get("end_time") is not None else None,
        })
    if current and len(grouped) < n_meetings:
        grouped.append((current_id, current))
    out = []
    for mid, segs in grouped[:n_meetings]:
        turns = _merge_ami_segments(segs)
        speakers = []
        for t in turns:
            if t.speaker not in speakers:
                speakers.append(t.speaker)
        out.append({
            "meeting_id": f"ami_{mid}",
            "title": NOT_SPECIFIED,
            "date": NOT_SPECIFIED,
            "participants": speakers or [NOT_SPECIFIED],
            "turns": [t.to_dict() for t in turns],
            "reference_summary": "",       # AMI abstractive summaries not in this config
            "source": "ami",
            "split": split,
        })
    return out


def _merge_ami_segments(segs: Sequence[dict], max_gap_words: int = 3) -> list[Turn]:
    turns: list[Turn] = []
    for s in segs:
        if not s["text"]:
            continue
        if turns and turns[-1].speaker == s["speaker"]:
            turns[-1].text = f"{turns[-1].text} {s['text']}".strip()
            turns[-1].end = s["end"] if s["end"] is not None else turns[-1].end
        else:
            turns.append(Turn(speaker=s["speaker"], text=s["text"],
                              start=s["start"], end=s["end"]))
    return turns


# --------------------------------------------------------------------------- generic I/O
def load_transcript_file(path: str | Path, meeting_id: str | None = None) -> dict:
    """Load a .txt ('Speaker: line' format) or .json meeting file into the schema."""
    path = Path(path)
    mid = meeting_id or path.stem
    if path.suffix.lower() == ".json":
        data = load_json(path)
        if "turns" in data:                       # already unified schema
            data.setdefault("meeting_id", mid)
            return data
        if isinstance(data, list):                # ASR-style segment list
            turns = [{"speaker": "Unknown", "text": (d.get("text") or "").strip(),
                      "start": d.get("start"), "end": d.get("end")} for d in data]
            turns = [t for t in turns if t["text"]]
        else:
            raise ValueError(f"Unrecognised JSON transcript format: {path}")
    else:
        from .preprocessing import parse_transcript_text
        body = "\n".join(line for line in
                         path.read_text(encoding="utf-8").splitlines()
                         if not line.lstrip().startswith("#"))   # drop comment lines
        turns = [t.to_dict() for t in parse_transcript_text(body, mid)]
    speakers = []
    for t in turns:
        if t["speaker"] not in speakers:
            speakers.append(t["speaker"])
    return {
        "meeting_id": mid,
        "title": NOT_SPECIFIED,
        "date": NOT_SPECIFIED,
        "participants": speakers or [NOT_SPECIFIED],
        "turns": turns,
        "reference_summary": "",
        "source": "user_input",
    }


def save_meeting(meeting: dict, path: str | Path) -> Path:
    return save_json(meeting, path)


def load_meetings(path: str | Path) -> list[dict]:
    """Load a JSON list of unified-schema meetings (processed data)."""
    data = load_json(path)
    if isinstance(data, dict):
        data = [data]
    return data
