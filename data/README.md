# Data

Dataset discovery was performed **2026-10-04** by querying the Hugging Face API,
the GitHub API and actually loading each candidate with the `datasets` library
(`scripts/probe_datasets.py`). Identifiers below were verified working at that
date; anything unverified is marked as such.

## Verified sources

| Dataset | Source / URL | Licence | Purpose | Subset used | Examples actually downloaded | Preprocessing |
|---|---|---|---|---|---|---|
| **QMSum** (primary) | Official GitHub: https://github.com/Yale-LILY/QMSum → `data/ALL/jsonl/{train,val,test}.jsonl` (11.9 / 2.7 / 2.7 MB). **No official HF mirror exists** (verified: `yale-nlp/QMSum` returns 401/absent). | MIT (repo `LICENSE`, verified via GitHub API) | main summarisation benchmark; speaker turns; reference summaries | all 3 splits downloaded; **first N meetings** per run (config: `smoke_test_meetings: 5`, `initial_eval_meetings: 30`) | 3 jsonl files, 17.4 MB total (`download_manifest.json` records exact bytes) | speaker-turn parsing → whitespace/filler cleaning (cue words preserved) → short-turn merging → sentence segmentation → token-aware chunking |
| **AMI** | HF: `edinburghcstr/ami` (configs `ihm`, `sdm`) | CC-BY-4.0 (HF card) | meeting structure; speaker/timestamped turns; optional ASR demo | **streaming, text columns only, 2 meetings** — the full download is **~15 GB**, so audio is *never* downloaded by default (`Audio(decode=False)` cast) | `n_meetings` from `--ami-meetings` (default 2); counts recorded in `download_manifest.json` | consecutive segments merged into turns (`speaker_id`, `begin_time`, `end_time`) |
| **DialogSum** (optional warm-up) | HF: `knkarthick/dialogsum` | CC-BY-NC-SA-4.0 (HF card) | optional fine-tuning warm-up | splits as served: train 12 460 / val 500 / test 1 500 (verified by loading) | downloaded on demand with `--with-dialogsum` (CSVs ~200 MB) | CSV export only (not used by the core pipeline) |
| **SAMSum** (optional) | Official `Samsung/samsum` was **NOT accessible** during verification (HTTP 401). Mirror `knkarthick/samsum` exists (CC-BY-NC-ND-4.0). | see left | warm-up only (not used) | not downloaded | 0 | — |
| **MeetingBank** (optional) | HF: `lytang/MeetingBank-transcript` (verified) | CC-BY-NC-SA-4.0 | extra long-document demo | **20 examples max** with `--with-meetingbank` (never the full corpus) | 0 by default | — |

Notes:

- **Why QMSum's official location changed:** older tutorials often cite a Hugging Face
  QMSum ID. The official dataset is distributed from the Yale-LILY GitHub repository;
  the HF identifiers that show up in search (`pszemraj/qmsum-cleaned`, `tau/scrolls`)
  are third-party conversions — `tau/scrolls` **no longer loads** with `datasets>=3`
  ("Dataset scripts are no longer supported"). We therefore download the official
  jsonl files directly — smaller (17 MB) and always current.
- `scripts/probe_datasets.py` re-runs all identifier checks and prints a report.
- **Storage:** datasets live in `data/raw/`, processed data in `data/processed/`,
  tiny synthetic demos in `data/samples/`. All contents are git-ignored except the
  lightweight text samples and this README. **Never commit audio or datasets.**
- **Optional Google Drive persistence (Colab):** mount Drive and point
  `paths.data_raw` / `paths.data_processed` in `configs/config.yaml` at a Drive
  folder so downloads survive session disconnects. This is optional.

## What is committed to Git

- `data/README.md`, `.gitkeep` files
- `data/samples/sample_meeting.txt` — a **synthetic** meeting (all names invented)
- `data/samples/manual_labels.json` — 30 sentences **hand-labelled by the project
  author** (0=other, 1=decision, 2=action); provenance: written and labelled by
  hand while building this repo. Used as a small human-validation set only —
  far too small to train on, and never described as a benchmark.

## Reproducing the downloads

```bash
python scripts/download_data.py                     # QMSum (~17 MB)
python scripts/download_data.py --with-dialogsum    # + DialogSum
python scripts/download_data.py --with-ami --ami-meetings 2   # AMI text subset
python scripts/download_data.py --all
```

Every run writes `data/raw/download_manifest.json` with URLs, licences, file
sizes and exact example counts actually downloaded.
