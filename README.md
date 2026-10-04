# Automated Meeting Minutes Generator

A college-level NLP mini project that turns a meeting recording (or transcript) into **draft** structured
minutes: executive summary, key discussion points, decisions, action items (owner, deadline) and follow-ups.
It uses only free, open-source pretrained models and is designed to run on a free Colab GPU (with CPU fallback).

> **Status: Phase 1 of 10 complete (repository and environment).** Nothing has been evaluated yet; this
> README contains **no results**. Results appear only after experiments are run (see *Results*).

## Overview
Pipeline: Audio → Whisper → transcript → preprocessing → turn handling → chunking → DistilBERT sentence
classifier (decision / action / other) → DistilBART hierarchical summarisation → spaCy + rules extraction →
Markdown minutes. Any field that cannot be reliably determined is output as **"Not specified"**.
The output is a *drafting assistant for human review*, not an authoritative record.

## Features
Implemented now: configuration, seeding, GPU/VRAM detection, automatic model-size policy, logging, resumability helpers, unit tests, smoke test.
Planned (phases 2–10): dataset download and validation, preprocessing, Whisper ASR, summarisation, extraction, baselines (TF-IDF, TextRank, LSTM), evaluation, demo, report. Each is marked complete here only after it has been run and tested.

## Architecture
See `reports/methodology.md` (written in Phase 10) and the pipeline above.

## Dataset
Not downloaded yet. Verified sources, licences and subset sizes are recorded in `data/README.md` (Phase 2).

## Installation
```bash
git clone <your-repo-url> meeting-minutes-generator
cd meeting-minutes-generator
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/smoke_test.py --strict
```

## Google Colab Setup
Open `notebooks/00_colab_setup.ipynb` in Colab, choose *Runtime → Change runtime type → GPU*, and run the cells.
Colab already provides torch/numpy/pandas/scikit-learn; `requirements.txt` deliberately does not pin them.

## GPU Requirements
A free Colab GPU (typically 15–16 GB VRAM) is sufficient; CPU also works, more slowly. With `device: auto`
the project picks CUDA when available and downgrades model sizes on CPU or low VRAM (`configs/config.yaml`).
Large models (`bart-large`, Whisper medium/large) are never selected automatically.

## Quick Start
```bash
python scripts/smoke_test.py          # environment check
python -m pytest -q                   # unit tests
```

## Training / Evaluation / End-to-End Demo / Results
Not available yet (Phases 5–9). Smoke-test results and final-experiment results will be reported separately,
and only from files in `outputs/metrics/`. Until then: **Not yet evaluated.**

## Project Structure
`configs/` settings · `src/` library code · `scripts/` command-line entry points · `notebooks/` Colab notebooks ·
`tests/` unit tests · `data/` (git-ignored contents) · `outputs/` generated artefacts · `models/` checkpoints ·
`reports/` methodology and case study.

## Privacy
Meeting recordings can be confidential. Obtain participant consent; do not upload confidential meetings to public
services; never commit recordings or private transcripts; anonymise personal information when appropriate; use
public or synthetic meetings for demonstrations; treat generated minutes as drafts and verify every decision and
action item manually.

## Limitations
ASR errors propagate; no speaker diarization in the core project (speaker = "Unknown" for raw audio); summarisers
can omit or distort content; owners and deadlines are often implicit. Details will be quantified in the report.

## Future Work
Speaker diarization, multilingual meetings, deadline normalisation, confidence calibration, human-in-the-loop correction.

## License
MIT (see `LICENSE`).

## Citation
Dataset and model citations are added in Phase 10 (`reports/final_case_study.md`).
