# CLAUDE.md — NeuroMPS project rules

You are building an entry for the AI4S Open Innovation: AI + Organ-on-a-Chip challenge (Kaggle / 5th Pazhou Algorithm Competition). Read PLAN.md for the phase schedule and PROGRESS.md for current state before doing anything.

## Project in one paragraph
Two layers: (A) a standardized, provenance-tracked dataset of neural MEA/chip recordings assembled from public sources, plus a rat→hPSC transfer study; (B) a Brian2 spiking-network digital twin of an asymmetric two-compartment neural chip, calibrated on (A), with `sbi` inference recovering connectivity and drug effects from MEA features. Deliverables: public repo, one-command reproduction, Streamlit demo, technical report, Kaggle Writeup, ≤5-min video.

## Non-negotiable rules
- Research decision-support only. Never write efficacy, safety, diagnostic, or clinical claims anywhere (code comments, docs, app text).
- Only use data with an explicit license permitting reuse. Record source URL, license, SHA256, and download time in `data/raw/MANIFEST.json`. Never scrape gated content.
- Every number that could appear in the report must be produced by a script under `src/` and written to `outputs/metrics.json`. Notebooks are exploration only.
- `features/mea.py` is the single MEA feature extractor. The digital twin must call the same function. Do not fork it.
- No synthetic data presented as real. Simulated data lives under `outputs/twin/` and is labeled as simulated in every filename and figure title.
- Pin dependencies in `pyproject.toml`. Prefer numpy/pandas/scikit-learn/brian2/sbi/streamlit. Ask before adding anything else.
- `python run_all.py --quick` must finish in under 10 minutes on CPU and regenerate all figures and metrics. Keep a `--full` path for the complete run.

## How to work
- At session start: state the current phase and gate from PLAN.md, list what exists, propose the next 3 concrete steps, then wait for approval.
- Write tests first for: feature extraction, QC flags, provenance hashing, MEA simulator.
- After running anything, paste the actual output. Never summarize a result you did not observe.
- When a result looks good, name one assumption that could make it wrong and add a check for it.
- One phase per branch; open a PR with a summary that links each claim to an output file.
- At session end: update `PROGRESS.md` (done / blocked / next first task).

## Style
- Python 3.11, type hints, docstrings on public functions, `ruff` clean.
- Small modules; functions under ~50 lines.
- Figures: matplotlib, colorblind-safe palette, titled with what is shown and whether data is real or simulated.
