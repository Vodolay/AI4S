# NeuroMPS

A standardized, provenance-tracked dataset of public neural MEA recordings, and a spiking-network
digital twin of an asymmetric two-compartment neural chip calibrated on it.
Entry for the AI4S Open Innovation: AI + Organ-on-a-Chip challenge (5th Pazhou Algorithm Competition).

**Research decision-support only.** Nothing in this repository makes efficacy, safety,
diagnostic or clinical claims.

## Setup

```bash
pip install uv
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Reproduce

```bash
.venv/bin/python run_all.py --quick     # CI path, < 10 min on CPU
.venv/bin/python run_all.py --full      # complete run, larger downloads
```

Fetch data only:

```bash
.venv/bin/python -m neuromps.fetch.runner [--full] [--only kapucu2022 toxcast_mea]
```

Downloads land in `data/raw/<source>/`. Every file's URL, license, SHA256, size and download time are recorded in
`data/raw/MANIFEST.json`. Source decisions are in [data/SOURCES.md](data/SOURCES.md).

## Documents

| document | what it is |
|---|---|
| [report/TECHNICAL_REPORT.md](report/TECHNICAL_REPORT.md) | the full write-up: standard, asset, models, twin, results, limitations |
| [report/KAGGLE_WRITEUP.md](report/KAGGLE_WRITEUP.md) | the competition summary |
| [report/LAB_NOTEBOOK.md](report/LAB_NOTEBOOK.md) | every decision with its date and commit, what was abandoned, 33 limitations |
| [report/RESULTS_DRAFT.md](report/RESULTS_DRAFT.md) | Phase 1-2 results in detail |
| [CITATIONS.md](CITATIONS.md) / [THIRD_PARTY.md](THIRD_PARTY.md) | data sources and dependency licenses, both generated |

Numbers in these documents cite the `outputs/metrics.json` key that produced them, and

```bash
.venv/bin/python -m neuromps.report.verify
```

fails if any citation disagrees with the live metrics. `CITATIONS.md` and `THIRD_PARTY.md` are
regenerated with `python -m neuromps.report.attribution`.

## Layout

See [PLAN.md](PLAN.md). `src/neuromps/{fetch,features,qc,models,twin,report}`, tests in `tests/`,
all report numbers in `outputs/metrics.json`.
