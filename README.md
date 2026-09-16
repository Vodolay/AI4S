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

## Layout

See [PLAN.md](PLAN.md). `src/neuromps/{fetch,features,qc,models,twin}`, tests in `tests/`,
all report numbers in `outputs/metrics.json`.
