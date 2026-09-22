# NeuroMPS

A standardized, provenance-tracked dataset of public neural MEA recordings, and a spiking-network
digital twin of an asymmetric two-compartment neural chip calibrated on it.
Entry for the AI4S Open Innovation: AI + Organ-on-a-Chip challenge (5th Pazhou Algorithm
Competition), submission category **End-to-End System**.

**Research decision-support only.** Nothing in this repository makes efficacy, safety,
diagnostic or clinical claims.

## What it found

The two layers together are a falsification harness, and the results it produced are as much
negative as positive. Four confounds that published practice would have carried:

- **Coupling asymmetry between two chambers reports firing rate, not connectivity direction.** In
  the twin, drive moves the measure several times further than wiring does, and pre-whitening the
  correlogram makes that worse. On ten real four-chamber devices whose microtunnels are themselves
  instrumented — so the direction an axon fired is *measured*, not inferred — the chamber measure
  agrees with the axons on a fraction 0.419
  <!-- metrics.json: axonal_directionality.headline.sign_agreement.fraction --> of 31 pairs, which is
  chance.
- **Pseudo-replication manufactured a dose trend.** An apparent bicuculline response (ρ = −0.197,
  p = 0.036) dissolved once 113 recordings were recognised as coming from 16 devices; a microchannel
  geometry effect dissolved once 57 recordings were recognised as coming from 26 chips
  <!-- metrics.json: microchannel_directionality.n_chips -->.
- **In-sample leakage inflated a classifier.** Fitting the rate-residualiser outside the training
  fold moved an AUC from 0.9359 to 0.8825
  <!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.auc -->
  and the cheapest qualifying configuration from 24 electrodes per compartment to 6
  <!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.electrodes_per_compartment -->.
- **Rate and burst-fraction measures survive a species change; synchrony and network-burst measures
  do not — and across *platforms* none of them is safe**, with mean firing rate itself reversing
  sign.
- **Two competent implementations of the same feature disagree.** Scored against the depositors' own
  meaRtools output on the same 336 recordings
  <!-- metrics.json: extractor_agreement.n_recordings -->, one of seven shared features reproduces;
  three rank recordings identically on a different scale; three burst measures disagree about which
  recordings burst more. "Bursts per minute" differs by a factor of two. That is the argument for a
  data standard, measured.

Every number above is produced by a script into `outputs/metrics.json` and re-checked against the
documents that cite it. Full account: [report/TECHNICAL_REPORT.md](report/TECHNICAL_REPORT.md).

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

The two runs write to separate roots, so the CI path cannot overwrite the run
the report cites:

| run | results | data asset |
|---|---|---|
| `--full` (and a bare `run_all.py`) | `outputs/` | `data/processed/` |
| `--quick` | `outputs/quick/` | `data/processed/quick/` |

`--quick` fetches only the small-tier sources, so analyses that need a
`--full`-only source log what they skipped and leave that section of
`outputs/quick/metrics.json` unwritten. The app shows the full run where it
exists and says so on the page when it is showing a quick one instead.

Fetch data only:

```bash
.venv/bin/python -m neuromps.fetch.runner [--full] [--only kapucu2022 toxcast_mea]
```

Downloads land in `data/raw/<source>/`. Every file's URL, license, SHA256, size and download time are recorded in
`data/raw/MANIFEST.json`. Source decisions are in [data/SOURCES.md](data/SOURCES.md).

`run_all.py` is the only thing that picks a run root. A stage module run on its
own follows the environment, so the quick root is reachable from anywhere:

```bash
NEUROMPS_RUN_MODE=quick .venv/bin/python -m neuromps.twin.calibrate --quick
```

## Interactive demo

```bash
streamlit run app/app.py     # after a run_all.py pass; reads its outputs, recomputes nothing
```

Three tabs: the data asset and its standard, mechanism prediction with a conformal set beside the
transfer results, and the twin with its posterior and what it does on real recordings. It reads
`outputs/metrics.json` and `outputs/demo/`, so it cannot disagree with the report. Screenshots are in
`outputs/figures/app_tab*.png`, and a 4 min 36 s walkthrough is in `outputs/video/neuromps_demo.mp4`.

## Inputs and outputs

| stage | reads | writes |
|---|---|---|
| `fetch` | eight public deposits over HTTP | `data/raw/<source>/`, `data/raw/MANIFEST.json` |
| `data` | `data/raw/`, `schema/neuromps_schema.json` | `data/processed/neuromps.parquet`, `provenance.json` |
| `models` | the asset | `outputs/metrics.json`, `outputs/figures/*.png` |
| `twin` | the asset (calibration targets) | `outputs/twin/*.json`, `outputs/twin/*.png`, metrics |
| `demo` | metrics, the trained posterior | `outputs/demo/` for the app |

No model weights need downloading: the only trained artefact is the `sbi` posterior, which
`run_all.py` fits from simulations in the twin stage and caches under `outputs/twin/`.

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
