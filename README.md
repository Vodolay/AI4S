# NeuroMPS

A standardized, provenance-tracked dataset of public neural MEA recordings, and a spiking-network
digital twin of an asymmetric two-compartment neural chip calibrated on it.
Entry for the AI4S Open Innovation: AI + Organ-on-a-Chip challenge (5th Pazhou Algorithm
Competition), submission category **End-to-End System**.

**Video walkthrough:** https://www.youtube.com/watch?v=WnnD-oSFdcI

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
- **The twin could not read bicuculline dose off a real chip, and neither could the features.** No
  feature moves monotonically with dose across 113 recordings
  <!-- metrics.json: feature_dose_response.n_recordings --> and 16 devices, absolute or differenced
  against each device's own baseline — so the inverse model was being asked to invert a measurement
  that does not contain the answer.
- **Pseudo-replication manufactured a dose trend.** An apparent bicuculline response (ρ = −0.197,
  p = 0.036) dissolved once 113 recordings were recognised as coming from 16 devices; a microchannel
  geometry effect dissolved once 57 recordings were recognised as coming from 26 chips
  <!-- metrics.json: microchannel_directionality.n_chips -->.
- **Much of the simulated direction signal is firing rate.** Telling asymmetric from symmetric
  wiring at an AUC of 0.8 takes 24 electrodes per compartment
  <!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_all_features.electrodes_per_compartment -->
  on all features, and 48
  <!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.electrodes_per_compartment -->
  once firing rate is regressed out inside each training fold (AUC 0.8969
  <!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.auc -->).
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

Linux or macOS, with `git` and `curl`. CI runs on Ubuntu 24.04. uv fetches Python 3.11 itself
if the machine does not have it.

```bash
git clone https://github.com/Vodolay/AI4S
cd AI4S
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv into ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"                 # so this terminal finds it
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

If uv is already installed, skip the `curl` line. `pip install uv` also works where pip may install
into the system Python; recent Ubuntu, Debian and Homebrew Pythons refuse it ("externally managed
environment"), which is why the installer comes first. On a machine without an NVIDIA GPU, add
`--torch-backend cpu` to the last command to skip several gigabytes of CUDA libraries.

Every command below is run from the `AI4S` folder.

## Reproduce

```bash
.venv/bin/python run_all.py --quick     # CI path: ~10 min on 20 cores, + ~3 min of downloads the first time
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
`data/raw/MANIFEST.json`, or in `data/raw/MANIFEST.quick.json` for the small tier fetched without
`--full`, so a partial fetch never rewrites the committed record of the full one. If a download
breaks off, the run stops after the fetch and names the source; rerun the same command and only the
missing files are fetched. Kapucu's quick tier falls back to a checksummed mirror when its host,
G-Node GIN, is down (see [data/SOURCES.md](data/SOURCES.md)). Source decisions are in [data/SOURCES.md](data/SOURCES.md).

`run_all.py` is the only thing that picks a run root. A stage module run on its
own follows the environment, so the quick root is reachable from anywhere:

```bash
NEUROMPS_RUN_MODE=quick .venv/bin/python -m neuromps.twin.validate --quick
```

## Experiment design

The §17 grid answers a question a chip user has *before* running an experiment, so it is exposed as
a lookup rather than left in a figure:

```bash
.venv/bin/python -m neuromps.design --target-auc 0.8               # the cheapest design that clears it
.venv/bin/python -m neuromps.design --electrodes 12 --duration 30  # what that recording could detect
.venv/bin/python -m neuromps.design                                # the whole grid
```

Every answer carries the caveat it needs: the grid is simulated, the classifier was trained on the
same generative model that produced its test set, and the equivalent question put to real devices
in §15 was null. Treat a row as an upper bound on what the measurement could do.

## Interactive demo

```bash
.venv/bin/streamlit run app/app.py     # after a run_all.py pass; reads its outputs, recomputes nothing
```

Use the virtual environment's `streamlit`, not a bare `streamlit`. The app imports `neuromps`, which
is installed into `.venv` by the setup above and nowhere else, so a system-wide Streamlit cannot
find it — and on a machine that has no system-wide Streamlit the bare command simply reports
`command not found`. Activating the environment first works equally well. The live Brian2 button
runs its simulation with `sys.executable`, so starting the app this way is also what gives that
subprocess the interpreter that has Brian2 in it.

Three tabs, in this order: mechanism prediction with a conformal set beside the transfer results,
the twin with its posterior and what it does on real recordings, and the data asset with its
standard. Each of the first two opens on a generated diagram of the model it is about. The app reads
`outputs/metrics.json` and `outputs/demo/`, so it cannot disagree with the report.


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
| [report/LAB_NOTEBOOK.md](report/LAB_NOTEBOOK.md) | every decision with its date and commit, what was abandoned, 33 limitations |
| [CITATIONS.md](CITATIONS.md) / [THIRD_PARTY.md](THIRD_PARTY.md) | data sources and dependency licenses, both generated |

Numbers in these documents cite the `outputs/metrics.json` key that produced them, and

```bash
.venv/bin/python -m neuromps.report.verify
```

fails if any citation disagrees with the live metrics. `CITATIONS.md` and `THIRD_PARTY.md` are
regenerated with `.venv/bin/python -m neuromps.report.attribution`.

## Layout

`src/neuromps/{fetch,features,qc,models,twin,report}`, tests in `tests/`,
all report numbers in `outputs/metrics.json`.
