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
- **The twin could not read bicuculline dose off a real chip, though the features can.** Burst rate
  and network burst rate fall with dose across 113 recordings
  <!-- metrics.json: feature_dose_response.n_recordings --> and 16 devices, absolute and differenced
  against each device's own baseline, while the twin's inferred inhibition does not follow them: it
  is the inverse model that fails. An earlier version said the opposite, because it ranked doses
  recorded in nM and µM as if they were on one scale.
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

### After round 1: from chip to animal

The twin's drug knobs, applied to a **simulated** whole adult fly brain (the connectome-constrained
model of Shiu et al. 2024 on FlyWire, 138,639 <!-- metrics.json: fly_models.flywire783.neurons -->
neurons), read at once as a chip (a virtual array scored by `features/mea.py`) and as an animal (the
neurons the published model validated against feeding and grooming). The predictions were committed
before the simulations ran ([preregistration/](preregistration/)). The chip's firing rate ranks
drugs by how much they disrupt behaviour (Spearman 0.863
<!-- metrics.json: fly_chip_to_animal.predictions.C1_chip_tracks_disruption.value -->), but in
every drug family behaviour failed at a lower dose than the registered chip threshold flagged; at
behavioural onset, GABA block had moved firing rate by only 0.149
<!-- metrics.json: fly_confirm_network_bursts.precision.abs_log2_mfr_change_at_behavioural_onset.flywire_sweep.gaba_block -->
log2. A second connectome (the male CNS) replicates the behaviour and ranking results. Account:
[report/TECHNICAL_REPORT.md](report/TECHNICAL_REPORT.md), Act VII; run with
`python run_all.py --stage fly` (a 0.4 GB download and about 7 GB of memory per worker).

### After round 1: the embodied fly

The same simulated fly, now with a body. Its nerve cord (the male CNS connectome) drives a
muscle-driven front leg (FlyMimic) and its brain drives the proboscis of a whole-fly physics model
(flybody), in MuJoCo; every stage was registered before it ran. Antennal input to the FlyWire
brain, relayed through matched descending neurons, brings the leg's claw to
0.5369 <!-- metrics.json: embodied_stage1f.predictions.F1_antennal_input_reaches_deepest.value.jon_ce --> mm of
the antenna; sugar input extends the proboscis fully
(0.3431 <!-- metrics.json: embodied_stage2b.predictions.G1_sugar_extends_the_proboscis.value.sugar --> mm) and
bitter input holds it back. Rerun with movement as the readout, the drug study gives the same
answer as before: the chip first flags GABA-A block at
0.6 <!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_block.chip -->
of the knob's range, after movement has already changed at 0.2
<!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_block.movement -->.
Registered reruns on fresh seeds and on a finer dose grid keep the chip late in every family; in
GABA-A potentiation the lateness rests on the bitter-gated proboscis alone.
The negative results are kept beside the positive ones: no walking rhythm from the male nerve
cord under leaky integrate-and-fire neurons; leg-sensor feedback that does not move the leg; a
flaw in our own rhythm measure, found and audited in the open. Account:
[report/ROADMAP_EMBODIED_FLY.md](report/ROADMAP_EMBODIED_FLY.md) §7; run with
`python run_all.py --stage embodied` after installing the `embodied` extra (below).

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

For the embodied fly (MuJoCo), install the extra as well:
`uv pip install --python .venv/bin/python -e ".[dev,embodied]"`.

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

A full run ends with the `fly` stage (the chip-to-animal study) and the `embodied` stage,
which reruns every later fly study in dependency order, each as its own process (a few hours on
a many-core machine, 6.4 GB of optional downloads, most of it the male CNS neuron table for the
volume size rule). Any single study can be rerun:

```bash
.venv/bin/python -m neuromps.fly.pipeline --list            # every step, with what it needs
.venv/bin/python -m neuromps.fly.pipeline --only stage4a    # one step
.venv/bin/python -m neuromps.fly.pipeline --from stage4a    # from a step on
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

Four tabs, in this order: mechanism prediction with a conformal set beside the transfer results,
the twin with its posterior and what it does on real recordings, the data asset with its
standard, and the fly-brain study ("From chip to animal"), which appears once the fly stage has run. Each of the first two opens on a generated diagram of the model it is about. The app reads
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
| [report/ROADMAP_EMBODIED_FLY.md](report/ROADMAP_EMBODIED_FLY.md) | the embodied fly: plan, and every stage's registered result |
| [report/POST_SUBMISSION_CHANGES.md](report/POST_SUBMISSION_CHANGES.md) | every change since round 1, one slide message each |
| [report/PAPER_DRAFT.md](report/PAPER_DRAFT.md) / [report/PAPER_OUTLINE.md](report/PAPER_OUTLINE.md) | the paper draft and its section-by-section status |
| [report/DECISION_LOG.md](report/DECISION_LOG.md) | every crossroads on the embodied-fly path, the option taken, and where to branch |
| [report/SLIDES_OUTLINE.md](report/SLIDES_OUTLINE.md) | a slide-by-slide outline of the post-submission story, every number cited |
| [preregistration/](preregistration/) | every registered study, committed before it ran |
| [CITATIONS.md](CITATIONS.md) / [THIRD_PARTY.md](THIRD_PARTY.md) | data sources and dependency licenses, both generated |

Numbers in these documents cite the `outputs/metrics.json` key that produced them, and

```bash
.venv/bin/python -m neuromps.report.verify
```

fails if any citation disagrees with the live metrics. `CITATIONS.md` and `THIRD_PARTY.md` are
regenerated with `.venv/bin/python -m neuromps.report.attribution`.

## Layout

`src/neuromps/{fetch,features,qc,models,twin,fly,report}`, tests in `tests/`,
all report numbers in `outputs/metrics.json`.
