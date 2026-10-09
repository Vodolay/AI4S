# NeuroMPS demo app

Three tabs, one for each claim the project makes: the data asset and its standard, mechanism
prediction and what transfers between species, and the digital twin with its inference.

> **Research decision-support only.** Nothing in the app is an efficacy, safety, diagnostic or
> clinical claim. The app repeats the banner on every page.

## Running it

```bash
pip install -e .
python run_all.py --quick        # writes outputs/quick/ and data/processed/quick/
streamlit run app/app.py
```

`--quick` is enough to open every tab. A full `python run_all.py` gives the same app with the full
5 × 5 × 3 twin grid and all 113 real recordings instead of the reduced quick versions.

The app shows the full run's files (`outputs/`, `data/processed/`) and falls back to the quick
run's where the full ones are absent, with a line on the page naming what it fell back to.

`?tab=models`, `?tab=twin` or `?tab=data` opens one tab on its own — useful for a link into a
specific result. The numeric form still works, but the tabs were reordered, so `?tab=1` no longer
means what an older link meant by it.

## What each tab shows

**Tab 3 — Data asset & standard.** The dataset card (records, sources, usable fraction, species,
licenses) straight from `outputs/metrics.json`; a filterable table of harmonized records with QC
flags as coloured badges and a per-record provenance chain on expand; the record schema; a worked
level-2 provenance check with the bug that made the rule worth enforcing; and download buttons for
the asset and its provenance file.

**Tab 1 — Mechanism prediction & transfer.** Opens on a generated diagram of the classifier. Left,
a prediction from either a real treated/baseline
pair or a nine-feature delta vector dialled in by hand, with class probabilities and the conformal
prediction set at α = 0.1 — including a sentence on what a set larger than one means. Right, the
rat→human transfer results with bootstrap intervals, the per-feature species shift (naming the one
feature whose interval excludes zero), the reliability diagram and the conformal set-size curve.

**Tab 2 — Digital twin & inference.** Opens on a diagram of the twin, and carries a second of the
inversion above the real-recording results. Three sliders that snap to the precomputed grid; for
each grid
point, the spike raster for both compartments, the simulated feature vector against the real
baseline distribution, and the posterior marginals with the generating value marked. Below, the
posterior over drug knobs for each real two-chamber recording — the HD15 result, including its
negative reading — and the HD16 directionality panel. A button runs one real Brian2 simulation at
the current slider values, so the twin is verifiably a simulator rather than a lookup table.

## Timings

Measured on this machine (16-core aarch64), through Streamlit's `AppTest` harness:

| action | time |
|---|---|
| cold page load | 1.9 s |
| warm rerun | 0.65 s |
| twin slider move | 0.64–0.85 s |
| real-recording selector | 0.84 s |
| switch to manual feature vector | 0.76 s |
| feature slider → new prediction | 0.62 s |
| **"Run one live simulation" (Brian2)** | **4.5 s**, then cached |

Every interaction except the live-simulation button stays under a second, which is the point of the
`demo` stage: no callback runs Brian2 or trains a posterior.

## Why the live simulation runs in a subprocess

Brian2 registers a SIGINT handler when it is imported, and `signal.signal` only works in a main
thread — so importing it inside Streamlit's script-runner thread raises `ValueError: signal only
works in main thread of the main interpreter`. A spawned multiprocessing child is no better, because
it re-imports the app script. The button therefore shells out to
`python -m neuromps.demo.simulate_one`, which also keeps the simulator out of the server process, so
a slow or failed run cannot take the app down. The result is cached on the parameter tuple.

## What the app never does

It never recomputes a reported number. Every figure quoted in a tab is read from
`outputs/metrics.json` or from `outputs/demo/`, both written by scripts under `src/`, so the app and
`report/TECHNICAL_REPORT.md` cannot disagree. The plots that already exist beside each analysis are
displayed as the files those scripts wrote; only the twin's raster, feature-strip and posterior
figures are drawn in the app, by `src/neuromps/demo/plots.py`.
