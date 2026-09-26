# NeuroMPS: a measurement standard for neural organ-on-a-chip data, and a harness that tests what it can support

## Submission category: **End-to-End System**

A complete path from public recordings to actionable output: provenance-enforced ingestion → a
standardized feature layer → models and a calibrated digital twin → a falsification harness that
reports which measurements the data can and cannot support, with an interactive demo.

**AI4S Open Innovation: AI + Organ-on-a-Chip — 5th Pazhou Algorithm Competition**

> **Research decision-support only.** No efficacy, safety, diagnostic or clinical claim is made.

## Demo video

<!-- FILL IN BEFORE SUBMITTING: paste the public URL (YouTube / Bilibili / Loom), or attach
     outputs/video/neuromps_demo.mp4 to the Kaggle Writeup. It must play without login. -->
**[PUBLIC VIDEO LINK]** — 4 min 36 s. Source file: `outputs/video/neuromps_demo.mp4`.

## Code repository

**https://github.com/Vodolay/AI4S** — MIT licensed, one-command reproduction, CI green on every
push.

## Project summary

Microphysiological systems produce electrophysiology that is rich and almost impossible to compare
across studies: feature names collide while their definitions differ, electrode counts and recording
lengths vary, and provenance is usually gone by the time data reach an analysis. Before anyone can
build a predictive digital twin of a neural chip, someone has to establish which measurements
survive being moved between a species, a platform and a laboratory.

NeuroMPS does both halves. The first is a standardized, provenance-tracked data asset: 21,887
records <!-- metrics.json: data_asset.n_records --> from eight public sources, each carrying the URL,
license, SHA256 and download time of the file it came from, validated against a JSON Schema whose
conformance level *requires* every provenance hash to match the download manifest — the build
refuses to emit an asset otherwise. Records declare an explicit data level and compare only within
it, and a single fixed feature extractor computes the nine-feature vector that everything downstream
consumes, including the twin.

The second is a Brian2 spiking-network digital twin of an asymmetric two-compartment chip,
calibrated against that asset and scored through the *same* extractor as the real recordings, with
simulation-based inference (`sbi`) attempting to recover connectivity and drug effects.

What the two together produce is a falsification harness, and it has found four confounds that
published practice would have carried: pseudo-replication that manufactured a dose trend,
firing-rate confounding that makes a standard directionality measure report the wrong thing,
in-sample leakage that inflated a classifier's AUC, and a chamber-level coupling measure that
disagrees with axonal traffic measured inside the connecting microtunnels. Each was caught by the
infrastructure rather than by inspection, and each is reported with the number that caught it.

## What this is

Two layers, reproduced by one command:

1. **A standardized data asset** of public neural microelectrode-array recordings — 21,887 records
   <!-- metrics.json: data_asset.n_records --> from eight sources — where every record carries the
   URL, license, SHA256 and download time of the file it came from, and the build refuses to emit an
   asset if any provenance hash fails to match the download manifest.

2. **A Brian2 spiking-network digital twin** of an asymmetric two-compartment neural chip,
   calibrated against that asset and scored through the *same* feature extractor as the real
   recordings, with `sbi` inference attempting to recover connectivity and drug effects.

## The argument

Public MEA data are easy to download and hard to compare: feature names collide while definitions
differ, electrode counts and recording lengths vary, and provenance is usually gone by the time data
reach an analysis. Our answer is a schema where a record's **data level** is explicit and records
only compare within a level, a single fixed feature extractor that the twin is also required to
call, and provenance enforced at build time rather than documented in prose.

The twin then asks the question a chip user actually has: *how much can be read off MEA features
about what is wired to what, and about what a compound did?*

## What we found

**Feature names do not determine feature values — and we measured by how much.** One source deposits
both its spike files and the authors' own meaRtools output, so the same 336 recordings
<!-- metrics.json: extractor_agreement.n_recordings --> can be scored by two independent
implementations of the same seven features. One reproduces on both ordering and scale. Three rank
the recordings identically but sit on a different scale, because one activity threshold differs.
Three — all burst measures — disagree about *which* recordings burst more: our "bursts per minute"
is a median 0.4423
<!-- metrics.json: extractor_agreement.by_feature.burst_rate_per_min.median_ratio_ours_over_theirs -->
of theirs, a factor of two.
Nothing in either paper would warn a reader pooling them. This is the argument for a standard,
measured rather than asserted.

**Rate and burst-fraction measures survive a change of species; synchrony and network-burst measures
do not.** Three independent analyses agree: the species-shift ranking, the feature ablations, and the
twin's residual calibration mismatch, which lands exactly on burst structure and synchrony. **Across
platforms the picture reverses** — in the one cross-platform contrast available, mean firing rate
itself flips sign while synchrony agrees, so portability is a species result, not a platform one.

**Coupling asymmetry tracks relative firing rate, not pathway direction.** At fixed wiring, changing
one chamber's drive swings the measured asymmetry across most of its range, and pre-whitening the
correlogram makes that worse rather than better, so it is the network and not the estimator. Any
claim about direction has to control for rate first — the single most useful negative result in the
project.

**And no chamber readout rescues it.** Fitting all four readouts a two-chamber device gives you —
asymmetry, peak correlation, lag, rate ratio — to predict the sign of the measured axonal flow, one
culture held out at a time, reaches balanced accuracy 0.3854
<!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.balanced_accuracy -->
against chance of one half, at a sample that could have detected 0.6438
<!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.detectable_balanced_accuracy -->.

**And it is wrong against ground truth.** One deposit instruments the microtunnels between its
chambers, two electrodes per tunnel, so a spike crossing is recorded twice and the sign of the delay
is the direction it physically travelled. The traffic is strongly directed — mean |flow index| 0.363
<!-- metrics.json: axonal_directionality.headline.median_abs_flow_index --> against 0.0093
<!-- metrics.json: axonal_directionality.headline.shift_control_median_abs_index --> for a
time-shifted control — and the chamber measure points the same way on a fraction 0.419
<!-- metrics.json: axonal_directionality.headline.sign_agreement.fraction --> of 31 chamber pairs,
a coin flip, at every latency window tested. Reading direction off chamber cross-correlation does
not work on real devices, and we can now say so against a measurement rather than a simulation.

**Neither the twin nor the features recover pharmacology from a real chip — and knowing which is
the point.** Before blaming the inference, we asked whether the summaries it was given carry a dose
response at all. Across 113 recordings
<!-- metrics.json: feature_dose_response.n_recordings --> and 16 devices, no feature moves
monotonically with bicuculline dose, absolute or differenced against each device's own baseline.
The twin was being asked to invert a measurement that does not contain the answer, which moves the
problem from the posterior to the feature layer.

**The twin does not yet recover pharmacology from a real chip.** Applied to 113 real two-chamber
recordings <!-- metrics.json: twin_hd15_real_inference.n_recordings -->, the inferred GABA-A scale
shows no bicuculline dose trend (Spearman −0.121
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_dose_trend.spearman_rho -->, 95% CI
[−0.3331, 0.1148]
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_dose_trend.spearman_rho_ci95 -->, clustered
over devices), and **untreated chips already infer 0.5219**
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median --> where a neutral value
is 1.0 — the parameter is absorbing model mismatch rather than tracking the compound.

**Direction is recoverable in simulation once rate is residualised out.** With features residualised
against both chambers' firing rates — a control that carries its own validity check — symmetric and
asymmetric wiring separate at AUC 0.8825
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.auc -->
from a 10 s recording
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.duration_s -->
on 6 electrodes per compartment
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.electrodes_per_compartment -->.
A second arm that adjusted drive to match rates reports a higher number on fewer electrodes and
fails its own validity check; we report it as inconclusive rather than as the result.

**And that result does not reproduce on real devices.** A late-arriving dataset — hippocampal
cultures on straight against asymmetric microchannels, with the geometry fabricated into the chip —
lets the same question be asked of 26 real devices
<!-- metrics.json: microchannel_directionality.n_chips -->. Propagation asymmetry does not separate
the geometries (Kruskal p = 0.64
<!-- metrics.json: microchannel_directionality.comparison.kruskal_p -->), and the statistic is
confounded with how many events a chip produced (Spearman −0.494
<!-- metrics.json: microchannel_directionality.event_count_confound.spearman_rho -->). Counting
*recordings* instead of chips would have produced an effect; the same pseudo-replication that
dissolved our earlier dose trend. The sample is too small to call this a refutation of the simulated
result, and too directly aimed to leave out.
Read as a contrast between two wiring regimes, not as a measurement: the same summaries give a
posterior contraction of only 0.438
<!-- metrics.json: twin_hd14_inference.contraction.log10_forward_ratio.contraction --> for the
continuous forward/backward ratio.

**Mechanism classification is above chance and not yet useful.** Balanced accuracy 0.537
<!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean -->
on six classes against a chance level of 0.167
<!-- metrics.json: mechanism_baselines.task1_within_platform.chance_balanced_accuracy -->, with
conformal sets that cover honestly (0.918
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep.0.empirical_coverage --> against a 0.95
target) at a mean size of 4.562
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep.0.mean_set_size --> classes out of six.
Two rate features match all nine.

## Why the negative results are the point

An entry that reported only what worked would be easy to write from this data and impossible to
trust. Three specific mechanisms keep this one honest:

- **Numbers cannot be typed into prose.** Every figure in the report is produced by a script into
  `outputs/metrics.json`, and `python -m neuromps.report.verify` re-reads every citation in the
  document and fails the build if it disagrees. It caught a stale timestamp and three
  mis-anchored citations in our own draft.
- **Intervals resample groups, never rows.** Cultures contribute many correlated records; a
  row bootstrap would have halved every interval. An earlier apparent dose trend (ρ = −0.197,
  p = 0.036) dissolved once 113 recordings were recognised as coming from 16 devices, and a
  geometry effect dissolved once 57 recordings were recognised as coming from 26 chips.
- **Caches refuse to be reused across a change in meaning.** The simulation cache stores its summary
  names, electrode count and a hash of the source of every function that defines a summary — added
  after a real defect where the twin measured coupling from every neuron while real recordings
  measure it from electrodes.

## Why this matters, and to whom

The supporting organization's stated roadmap is standardization, then data assetization, then
AI-driven digital twins for neural drug development. That order is the right one, and this entry is
an argument about the first two steps: a digital twin built on measurements that have not been
checked for portability will be confidently wrong, and the checking is neither expensive nor
optional.

Three outputs are directly usable by a chip laboratory today.

- **A schema and feature standard with enforced provenance.** Any group pooling MEA data across
  studies faces the comparability problem; the record model, the explicit data level, and the
  build-time hash check are transferable as-is and are the part of this work most likely to outlive
  it.
- **A negative result that changes practice.** Coupling asymmetry between two chambers is a
  standard readout for connectivity direction. We show it tracks relative firing rate rather than
  direction in simulation, that pre-whitening makes that worse rather than better, and that on real
  devices it agrees with the axons' own measured direction at chance. Anyone reporting direction
  from a two-chamber MEA needs to residualise rate first, and should not expect chamber
  cross-correlation alone to carry the claim.
- **A cost curve for experiment design, as a tool.** `python -m neuromps.design --target-auc 0.8`
  returns the cheapest electrode count and recording length whose interval *lower bound* clears a
  target, and `--electrodes 12 --duration 30` says what a given recording could detect. It is a
  lookup over the §7.6 grid, and every answer carries the caveat that the grid is simulated and the
  equivalent real-device test was null — a tool that overstates itself would be worse than none.

## Reproducing

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
python run_all.py --quick     # path test: small-tier sources, ~400 s on CPU
streamlit run app/app.py      # interactive demo, reads the run's outputs
```

## Optional demo

The Streamlit app runs locally after the command above — three tabs covering the data asset and its
standard, mechanism prediction with a conformal set beside the transfer results, and the twin with
its posterior and what it does on real recordings. It reads `outputs/metrics.json` and recomputes
nothing, so it cannot disagree with the report. Screenshots are in `outputs/figures/app_tab*.png` for
reviewers who prefer not to run it.

## Technical report

**`report/TECHNICAL_REPORT.md`** in the repository, and `report/TECHNICAL_REPORT.pdf` built from it
by `report/build_report.sh`. Every computed number in it names the `outputs/metrics.json` key that
produced it, and `python -m neuromps.report.verify` fails the build if the document and the metrics
disagree.

Also present: the data asset and schema, the feature extractor and QC, the model baselines with
calibration and conformal sets, the full twin (network, MEA simulator, calibration, drug knobs,
inference, directionality experiment), and the decision record in `report/LAB_NOTEBOOK.md` with 33
numbered limitations and everything that was tried and abandoned.

## Team

<!-- FILL IN BEFORE SUBMITTING. The rules give a +0.5 bonus in the interpretability and reliability
     dimension to teams covering both AI/CS and biology/bioengineering/clinical expertise, and ask
     that team composition be declared in the technical report as well as here. Replace the lines
     below with the real roster; if the relevant background is present, name it explicitly. -->

| member | role | background |
|---|---|---|
| **[NAME]** (team leader) | **[ROLE]** | **[AI/CS and/or biology/bioengineering/clinical]** |

**AI assistance.** Built in a paired workflow with Anthropic's Claude (Claude Opus 5) via Claude
Code, under step-by-step human direction; §11 of the technical report states what the model did and
the three practices used to keep it honest.
