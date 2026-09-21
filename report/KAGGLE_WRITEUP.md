# NeuroMPS: a provenance-tracked MEA data asset and a digital twin that reports what it cannot do

**AI4S Open Innovation: AI + Organ-on-a-Chip — 5th Pazhou Algorithm Competition**

> **Research decision-support only.** No efficacy, safety, diagnostic or clinical claim is made.

## What this is

Two things, built to be reproduced by one command:

1. **A standardized data asset** of public neural microelectrode-array recordings — 21,380 records
   <!-- metrics.json: data_asset.n_records --> from five sources — where every record carries the
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

**Rate and burst-fraction measures survive a change of species; synchrony and network-burst measures
do not.** Three independent analyses agree: the species-shift ranking, the feature ablations, and the
twin's residual calibration mismatch, which lands exactly on burst structure and synchrony. **Across
platforms the picture reverses** — in the one cross-platform contrast available, mean firing rate
itself flips sign while synchrony agrees, so portability is a species result, not a platform one.

**Coupling asymmetry tracks relative firing rate, not pathway direction.** At fixed wiring, changing
one chamber's drive swings the measured asymmetry across most of its range. Any claim about
direction has to control for rate first — the single most useful negative result in the project.

**The twin does not yet recover pharmacology from a real chip.** Applied to 113 real two-chamber
recordings <!-- metrics.json: twin_hd15_real_inference.n_recordings -->, the inferred GABA-A scale
shows no bicuculline dose trend (Spearman 0.118
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_dose_trend.spearman_rho -->, 95% CI
[−0.168, 0.419]
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_dose_trend.spearman_rho_ci95 -->, clustered
over devices), and **untreated chips already infer 0.3781**
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median --> where a neutral value
is 1.0 — the parameter is absorbing model mismatch rather than tracking the compound.

**Direction is recoverable in simulation once rate is residualised out.** With features residualised
against both chambers' firing rates — a control that carries its own validity check — symmetric and
asymmetric wiring separate at AUC 0.9359
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.auc -->
from a 10 s recording
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.duration_s -->
on 24 electrodes per compartment
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
posterior contraction of only 0.311
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

## Reproducing

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
python run_all.py --quick     # 463 s on CPU: every figure and metric
```

## What exists, and what does not

Present in the repository: the data asset and schema, the feature extractor and QC, the model
baselines with calibration and conformal sets, the full twin (network, MEA simulator, calibration,
drug knobs, inference, directionality experiment), the technical report, and the decision record in
`report/LAB_NOTEBOOK.md` with 33 numbered limitations.

The interactive demo is built: `streamlit run app/app.py` after `python run_all.py --quick`, with
screenshots in `outputs/figures/app_tab*.png`. Not yet recorded: the demo video.

**Full details:** `report/TECHNICAL_REPORT.md`.
