# NeuroMPS — a provenance-tracked MEA data asset and a calibrated two-compartment digital twin

**Technical report.** Entry for the AI4S Open Innovation: AI + Organ-on-a-Chip challenge
(5th Pazhou Algorithm Competition).

> **Research decision-support only.** Nothing in this report is an efficacy, safety, diagnostic or
> clinical claim. The models here describe public *in vitro* recordings and a simulation of them.

Every computed number below carries an HTML comment naming the `outputs/metrics.json` key it
came from. The few figures that describe a source deposit rather than a result of this
pipeline - a recording length, an electrode count - are attributed in the text instead.
`python -m neuromps.report.verify` re-reads each citation and fails if the document and the metrics
disagree, so a figure that goes stale when a stage is regenerated cannot quietly survive in the
prose. The decision record behind these results, including what was tried and abandoned, is
`report/LAB_NOTEBOOK.md`.

---

## 1. Summary

Two layers were built:

**(A) A standardized, provenance-tracked data asset** of public neural microelectrode-array (MEA)
recordings: 21,887 records <!-- metrics.json: data_asset.n_records --> from eight sources, each
record carrying its source URL, license, SHA256 and download time, validated against a JSON Schema
at a conformance level that requires every provenance hash to match the download manifest.

**(B) A Brian2 spiking-network digital twin** of an asymmetric two-compartment neural chip,
calibrated against the asset, with simulation-based inference (`sbi`) attempting to recover
connectivity and drug effects from MEA summary features.

The results that matter are as much negative as positive, and the report leads with both:

1. **Rate and burst-fraction measures survive a change of species; synchrony and network-burst
   measures do not** (§6.3, §6.5, §6.6, §7.3). **Across *platforms* none of them is safe** — mean
   firing rate itself reverses sign (§6.4), which is the sharper limitation of the two. That is a
   claim about drug-response deltas: *untreated* human and rat activity differs on 7 of the 9
   features even at matched platform and preparation (§6.8).
2. **Coupling asymmetry tracks relative firing rate, not pathway direction.** At fixed wiring,
   changing one chamber's drive moves the measure several times further than changing the wiring
   does (§7.2).
3. **The twin's posterior does not recover drug-knob direction from real recordings.** The inferred
   GABA-A scale shows no bicuculline dose trend, and untreated chips already infer 0.3781
   <!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median -->, so the parameter
   is absorbing model mismatch rather than tracking the compound (§7.5).
4. **Direction *is* recoverable from simulated features once firing rate is residualised out** —
   AUC 0.8825
   <!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.auc -->,
   95% CI [0.8184, 0.9359]
   <!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.auc_ci95 -->,
   from a 10 s recording on 6 electrodes per compartment
   <!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.electrodes_per_compartment -->
   — but as a *contrast* between two well-spaced wiring classes, not as a measurement of a
   continuous ratio (§7.6). A second arm, which adjusted drive to match rates, reaches a higher AUC
   and **fails its own validity check**; it is reported as inconclusive rather than as the result.
5. **That result does not reproduce on real devices.** On 26 chips whose microchannel geometry was
   fabricated rather than assumed, propagation asymmetry does not separate straight channels from
   three asymmetric designs (Kruskal p = 0.64
   <!-- metrics.json: microchannel_directionality.comparison.kruskal_p -->), and the statistic is
   confounded with how many events a chip produced (§7.7). The sample is too small to call that a
   refutation of the simulated result, and too directly aimed to omit.

The honest summary of the twin is therefore: it reproduces baseline features and most drug
directions, it can tell two wiring regimes apart in simulation, it cannot yet read pharmacology off
a real chip, and the one real-data test of its directionality claim is null.

---

## 2. Problem and scope

Microphysiological systems produce electrophysiology that is rich but hard to compare: different
platforms, electrode counts, durations, species and feature definitions. Two consequences follow for
anyone trying to use public MEA data for compound screening.

*Data are not comparable by default.* Feature names collide while definitions differ — "synchrony"
from a spike-time tiling coefficient is not the Axion synchrony index — and provenance is usually
lost by the time data reach an analysis.

*A model of the chip is not a model of the measurement.* A spiking network that reproduces a
culture's firing does not necessarily reproduce what 48 electrodes detect from it.

This entry addresses both: a schema and feature standard with enforced provenance, and a digital
twin that is scored through the *same* feature extractor as the real recordings.

**Out of scope.** No clinical, diagnostic, efficacy or safety claim is made or implied. Compound
labels are mechanism classes drawn from the source publications, not indications.

---

## 3. The data standard

### 3.1 Record model and data levels

Records validate against `schema/neuromps_schema.json` (JSON Schema 2020-12). The central decision is
that records carry an explicit **data level**, and that records only compare within a level:

| level | meaning | records |
|---|---|---|
| `spike_times` | per-electrode spike trains; our feature vector is computable | 1,512 <!-- metrics.json: data_asset.counts.by_data_level.spike_times --> |
| `fitted_curve` | published concentration-response fits only | 20,375 <!-- metrics.json: data_asset.counts.by_data_level.fitted_curve --> |
| `per_well_features` | vendor feature tables without spike times | — |

This distinction is load-bearing. Most of the asset by count is ToxCast curve fits, from which our
nine-feature vector cannot be recomputed; only 897 records support the feature pipeline. A schema
that hid that behind a single "features" field would invite exactly the wrong comparison.

### 3.2 Provenance and conformance

Conformance **level 2** requires that every record's provenance hash matches `data/raw/MANIFEST.json`,
and the build refuses to emit an asset otherwise. Each fetcher records URL, license, SHA256, byte
count and download time, and re-checks the source's advertised license at fetch time so a
silently-relicensed upstream blocks rather than propagates.


### 3.3 Quality control

QC is recorded, never applied silently. Each record carries flags and a single `qc.usable` verdict,
and the flags are informational: a flagged record is still in the asset, with the reason attached, so
a downstream analysis can make its own decision.

| flag | records | what it means |
|----------------------------|-------|--------------------------------------------|
| `incomplete_dose_series` | 272 <!-- metrics.json: data_card.flag_counts.incomplete_dose_series --> | the compound's dose series is missing points |
| `too_few_active_electrodes` | 265 <!-- metrics.json: data_card.flag_counts.too_few_active_electrodes --> | under 10 electrodes cleared 5 spikes/min |
| `replicate_divergence` | 221 <!-- metrics.json: data_card.flag_counts.replicate_divergence --> | one replicate departs from its group |
| `outlier_feature_value` | 221 <!-- metrics.json: data_card.flag_counts.outlier_feature_value --> | a feature sits far outside its source's distribution |
| `excluded_well_upstream` | 90 <!-- metrics.json: data_card.flag_counts.excluded_well_upstream --> | the source's own log excluded the well |
| `noisy_electrodes_present` | 11 <!-- metrics.json: data_card.flag_counts.noisy_electrodes_present --> | the source marked electrodes as noisy |

Two decisions here were driven by defects found in the data rather than by design. The
replicate-divergence test originally used a robust z-score, which collapses when more than half a
replicate group shares a value — common in sparse recordings, where the median absolute deviation is
exactly zero; it now falls back to a mean absolute deviation and names the diverging feature. And
the 90 wells excluded by an upstream log were at first missing from the asset altogether, because
records were built from the feature tables alone; they are now built from the union of the log and
the spike files, so an exclusion is represented rather than absent.

### 3.4 Feature set

Nine features form the core vector, computed by a single extractor (`features/mea.py`) that the twin
must also call:

| feature | definition as computed |
|------------------------------|-------------------------------------------|
| `mfr_hz` | spikes per second per active electrode |
| `burst_rate_per_min` | bursts per minute per active electrode |
| `burst_duration_s` | mean burst duration |
| `spikes_per_burst` | mean spikes in a burst |
| `pct_spikes_in_bursts` | share of spikes falling inside bursts |
| `network_burst_rate_per_min` | network bursts per minute, chance-corrected participation |
| `synchrony_index` | spike-time tiling coefficient, mean over electrode pairs |
| `isi_cv` | coefficient of variation of inter-spike intervals |
| `active_electrodes` | electrodes clearing the activity threshold |

Thresholds are fixed and recorded per record: an electrode is active at ≥5 spikes/min, a well is
usable at ≥10 active electrodes, a burst is ≥5 spikes with ISI ≤0.1 s, and the tiling coefficient
uses Δt = 0.05 s.

The network-burst threshold deserves its own note, because the obvious implementation is wrong. A
fixed participation threshold — *n* electrodes firing in the same window — counts coincidences that
independent electrodes produce anyway; at 5 Hz on 16 electrodes that is on the order of fifty
spurious "network bursts" per minute, which would dominate the measure. The threshold is therefore
set per recording so that the expected number of chance detections stays below 0.05, from the
recording's own electrode count and firing rates. This matters beyond bookkeeping: because the
correction scales with electrode count, it is a candidate explanation for part of the cross-platform
disagreement reported in §6.4, and that remains untested.

---

## 4. The data asset

| source | records | level | species | license |
|--------------------------|-------|------------|-----------|---------|
| `toxcast_mea` | 20,375 <!-- metrics.json: data_asset.counts.by_source.toxcast_mea --> | fitted_curve | rat | CC0-1.0 |
| `screenneuropharm_bic` | 580 <!-- metrics.json: data_asset.counts.by_source.screenneuropharm_bic --> | spike_times | rat | MIT |
| `kapucu2022` | 288 <!-- metrics.json: data_asset.counts.by_source.kapucu2022 --> | spike_times | human, rat | CC-BY-4.0 |
| `sharf2022` | 29 <!-- metrics.json: data_asset.counts.by_source.sharf2022 --> | spike_times | human | CC0-1.0 |
| `mateus2024` | 108 <!-- metrics.json: data_asset.counts.by_source.mateus2024 --> | spike_times | rat | CC-BY-4.0 |
| `trujillo2019` | 184 <!-- metrics.json: data_asset.counts.by_source.trujillo2019 --> | spike_times | human | CC-BY-4.0 |
| `brewer2021` | 40 <!-- metrics.json: data_asset.counts.by_source.brewer2021 --> | spike_times | rat | CC-BY-4.0 |
| `vassallo2017` | 283 <!-- metrics.json: data_asset.counts.by_source.vassallo2017 --> | spike_times | rat | US Public Domain |

Totals: 21,887 records <!-- metrics.json: data_asset.n_records -->, of which 21,350 are marked usable
<!-- metrics.json: data_asset.counts.by_usable.True --> and 537 are not
<!-- metrics.json: data_asset.counts.by_usable.False -->. By species, 21,530 records are rat
<!-- metrics.json: data_asset.counts.by_species.rat --> and 357 are human
<!-- metrics.json: data_asset.counts.by_species.human -->.

**The species imbalance is the asset's main structural weakness** and is stated wherever human
results appear. Adding Trujillo 2019 took the human count from 173 to 357
<!-- metrics.json: data_asset.counts.by_species.human --> across three sources, and changed no
modelling number at all: those records are untreated baselines, and every model in §6 consumes
exposure-minus-baseline pairs. §6.8 is the comparison they do support.

`brewer2021` is here for §7.7's sake rather than for a result: ten rat hippocampal cultures in a
four-chamber device whose **tunnels are themselves over electrodes**, so a crossing is measured
rather than inferred from the chambers either side. Its anatomical labels are not trusted — 4 of its
40 files carry the deposit's own subregion field and 2 of those disagree with the file name, which
each record's QC note states.

### 4.1 A rejection, as a worked example of the license discipline

Tukker 2020 (rat and hiPSC MEA pharmacology) is a good fit scientifically and is **excluded**. It is
CC BY-NC 4.0, and the competition rules assign joint ownership of submissions rather than preserving
an entrant's own license, so the non-commercial condition cannot be guaranteed downstream. It is
recorded as `skipped` in the manifest with that reason rather than dropped silently. The cost is
real: without it, the rat→human transfer study rests on one source.

### 4.2 A licence conflict, resolved rather than avoided

Zenodo 14525182 (Mateus et al. 2024) is the closest published analogue of the chip the twin models:
hippocampal cultures on straight against asymmetric microchannels, 57 spike-data recordings, with the
geometry fabricated into the device. It was **rejected on 2026-09-16** because the deposit
contradicts itself — the Zenodo metadata licence field says `cc-by-4.0` and the record is open
access, while the README inside says, in capitals, CC BY-NC-ND. Under NoDerivatives, publishing
derived features would not have been permitted, which is the whole of what this project does with a
recording.

On 2026-09-21 the corresponding author confirmed by email that the metadata governs and the README
line was an unintentional default. The source is included on that basis, and the reasoning is
recorded where a reader can check it: both halves of the contradiction are public, the fetcher
blocks if the metadata licence ever moves, and it reads the README on every fetch and writes
whatever licence claim it finds into `data/raw/MANIFEST.json`. **The resolution itself is private
correspondence, which a reader cannot verify**; `data/SOURCES.md` says so in those words. This is the
one source in the asset whose licence rests on anything other than a public record.

Its 108 records <!-- metrics.json: data_asset.counts.by_source.mateus2024 --> are in the asset:
two chambers per recording, split at the microchannel band recovered from the spike times, with
features from the same extractor as every other source. They are analysed in §7.7.

---

## 5. Methods

### 5.1 Cross-validation and resampling

Every model is evaluated with **grouped** cross-validation, grouping by culture/well so replicates
of one culture never span the train/test boundary, and every interval is a **cluster bootstrap over
those groups**, never over rows — with 1,000 resamples
<!-- metrics.json: mechanism_baselines.task1_within_platform.bootstrap_resamples --> in a full run and
fewer in `--quick`, the count recorded alongside each estimate. Bootstrapping rows here would report
intervals roughly a factor of two too narrow, because rows within a culture are not independent.

### 5.2 Tasks

- **Task 1 — mechanism classification** (within platform): predict mechanism class from the
  nine-feature delta vector of a treated/baseline pair.
- **Task 2 — external detection**: apply the Task 1 model to a source it never saw, with a negative
  control of the opposite mechanism.
- **Task 3 — platform contrast**: compare the same mechanism measured on two platforms.
- **Transfer**: rat → human, with and without domain adaptation.

---

## 6. Results: data and models

### 6.1 Task 1 — mechanism classification

Six classes, 73 treated/baseline pairs
<!-- metrics.json: mechanism_baselines.task1_within_platform.n_pairs --> across 50 culture groups
<!-- metrics.json: mechanism_baselines.task1_within_platform.n_groups -->, 4-fold grouped CV
<!-- metrics.json: mechanism_baselines.task1_within_platform.n_splits -->.

| model | balanced accuracy | 95% CI |
|---|---|---|
| logistic regression | 0.537 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean --> | [0.4216, 0.6845] <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_ci95 --> |
| gradient boosting | 0.398 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_mean --> | [0.2762, 0.5178] <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_ci95 --> |
| most-frequent baseline | 0.167 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.dummy_most_frequent.balanced_accuracy_mean --> | — |

Chance is 0.167 <!-- metrics.json: mechanism_baselines.task1_within_platform.chance_balanced_accuracy -->.
Logistic regression beats chance with an interval that excludes it. Its interval and gradient
boosting's overlap heavily, so the gap between the two models is within noise; with 73 pairs the
simpler model is the one to prefer, not the one shown to be better.

![Task 1 confusion matrix, REAL data: out-of-fold predictions from grouped cross-validation over 73 treated/baseline pairs. Errors concentrate between the two GABA-A classes.](../outputs/figures/mechanism_confusion.png){width=55%}


**Calibration.** Expected calibration error on out-of-fold predictions is 0.1114
<!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.ece -->, 95% CI [0.0773, 0.2557]
<!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.ece_ci95 -->
(`outputs/figures/task1_reliability_real.png`). The upper end of that interval is large: predicted
probabilities should be read as ordering, not as calibrated risk — and not even as ordering above
0.8 confidence, where observed accuracy falls rather than rises, on a handful of predictions.

Both are plotted in `outputs/figures/task1_reliability_real.png` and `conformal_set_size.png`; the numbers they show are quoted above.


**Conformal sets.** At a 95% target the empirical coverage is 0.918
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep.0.empirical_coverage --> with a mean set
size of 4.562 <!-- metrics.json: uncertainty_and_ablations.conformal_sweep.0.mean_set_size --> classes
out of six — honest coverage at the cost of sets too large to act on. That is the correct reading of
73 calibration points, not a defect to tune away.

**The same task on an independent platform.** `vassallo2017` is a different laboratory and vendor,
with six compounds Kapucu never used and each baseline on the same chip as its doses: 171 pairs
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.n_pairs --> over 26 chips
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.n_groups --> in four mechanism
classes. **Logistic regression reaches 0.333
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.models.logistic_regression.balanced_accuracy_mean -->
there, 95% CI [0.2317, 0.4204]
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.models.logistic_regression.balanced_accuracy_ci95 -->,
against a chance level of 0.25
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.chance_balanced_accuracy -->** —
an interval that contains chance. The two tasks differ in class count and in compound, so the
accuracies are not directly comparable; what is comparable is each against its own chance level, and
only the first clears it. The mechanism claim is weaker than this section's Kapucu numbers alone
suggest.

**Why this is not one model over all 754 pairs.** Pooling the sources would measure the wrong
thing: a pair's mechanism label very nearly determines which laboratory produced it. The label
predicts the source for 97.35%
<!-- metrics.json: mechanism_baselines.label_source_confounding.source_determined_by_label --> of
the 754 pairs <!-- metrics.json: mechanism_baselines.label_source_confounding.n_pairs -->, and 6
<!-- metrics.json: mechanism_baselines.label_source_confounding.n_labels_from_one_source_only -->
of the nine labels occur in one source only, so a pooled model would be scored on recognising the
platform - which §6.4 shows these features do well. The asset supports within-source classification
and cross-source *detection* of one mechanism at a time (§6.2), not a cross-platform benchmark.
That is a property of what the field has published, not of the model.

---

### 6.2 Task 2 — external detection and its negative control

Applied to bicuculline recordings from a source the model never saw, the GABA-A antagonist class is
detected in 0.366 <!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.detection_rate -->
of 497 records <!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.n -->,
and the detection does **not** track dose (Spearman 0.014
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman -->,
p = 0.751 <!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman_p -->).

The negative control behaves as it should: diazepam, a positive modulator rather than an antagonist,
is called GABA-A antagonist in only 0.077
<!-- metrics.json: mechanism_baselines.task2_external_detection.sharf_diazepam_negative_control.detection_rate -->
of its records.


**This is a weak result reported as a weak result.** A detection rate near a third with no dose
relationship is not a screening tool; it is evidence that the mechanism signal partially survives a
platform change.

### 6.3 Species transfer (rat → human)

| setting | balanced accuracy | 95% CI |
|---|---|---|
| human, within-species | 0.646 <!-- metrics.json: species_transfer.transfer.none.human_within.balanced_accuracy --> | — |
| rat, within-species | 0.543 <!-- metrics.json: species_transfer.transfer.none.rat_within.balanced_accuracy --> | — |
| rat → human, no adaptation | 0.55 <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy --> | [0.4, 0.6] <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy_ci95 --> |
| rat → human, CORAL | 0.442 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy --> | [0.302, 0.591] <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy_ci95 --> |
| rat → human, per-species standardisation | 0.408 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy --> | [0.267, 0.568] <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy_ci95 --> |

Chance is 0.2 <!-- metrics.json: species_transfer.transfer.none.chance_balanced_accuracy -->.
Transfer works above chance (permutation p = 0.0002
<!-- metrics.json: species_transfer.transfer.none.rat_to_human.permutation_p -->), and **both
adaptation methods made it worse**. With 37 training and 32 test records
<!-- metrics.json: species_transfer.transfer.none.rat_to_human.n_test -->, CORAL is estimating a
covariance it cannot support. Reporting adaptation as a negative result is the useful outcome here.

### 6.4 Task 3 — platform contrast

The same mechanism at ≥30 µM
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.min_dose_uM --> measured on two
platforms: gabazine on 16-electrode Axion wells, n = 10
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.by_source.kapucu2022.n -->, against
bicuculline on multi-electrode arrays, n = 167
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.by_source.screenneuropharm_bic.n -->.
Five features agree in sign and four disagree
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.features_agreeing_in_sign -->
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.features_disagreeing_in_sign -->.

![The same mechanism measured on two platforms, REAL data: per-feature log2 change for GABA-A antagonists. Mean firing rate reverses sign across platforms; synchrony agrees.](../outputs/figures/platform_contrast_gabaa_antagonist.png){width=63%}

**The disagreement does not fall where the species analysis puts it.** `mfr_hz` reverses, +0.348
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.by_source.kapucu2022.median_delta.mfr_hz -->
against −0.684
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.by_source.screenneuropharm_bic.median_delta.mfr_hz -->,
while `synchrony_index` agrees, +1.743
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.by_source.kapucu2022.median_delta.synchrony_index -->
against +0.148
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.by_source.screenneuropharm_bic.median_delta.synchrony_index -->.
That is the opposite of the pattern in §6.5 and §6.6, and it is the reason this report separates the
two claims: **the portable-feature result is about species, and does not carry across platforms.**

Three cautions on how far this goes. The contrast confounds compound (gabazine against bicuculline)
with electrode count, recording duration and culture age, so it bounds portability rather than
explaining it. The two arms are ten wells against 167 records. And `active_electrodes` is counted as
a disagreement while its median delta is exactly 0.000
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.by_source.kapucu2022.median_delta.active_electrodes -->
on both platforms — a zero has no sign, so the substantive count is three disagreements, not four.

### 6.5 Which features survive a species change

The most shifted features between species are `synchrony_index`
<!-- metrics.json: species_transfer.species_shift.most_shifted_features -->, `isi_cv` and
`network_burst_rate_per_min`; the most invariant are `mfr_hz`
<!-- metrics.json: species_transfer.species_shift.most_invariant_features -->,
`pct_spikes_in_bursts` and `burst_rate_per_min` (`outputs/figures/species_shift.png`). The synchrony
shift is large — a difference of −4.179
<!-- metrics.json: species_transfer.species_shift.by_feature.synchrony_index.difference --> in log2
units — though its interval [−5.7435, 0.1868]
<!-- metrics.json: species_transfer.species_shift.by_feature.synchrony_index.difference_ci95 -->
reaches zero, so the ordering is better supported than the magnitude. The same hedge applies to
`isi_cv`, whose interval [−1.6738, 0.15]
<!-- metrics.json: species_transfer.species_shift.by_feature.isi_cv.difference_ci95 --> also reaches
zero; only `network_burst_rate_per_min` has an interval that excludes it,
[−9.7879, −0.3389]
<!-- metrics.json: species_transfer.species_shift.by_feature.network_burst_rate_per_min.difference_ci95 -->.
The ranking is by standardised difference, so "most invariant" does not mean small: `burst_rate_per_min`
is named invariant while carrying a median difference of 1.04
<!-- metrics.json: species_transfer.species_shift.by_feature.burst_rate_per_min.difference --> log2
units, a two-fold shift whose interval also spans zero.

![Per-feature species shift, REAL data: rat against human baseline distributions, with cluster-bootstrap intervals. Rate and burst-fraction features move least; synchrony moves most.](../outputs/figures/species_shift.png){width=52%}


### 6.6 Ablations: the feature set is mostly one feature

| feature subset | balanced accuracy | 95% CI |
|---|---|---|
| all nine | 0.514 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.all.balanced_accuracy_mean --> | [0.3983, 0.6619] <!-- metrics.json: uncertainty_and_ablations.feature_subsets.all.balanced_accuracy_ci95 --> |
| rate only | 0.53 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.rate_only.balanced_accuracy_mean --> | [0.3964, 0.6235] <!-- metrics.json: uncertainty_and_ablations.feature_subsets.rate_only.balanced_accuracy_ci95 --> |
| species-invariant three | 0.516 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.species_invariant.balanced_accuracy_mean --> | [0.371, 0.6295] <!-- metrics.json: uncertainty_and_ablations.feature_subsets.species_invariant.balanced_accuracy_ci95 --> |
| synchrony only | 0.455 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.synchrony_only.balanced_accuracy_mean --> | [0.3554, 0.586] <!-- metrics.json: uncertainty_and_ablations.feature_subsets.synchrony_only.balanced_accuracy_ci95 --> |
| network measures only | 0.417 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.network_only.balanced_accuracy_mean --> | [0.2983, 0.5573] <!-- metrics.json: uncertainty_and_ablations.feature_subsets.network_only.balanced_accuracy_ci95 --> |

Two rate features match all nine. Leave-one-out agrees: dropping `mfr_hz` changes balanced accuracy
by −0.05 <!-- metrics.json: uncertainty_and_ablations.leave_one_out.mfr_hz.delta_vs_all -->, while
dropping `burst_rate_per_min` changes it by 0.0
<!-- metrics.json: uncertainty_and_ablations.leave_one_out.burst_rate_per_min.delta_vs_all -->.

### 6.7 Compartment asymmetry in real devices

Across 16 two-chamber devices <!-- metrics.json: compartment_asymmetry.n_devices -->, the statistic
that separates device types is coupling **lag**, not asymmetry. Median absolute lag is 11.25 ms
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.CxHp.median -->
for cortical–hippocampal devices against 5.0 ms
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.CxCx.median -->
and 5.0 ms
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.HpHp.median -->
for the two same-tissue types (Kruskal p = 0.0082
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.kruskal_p -->; Holm-adjusted
0.0413
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.pairwise_holm.CxCx_vs_CxHp -->
and 0.0257
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.pairwise_holm.CxHp_vs_HpHp -->
against the two homogeneous types).

**The internal control behaves.** The two same-tissue types are indistinguishable from each other,
Holm 0.4785
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.pairwise_holm.CxCx_vs_HpHp -->,
which is what a real device-type effect should look like and what a pooled heterogeneous-against-
homogeneous split cannot show.

![Inter-compartment coupling lag by device type, REAL data, one value per device: five CxCx, five HpHp, six CxHp. The two same-tissue types are indistinguishable from each other; the heterogeneous type sits apart.](../outputs/figures/compartment_lag_by_device_type.png){width=61%}

Two things this section does **not** claim. On absolute asymmetry rather than lag, the same per-type
comparison does not survive correction, Holm 0.0519
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_asymmetry.pairwise_holm.CxCx_vs_CxHp -->,
so lag is the claim and asymmetry is not. And a pooled heterogeneous-against-homogeneous test of
asymmetry *does* cross the line, p = 0.042
<!-- metrics.json: compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.p_value -->; it is kept
in `metrics.json` and is deliberately not the headline here, because pooling CxCx with HpHp is an
artefact — the two homogeneous types differ from each other on peak correlation, so pooling them
manufactures a contrast. An earlier draft of this report led with that pooled test; the lab notebook
had already recorded it as abandoned, and this section now reports the analysis that replaced it.

With five to six devices per group, this supports an ordering of device types, not an effect size,
and §7.2 gives a specific reason to distrust the asymmetry measure in particular.

### 6.8 Untreated baselines: what differs before any drug is applied

Everything above consumes pairs — an exposure minus its own baseline. That is why adding 184 human
organoid recordings <!-- metrics.json: data_asset.counts.by_source.trujillo2019 --> changed none of
the numbers in §6.1 to §6.7: they are untreated baselines with nothing to pair against. They answer
a different question — before any compound, does human tissue sit inside the rat distribution?

Species and platform are confounded here — every MaxWell recording is human, every Multi Channel
Systems recording is rat — so the comparison is made inside one vendor, resampling over culture
units rather than rows: an organoid recorded twenty times is one culture.

**At matched platform and matched preparation, 7
<!-- metrics.json: baseline_activity.matched_preparation.n_features_separating_species -->
of the nine features separate the species** (Axion, dissociated monolayers, 28 human cultures
<!-- metrics.json: baseline_activity.matched_preparation.n_human_cultures --> against 44 rat
<!-- metrics.json: baseline_activity.matched_preparation.n_rat_cultures -->). Human cultures fire
faster, put more of their spikes into bursts — a median of 70.9
<!-- metrics.json: baseline_activity.matched_preparation.features.pct_spikes_in_bursts.median_human -->
against 40.05
<!-- metrics.json: baseline_activity.matched_preparation.features.pct_spikes_in_bursts.median_rat -->
percent — and are four times as synchronous. Only burst rate overlaps. Dropping the untreated wells
recorded during a plate's exposure phase leaves the same 7
<!-- metrics.json: baseline_activity.matched_preparation_baseline_phase_only.n_features_separating_species -->,
so this is not an artifact of session order.

This qualifies §6.5 rather than contradicting it: that section found rate-like **drug-response
deltas** portable, and a delta can travel between two populations whose resting activity does not
resemble each other at all. Preparation separates from species too — within human tissue on one
vendor, 3D organoids differ from dissociated monolayers on 5
<!-- metrics.json: baseline_activity.preparation_within_human.n_features_separating_species -->
of the nine features, so "human" does not name one kind of preparation.

**A confound the module refuses rather than reports.** Active-electrode count looked like the
strongest separator of all, reaching 49 electrodes
<!-- metrics.json: baseline_activity.within_vendor.features.active_electrodes.difference_ci95.1 -->
— an artifact of Trujillo's 64-electrode wells against Kapucu's 16. It is flagged
`format_confounded` and excluded; the fraction of electrodes active gives an interval of −0.031 to
0.063 <!-- metrics.json: baseline_activity.within_vendor.features.active_fraction.difference_ci95 -->,
astride zero —
the fourth apparent effect here dissolved by asking what else differs between the groups.

![Untreated baseline activity on Axion platforms, by species and preparation. REAL recordings. The
three groups are shown separately because the species contrast and the preparation contrast are both
present in this slice; two groups would let the whole gap be read as species. Burst rate is the one
feature flat across all three.](../outputs/figures/baseline_species_comparison.png){width=81%}

---

## 7. Results: the digital twin

### 7.1 Model

Two compartments of conductance-based leaky integrate-and-fire neurons with spike-frequency
adaptation, 80/20 excitatory/inhibitory, recurrent intra-compartment connectivity and **directional**
inter-compartment projections whose forward/backward ratio is the parameter of interest. Poisson
background drive. Connectivity is parameterised by **in-degree rather than probability**, so a
calibration done at one network size still holds at another.

The parameters carried by the network, and how each was fixed:

| parameter | role | how it was set |
|--------------------------------------|----------------------|----------------------|
| `intra_in_degree` | recurrent inputs per neuron | calibrated, searched over 10-35 |
| `inter_in_degree` | cross-compartment inputs per neuron | inferred, prior 4-16 |
| `forward_ratio` | forward against backward pathway strength | inferred, prior 1-16 in log space |
| `excitatory_weight`, `inhibitory_weight` | synaptic conductance steps | calibrated |
| `background_rate_hz`, `background_weight` | Poisson drive | calibrated; drive to the target chamber is a separate scale |
| `adaptation_increment`, `adaptation_tau_ms` | spike-frequency adaptation | calibrated |
| `threshold_spread` | per-neuron excitability heterogeneity | calibrated, searched over 0-0.3 |
| `detection_radius_um` | how far an electrode hears | calibrated, searched over 100-300 µm |
| `gabaa_scale`, `ampa_scale`, `threshold_scale` | drug knobs, 1.0 is untreated | inferred |

Identical neurons synchronise almost perfectly, which no real culture does, so `threshold_spread`
exists to stop the twin being trivially more synchronous than any recording. `detection_radius_um`
is a property of the virtual electrode rather than of the culture, and is called out as such in the
calibration record so it is not read as a biological parameter.

Neurons are placed on a virtual electrode grid and only spikes within a detection radius reach an
electrode; the resulting per-electrode spike trains go through the same `features/mea.py` as real
recordings, and — after a defect found in HD16 — through the same coupling measure on the same
per-electrode input.

### 7.2 The measure the twin exists to use is confounded

Coupling asymmetry was expected to report pathway direction. It reports relative firing rate. At
**fixed** wiring, lowering the receiving chamber's background drive moves asymmetry from 0.109
<!-- metrics.json: twin_hd11_validation.directionality_sweep.0.asymmetry_mean --> to 0.681
<!-- metrics.json: twin_hd11_validation.directionality_sweep.2.asymmetry_mean -->; at **fixed**
drive, raising the forward ratio from 1 to 16 moves it only from 0.109 to 0.143
<!-- metrics.json: twin_hd11_validation.directionality_sweep.6.asymmetry_mean -->. The measure's own
range is [−1, 1], so this is not the whole of it — but drive moves the statistic several times
further than the wiring it is supposed to be reporting. This is the most
consequential single finding about the twin, and it propagates into every later section: any analysis
that reads direction off coupling must control for rate first.

Three checks on the real devices confirmed the confound does not obviously bite there (firing-rate
imbalance does not differ by device type), which is why §6.6 survives as a hypothesis.

![SIMULATED: measured coupling asymmetry against the forward/backward wiring ratio, at three levels of drive to the receiving chamber. Drive moves the measure further than wiring does.](../outputs/twin/hd11_directionality_simulated.png){width=60%}


### 7.3 Calibration

A random search over seven tuned parameters
<!-- metrics.json: twin_hd12_calibration.tuned_parameters --> brought the simulated baseline into the
real band for `mfr_hz`, `burst_rate_per_min` and `network_burst_rate_per_min`, and **not** for
`pct_spikes_in_bursts` or `synchrony_index`
<!-- metrics.json: twin_hd12_calibration.inside_real_band -->. The residual mismatch is precisely the
burst-structure and synchrony group that §6.5 and §6.6 independently identify as the unreliable part
of the feature set across species. Three analyses agree about species; the platform contrast of §6.4
disagrees, and that disagreement is itself the finding.

![SIMULATED against REAL: calibrated baseline features of the twin compared with the real baseline band. Rate and burst-rate features land inside it; burst fraction and synchrony do not.](../outputs/twin/hd12_calibration_simulated.png){width=60%}


### 7.4 Drug knobs

Each knob is compared against the real direction for its mechanism, scoring only features whose real
direction is established by a cluster bootstrap:

| knob | agrees / scored |
|---|---|
| GABA-A block | 4 <!-- metrics.json: twin_hd13_perturbation.knobs.0.n_features_agreeing --> / 5 <!-- metrics.json: twin_hd13_perturbation.knobs.0.n_features_scored --> |
| GABA-A potentiation | 2 <!-- metrics.json: twin_hd13_perturbation.knobs.1.n_features_agreeing --> / 4 <!-- metrics.json: twin_hd13_perturbation.knobs.1.n_features_scored --> |
| AMPA block | 4 <!-- metrics.json: twin_hd13_perturbation.knobs.2.n_features_agreeing --> / 4 <!-- metrics.json: twin_hd13_perturbation.knobs.2.n_features_scored --> |
| sodium-channel block | 3 <!-- metrics.json: twin_hd13_perturbation.knobs.3.n_features_agreeing --> / 3 <!-- metrics.json: twin_hd13_perturbation.knobs.3.n_features_scored --> |

Directions only: each knob is compared against a real reference from a different platform, so this
supports direction, never magnitude.

![SIMULATED drug knobs against REAL pharmacology directions, per feature. Only features whose real direction is established by a cluster bootstrap are scored.](../outputs/twin/hd13_drug_knob_validation_simulated.png){width=56%}


### 7.5 Simulation-based inference, and what it cannot do

A neural posterior estimator was trained on 3,500 simulations
<!-- metrics.json: twin_hd14_inference.n_simulations -->.

| parameter | posterior contraction | KS statistic |
|------------------------|-----------------------|--------------|
| `threshold_scale` | 0.8 <!-- metrics.json: twin_hd14_inference.contraction.threshold_scale.contraction --> | 0.0571 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.threshold_scale.ks_statistic --> |
| `ampa_scale` | 0.443 <!-- metrics.json: twin_hd14_inference.contraction.ampa_scale.contraction --> | 0.0871 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.ampa_scale.ks_statistic --> |
| `gabaa_scale` | 0.323 <!-- metrics.json: twin_hd14_inference.contraction.gabaa_scale.contraction --> | 0.1093 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.gabaa_scale.ks_statistic --> |
| `inter_in_degree` | 0.32 <!-- metrics.json: twin_hd14_inference.contraction.inter_in_degree.contraction --> | 0.1029 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.inter_in_degree.ks_statistic --> |
| `log10_forward_ratio` | 0.311 <!-- metrics.json: twin_hd14_inference.contraction.log10_forward_ratio.contraction --> | 0.0957 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.log10_forward_ratio.ks_statistic --> |

Only excitability is well identified, and only `threshold_scale` passes a rank-uniformity check
(p = 0.1956 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.threshold_scale.ks_p -->;
every other parameter is below 0.01). **Posterior intervals from this estimator should not be quoted
as calibrated.**

That statement is weaker than it looks, and the reason is worth recording. Posterior *training* was
unseeded until this report was being built: rerunning identical code on the identical simulation
cache moved contraction by about 0.02 and moved this p-value between 0.46, 0.0092 and 0.1956 — on
one of those runs `threshold_scale` failed the check that it passes here. Training is now seeded and
verified to reproduce across processes, but the run-to-run spread is the honest measure of how much
weight a single rank-uniformity p-value can carry. Contraction was stable across the same runs; the
rank test was not. A learning curve
showed connectivity contraction still rising with training-set size but only slightly: more
simulations alone will not fix it.

![SIMULATED posterior calibration: rank of the true parameter among posterior samples. Flat is calibrated, U-shaped is over-confident. Only `threshold_scale` is close to flat.](../outputs/twin/hd14_posterior_ranks_simulated.png){width=67%}


**Applied to 113 real two-chamber recordings**
<!-- metrics.json: twin_hd15_real_inference.n_recordings --> from 16 devices, with 99.2% of real
summary values inside the simulated range
<!-- metrics.json: twin_hd15_real_inference.distribution_check.mean_fraction_inside -->:

- The inferred GABA-A scale shows **no dose trend**: Spearman 0.118
  <!-- metrics.json: twin_hd15_real_inference.gabaa_scale_dose_trend.spearman_rho -->,
  95% CI [−0.168, 0.419]
  <!-- metrics.json: twin_hd15_real_inference.gabaa_scale_dose_trend.spearman_rho_ci95 -->,
  clustered over devices.
- **Untreated recordings already infer 0.3781**
  <!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median -->,
  CI [0.3149, 0.4677]
  <!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median_ci95 -->, excluding the
  neutral value of 1.0. The knob is absorbing general model mismatch.
- Four summary dimensions — burst rate and active electrodes in both chambers —
  <!-- metrics.json: twin_hd15_real_inference.distribution_check.unsupported_dimensions --> pass a
  range check but have almost no simulated draws where the real values sit. Range containment is not
  density containment.

An earlier version of this analysis produced an apparent dose trend (ρ = −0.197, p = 0.036). It was
an artefact of pseudo-replication — 113 recordings from 16 devices treated as independent — and of
posterior extrapolation on out-of-distribution dimensions. Both are now checks in code.

![Inferred GABA-A scale for REAL bicuculline recordings, from a posterior trained on SIMULATIONS. There is no dose trend, and untreated chips already sit far below the neutral value of 1.](../outputs/twin/hd15_real_dose_response.png){width=56%}


### 7.6 Directionality: what it takes to see asymmetry

120 simulated networks per arm <!-- metrics.json: twin_hd16_directionality.n_networks --> with known
wiring, symmetric against asymmetric. Electrode count is varied by subsampling one fixed array and
recording length by truncating one simulation, so neither varies the culture.

The naive version is confounded exactly as §7.2 predicts: the full feature set separates the classes
at AUC from 0.9219
<!-- metrics.json: twin_hd16_directionality.summary_by_feature_set.free_rate.all.min --> to 0.9863
<!-- metrics.json: twin_hd16_directionality.summary_by_feature_set.free_rate.all.max -->, and three
firing-rate features alone reach from 0.8644
<!-- metrics.json: twin_hd16_directionality.summary_by_feature_set.free_rate.rate_only.min --> to 0.9841
<!-- metrics.json: twin_hd16_directionality.summary_by_feature_set.free_rate.rate_only.max -->.

Residualising every feature against both chambers' firing rates removes that, and carries its own
validity check: rate features after residualising score from 0.4772
<!-- metrics.json: twin_hd16_directionality.summary_by_feature_set.free_rate.rate_only_rate_residualized.min -->
to 0.6213
<!-- metrics.json: twin_hd16_directionality.summary_by_feature_set.free_rate.rate_only_rate_residualized.max -->
and do not clear chance in every cell. Direction still separates at 0.8825
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.auc -->,
95% CI [0.8184, 0.9359]
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.auc_ci95 -->,
from a 10 s recording
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.duration_s -->
on 6 electrodes per compartment
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.electrodes_per_compartment -->.

**The rate-matched arm does not pass its own check, and is not the result.** `metrics.json` states
the criterion: in that arm, rate-only features should fall towards chance, and if they do not, the
matching failed and the arm says nothing about direction. They do not — rate-only AUC there runs
from 0.8587
<!-- metrics.json: twin_hd16_directionality.summary_by_feature_set.rate_matched.rate_only.min -->
to 0.9859
<!-- metrics.json: twin_hd16_directionality.summary_by_feature_set.rate_matched.rate_only.max -->.
The drive adjustment did not remove the rate imbalance so much as reverse its sign, from −0.2756 and
−0.5180 before to +0.3116 and +0.2377 after
<!-- metrics.json: twin_hd16_directionality.rate_imbalance_by_arm -->, leaving a worst relative gap
of 0.26
<!-- metrics.json: twin_hd16_directionality.matching_quality.rate_matched.worst_relative_rate_gap -->
against 0.2248
<!-- metrics.json: twin_hd16_directionality.matching_quality.free_rate.worst_relative_rate_gap -->
before it. That arm reaches AUC 0.9869
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.rate_matched_residualized.auc -->
at the same electrode count, and an earlier draft of this report quoted its predecessor as the
headline. It is withdrawn: the guardrail written into `metrics.json` says the arm is
uninterpretable, and the correct response is to report the arm that passes.

**The residualisation is fitted inside each fold.** An earlier version fitted the rate regression
once across the whole cell before cross-validating, letting test rows help fit the coefficients that
built their own features. Moving it inside the training fold took this section's AUC from 0.9359 to
0.8825 and, because the criterion is the interval's *lower* bound, moved the cheapest qualifying
configuration from 24 electrodes per compartment to 6.

**Read this as a contrast, not a measurement.** The same summaries that classify two well-spaced
wiring regimes at 0.98 give a posterior contraction of only 0.311 for the continuous ratio (§7.5).
Classifying is a far weaker demand than estimating.

![SIMULATED, drive left alone: telling symmetric from asymmetric wiring, by recording length and electrode count. `all_rate_residualized` is the controlled comparison this section reports; `rate_only_rate_residualized` is the validity check it has to pass. The rate-matched arm, which does not pass it, is plotted separately in `outputs/twin/hd16_directionality_rate_matched_simulated.png` — split out because eight panels across one text block were unreadable at print size.](../outputs/twin/hd16_directionality_simulated.png){width=78%}


### 7.7 The same question on real devices, and a null

§7.6 could only ask its question in simulation, where the ground truth is something we set. The
Mateus deposit (§4.2) has it fabricated into the device: straight `control` microchannels against
`arrows`, `rams` and `tesla`-valve geometries, 57 recordings
<!-- metrics.json: microchannel_directionality.n_recordings --> from 26 chips
<!-- metrics.json: microchannel_directionality.n_chips -->.

The deposit publishes spike times and the measurement but not the electrode map it was computed on,
so the geometry is recovered with the authors' own criterion — an event on all five electrodes of a
microchannel with at most 1 ms between consecutive electrodes. That finds 10–15 channels per chip in
a single row band, which is what the published five-electrode design predicts. Direction is reported
as a magnitude: the deposit does not record how each device was seated on the array, so |asymmetry|
says how one-sided propagation is, not which way it points.

![REAL data: propagation asymmetry per chip by microchannel geometry. One point per chip, bars are medians. The control group spans almost the whole range on its own.](../outputs/figures/microchannel_asymmetry_real.png){width=60%}

**The unit decides the answer.** A chip recorded at DIV 12, 17 and 21 supplies three recordings and
one device. Counting recordings, median |asymmetry| runs from 0.339
<!-- metrics.json: microchannel_directionality.per_recording_naive.by_condition.control.median_abs_asymmetry -->
for control to 0.854
<!-- metrics.json: microchannel_directionality.per_recording_naive.by_condition.rams.median_abs_asymmetry -->
for rams, which looks like a geometry effect. Counting chips, it is 0.617
<!-- metrics.json: microchannel_directionality.comparison.by_condition.control.median_abs_asymmetry -->
against 0.875
<!-- metrics.json: microchannel_directionality.comparison.by_condition.rams.median_abs_asymmetry -->
with Kruskal p = 0.64
<!-- metrics.json: microchannel_directionality.comparison.kruskal_p --> and every Holm-adjusted
comparison against control at 1.0
<!-- metrics.json: microchannel_directionality.comparison.vs_reference_holm.rams -->. Both are kept
in `metrics.json`, the per-recording one labelled as the wrong unit, because the gap between them is
the result. This is the pseudo-replication of §7.5 a third time, on a different dataset.

**The statistic is also confounded with sampling.** |asymmetry| is a ratio of counts, so a chip with
few propagation events looks one-sided by chance; across chips the correlation between event count
and |asymmetry| is −0.494
<!-- metrics.json: microchannel_directionality.event_count_confound.spearman_rho -->, p = 0.010
<!-- metrics.json: microchannel_directionality.event_count_confound.p_value -->. Restricting to the
better-sampled half leaves 13 chips
<!-- metrics.json: microchannel_directionality.well_sampled_half.n_chips --> and p = 0.086
<!-- metrics.json: microchannel_directionality.well_sampled_half.kruskal_p -->, which is not a
result either.

**What this does not say.** It does not contradict the deposit's own paper: those authors had the
electrode map, measured per-channel propagation velocities, and asked a different question. The
claim here is narrower — *this* statistic, aggregated per chip, does not separate these geometries,
on 26 chips in five groups of three to eight, with control chips confined to experiments 9–11 so
that condition is confounded with culture batch by the deposit's design.

**The null is not an artefact of the geometry assumptions.** The detector assumes the channels run
down the array's columns and that all five of their electrodes fired; the deposit settles neither.
Running it under both orientations, and again with one electrode allowed to miss, gives four
detectors and none separates the conditions — Kruskal p = 0.13074
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_one_missing.kruskal_p -->,
0.33528
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.row_one_missing.kruskal_p -->
and 0.77195
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.row_strict.kruskal_p -->
beside the headline's 0.63969. The orientation is also more than an assumption: down the columns the
detector finds a median of 14
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_strict.median_channels_detected -->
channels per recording against 1
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.row_strict.median_channels_detected -->
along the rows, so the devices were seated the way the detector assumes. Tolerating a missing
electrode triples the events to 1,101,233
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_one_missing.total_events -->
and still finds nothing.

**What it says about the twin.** §7.6's directionality result is simulation-only, and this is the
first time that question has been put to devices whose asymmetry was fabricated rather than
simulated. It does not reproduce. The sample is too small and too confounded to call that a
refutation, and too directly aimed to leave out.

**A second real-device test, of the confound rather than the result.** §7.2's finding — that the
asymmetry measure reads relative activity rather than pathway direction — had only ever been shown
in the twin, where drive could be moved at fixed wiring. `brewer2021`'s four-chamber cultures allow
a version of it on real tissue whose chambers are joined by instrumented tunnels. Across 46 chamber
pairs <!-- metrics.json: tunnel_rate_tracking.n_pairs --> in 9 cultures
<!-- metrics.json: tunnel_rate_tracking.n_devices -->, coupling asymmetry has no rank relationship
with the pair's firing-rate imbalance: Spearman ρ = −0.0527
<!-- metrics.json: tunnel_rate_tracking.spearman_rho -->, 95% CI [−0.5988, 0.4665]
<!-- metrics.json: tunnel_rate_tracking.spearman_rho_ci95 -->, over rate imbalances reaching a log2 ratio of 4.947
<!-- metrics.json: tunnel_rate_tracking.abs_log2_rate_ratio.max -->, a thirty-fold difference. This is **not** a refutation of §7.2: that section moved drive with wiring held
fixed, while these pairs differ in wiring and rate at once, so a null across them is consistent with
a strong within-device effect. What it says is that rate imbalance alone does not predict measured
asymmetry in real tissue, over an interval wide enough to admit a substantial effect either way. The
scatter is in `outputs/figures/tunnel_asymmetry_vs_rate.png`.

---

## 8. Limitations

**Data.** The asset is 21,530 rat records
<!-- metrics.json: data_asset.counts.by_species.rat --> against 357 human
<!-- metrics.json: data_asset.counts.by_species.human -->; every human claim rests on three sources.
That rat figure had been stale since `mateus2024` was added, because it carried no citation comment
— the checker in §9.2 guards only marked numbers, which is an argument for marking them all.
Most records are curve fits from which our features cannot be recomputed. Dropping Tukker 2020 on
license grounds leaves the *transfer* study on a single source, and Trujillo 2019 does not relieve
it: its drug sessions say a compound was applied but not which well received which, so those 24
records are flagged and excluded from pairing rather than labelled by guesswork. Nineteen of its 42
sessions are MATLAB 5 files that cannot be read in place, so its developmental series starts late.
The §6.8 human side rests on 8 organoid cultures
<!-- metrics.json: baseline_activity.preparation_within_human.n_human_cultures --> recorded many
times each, not 160 independent ones. Compound coverage is no longer thin - `vassallo2017` took the
modelling table from 73 pairs to 754 across nine mechanism classes - but it is uneven: over half of
those pairs are one compound at many doses, and the EPA chips carry four classes of which three are
its negative controls.

**Models.** Balanced accuracies around 0.5 on six classes are above chance but far from usable;
intervals are wide and overlapping, and on §6.1's independent platform the interval contains chance
outright. Conformal sets cover honestly but are too large to act on.
Calibration error may be as high as 0.2557
<!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.ece_ci95.1 -->. The nine-feature
vector is effectively a rate feature plus noise for this task. The implementation uses single-level
grouped CV where the plan specified nested CV, so hyperparameter choice is not fully held out.

**The twin.** Coupling asymmetry tracks rate, not direction (§7.2). Calibration misses burst
structure and synchrony (§7.3). Connectivity parameters are weakly identified and no parameter's
posterior is calibrated except by the weakest test (§7.5). On real data the drug knob absorbs model
mismatch (§7.5). Simulated recordings are 10 s against real recordings of 480 s, the length ScreenNeuroPharm deposited. The simulated array
is one fixed geometry while real chambers carry 48, 36 or 60 electrodes. HD16's positive result is
identifiability *under a correct model*, since the same simulator generated both classes, and its
rate controls remove only linear rate dependence.

**The real-device test.** The §7.7 null rests on 26 chips in five groups, with condition confounded
with culture batch by the deposit's design, an unanchored direction, and a statistic that varies with
how many propagation events a chip produced. It bounds the effect rather than excluding it. The
microchannel geometry is also recovered from the spike times rather than read from a map the deposit
does not publish, so it approximates the authors' own analysis rather than reproducing it.

**Process.** CI runs on every push and both jobs pass, at commit `1a28703`; the quick pipeline
takes 1,126 s on the two-vCPU runner a private repository gets, against 401 s here, so the
ten-minute budget is a claim about this machine. The demo video has been recorded. Every twin metric
produced before the coupling-definition fix is incomparable with those after it, and was regenerated.

A fuller list, with dates and commits, is in `report/LAB_NOTEBOOK.md` (33 numbered limitations).

---

## 9. Reproducibility

```bash
pip install uv
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
python run_all.py --quick     # path test: every stage, reduced settings (see below)
python run_all.py             # full run
```

- `--quick` completes in **401 s on a fresh clone** of this repository — install, download, build,
  model, twin and demo — against a 600 s budget. That is a 20-core machine; the same pipeline takes
  1,126 s on the two-vCPU runner CI provides, which is the number to plan against. Each run records
  its own per-stage timings
  under `metrics.run_all_timings_quick`. Most of the variable part is downloading: the same clone
  measured 52 s and 133 s for the fetch stage on two runs; the full twin stage takes roughly 2,000 s
  and records `metrics.run_all_timings_full`. Both are logged under `outputs/logs/` as well.
- **`--quick` is a path test, not a reproduction of the report.** Measured on a fresh clone, it
  fetches the small-tier sources only and therefore builds a much smaller asset than the one this
  report describes; it produces 15 of the 16 figures the pipeline draws - the exception is the rate-matched
  directionality arm, which the quick tier does not simulate - and omits six
  `metrics.json` entries —
  `toxcast_mea_inventory`, `twin_hd12_calibration`, `twin_hd15_real_inference`,
  `compartment_asymmetry`, `microchannel_directionality` and the full-run timings. The analyses
  behind those either need a full-tier download (the 109 MB microchannel archive, the ToxCast
  release) or need compartment records the small-tier asset does not contain, and each skips itself
  with a logged reason rather than failing. The §7.6 sweep additionally has too few simulations per
  cell to fit a discriminator, so its `cheapest_configuration` entries are written as `null`.
- **Reproducing the numbers in this report takes the full path**: `python run_all.py` after a
  full-tier fetch. `metrics.json` records which mode produced each stored value, and quick and full
  runs write to separate roots (`outputs/` against `outputs/quick/`) so neither can overwrite the
  other.
- Dependencies are pinned in `pyproject.toml`; numpy is held at 2.3.5 because brian2 2.9.0 does not
  import against 2.4.
- Every number in this report is produced by a script under `src/` and written to
  `outputs/metrics.json`; `python -m neuromps.report.verify` checks the report against it.
- Caches refuse to be reused across a change in what they mean: the simulation cache stores its
  summary names, electrode count and a hash of the source of every function that defines a summary.
- Tests run with `pytest` (feature extraction, QC flags, provenance hashing, the MEA simulator, and
  a contract test that fails if any headline metric loses its confidence interval).

### 9.1 Where the time goes

Each run writes its own stage timings, so this is generated rather than transcribed: see
`metrics.run_all_timings_quick.by_stage_s` and `run_all_timings_full`. On the fresh-clone run above,
the twin stage dominates, then downloading, then the asset build.

### 9.2 What the repository enforces, rather than documents

- **Provenance.** The asset build fails if any record's provenance hash disagrees with the manifest.
- **One feature extractor.** The twin computes features through the same `features/mea.py` as real
  recordings, and — after a defect that survived for several phases — through the same coupling
  measure on the same per-electrode input.
- **Interval discipline.** A contract test fails if any headline metric loses its confidence
  interval, and the resample count is recorded next to every estimate.
- **Cache meaning.** The simulation cache stores its summary names, its electrode count and a hash
  of the source of every function that defines a summary; it refuses to load across a change in any
  of them.
- **Report numbers.** `neuromps.report.verify` re-reads every citation in this document and fails on
  disagreement.
- **No clinical claims.** A test scans the reports for efficacy, diagnostic and treatment language
  and fails unless the line is a disclaimer ruling it out.

---

## 10. The demo

`streamlit run app/app.py`, after `python run_all.py --quick`, opens three tabs — the data asset and
its standard, mechanism prediction with a conformal set beside the transfer results, and the twin
with its posterior and what it does on real recordings.

No interaction waits on a simulator: `run_all.py --stage demo` precomputes a grid of 75
<!-- metrics.json: demo_precompute.n_grid_points --> twin simulations spanning the prior with
posterior samples per point, posterior samples for all 113 real two-chamber recordings
<!-- metrics.json: demo_precompute.n_real_recordings -->, and a classifier fitted once, in 66.3 s
<!-- metrics.json: run_all_timings_full.by_stage_s.demo -->. Through Streamlit's own test harness
every interaction lands between 0.6 and 0.9 s; the exception is a button running one Brian2
simulation in about 4.5 s, so the twin is verifiably a simulator and not a lookup table. The app
reads `outputs/metrics.json` and `outputs/demo/` and recomputes nothing, so it cannot disagree with
this report. Detail is in `app/README.md`, screenshots in `outputs/figures/app_tab*.png`.

---

## 11. AI assistance disclosure

This work was produced in a paired workflow with Anthropic's Claude (Claude Opus 5) via Claude Code.
The model wrote the large majority of the code, tests and prose in this repository, under
step-by-step human direction: the human set the plan and phase gates, chose the scope, approved or
rejected each step, and directed the scientific decisions.

Specific practices used to keep that honest, all visible in the repository history:

- Numbers are never written by hand into documents. Every figure in this report is emitted by a
  script into `outputs/metrics.json` and re-checked against it by `neuromps.report.verify`, which
  caught a stale timestamp and three mis-anchored citations while this report was being written.
- Results were reported as observed, including when they contradicted the plan's expectation — the
  negative findings in §6.2, §6.3, §7.2 and §7.5 are examples.
- `report/LAB_NOTEBOOK.md` records what was tried and abandoned, with dates and commit hashes,
  including several defects the model introduced and then found in its own work.

---

## 12. Sources, licenses, and where each result lives

Data sources, with licenses and citations, are listed in `CITATIONS.md` and recorded per file in
`data/raw/MANIFEST.json`. Third-party software licenses are in `THIRD_PARTY.md`. This repository is
released under the MIT license (`LICENSE`).

Every number here comes from a key in `outputs/metrics.json`, written by a script under `src/`;
`report/RESULT_INDEX.md` maps each result to its key and its script. Figures go to
`outputs/figures/` (real) and `outputs/twin/` (simulated, marked in the filename except
`hd15_real_dose_response.png`, real recordings under a simulation-trained model) — the rule is
enforced in the captions, not the filenames.
