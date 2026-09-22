# Results (Phase 1–2)

> **Working document, superseded by `report/TECHNICAL_REPORT.md`.** This is the Phase 1-2 results
> draft, kept because it carries per-result detail the report had to cut for length. Its numbers are
> live — `python -m neuromps.report.verify` checks them against `outputs/metrics.json` on every run,
> the same as the report's. Where the two differ in *framing*, the technical report is the one that
> was reviewed and is the submission.

Scope: the data asset, the draft standard, mechanism classification, rat→hPSC transfer, compartment
asymmetry, and uncertainty. Phase 3 (the twin) is out of scope here; `twin_hd11_validation` and
`twin_hd12_calibration` are not cited below.

Research decision-support only. Nothing in this section is an efficacy, safety, diagnostic or
clinical claim.

Every number carries an HTML comment naming the `outputs/metrics.json` key or the
`outputs/tables/` file and column it came from. Where a claim would need a number that does not
exist in `outputs/`, the text says `[MISSING: ...]` instead of estimating.

---

## 1. The data asset

**A single schema-validated asset holds 21,272 records from four openly licensed sources, but it is
dominated by one source and by one data level, and the two levels cannot be compared.**

The build produces 21,887 records <!-- metrics.json: data_asset.n_records --> validated against
`schema/neuromps_schema.json` at conformance level 2 (every provenance hash matches
`data/raw/MANIFEST.json`), generated 2026-09-22T11:20:51+00:00
<!-- metrics.json: data_asset.generated_at -->. Of these, 21,350 are marked usable
<!-- metrics.json: data_asset.counts.by_usable.True --> and 537 are not
<!-- metrics.json: data_asset.counts.by_usable.False -->.

Composition by source:

| Source | Records | Level | Species | License |
|---|---:|---|---|---|
| `toxcast_mea` | 20,375 <!-- metrics.json: data_asset.counts.by_source.toxcast_mea --> | fitted_curve | rat | CC0-1.0 <!-- outputs/DATA_CARD.md, Sources table, license column --> |
| `screenneuropharm_bic` | 580 <!-- metrics.json: data_asset.counts.by_source.screenneuropharm_bic --> | spike_times | rat | MIT <!-- outputs/DATA_CARD.md, Sources table, license column --> |
| `kapucu2022` | 288 <!-- metrics.json: data_asset.counts.by_source.kapucu2022 --> | spike_times | human, rat | CC-BY-4.0 <!-- outputs/DATA_CARD.md, Sources table, license column --> |
| `sharf2022` | 29 <!-- metrics.json: data_asset.counts.by_source.sharf2022 --> | spike_times | human | CC0-1.0 <!-- outputs/DATA_CARD.md, Sources table, license column --> |

The imbalance is the first thing to state plainly. Of 21,272 records, 20,375
<!-- metrics.json: data_asset.counts.by_data_level.fitted_curve --> are ToxCast curve fits, leaving 1512 records at
`spike_times` level <!-- metrics.json: data_asset.counts.by_data_level.spike_times --> — the only
records from which our own feature vector can be computed. By species, 21,530 records are rat
<!-- metrics.json: data_asset.counts.by_species.rat --> and 357 are human
<!-- metrics.json: data_asset.counts.by_species.human -->. By mechanism class, 20,933 records carry
no mechanism label <!-- metrics.json: data_asset.counts.by_mechanism_class.none -->; the labelled
classes are GABA-A antagonist 537 <!-- metrics.json: data_asset.counts.by_mechanism_class.gabaa_antagonist -->,
Na-channel blocker 113 <!-- metrics.json: data_asset.counts.by_mechanism_class.na_channel_blocker -->,
GABA-A agonist/PAM 26 <!-- metrics.json: data_asset.counts.by_mechanism_class.gabaa_agonist_or_pam -->,
glutamatergic antagonist 22 <!-- metrics.json: data_asset.counts.by_mechanism_class.glutamatergic_antagonist -->
and glutamatergic agonist 55 <!-- metrics.json: data_asset.counts.by_mechanism_class.glutamatergic_agonist -->.
Every downstream model in this section is therefore built on hundreds of records, not twenty
thousand.

The asset also carries 377 compartment records from 16 multi-chamber devices
<!-- outputs/DATA_CARD.md, "Multi-compartment devices" section -->, split into 246 chamber and 131
interface records <!-- outputs/DATA_CARD.md, "Multi-compartment devices" section -->. These are the
basis of §5.

Cross-record QC flagged, across the whole asset: 272 `incomplete_dose_series`
<!-- metrics.json: data_card.flag_counts.incomplete_dose_series -->, 265
`too_few_active_electrodes` <!-- metrics.json: data_card.flag_counts.too_few_active_electrodes -->,
221 `replicate_divergence` <!-- metrics.json: data_card.flag_counts.replicate_divergence -->, 221
`outlier_feature_value` <!-- metrics.json: data_card.flag_counts.outlier_feature_value -->, 90
`excluded_well_upstream` <!-- metrics.json: data_card.flag_counts.excluded_well_upstream --> and 11
`noisy_electrodes_present` <!-- metrics.json: data_card.flag_counts.noisy_electrodes_present -->.
Flags are informational and do not by themselves set `qc.usable`.

**Figure:** none. The asset composition is tabulated in `outputs/DATA_CARD.md`; there is no figure
for this subsection. [MISSING: a data-asset composition figure (records by source × level × species)
regenerated by `run_all.py`.]

Also missing for a complete account of the asset: [MISSING: record counts by conformance level —
how many records satisfy level 2 versus level 3 — currently asserted as an all-or-nothing build gate
with no key in `metrics.json`.]

---

## 2. The draft standard, tested against the data it was written for

**The standard's central claim — that records are comparable only within a data level, and that
feature names do not imply comparable measurements — is borne out by the coverage table: one source
supports none of the nine core features, and the largest single-platform source supports all of
them.**

Feature coverage, as percentage of a source's records carrying a non-null value, is the direct
evidence. `toxcast_mea` records 0% coverage on all nine core features
<!-- outputs/DATA_CARD.md, "Feature coverage" table, toxcast_mea row, all nine feature columns -->,
because ToxCast publishes concentration-response curve fits, not per-well features.
`screenneuropharm_bic` records 100% on all nine
<!-- outputs/DATA_CARD.md, "Feature coverage" table, screenneuropharm_bic row -->. `sharf2022`
records 100% except `burst_duration_s` and `spikes_per_burst` at 93%
<!-- outputs/DATA_CARD.md, "Feature coverage" table, sharf2022 row -->. `kapucu2022` ranges from 52%
(`burst_duration_s`, `spikes_per_burst`) to 64% on most features, with `active_electrodes` at 100%
<!-- outputs/DATA_CARD.md, "Feature coverage" table, kapucu2022 row -->. The nulls are the result:
a source that cannot support a feature records `null` rather than a substitute, so the gaps are
visible instead of being imputed away.

The ToxCast arm is retained at its own level and gives chemical-space scale that the spike-time
sources cannot: 36 MEA endpoints total <!-- metrics.json: toxcast_mea_inventory.endpoints_total -->,
17 acute <!-- metrics.json: toxcast_mea_inventory.endpoints_acute --> and 19 developmental
<!-- metrics.json: toxcast_mea_inventory.endpoints_dev -->, over 508 chemicals in the acute assay
<!-- metrics.json: toxcast_mea_inventory.chemicals_acute --> and 463 in the developmental assay
<!-- metrics.json: toxcast_mea_inventory.chemicals_dev -->, for 20,375 curve-fit rows
<!-- metrics.json: toxcast_mea_inventory.rows_mea -->. At a hitcall threshold of 0.9
<!-- metrics.json: toxcast_mea_inventory.hit_threshold --> this yields 2,117 acute hits
<!-- metrics.json: toxcast_mea_inventory.hits_acute --> and 3,853 developmental hits
<!-- metrics.json: toxcast_mea_inventory.hits_dev -->. Per-endpoint hit rates span 0.013 for acute
LDH <!-- outputs/tables/toxcast_mea_endpoints.csv, aeid 2540, hit_rate --> to 0.524 for
developmental `network_spike_peak`
<!-- outputs/tables/toxcast_mea_endpoints.csv, aeid 2512, hit_rate -->.

The standard's requirement that effect direction be declared per assay rather than assumed is
testable, and the test passes: the sign convention re-derived from cytotoxicity anchors on every ETL
run agrees with the mapping for 92.4% of acute chemicals
<!-- metrics.json: toxcast_mea_inventory.direction_check_agreement.acute --> and 99.1% of
developmental chemicals
<!-- metrics.json: toxcast_mea_inventory.direction_check_agreement.dev -->, against a 90% raise
threshold. The acute figure is 2.4 points above that threshold, which is a narrow margin and should
be read as such.

**Figure:** none. Evidence is `outputs/DATA_CARD.md` (coverage, licenses, flags) and
`outputs/tables/toxcast_mea_endpoints.csv` (per-endpoint inventory).

**Documentation defect to fix before publication:** `schema/STANDARD.md` §8 still lists "No
representation for compartmentalized chips yet" as a known gap, while §5.2 of the same document
specifies the `device` and `coupling` blocks in full and 377 compartment records are in the asset.
§8 is stale.

---

## 3. Mechanism classification from MEA feature deltas

**Within a single platform, MEA feature deltas carry real but modest mechanism information — roughly
three times chance and well short of a usable classifier — and they do not survive a change of
platform.**

### 3.1 Within-platform multiclass (Kapucu 2022)

On 73 exposure–baseline pairs <!-- metrics.json: mechanism_baselines.task1_within_platform.n_pairs -->
across 50 culture groups <!-- metrics.json: mechanism_baselines.task1_within_platform.n_groups -->
and 6 mechanism classes, with 4 grouped CV splits
<!-- metrics.json: mechanism_baselines.task1_within_platform.n_splits --> and a chance balanced
accuracy of 0.167 <!-- metrics.json: mechanism_baselines.task1_within_platform.chance_balanced_accuracy -->:

| Model | Balanced accuracy | Macro F1 |
|---|---|---|
| Logistic regression | 0.537 ± 0.102 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean and .balanced_accuracy_sd --> | 0.438 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.macro_f1_mean --> |
| Gradient boosting | 0.398 ± 0.089 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_mean and .balanced_accuracy_sd --> | 0.363 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.macro_f1_mean --> |
| Dummy (most frequent) | 0.167 ± 0.000 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.dummy_most_frequent.balanced_accuracy_mean and .balanced_accuracy_sd --> | 0.087 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.dummy_most_frequent.macro_f1_mean --> |

The class support is small and unbalanced: Na-channel blocker 26
<!-- metrics.json: mechanism_baselines.task1_within_platform.class_counts.na_channel_blocker -->,
glutamatergic antagonist 17
<!-- metrics.json: mechanism_baselines.task1_within_platform.class_counts.glutamatergic_antagonist -->,
control 10 <!-- metrics.json: mechanism_baselines.task1_within_platform.class_counts.control -->,
GABA-A antagonist 10
<!-- metrics.json: mechanism_baselines.task1_within_platform.class_counts.gabaa_antagonist -->,
glutamatergic agonist 6
<!-- metrics.json: mechanism_baselines.task1_within_platform.class_counts.glutamatergic_agonist -->,
GABA-A agonist/PAM 4
<!-- metrics.json: mechanism_baselines.task1_within_platform.class_counts.gabaa_agonist_or_pam -->.
The standard deviation on the headline number is ±0.102 across four folds, so the interval around
0.537 is wide. This result supports the statement "MEA feature deltas carry mechanism information
within a platform"; it does not support a performance claim.

Gradient boosting underperforms logistic regression by 0.139 balanced accuracy
<!-- metrics.json: derived from mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_mean subtracted from .logistic_regression.balanced_accuracy_mean -->,
which at this sample size is the expected direction: the flexible model has nothing to fit but noise.

**Two estimates exist for the same nominal model and must be reconciled before publication.** The
classification script reports 0.537
<!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean -->
± 0.102
<!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_sd -->
for logistic regression on all nine features,
while the ablation script reports 0.514 ± 0.107 for the `all` feature subset
<!-- metrics.json: uncertainty_and_ablations.feature_subsets.all.balanced_accuracy_mean and .balanced_accuracy_sd -->,
on the same 73 pairs <!-- metrics.json: uncertainty_and_ablations.n_pairs --> and the same 6 classes
<!-- metrics.json: uncertainty_and_ablations.n_classes -->. The 0.023 gap is smaller than either
standard deviation, but its cause is not recorded anywhere in `outputs/` or the lab notebook.
[MISSING: the reason the two scripts' CV estimates differ — seed, split count, or pipeline. Until
this is resolved, quote both and do not present either as *the* number.]

**Figure:** `outputs/figures/mechanism_confusion.png` (logistic regression, CV grouped by culture).
The diagonal is strong for control and GABA-A antagonist and weak for Na-channel blocker, the
largest class. [MISSING: per-class recall as `metrics.json` keys — the confusion matrix values exist
only as rendered text inside the figure, so no per-class recall is quoted here. Per-class *conformal
coverage* is available and is used instead in §6.]

[MISSING: a calibration measure for Task 1 (e.g. expected calibration error or a reliability curve).
PLAN.md HD7 called for "balanced accuracy + calibration"; only balanced accuracy and macro F1 were
written to `metrics.json`.]

[MISSING: a bootstrap confidence interval on the Task 1 balanced accuracy. PLAN.md HD9 called for
"bootstrap CIs everywhere"; CIs exist only for the transfer results in §4. What is reported here is
the standard deviation across four folds, which is not the same quantity.]

### 3.2 External detection on unseen labs — weak, and with no dose trend

**A GABA-A-antagonist detector fitted on Kapucu fires on only about a third of bicuculline exposures
from another lab, and its probability does not increase with dose.**

Applied to 497 ScreenNeuroPharm bicuculline pairs
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.n -->,
the detection rate for the positive class `gabaa_antagonist`
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.positive_class -->
is 0.366
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.detection_rate -->
at a mean probability of 0.418
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.mean_probability -->.

**There is no monotone dose relationship.** Spearman ρ between dose and detection is 0.014
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman -->
at p = 0.751
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman_p -->.
Per-dose detection does rise across most of the series and then collapses at the top dose:

| Dose (µM) | n | Detection rate |
|---:|---:|---:|
| 0.1 | 67 | 0.179 |
| 0.3 | 56 | 0.304 |
| 1 | 71 | 0.366 |
| 3 | 70 | 0.386 |
| 10 | 66 | 0.424 |
| 30 | 61 | 0.426 |
| 100 | 50 | 0.480 |
| 300 | 45 | 0.467 |
| 500 | 11 | 0.091 |

<!-- whole table: metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose (fields dose_uM, n, detection_rate) -->

The 500 µM point rests on 11 pairs
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose, dose_uM 500, n -->
and is the most likely cause of the null rank statistic, but we have not tested that: [MISSING: the
Spearman recomputed with the 500 µM dose excluded, and any test of whether the top-dose collapse is
a distinct effect (e.g. activity abolished) rather than sampling noise.] The honest statement is the
one supported by the outputs: detection rate 0.366, no significant dose trend across the series as
run.

The negative control behaves as intended. Diazepam — a GABA-A *positive* modulator — is detected as
a GABA-A antagonist in 0.077 of 13 pairs
<!-- metrics.json: mechanism_baselines.task2_external_detection.sharf_diazepam_negative_control.detection_rate and .n -->
at mean probability 0.210
<!-- metrics.json: mechanism_baselines.task2_external_detection.sharf_diazepam_negative_control.mean_probability -->,
with probability falling as dose rises: ρ = −-0.799
<!-- metrics.json: mechanism_baselines.task2_external_detection.sharf_diazepam_negative_control.dose_spearman -->,
p = 0.00104
<!-- metrics.json: mechanism_baselines.task2_external_detection.sharf_diazepam_negative_control.dose_spearman_p -->.
At n = 13 this is a direction, not an effect size.

**Specificity is not reported and cannot be**: neither external dataset contains vehicle controls,
so there is no negative-condition denominator. The diazepam arm substitutes a different mechanism
for a vehicle, which is a weaker control.

**Figure:** none. [MISSING: a figure for the external-detection result (detection rate and mean
probability against dose, with the negative-control arm overlaid).]

### 3.3 Platform contrast — the harmonization limit, measured

**The same mechanism at the same nominal dose moves four of nine features in opposite directions on
two platforms, including mean firing rate and network burst rate.**

Comparing GABA-A antagonists at ≥ 30 µM
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.min_dose_uM -->
<!-- mechanism: metrics.json: mechanism_baselines.task3_platform_contrast.mechanism -->
— gabazine on Kapucu's 16-electrode Axion wells, n = 10 pairs
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.by_source.kapucu2022.n; compound from .compounds -->,
versus bicuculline on ScreenNeuroPharm's 60-electrode MCS arrays, n = 167 pairs
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.by_source.screenneuropharm_bic.n; compound from .compounds -->
— five features agree in sign and four disagree
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.features_agreeing_in_sign (5 entries) and .features_disagreeing_in_sign (4 entries) -->.

The disagreeing features are `mfr_hz`, `network_burst_rate_per_min`, `isi_cv` and
`active_electrodes` <!-- metrics.json: mechanism_baselines.task3_platform_contrast.features_disagreeing_in_sign -->.
Median log2 exposure/baseline ratios:

| Feature | Kapucu (gabazine) | ScreenNeuroPharm (bicuculline) |
|---|---:|---:|
| `mfr_hz` | +0.348 | −0.684 |
| `network_burst_rate_per_min` | +1.536 | −0.920 |
| `isi_cv` | +0.573 | −0.127 |
| `synchrony_index` | +1.743 | +0.148 |
| `burst_rate_per_min` | −0.277 | −0.915 |
| `pct_spikes_in_bursts` | +0.476 | +0.050 |
| `spikes_per_burst` | +0.635 | +0.314 |
| `burst_duration_s` | +0.203 | +0.086 |
| `active_electrodes` | 0.000 | 0.000 |

<!-- whole table: metrics.json: mechanism_baselines.task3_platform_contrast.by_source.kapucu2022.median_delta.* and .by_source.screenneuropharm_bic.median_delta.* -->

Note that `active_electrodes` is 0.000 on both platforms and is nonetheless listed among the
disagreeing features by the sign test
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.features_disagreeing_in_sign -->;
a median delta of exactly zero has no sign, so the "4 disagree" count should be read as at most
three substantive disagreements plus one degenerate case. [MISSING: a sign-test rule that excludes
zero-valued medians, or a recomputed agreement count that does.]

Gabazine *raises* network burst rate on the 16-electrode platform and bicuculline *lowers* it on the
60-electrode platform — a sign reversal in the feature most often used to summarise disinhibition.
Candidate causes are confounded in this contrast and none has been isolated: electrode count and
pitch, recording duration, culture age, compound identity, and the chance-corrected network-burst
threshold, which itself scales with electrode count.
[MISSING: any experiment varying one of these factors while holding the others fixed — recording
duration matched, or the network-burst threshold held constant across electrode counts. Without one,
"platform" is a label for several unseparated differences, including the fact that the two arms use
two different compounds.]

**Figure:** `outputs/figures/platform_contrast_gabaa_antagonist.png`.

---

## 4. Rat → hPSC transfer

**Training on rat and testing on hPSC-derived cultures transfers at 0.550 balanced accuracy against
a chance level of 0.200 — better than chance, only slightly below what the rat data supports
within-species, and materially below the human-only model — and both domain-adaptation methods we
tried made it worse.**

All transfer results use Kapucu 2022 alone
<!-- metrics.json: species_transfer.source -->, the one source with both species on one platform in
one lab, scored on the 5 mechanism labels present in both species
<!-- metrics.json: species_transfer.transfer.none.shared_labels (5 entries) --> against a chance
balanced accuracy of 0.200 <!-- metrics.json: species_transfer.transfer.none.chance_balanced_accuracy -->.

| Setting | Balanced accuracy | 95% CI | n |
|---|---:|---|---:|
| rat → human, no adaptation | 0.550 <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy --> | [0.400, 0.600] <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy_ci95 --> | 37 train / 32 test <!-- metrics.json: species_transfer.transfer.none.rat_to_human.n_train and .n_test --> |
| human within-species (5 repeats × 4 splits) | 0.646 ± 0.139 <!-- metrics.json: species_transfer.transfer.none.human_within.balanced_accuracy and .balanced_accuracy_sd --> | — | 32 <!-- metrics.json: species_transfer.transfer.none.human_within.n --> |
| rat within-species (5 repeats × 2 splits) | 0.543 ± 0.067 <!-- metrics.json: species_transfer.transfer.none.rat_within.balanced_accuracy and .balanced_accuracy_sd --> | — | 37 <!-- metrics.json: species_transfer.transfer.none.rat_within.n --> |
| rat → human, CORAL | 0.442 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy --> | [0.302, 0.591] <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy_ci95 --> | 37 / 32 |
| rat → human, per-species standardisation | 0.408 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy --> | [0.267, 0.568] <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy_ci95 --> | 37 / 32 |

The unadapted transfer is above chance by permutation test, p = 0.0002
<!-- metrics.json: species_transfer.transfer.none.rat_to_human.permutation_p -->. Two things should
be said about the headline number rather than around it. First, 0.550 sits within one standard
deviation of the rat within-species baseline of 0.543 ± 0.067, so crossing species costs little
relative to what the rat data itself supports — but the rat baseline is low, so this is a statement
about a weak model transferring, not a strong one. Second, the 95% CI is [0.400, 0.600], which
overlaps the CIs of both adaptation methods; the ordering is clear but the separations are not.

**Both domain-adaptation methods hurt, and are reported as found.** CORAL costs 0.108 balanced
accuracy and per-species standardisation costs 0.142 relative to no adaptation
<!-- metrics.json: derived from species_transfer.transfer.{coral,standardize}.rat_to_human.balanced_accuracy subtracted from .none.rat_to_human.balanced_accuracy -->.
With 32 target samples, both estimate scale and covariance from noise. They remain in the codebase
and are reported here rather than dropped.

### Per-feature species shift

Computed over 37 rat and 36 human pairs
<!-- metrics.json: species_transfer.species_shift.n_rat and .n_human -->. (Note that the shift
analysis uses 36 human pairs while the transfer test uses 32 — the transfer is restricted to the 5
shared labels. The two are not the same sample.)

The three most species-invariant features are `mfr_hz`, `pct_spikes_in_bursts` and
`burst_rate_per_min` <!-- metrics.json: species_transfer.species_shift.most_invariant_features -->;
the three most shifted are `synchrony_index`, `isi_cv` and `network_burst_rate_per_min`
<!-- metrics.json: species_transfer.species_shift.most_shifted_features -->. Standardised
differences:

| Feature | Rat median | Human median | Standardised difference |
|---|---:|---:|---:|
| `synchrony_index` | −0.143 | −4.322 | −1.800 |
| `isi_cv` | −0.029 | −1.262 | −1.630 |
| `network_burst_rate_per_min` | +0.585 | −4.322 | −1.291 |
| `spikes_per_burst` | −0.031 | −1.151 | −1.193 |
| `burst_duration_s` | −0.103 | −0.791 | −1.107 |
| `active_electrodes` | −0.093 | 0.000 | +0.713 |
| `burst_rate_per_min` | −1.558 | −0.519 | +0.636 |
| `pct_spikes_in_bursts` | −0.131 | −0.881 | −0.510 |
| `mfr_hz` | −0.442 | −0.764 | −0.321 |

<!-- whole table: metrics.json: species_transfer.species_shift.by_feature.<feature>.{rat_median, human_median, standardized_difference} -->

The pattern matches §3.3: the features that shift most across species — synchrony and network-burst
measures — are the same ones that reverse sign across platforms. Rate and burst-fraction measures
are the portable ones on both axes. This is the single most reusable finding in Phase 2, and it is
tested rather than asserted: the `species_invariant` feature subset is scored in §6.

Two cautions on the table. The human medians for `synchrony_index` and
`network_burst_rate_per_min` are identical at −4.322, which is what a floored log2 ratio looks like
when the exposure value is at or near zero; these are "feature collapsed" cases, not measured
four-fold reductions. [MISSING: the count of floored (clipped) ratios per feature per species, which
would show how much of the shift is a floor artefact.]

**Figure:** `outputs/figures/species_shift.png`.

---

## 5. Compartment asymmetry in two-chamber devices

**Devices with two different tissue types show roughly twice the inter-chamber coupling lag of
same-tissue devices, and lower peak coupling — with same-tissue devices indistinguishable from each
other, which is the internal control this test needs.**

16 devices <!-- metrics.json: compartment_asymmetry.n_devices -->, one value per device (the median
over that device's bicuculline conditions), magnitudes only: 5 CxCx
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.CxCx.n_devices -->,
5 HpHp <!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.HpHp.n_devices -->,
6 CxHp <!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.CxHp.n_devices -->.

**Lag.** Median |lag| is 11.25 ms for CxHp
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.CxHp.median -->
against 5.0 ms for both CxCx
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.CxCx.median -->
and HpHp <!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.HpHp.median -->.
Kruskal–Wallis p = 0.0082
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.kruskal_p -->
(statistic 9.61 <!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.kruskal_statistic -->).
Holm-corrected pairwise: CxHp vs HpHp 0.0257
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.pairwise_holm.CxHp_vs_HpHp -->,
CxCx vs CxHp 0.0413
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.pairwise_holm.CxCx_vs_CxHp -->,
and — the control — CxCx vs HpHp 0.4785
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.pairwise_holm.CxCx_vs_HpHp -->.
The two same-tissue device types are not distinguishable from each other, which is what should
happen if the measure tracks tissue heterogeneity rather than device-to-device variation.

**Peak coupling falls with heterogeneity.** Median peak correlation: CxCx 0.962
<!-- metrics.json: compartment_asymmetry.by_device_type.median_peak_correlation.by_type.CxCx.median -->,
HpHp 0.908 <!-- metrics.json: compartment_asymmetry.by_device_type.median_peak_correlation.by_type.HpHp.median -->,
CxHp 0.822 <!-- metrics.json: compartment_asymmetry.by_device_type.median_peak_correlation.by_type.CxHp.median -->;
Kruskal p = 0.0053
<!-- metrics.json: compartment_asymmetry.by_device_type.median_peak_correlation.kruskal_p -->,
Holm CxCx vs CxHp 0.0130
<!-- metrics.json: compartment_asymmetry.by_device_type.median_peak_correlation.pairwise_holm.CxCx_vs_CxHp -->
and CxCx vs HpHp 0.0159
<!-- metrics.json: compartment_asymmetry.by_device_type.median_peak_correlation.pairwise_holm.CxCx_vs_HpHp -->.
Here the control pair *is* separated, so unlike lag, peak coupling distinguishes the two same-tissue
types too — the ordering is not purely a heterogeneity effect.

**Asymmetry magnitude: unresolved, deliberately not claimed.** Median |asymmetry| is 0.047 for CxCx
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_asymmetry.by_type.CxCx.median -->,
0.096 for HpHp
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_asymmetry.by_type.HpHp.median -->
and 0.138 for CxHp
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_asymmetry.by_type.CxHp.median -->,
with Kruskal p = 0.0336
<!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_asymmetry.kruskal_p -->.
[MISSING: the Holm-corrected pairwise verdict for |asymmetry|. `report/LAB_NOTEBOOK.md` and
`PROGRESS.md` both state "Holm 0.04" for CxCx vs CxHp; `outputs/metrics.json` gives 0.0519 for the
same comparison. The two disagree across the 0.05 line and the analysis is pending a rerun. No
significance claim for |asymmetry| is made here until it is resolved.]

A pooled heterogeneous-vs-homogeneous comparison also exists in the outputs — medians 0.138 vs 0.094,
p = 0.042, rank-biserial 0.633, 6 vs 10 devices
<!-- metrics.json: compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.{median_heterogeneous, median_homogeneous, p_value, rank_biserial, n_heterogeneous, n_homogeneous} -->
— and is **not** the headline. It is an artefact of pooling, because HpHp overlaps CxHp. It is
retained in `metrics.json` so a reader can see the difference between the two framings.

At 5–6 devices per group this analysis supports an ordering of device types, not a quotable effect
size. [MISSING: an effect size (e.g. rank-biserial) for the per-device-type pairwise comparisons;
`rank_biserial` exists only for the abandoned pooled test.]
[MISSING: the directional consistency claim — `PROGRESS.md` records "4 of 6 devices leading the same
way, 1 opposite" for CxHp, but `metrics.json` stores only magnitudes
(`compartment_asymmetry.per_device.median_abs_asymmetry`), so signed agreement cannot be verified
from the outputs.]

**A confound identified in Phase 3 bears on how far this result can be pushed.** Coupling asymmetry
can track the two chambers' relative firing rate rather than the pathway between them. Three checks
against the real devices were run ad hoc and reportedly showed the confound does not bite here, but
they are not produced by any script under `src/` and are not in `metrics.json`.
[MISSING: the three rate-imbalance confound checks as script-produced metrics — (a) rate imbalance
by device type, (b) correlation of rate imbalance with |lag|, (c) signed rate ratio against signed
asymmetry. `report/LAB_NOTEBOOK.md` quotes Kruskal p = 0.96, ρ = −0.05 (p = 0.85) and ρ = +0.04
(p = 0.73), and states explicitly that these "must be" scripted before entering the report. They are
therefore not quoted above.]

**Figure:** `outputs/figures/compartment_lag_by_device_type.png`.

---

## 6. Uncertainty and feature ablations

**The honest summary of the mechanism model is its conformal set size, not its accuracy: to be right
90% of the time the model must name 4.6 of 6 mechanisms and never commits to a single one.**

### 6.1 Conformal prediction sets

Split conformal, calibration split grouped by culture, on 73 pairs and 6 classes
<!-- metrics.json: uncertainty_and_ablations.n_pairs and .n_classes -->:

| α | Target | Empirical coverage | Mean set size | Singleton fraction | Empty fraction |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.95 | 0.918 | 4.562 | 0.000 | 0.000 |
| 0.10 | 0.90 | 0.918 | 4.562 | 0.000 | 0.000 |
| 0.20 | 0.80 | 0.822 | 3.014 | 0.068 | 0.000 |
| 0.30 | 0.70 | 0.712 | 2.315 | 0.151 | 0.000 |

<!-- whole table: metrics.json: uncertainty_and_ablations.conformal_sweep (fields alpha, target_coverage, empirical_coverage, mean_set_size, fraction_singleton, fraction_empty) -->

Coverage tracks the target at α = 0.2 and 0.3 and overshoots it at α = 0.05 and 0.1, where the two
rows are identical because the small calibration set gives them the same threshold (0.019)
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep, alpha 0.05 and 0.1, mean_threshold -->
— recorded in the outputs as a known consequence of 73 pairs
<!-- metrics.json: uncertainty_and_ablations.note -->. The method is working; the set size is the
result.

Per-class coverage at α = 0.1 shows where the model fails rather than where it is merely uncertain:
control, GABA-A agonist/PAM and GABA-A antagonist all at 1.000, glutamatergic antagonist 0.941,
Na-channel blocker 0.923, and **glutamatergic agonist 0.500**
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep, alpha 0.1, coverage_by_class -->.
At α = 0.3 glutamatergic agonist coverage falls to 0.000
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep, alpha 0.3, coverage_by_class.glutamatergic_agonist -->
on 6 training members
<!-- metrics.json: mechanism_baselines.task1_within_platform.class_counts.glutamatergic_agonist -->.
The model does not cover that class at any useful rate.

**Figure:** `outputs/figures/conformal_set_size.png`.

### 6.2 Feature ablations

**Two rate features do as well as all nine, and the synchrony and network-burst features contribute
nothing detectable at this sample size.**

| Subset | Features | Balanced accuracy |
|---|---:|---|
| `rate_only` (`mfr_hz`, `active_electrodes`) | 2 | 0.530 ± 0.040 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.rate_only.balanced_accuracy_mean and .balanced_accuracy_sd --> |
| `species_invariant` (`mfr_hz`, `pct_spikes_in_bursts`, `burst_rate_per_min`) | 3 | 0.516 ± 0.099 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.species_invariant.balanced_accuracy_mean and .balanced_accuracy_sd --> |
| `all` | 9 | 0.514 ± 0.107 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.all.balanced_accuracy_mean and .balanced_accuracy_sd --> |
| `synchrony_only` | 1 | 0.455 ± 0.089 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.synchrony_only.balanced_accuracy_mean and .balanced_accuracy_sd --> |
| `mfr_only` | 1 | 0.442 ± 0.098 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.mfr_only.balanced_accuracy_mean and .balanced_accuracy_sd --> |
| `network_only` | 2 | 0.417 ± 0.084 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.network_only.balanced_accuracy_mean and .balanced_accuracy_sd --> |
| `burst_only` | 4 | 0.411 ± 0.110 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.burst_only.balanced_accuracy_mean and .balanced_accuracy_sd --> |

Two features reach 0.530 where nine reach 0.514, and the two-feature model has the smallest spread
across folds (±0.040 against ±0.107). The `species_invariant` subset — named after the §4 result
before being scored here — reaches 0.516, indistinguishable from the full set. The claim "these are
the portable features" survives its own test, in the weak sense that dropping the non-portable ones
costs nothing.

Leave-one-out confirms this from the other direction. Only `mfr_hz` clearly matters: dropping it
changes balanced accuracy by −-0.05
<!-- metrics.json: uncertainty_and_ablations.leave_one_out.mfr_hz.delta_vs_all -->. Dropping
`synchrony_index` changes it by −-0.014
<!-- metrics.json: uncertainty_and_ablations.leave_one_out.synchrony_index.delta_vs_all -->,
`active_electrodes` by −-0.011
<!-- metrics.json: uncertainty_and_ablations.leave_one_out.active_electrodes.delta_vs_all -->, and
`isi_cv` by −-0.006
<!-- metrics.json: uncertainty_and_ablations.leave_one_out.isi_cv.delta_vs_all -->. Dropping
`burst_rate_per_min`, `burst_duration_s` or `spikes_per_burst` changes nothing at all (0.000)
<!-- metrics.json: uncertainty_and_ablations.leave_one_out.{burst_rate_per_min,burst_duration_s,spikes_per_burst}.delta_vs_all -->,
and dropping `pct_spikes_in_bursts` or `network_burst_rate_per_min` slightly *helps*, by 0.010 and
0.007
<!-- metrics.json: uncertainty_and_ablations.leave_one_out.pct_spikes_in_bursts.delta_vs_all and .network_burst_rate_per_min.delta_vs_all -->.
Every one of these deltas is far inside the fold-to-fold standard deviation of roughly ±0.10, so the
correct reading is that eight of the nine features are not doing measurable work here — not that
removing them is beneficial.

**Figure:** none for the ablations. [MISSING: a feature-subset / leave-one-out figure; only the
conformal figure exists for this subsection.]

---

## 7. What Phase 1–2 does and does not establish

Stated plainly, because several of the results above are weak or negative:

- The asset and standard are the solid part. 21,887 validated, hash-traceable records
  <!-- metrics.json: data_asset.n_records -->, seven sources, explicit data levels, and a coverage
  table that shows the gaps instead of hiding them.
- Mechanism information is present in MEA feature deltas within one platform (0.537 vs 0.167 chance)
  <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean and .chance_balanced_accuracy -->
  but is not enough to identify a mechanism: the conformal sets contain 4.56 of 6 classes at 90%
  coverage <!-- metrics.json: uncertainty_and_ablations.conformal_sweep, alpha 0.1, mean_set_size -->.
- Feature deltas do not transfer across platforms unchanged; four of nine features disagree in sign
  <!-- metrics.json: mechanism_baselines.task3_platform_contrast.features_disagreeing_in_sign -->,
  and the platform contrast confounds compound, electrode count, duration and threshold.
- Rat→hPSC transfer is real but modest (0.550, CI [0.400, 0.600])
  <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy and .balanced_accuracy_ci95 -->,
  and both domain-adaptation methods reduced it.
- Rate and burst-fraction features are portable across both species and platform; synchrony and
  network-burst features are not.
- Two-chamber devices of mixed tissue show longer coupling lag and weaker peak coupling than
  same-tissue devices, on 5–6 devices per group — an ordering, not an effect size.

---

## Claims to verify

Every quantitative statement above, with its source. Audit line by line.

### §1 Data asset

| # | Claim | Source |
|---:|---|---|
| 1 | 21,272 records | `metrics.json` → `data_asset.n_records` |
| 2 | Built 2026-09-20T16:17:51+00:00 | `data_asset.generated_at` |
| 3 | 20,831 usable / 441 not | `data_asset.counts.by_usable.True` / `.False` |
| 4 | toxcast_mea 20,375 / screenneuropharm_bic 580 / kapucu2022 288 / sharf2022 29 | `data_asset.counts.by_source.*` |
| 5 | fitted_curve 20,375; spike_times 897 | `data_asset.counts.by_data_level.*` |
| 6 | rat 21,099; human 173 | `data_asset.counts.by_species.*` |
| 7 | none 20,563; gabaa_antagonist 537; na_channel_blocker 113; gabaa_agonist_or_pam 26; glutamatergic_antagonist 22; glutamatergic_agonist 11 | `data_asset.counts.by_mechanism_class.*` |
| 8 | Licenses CC0-1.0 / MIT / CC-BY-4.0 / CC0-1.0 | `outputs/DATA_CARD.md`, Sources table, license column |
| 9 | 377 compartment records, 16 devices, 246 chamber + 131 interface | `outputs/DATA_CARD.md`, "Multi-compartment devices"; device count cross-checks `compartment_asymmetry.n_devices` |
| 10 | Flags: 272 / 169 / 151 / 151 / 90 / 11 | `data_card.flag_counts.*` |

### §2 Standard

| # | Claim | Source |
|---:|---|---|
| 11 | toxcast_mea 0% coverage on all nine features | `outputs/DATA_CARD.md`, Feature coverage table, toxcast_mea row |
| 12 | screenneuropharm_bic 100% on all nine | same table, screenneuropharm_bic row |
| 13 | sharf2022 100% except burst_duration_s / spikes_per_burst at 93% | same table, sharf2022 row |
| 14 | kapucu2022 52–64%, active_electrodes 100% | same table, kapucu2022 row |
| 15 | 36 endpoints (17 acute, 19 dev) | `toxcast_mea_inventory.endpoints_total` / `.endpoints_acute` / `.endpoints_dev` |
| 16 | 508 acute chemicals; 463 dev chemicals | `toxcast_mea_inventory.chemicals_acute` / `.chemicals_dev` |
| 17 | 20,375 curve-fit rows | `toxcast_mea_inventory.rows_mea` |
| 18 | 2,117 acute hits; 3,853 dev hits; threshold 0.9 | `toxcast_mea_inventory.hits_acute` / `.hits_dev` / `.hit_threshold` |
| 19 | Hit rates 0.013 (acute LDH) to 0.524 (dev network_spike_peak) | `outputs/tables/toxcast_mea_endpoints.csv`, aeid 2540 and 2512, `hit_rate` |
| 20 | Direction agreement 92.4% acute / 99.1% dev | `toxcast_mea_inventory.direction_check_agreement.acute` / `.dev` |

### §3 Mechanism classification

| # | Claim | Source |
|---:|---|---|
| 21 | 73 pairs / 50 groups / 4 splits / chance 0.167 | `mechanism_baselines.task1_within_platform.{n_pairs,n_groups,n_splits,chance_balanced_accuracy}` |
| 22 | Logistic regression 0.537 ± 0.102, macro F1 0.438 | `...models.logistic_regression.{balanced_accuracy_mean,balanced_accuracy_sd,macro_f1_mean}` |
| 23 | Gradient boosting 0.398 ± 0.089, macro F1 0.363 | `...models.gradient_boosting.*` |
| 24 | Dummy 0.167 ± 0.000, macro F1 0.087 | `...models.dummy_most_frequent.*` |
| 25 | Class counts 26 / 17 / 10 / 10 / 6 / 4 | `...class_counts.*` |
| 26 | GB underperforms LR by 0.139 | derived from #22 and #23 |
| 27 | **Unreconciled:** 0.537 (task1) vs 0.514 (ablation `all`) on the same 73 pairs | `mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean` vs `uncertainty_and_ablations.feature_subsets.all.balanced_accuracy_mean` — cause not recorded |
| 28 | External detection rate 0.366, mean probability 0.418, n = 497 | `mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.{detection_rate,mean_probability,n}` |
| 29 | No dose trend: ρ = 0.014, p = 0.751 | `...screenneuropharm_bicuculline.dose_spearman` / `.dose_spearman_p` |
| 30 | Per-dose table (0.1 → 500 µM, 9 rows) | `...screenneuropharm_bicuculline.by_dose` |
| 31 | 500 µM point has n = 11 | `...by_dose`, `dose_uM` 500, `n` |
| 32 | Diazepam control: 0.077 detection, mean probability 0.210, ρ = −0.799, p = 0.00104, n = 13 | `...sharf_diazepam_negative_control.*` |
| 33 | Platform contrast at ≥ 30 µM, mechanism gabaa_antagonist | `mechanism_baselines.task3_platform_contrast.min_dose_uM` / `.mechanism` |
| 34 | Kapucu n = 10 (gabazine); ScreenNeuroPharm n = 167 (bicuculline) | `...task3_platform_contrast.by_source.{kapucu2022,screenneuropharm_bic}.n` / `.compounds` |
| 35 | 5 features agree, 4 disagree in sign | `...features_agreeing_in_sign` / `.features_disagreeing_in_sign` |
| 36 | All 18 median log2 deltas in the contrast table | `...by_source.{kapucu2022,screenneuropharm_bic}.median_delta.*` |
| 37 | `active_electrodes` median delta is 0.000 on both platforms yet counted as disagreeing | `...median_delta.active_electrodes` (both sources) and `.features_disagreeing_in_sign` |

### §4 Transfer

| # | Claim | Source |
|---:|---|---|
| 38 | Source is kapucu2022 only; 5 shared labels; chance 0.200 | `species_transfer.source`; `...transfer.none.shared_labels`; `...chance_balanced_accuracy` |
| 39 | rat → human 0.550, CI [0.400, 0.600], p = 0.0002, 37 train / 32 test | `species_transfer.transfer.none.rat_to_human.*` |
| 40 | human within 0.646 ± 0.139 (n = 32, 4 splits × 5 repeats) | `...transfer.none.human_within.*` |
| 41 | rat within 0.543 ± 0.067 (n = 37, 2 splits × 5 repeats) | `...transfer.none.rat_within.*` |
| 42 | CORAL 0.442, CI [0.280, 0.597], p = 0.002 | `...transfer.coral.rat_to_human.*` |
| 43 | Standardise 0.408, CI [0.246, 0.560], p = 0.0054 | `...transfer.standardize.rat_to_human.*` |
| 44 | CORAL costs 0.108; standardisation costs 0.142 | derived from #39, #42, #43 |
| 45 | Shift computed on 37 rat / 36 human pairs | `species_transfer.species_shift.n_rat` / `.n_human` |
| 46 | Invariant: mfr_hz, pct_spikes_in_bursts, burst_rate_per_min | `...species_shift.most_invariant_features` |
| 47 | Shifted: synchrony_index, isi_cv, network_burst_rate_per_min | `...species_shift.most_shifted_features` |
| 48 | All 27 values in the species-shift table | `...species_shift.by_feature.<feature>.{rat_median,human_median,standardized_difference}` |
| 49 | Human medians for synchrony_index and network_burst_rate_per_min are both −4.322 (floor artefact) | `...species_shift.by_feature.{synchrony_index,network_burst_rate_per_min}.human_median` |

### §5 Compartments

| # | Claim | Source |
|---:|---|---|
| 50 | 16 devices; 5 CxCx / 5 HpHp / 6 CxHp | `compartment_asymmetry.n_devices`; `...by_device_type.median_abs_lag_ms.by_type.*.n_devices` |
| 51 | Median \|lag\|: CxHp 11.25 ms, CxCx 5.0, HpHp 5.0 | `...median_abs_lag_ms.by_type.*.median` |
| 52 | Lag Kruskal p = 0.0082, statistic 9.61 | `...median_abs_lag_ms.kruskal_p` / `.kruskal_statistic` |
| 53 | Lag Holm: CxHp–HpHp 0.0257, CxCx–CxHp 0.0413, CxCx–HpHp 0.4785 | `...median_abs_lag_ms.pairwise_holm.*` |
| 54 | Peak correlation: CxCx 0.962, HpHp 0.908, CxHp 0.822 | `...median_peak_correlation.by_type.*.median` |
| 55 | Peak-correlation Kruskal p = 0.0053; Holm CxCx–CxHp 0.0130, CxCx–HpHp 0.0159 | `...median_peak_correlation.kruskal_p` / `.pairwise_holm.*` |
| 56 | Median \|asymmetry\|: CxCx 0.047, HpHp 0.096, CxHp 0.138; Kruskal p = 0.0336 | `...median_abs_asymmetry.by_type.*.median` / `.kruskal_p` |
| 57 | **Disputed, left as [MISSING]:** Holm CxCx–CxHp for \|asymmetry\| — notebook says 0.04, `metrics.json` says 0.0519 | `LAB_NOTEBOOK.md` / `PROGRESS.md` vs `...median_abs_asymmetry.pairwise_holm.CxCx_vs_CxHp` |
| 58 | Pooled test (not the headline): 0.138 vs 0.094, p = 0.042, rank-biserial 0.633, 6 vs 10 | `compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.*` |

### §6 Uncertainty and ablations

| # | Claim | Source |
|---:|---|---|
| 59 | 73 pairs, 6 classes | `uncertainty_and_ablations.n_pairs` / `.n_classes` |
| 60 | Conformal table (4 α rows: coverage, set size, singleton, empty) | `uncertainty_and_ablations.conformal_sweep` |
| 61 | α = 0.05 and 0.10 share threshold 0.019 | `...conformal_sweep`, `mean_threshold` at both α |
| 62 | Shared-threshold caveat is recorded in outputs | `uncertainty_and_ablations.note` |
| 63 | Per-class coverage at α = 0.1 (1.000 / 1.000 / 1.000 / 0.941 / 0.923 / 0.500) | `...conformal_sweep`, α 0.1, `coverage_by_class` |
| 64 | Glutamatergic agonist coverage 0.000 at α = 0.3 | `...conformal_sweep`, α 0.3, `coverage_by_class.glutamatergic_agonist` |
| 65 | Seven feature-subset scores | `uncertainty_and_ablations.feature_subsets.*.balanced_accuracy_mean` / `.balanced_accuracy_sd` |
| 66 | Leave-one-out deltas for all nine features | `uncertainty_and_ablations.leave_one_out.*.delta_vs_all` |

### Open `[MISSING]` items, collected

| # | What is needed | Where it would go |
|---:|---|---|
| M1 | Data-asset composition figure (source × level × species) | §1 |
| M2 | Record counts by conformance level (2 vs 3) | §1 |
| M3 | Per-class recall as `metrics.json` keys (currently only rendered in `mechanism_confusion.png`) | §3.1 |
| M4 | Calibration measure for Task 1 (ECE or reliability curve) — PLAN.md HD7 asked for it | §3.1 |
| M5 | Bootstrap CI on Task 1 balanced accuracy — PLAN.md HD9 asked for CIs "everywhere" | §3.1 |
| M6 | Reconciliation of 0.537 vs 0.514 | §3.1 |
| M7 | External-detection Spearman recomputed without the n = 11 top dose | §3.2 |
| M8 | Figure for the external-detection result | §3.2 |
| M9 | Sign-test rule excluding zero-valued medians (the `active_electrodes` case) | §3.3 |
| M10 | Any experiment separating electrode count, duration, culture age, compound and burst threshold | §3.3 |
| M11 | Count of floored (clipped) log2 ratios per feature per species | §4 |
| M12 | Holm-corrected pairwise verdict for \|asymmetry\| after rerun | §5 |
| M13 | Effect size for the per-device-type pairwise comparisons | §5 |
| M14 | Signed asymmetry per device, to verify "4 of 6 leading the same way" | §5 |
| M15 | The three rate-imbalance confound checks as script-produced metrics | §5 |
| M16 | Feature-subset / leave-one-out figure | §6.2 |

### Documentation defects found while drafting

| # | Defect | File |
|---:|---|---|
| D1 | §8 "Known gaps" still says compartmentalized chips are unrepresented, contradicting §5.2 of the same file and 377 records in the asset | `schema/STANDARD.md` |
| D2 | "Holm 0.04" for \|asymmetry\| CxCx vs CxHp contradicts `metrics.json` (0.0519) | `report/LAB_NOTEBOOK.md`, `PROGRESS.md` |
| D3 | "rising 0.18 → 0.48 with dose" for external bicuculline detection is not supported by the rank test in `metrics.json` (ρ = 0.014, p = 0.751) | `report/LAB_NOTEBOOK.md`, `PROGRESS.md` |
| D4 | `outputs/twin/hd12_calibration_simulated.json` exists with no corresponding notebook entry; `PROGRESS.md` still lists HD12 as the next task | `report/LAB_NOTEBOOK.md`, `PROGRESS.md` |
