# NeuroMPS lab notebook

Every decision behind the technical report, in the order it was taken, with the commit where it
landed. **Results are not restated here** — each entry points at the `outputs/metrics.json` key or
the `outputs/` file that holds the numbers, so there is one source of truth for every figure quoted
in the report.

Entries marked **[UNCERTAIN]** are ones I could not fully verify from the repository or the session
record; treat them as needing a check before they enter the report.

Sessions ran on 2026-09-16 (HD1–HD5) and 2026-09-20 (HD6–HD11). Commit dates are authoritative;
where a decision was taken during a session but committed later, the commit is still what is cited.

---

## Phase 0 — acquisition and provenance

### 2026-09-16 · Environment and pinning · `6c940fb`
- **Python 3.11 in a `uv`-managed venv.** The machine has only 3.12; `CLAUDE.md` specifies 3.11.
- **numpy held at 2.3.5.** brian2 2.9.0 fails to import against numpy 2.4 (`ndarray.ptp` removed).
  This pin blocks numpy upgrades until brian2 changes.
- **CI installs CPU-only torch** from the PyTorch CPU index to avoid multi-GB CUDA wheels.
- CI has never run: the repository still has no remote. **[UNCERTAIN]** whether the workflow passes
  as written; it has only been exercised locally as `ruff` + `pytest` + `run_all.py --quick`.

### 2026-09-16 · Provenance model · `6c940fb`
- Each raw file records URL, license, SHA256, byte size and download time in
  `data/raw/MANIFEST.json`. A source that cannot be fetched is recorded with `status: blocked` or
  `skipped` and a reason, rather than vanishing from the record.
- Downloads verify the publisher's MD5 where one is published, and reuse a cached file only when its
  SHA256 still matches the manifest.

### 2026-09-16 · Source selection, first pass · `baa1597`
Accepted: `kapucu2022` (G-Node GIN, CC-BY-4.0), `toxcast_mea` (EPA invitrodb v4.3, CC0-1.0),
`axon_diode` (three open-access papers, CC-BY, full text only — design parameters, no data).

- **Kapucu raw vs preprocessed: preprocessed.** The raw 12.5 kHz `.h5` recordings total ~2.4 TB
  (75 files, up to 120 GB each). We use the authors' spike-time CSVs and per-well meaRtools outputs.
  Consequence: **spike detection is theirs, not ours**, which is a permanent limitation of the asset.
- Non-open papers in the axon-diode lineage (Peyrin 2011 Lab Chip; Courte 2018 Methods Cell Biol)
  are cited only and never downloaded.

### 2026-09-16 · Two EPA Clowder server bugs · `388477b`
Both were found by running against the real server, and both have regression tests:
- suffix byte ranges (`bytes=-1`) return HTTP 500; explicit ranges are required;
- when `Accept-Encoding: gzip` is offered, the server returns HTTP 206 with a plausible
  `Content-Range` but streams the **whole 7.5 GB file** from byte 0. Range requests now send
  `identity` and verify status, start offset and encoding.

### 2026-09-16 · `iras_mea` (Tukker 2020) rejected on license · `388477b`
Fetched successfully, license verified from the article XML as **CC BY-NC 4.0**, then excluded.
Reasoning: the submission competes for cash prizes and is released under MIT, and a non-commercial
term on an input cannot be carried into that. The competition rules (Kaggle Rules tab, read
2026-09-16) give joint ownership of submissions to the team and the Organizing Committee and keep
third-party datasets under their own licenses, so the "entrant keeps their own license" condition
that would have justified keeping it does not hold. Recorded as `status: skipped` with the reason in
the manifest via `src/neuromps/fetch/excluded.py`; rationale in `data/SOURCES.md`.

### 2026-09-16 · License policy · `388477b`, widened in `2050862`
Accept only licenses permitting commercial reuse and redistribution with attribution: CC0, CC BY,
public domain, and (after `2050862`) permissive OSI licenses — MIT, BSD-3-Clause, Apache-2.0 — where
a repository applies one to data. Exclude NC, ND and unlicensed material. **Where repository metadata
and the dataset's own license file disagree, take the more restrictive one and record the conflict.**
Every fetcher re-checks its source's license at fetch time and blocks the source if it changed
upstream.

### 2026-09-16 · ToxCast: select MEA rows by name, never by AEID range · `83e190f`
AEIDs 2454–2541 also contain 11 unrelated assays (`CCTE_Deisenroth_AIME_*`, `CCTE_GLTED_hDIO2/3`).
Selection is by `aenm` prefix `CCTE_Shafer_MEA_`. Endpoint inventory:
`outputs/tables/toxcast_mea_endpoints.csv`; counts under `metrics.toxcast_mea_inventory`.

### 2026-09-16 · Effect-direction sign convention (OQ-1) · `b613b45`
The sign of ToxCast `top` means **opposite things in the two MEA assays**, so it is mapped per assay
into `activity_change` (positive = the measured quantity rose):

- acute: `activity_change = -top`. Normalised against a positive control of −100, so +100 means
  activity was abolished.
- dev: `activity_change = +top`. Percent of control on AUC metrics, so −100 means every metric
  collapsed to zero.

**Why not the annotation.** `signal_direction` in the EPA assay annotations does not settle it: all
32 activity endpoints are annotated `bidirectional`; only the 4 viability endpoints carry
`gain`/`loss`. openpyxl was added (PI-approved) to read that file.

**Verification, and the numbers behind it.** Derived from cytotoxicity anchors — chemicals that kill
cells must reduce electrical activity — over **16 acute** and **221 dev** cytotoxic chemicals. A
global sign flip was rejected because it inverts the viability endpoints (LDH must rise when cells
die). `check_direction_convention()` re-derives the mapping on every ETL run and raises below 90%
agreement; the current agreement values are `metrics.toxcast_mea_inventory.direction_check_agreement`
(acute and dev keys). Interval-type endpoints (interburst interval and similar) are excluded from
that check, because intervals lengthen as activity dies.

### 2026-09-16 · Three sources added after the Tukker exclusion · `2050862`
PI-approved. `sharf2022` (human iPSC organoid HD-MEA, diazepam series, CC0), `fofie2025` (hiPSC
nociceptors, CC0), `screenneuropharm_bic` (rat bicuculline series, MIT).

- **Rejected in the same search:** Zenodo 14525182 (Mateus & Aguiar), a compartmentalised
  asymmetric-microchannel MEA dataset and the closest published analogue of our two-compartment
  chip — Zenodo metadata says CC BY 4.0, the dataset's own README says CC BY-NC-ND, so the conflict
  rule excludes it. Also rejected: Carstens et al. 2025 3D hiPSC HD-MEA and a GABA-A seizure-liability
  study (data "upon reasonable request" — gated); Zenodo 14363732 and IMI2-NeuroDeRisk (NC/ND);
  DANDI 001872 and 001336 (CC-BY human organoid MEA but **no pharmacology**).
- **Archive check, recorded because it affects design balance:** the Sharf diazepam arm has 19 of the
  20 files the design implies — organoid 2957 has no 30 µM condition.

---

## Phase 1 — standard, features, ETL

### 2026-09-16 · Record schema and data standard · `efbdf42`
`schema/neuromps_schema.json` (JSON Schema 2020-12) is normative; `schema/STANDARD.md` is the
human-readable draft. Decisions worth knowing:

- **A record is one unit, one condition, one data level.** The three levels — `spike_times`,
  `per_well_features`, `fitted_curve` — exist because public datasets publish at different stages of
  processing, and **records are comparable only within a level**.
- A dose value requires a dose unit (`dependentRequired`); a non-null `synchrony_index` requires its
  named method; provenance requires at least one source file with a SHA256.
- **Three conformance levels**, with Level 2 (every provenance hash matches `MANIFEST.json`) enforced
  at build time rather than only by a CLI.
- `jsonschema` 4.26.0 pinned (already present transitively via `sbi`) so the standard is
  machine-checkable rather than prose.
- **Known modelling compromise:** `culture_age_days` flattens organoid age (months, converted at 30
  days/month) and days in vitro into one field. The conversion is recorded in
  `provenance.transform`, but the two concepts are not the same thing.
- Example records carry **null feature values by design** until HD4, so no plausible-looking numbers
  entered the repository; `schema/examples/README.md` says so.

### 2026-09-16 · Feature definitions · `dd3d59b`
`src/neuromps/features/mea.py` is the single extractor; the twin calls the same function. Choices:

- Active electrode: **≥ 5 spikes/min**; a unit is usable at **≥ 10 active electrodes**. Both adopted
  from the EPA acute MEA assay for comparability, and both recorded per record.
- Burst detector: `maxinterval_simple`, ISI ≤ **0.1 s**, minimum **5 spikes**.
- Synchrony: **STTC** (Cutts & Eglen 2014) with dt = **0.05 s**, mean over electrode pairs.
- Network bursts: **coincidence-based**, in the spirit of Axion's "network spike", not a
  rate-elevation detector — synchronous tonic firing counts, and a test documents that.
- **Chance-corrected participation threshold.** A fixed threshold counted chance coincidences: 16
  independent 5 Hz electrodes produced roughly 50 spurious network bursts per minute. The threshold
  is now the smallest participation whose expected false positives over the recording stay below
  **0.05**, with electrodes modelled as independent Poisson processes.
- Undefined features are `null`, never 0.0.

### 2026-09-16 · ETL and three data problems · `be0551e`
`run_all.py --stage data` rebuilds the asset from raw with no manual steps. Problems the build
caught, all fixed with tests:

1. **272 ToxCast rows are single-concentration fits.** The schema now allows `n_concentrations ≥ 1`
   and those records carry `incomplete_dose_series` + `usable: false` instead of being dropped.
2. **309 ToxCast rows have no chemical name**: they are the assay's controls, named in `spid` —
   DMSO, ethanol, acetic acid, HCl (vehicles) and bicuculline (reference GABA-A antagonist). Now
   named and typed by role.
3. **90 excluded wells were silently missing.** Upstream `ExcludedWell` rows have no spike data at
   all, so they produced no records, contradicting STANDARD.md §5. Records are now built from the
   union of experiment log and spike file.

Also fixed: `check_traceable()` indexed by hash, so two files with identical content (the per-phase
noisy-electrode lists) produced a false mismatch; it compares by path first now.
Asset composition: `metrics.data_asset.counts`.

### 2026-09-20 · Cross-record QC · `a91ee0d`
`replicate_divergence` / `outlier_feature_value` at robust z > **3.5** against the well's replicate
group (same source, plate, phase, compound, dose); groups smaller than **3** are never flagged;
`missing_baseline_pair` for an exposure with no baseline. **Flags never change `qc.usable` on their
own.** Two fixes found by testing:

- robust z collapsed when more than half a group shared a value exactly (MAD = 0), hiding the single
  divergent well; the **mean absolute deviation** (scaled by 1.2533) is now the fallback scale;
- the flag did not say *which* feature diverged, so a well at the group median for firing rate looked
  wrongly flagged when it had diverged on synchrony. `qc.note` now records feature and z-score.

Flag counts: `metrics.data_card.flag_counts` and `outputs/DATA_CARD.md`.

### 2026-09-20 · Per-source ETL decisions · `057461f`, `bb48766`, `174eb2f`
- **`screenneuropharm_bic`:** species confirmed as **E18 rat** cortical and hippocampal cultures from
  the source paper's methods (doi:10.3390/biom13010074), not inferred from file names. 8-minute
  recordings at 10 kHz, 60 electrodes.
- **`sharf2022`:** the archive publishes **sorted units**, but our features are per electrode, so
  units are merged onto their peak electrode and spikes of different units within **1 ms** are
  collapsed — an electrode could not have resolved them, and sorters do split one neuron across
  units. Recording duration is taken as the **last spike time** (every file within 0.2 s of 180 s),
  which is a lower bound, so rates are marginally high. Both recorded in `qc.note`.
  The Neuropixels files (different probe) and the recording-interval experiments are excluded.
- **`fofie2025` fetched but not converted.** The workbook is structurally parseable, but **dose units
  appear nowhere in the file** (TTX 30 nM, HWTX 20 nM, lidocaine 500 µM are stated only in the
  paper), and the sheet named `Curves TTX-HWTX-Lidocaine` contains only TTX-labelled blocks, so the
  other compounds cannot be mapped from the file alone. Converting it would mean guessing a unit or a
  compound. The data card lists sources fetched but not in the asset so the gap is visible.

### 2026-09-20 · Compartment extension to the schema · `8cb2ac3`
Driven by a discovery: the ScreenNeuroPharm archive contains 16 multi-chamber devices (5 `CxCx`,
5 `HpHp`, 6 `CxHp`) with the full bicuculline series.

- `device` block: `device_id`, `layout`, `compartment_id`, `compartment_role` (`chamber` vs the
  `interface` strip over the connecting region), and microchannel geometry. Geometry (28 channels,
  10 µm × 250 µm, 12×12 array at 200 µm pitch) comes from Brofiga et al. 2022,
  doi:10.3390/mi13081212. Electrode groups: Red = columns 1–5, Green = 8–12, Gray = 6–7.
- `coupling` block: `peak_correlation`, `lag_ms` (positive = this compartment leads), `asymmetry`,
  and the named method. **Descriptive, not causal** — common drive produces correlation without any
  projection.
- **`cell_source.system` accepts `unknown`, but only with a note explaining why.** Used for `CxHp`,
  where neither the data nor the repository's own code records which electrode-colour group holds
  cortical and which hippocampal cells. Their asymmetry is still measurable, which is why the records
  are kept rather than dropped or guessed.
- Coupling measure (`features/coupling.py`): cross-correlation of binned population rates, **5 ms
  bins, ±200 ms**. Direction is reported only when the peak correlation clears a noise floor of **5
  standard errors** (1/√bins) — otherwise uncoupled compartments yield a confident direction computed
  from noise. Implemented with an FFT correlation after a per-lag loop proved far too slow.

---

## Phase 2 — models

### 2026-09-20 · Modelling table · `1a43c27`
`models/dataset.py`. One row per exposure recording paired with its own unit's baseline, described by
**log2 feature ratios** — absolute rates differ by platform and culture far more than drug effects do.

- A silent exposure is floored at **0.05** of its baseline rather than running to −∞, so "activity
  abolished" stays finite and comparable.
- A **zero or missing baseline gives NaN**: that ratio does not exist, and inventing one would feed a
  fabricated effect size to the model.
- `group` = source : species : unit, so a culture never straddles a cross-validation split.

### 2026-09-20 · Algorithms, CV and hyperparameters · `1a43c27`
- Pipeline for every model: median imputation → standard scaling → estimator.
- **Logistic regression**: `max_iter=2000`, `class_weight="balanced"`, `C=1.0`. **Gradient
  boosting**: `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)`.
  **Dummy**: most-frequent, as the chance reference.
- **`StratifiedGroupKFold`**, shuffled, seeded; the number of splits is capped by the smallest class.
- **No hyperparameter search was run.** With 73 pairs, tuning would have been fitted to noise; the
  values above are library defaults except where stated. This is a deliberate limitation, not an
  oversight.
- **Three tasks are kept separate** because the classes are confounded with platform: only Kapucu has
  all six mechanism classes, so a pooled model would largely learn which lab a record came from.
  Results: `metrics.mechanism_baselines.task1_within_platform`, `.task2_external_detection`,
  `.task3_platform_contrast`; figures `outputs/figures/mechanism_confusion.png` and
  `platform_contrast_gabaa_antagonist.png`.
- **No specificity is reported for the external check.** Those datasets contain no vehicle controls,
  so specificity is undefined; a different mechanism (diazepam, a GABA-A *positive* modulator) serves
  as the negative control instead.

### 2026-09-20 · Transfer study · `e6f8e60`
`models/transfer.py`. Kapucu only, because it is the one source with both species on one platform in
one lab, which isolates species rather than platform.

- Scored on the labels present in **both** species; a class the target species never received cannot
  be predicted or credited. Fewer than two shared classes returns "cannot be scored" rather than a
  number.
- **Bootstrap** percentile CI with **2000** resamples; **permutation test** with **5000** shuffles of
  the target labels.
- Within-species baselines use **repeated** grouped CV (5 repeats) because the smallest class has a
  handful of members and a single split assignment is mostly noise.
- **Domain adaptation tried and reported as found:** per-species standardisation and CORAL (Sun et
  al. 2016). Both *hurt* — with 32 target samples they estimate scale and covariance from noise.
  They are kept in the code and reported rather than quietly dropped.
- Results: `metrics.species_transfer.transfer` and `.species_shift`; figure
  `outputs/figures/species_shift.png`.

### 2026-09-20 · Compartment asymmetry test · `e6f8e60`
`models/asymmetry.py`. Deliberately conservative:

- **One value per device** (conditions of one chip are not independent) — the device median.
- **Magnitudes only.** The colour-to-population mapping is unrecorded, so the sign is arbitrary per
  device.
- **Chamber-to-chamber links only**; the interface strip is not a chamber.
- Mann-Whitney U, two-sided, with rank-biserial effect size; **Holm correction** across the pairwise
  tests within each metric; groups smaller than **3** devices are not tested.
- **A pooled "heterogeneous vs homogeneous" comparison was computed and then abandoned as the
  headline**: it is an artefact of pooling, because HpHp overlaps CxHp. The per-device-type analysis
  replaced it. Both are still in `metrics.compartment_asymmetry`
  (`pooled_heterogeneous_vs_homogeneous` and `by_device_type`), so a reader can see the difference.
- Figure: `outputs/figures/compartment_lag_by_device_type.png`.

### 2026-09-20 · Uncertainty and ablations · `964d6e8`
- **Split conformal prediction**, calibration split grouped by culture so calibration scores never
  come from wells the model trained on. Non-conformity is 1 − probability of the true class; the
  threshold is the (1−α)(n+1)/n empirical quantile. Swept over α ∈ {0.05, 0.1, 0.2, 0.3}.
  **With 73 pairs the calibration set is small, so neighbouring α can share a threshold** — noted in
  `metrics.uncertainty_and_ablations.note`.
- **Feature subsets**: `all`, `rate_only`, `burst_only`, `network_only`, `synchrony_only`,
  `mfr_only`, and `species_invariant` — the last named after the HD9 result so that the claim "these
  are the portable features" is tested rather than asserted. Plus leave-one-out.
- Results: `metrics.uncertainty_and_ablations.{conformal_sweep, feature_subsets, leave_one_out}`;
  figure `outputs/figures/conformal_set_size.png`.

---

## Phase 3 — the twin

### 2026-09-20 · Network design · `944a689`
`twin/network.py`. Conductance-based LIF with spike-frequency adaptation, 80/20 E/I, two
compartments, recurrent within each and a directed pathway between them.

- **Connectivity is set by in-degree, not probability** (`intra_in_degree=20`, `inter_in_degree=10`).
  With fixed probability the number of inputs per neuron grows with network size, so a calibration at
  150 neurons did not hold at 250. This was found by validating, not by design.
- `forward_ratio` tilts the pathway **without changing the total number of inter-compartment
  synapses**, so asymmetry and overall coupling stay separable parameters.
- Defaults after manual tuning: 250 neurons/compartment, excitatory weight 0.25, inhibitory 2.6,
  background 100 synapses × 10 Hz × weight 0.042, adaptation increment 1.5 with τ 380 ms, intra delay
  1.5 ms, inter delay 4.0 ms. **Tuned by hand against the real feature ranges, not fitted.**
- Drug knobs `gabaa_scale`, `ampa_scale`, `threshold_scale` multiply conductances or threshold; 1.0
  is untreated.
- **Brian2 code generation: `numpy`, not `cython`.** Cython pays a compile cost on every run
  (11.6 s vs 0.5 s for a 400-neuron, 2 s benchmark) and needs a compiler in CI.
- The virtual-electrode mapping in `twin/validate.py` is a **placeholder** until HD12: 16 electrodes
  per compartment, each listening to **8** neurons. An earlier version pooled *every* neuron into 16
  electrodes, which made per-electrode firing rate scale with network size — an artefact of the
  mapping, not the network.
- Validation output: `metrics.twin_hd11_validation`, `outputs/twin/hd11_validation_simulated.json`,
  `outputs/twin/hd11_directionality_simulated.png`. All simulated outputs are labelled SIMULATED in
  filename and figure title.

### 2026-09-20 · The confound the twin exposed · `944a689`
**Coupling asymmetry tracks the compartments' relative firing rate at least as much as the pathway
direction**, because a directed pathway also excites its target and makes it burst sooner. In the
simulation, the same wiring reads as either direction depending on the target's background drive
(`metrics.twin_hd11_validation.directionality_sweep`). A regression test locks this in.

Checked against the real devices, where it does not bite — **these three checks are not yet produced
by a script under `src/`** and must be before they enter the report:
- chamber rate imbalance does not differ by device type (Kruskal p = 0.96);
- rate imbalance does not correlate with measured |lag| (ρ = −0.05, p = 0.85);
- signed rate ratio does not predict signed asymmetry (ρ = +0.04, p = 0.73).

Consequence for Phase 3: **any inference of direction from coupling must control for firing rate.**

### 2026-09-20 · Dose-unit normalization centralised, mM bug fixed · `f821bb4`
Writing this notebook surfaced a latent bug: `etl/kapucu.py` and `etl/screenneuropharm.py` each
mapped a `mM` dose suffix to `uM` **without multiplying the value by 1000**, so a 5 mM exposure would
have been recorded as 5 uM — the number kept, the unit changed, no error raised.

- The conversion now lives in one place, `src/neuromps/etl/units.py`, because the same wrong mapping
  had been written twice and would have drifted again.
- An unknown unit now raises rather than being guessed.
- **No current record was affected**: the asset holds only `uM`, `nM` and null doses, and rebuilding
  produced byte-identical counts (21,272 records; uM 20,927 / nM 123 / null 222). This was verified
  by rebuilding, not assumed.
- Regression tests at both levels (the helper, and each ETL's own parser) were confirmed to fail
  against the old mapping before the fix was applied.

---

### 2026-09-20 · Calibration and interval policy · `4cc93cf`
Two gaps against PLAN.md HD9, closed before starting HD13.

- **Calibration.** Expected calibration error over out-of-fold predictions, 10 confidence bins,
  under the same grouped CV as everything else; confidence is the probability of the predicted class
  (standard multiclass ECE). Value, interval and per-bin counts:
  `metrics.mechanism_baselines.task1_within_platform.calibration`; diagram
  `outputs/figures/task1_reliability_real.png`.
- **Interval policy, applied everywhere from here on:** a **cluster bootstrap over culture groups**,
  never over rows. Two conditions of one well share a baseline and a culture, so row resampling
  treats them as independent evidence and reports intervals that are too narrow. Percentile
  intervals; **1000 resamples, 200 under `--quick`**, and the count is written next to every
  interval so a reader can tell which run produced a number.
- Metrics carrying `*_ci95` from now on: Task 1 balanced accuracy per model, per-feature species
  shift, feature subsets, leave-one-out, ECE.
- `tests/test_metrics_contract.py` enforces this at two levels, so an interval cannot be dropped
  silently by a later change.
- **Consequence worth stating in the report:** several comparisons do not survive their own
  intervals. `rate_only` (0.530 [0.396, 0.624]) and `all` (0.514 [0.398, 0.662]) overlap almost
  entirely, and only one of nine species-shift features has an interval excluding zero. Point
  estimates alone overstated both results.

### 2026-09-20 · Quick-tier provenance bug · `4cc93cf`
Running the whole pipeline end to end exposed a bug that had been invisible while stages were run
individually: a `--quick` fetch marked the 582 MB ScreenNeuroPharm archive `skipped`, which removed
its files from `MANIFEST.json`, and every record built from the cached features then failed the
conformance-level-2 hash check. The quick tier now keeps files an earlier `--full` run left on disk,
and re-hashes files that are on disk but missing from the manifest, so a quick run repairs
provenance instead of degrading it. Regression test in `tests/test_new_sources.py`.

---

### 2026-09-20 · Drug-knob validation design · `66e650e`
- The real direction for each mechanism is **computed from the asset**, not asserted: median log2
  change of that mechanism's real pairs, with a cluster bootstrap over cultures.
- **A feature is scored only if the real interval excludes zero.** Of 24 knob-feature comparisons, 8
  were left unscored on this rule. Agreeing with a direction the data cannot resolve would be a free
  pass, and half the twin's "agreements" would have been with noise.
- Silence convention: a knob that stops all firing gives rate-like features of zero rather than
  undefined, so the ratio floors at log2(0.05) = -4.32 — the same floor `models/dataset.py` applies
  to an abolished real well. Before this, the sodium block produced only missing values and scored
  0/0.
- Known asymmetry in the comparison: each mechanism's real reference comes from whichever source has
  that drug (bicuculline from 60-electrode MCS chambers; TTX and the glutamatergic drugs from
  16-electrode Axion wells), and HD8 showed deltas do not transfer unchanged across platforms.
  **Agreement here is evidence about direction, not magnitude.**
- Result and the one disagreement: `metrics.twin_hd13_perturbation`, figure
  `outputs/twin/hd13_drug_knob_validation_simulated.png`.

---

---

## Tried and abandoned

| What | Why it was dropped | Where |
|---|---|---|
| `iras_mea` (Tukker 2020) | CC BY-NC 4.0, incompatible with an MIT release competing for prizes | `388477b` |
| Zenodo 14525182 compartmentalised MEA | Metadata says CC BY, README says CC BY-NC-ND | `2050862` |
| `fofie2025` conversion | Dose units absent from the file; sheet titles disagree with content | `174eb2f` |
| Global sign flip for ToxCast `top` | Inverts the viability endpoints; the convention is per assay | `b613b45` |
| AEID-range selection of MEA rows | The range contains 11 unrelated assays | `83e190f` |
| Fixed participation threshold for network bursts | Counted chance coincidences (~50/min for independent 5 Hz electrodes) | `dd3d59b` |
| Per-lag loop for cross-correlation | ~10⁸ operations per compartment pair; replaced by FFT | `8cb2ac3` |
| Pooled heterogeneous-vs-homogeneous asymmetry test | Artefact of pooling; HpHp overlaps CxHp | `e6f8e60` |
| Per-species standardisation and CORAL | Both reduced transfer accuracy; kept in code, reported as negative | `e6f8e60` |
| Gradient boosting as the headline model | Worse than logistic regression at this sample size | `1a43c27` |
| Connectivity by probability in the twin | In-degree grew with network size; replaced by in-degree | `944a689` |
| Pooling all neurons into 16 virtual electrodes | Per-electrode rate scaled with network size | `944a689` |
| brian2 `cython` code generation | Compile cost per run; needs a compiler in CI | `944a689` |

---

## Known limitations

**Data**
1. Spike detection is the publishers' in every source; we never detect spikes ourselves.
2. Records are comparable only within a data level (spike times / per-well features / fitted curves).
3. Electrode counts span 8 to 26,400 across sources; electrode-count-dependent features are not
   comparable across platforms without explicit rescaling.
4. Kapucu pharmacology is **single-concentration** (5 compounds + TTX), so no dose-response there.
5. ToxCast `mc5-6` holds curve fits and hit calls, **not per-concentration responses**.
6. `CxHp` devices carry `cell_source.system: unknown`: the colour-to-population mapping is unrecorded,
   so the measured direction cannot be read as cortical→hippocampal or the reverse.
7. `culture_age_days` conflates organoid months (÷30) with days in vitro.
8. `sharf2022` rates are marginally high (duration = last spike time) and its diazepam design is
   incomplete for organoid 2957.
9. `fofie2025` is fetched but not in the asset.

**Models**
10. Task 1 rests on **73 pairs with 4–26 members per class**; it supports "features carry mechanism
    information within a platform", not a performance claim.
11. No hyperparameter search; no nested CV. The PLAN called for nested CV — **the implemented scheme
    is single-level grouped CV with fixed hyperparameters**, which is defensible only because nothing
    was tuned.
11b. Intervals are percentile cluster bootstraps over a few dozen groups, so they are themselves
    noisy; they bound a claim, they do not make one precise.
12. Conformal calibration sets are tiny, so neighbouring α share a threshold.
13. The external detection check has no negative controls of its own.
14. The compartment tests have 5–6 devices per group and support an ordering, not an effect size.

**Twin**
15. Simulated network bursts (27/min) and synchrony (0.84) sit slightly above the real range;
    calibration is HD12's job.
16. The electrode mapping is a placeholder with no geometry.
17. Coupling asymmetry confounds connectivity with relative excitability (see above).
18. Twin parameters were tuned by hand in HD11, then by seeded random search in HD12; no formal fit
    (likelihood or posterior) yet - that is HD14.
19. The twin over-responds in rate: under disinhibition it raises firing where the real data show it
    flat or slightly down. Bursting reorganises correctly. Same direction of error as the HD12
    over-synchrony.
20. Each drug knob is compared against a real reference from a different platform, so knob agreement
    supports direction, not magnitude.

---

## Open questions

1. **Can the `CxHp` colour-to-population mapping be recovered?** One email to the ScreenNeuroPharm
   authors would turn an unlabelled direction into a cortical→hippocampal claim.
2. **Should the real-data confound checks become a script?** The three rate-imbalance checks behind
   the HD11 caveat are currently ad-hoc queries and cannot enter the report until they are under
   `src/` writing to `metrics.json`.
3. **Does the platform disagreement survive matched conditions?** Gabazine and bicuculline differ in
   recording duration (1800 s vs 480 s), electrode count and culture age; none of these has been
   varied to see which drives the sign disagreement in
   `metrics.mechanism_baselines.task3_platform_contrast`.
4. **Is the chance-corrected network-burst threshold itself responsible?** It scales with electrode
   count, so it is a candidate explanation for the cross-platform disagreement in exactly the
   features that disagree.
5. **Would CI pass?** The workflow has never run; there is no remote.
6. **Does the asymmetry measure need replacing for inference?** Given the excitability confound, HD14
   may need a rate-controlled statistic rather than the current one.
7. ~~A latent dose-unit bug.~~ **Fixed** on 2026-09-20, see the entry above. No further action.
