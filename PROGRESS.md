# PROGRESS

## 2026-09-16: Phase 0, HD1 (branch `phase0-setup`)

### Done
- Repo scaffold per PLAN.md; `pyproject.toml` pinned (Python 3.11 venv via `uv`; numpy held at 2.3.5
  because brian2 2.9.0 does not import with numpy 2.4).
- CI workflow `.github/workflows/ci.yml`: ruff, pytest, `run_all.py --quick` smoke (CPU-only torch).
  Not yet exercised on GitHub (no remote configured).
- `src/neuromps/fetch/`: one module per source (`kapucu2022`, `toxcast_mea`, `axon_diode`), shared
  downloader with hash cache, remote-zip member extraction, Europe PMC license check, exclusion
  records, `data/raw/MANIFEST.json` writer. 18 tests passing, ruff clean.
- Quick fetch: all sources `ok`; rerun re-downloads nothing and hashes are stable
  (`outputs/logs/fetch_quick.log`). Kapucu `--full`: 239 files, 1.2 GB, `ok` (`outputs/logs/fetch_full.log`).
  ToxCast `--full`: 42 files, 2.57 GB, `ok`, mc5-6 size matches the zip listing (`outputs/logs/fetch_full_toxcast.log`).
- `data/SOURCES.md` records the raw-vs-preprocessed decision (spike-time CSVs, not the 2.4 TB raw h5)
  and the license policy.
- **Decision:** Tukker 2020 (`iras_mea`, CC BY-NC 4.0) excluded from the data asset. The competition
  rules give joint ownership to team and Organizing Committee, not the entrant's own license, so the
  revisit condition is not met. Recorded as `skipped` in MANIFEST via `fetch/excluded.py`.
- **Gate (≥2 sources with real files and licenses recorded): met.**

### Bugs found against the real EPA Clowder server (both fixed, with regression tests)
- Suffix ranges (`bytes=-1`) return HTTP 500. Use explicit ranges.
- If `Accept-Encoding: gzip` is sent, Clowder replies 206 with a plausible Content-Range but streams
  the whole 7.5 GB file gzipped from byte 0. Range requests now send `identity` and verify status,
  start offset and encoding.

### Blocked / open
- Without Tukker 2020, the rat→hPSC transfer study rests on Kapucu 2022 alone (one pharmacology plate
  per species, 5 single-concentration compounds). Consider searching for another CC BY / CC0
  hiPSC MEA pharmacology set (e.g. DANDI) during HD2.

## 2026-09-16: Phase 0, HD2

### Done
- `src/neuromps/etl/toxcast_mea.py` + `src/neuromps/metrics.py`: MEA rows selected by name, written to
  `data/interim/toxcast_mea_mc56.parquet` (20,375 rows), per-endpoint table in
  `outputs/tables/toxcast_mea_endpoints.csv`, counts in `outputs/metrics.json`. 23 tests pass.
- Read the EPA assay descriptions. Same MEA platform as Kapucu (Axion Maestro 48-well, 16 electrodes,
  12.5 kHz); acute = 40 min baseline vs 40 min exposure; dev = 12-day exposure, AUC over DIV 5/7/9/12.
- `schema/FEATURE_SET.md`: draft 9-feature core vector mapped to both meaRtools outputs and ToxCast
  acute endpoints, with the non-interchangeable items (STTC vs Axion synchrony) called out.
- Found: no compound overlap between Kapucu and ToxCast acute (TTX is ToxCast's positive control).
- **OQ-1 resolved:** ToxCast `top` sign is per assay (acute `-top`, dev `+top`), derived from
  cytotoxicity anchors and re-checked every run (acute 92.4%, dev 99.1% agreement). openpyxl pinned
  (approved) to read the EPA annotations; `signal_direction` alone was not enough, since all 32
  activity endpoints are annotated `bidirectional`.

- **Three sources added (PI decision):** `sharf2022` (human iPSC organoid HD-MEA, diazepam
  control/3/10/30/50 uM in 4 organoids, CC0), `fofie2025` (hiPSC nociceptors on Axion, CC0, compound
  overlap incl. TTX/GABA/lidocaine/DNQX), `screenneuropharm_bic` (rat bicuculline series, MIT).
  License policy widened to permissive OSI licenses; each fetcher now re-checks its source's license
  at fetch time and blocks if it changed upstream. 34 tests pass.

### Blocked / open
- `fofie2025` has total spike counts only: firing-rate features only, no bursts. `sharf2022` electrode
  geometry (26,400 CMOS electrodes) is not comparable to 16-electrode Axion wells; the schema must
  record the data *level* (spike times / per-well features / fitted curves) per record.
- A near-perfect match for the two-compartment chip (Zenodo 14525182, compartmentalized asymmetric
  microchannels) was rejected: Zenodo says CC BY 4.0, its own README says CC BY-NC-ND. Revisit only if
  the authors clarify.

## 2026-09-16: Phase 1, HD3

### Done
- `schema/neuromps_schema.json` (JSON Schema 2020-12): record = one unit, one condition, one data
  level. Conditional rules per level; dose requires a unit; a synchrony value requires its named
  method; provenance requires at least one hashed source file.
- `schema/STANDARD.md` (draft v0.1): scope, terminology, the three data levels, technical
  requirements, QC vocabulary with EPA-derived defaults (active electrode >= 5 spikes/min, well usable
  with >= 10 active electrodes), provenance, 3 conformance levels, and known gaps.
- `src/neuromps/qc/schema.py`: validator + `check_traceable()` (Level 2: hashes must match
  MANIFEST.json) + CLI `python -m neuromps.qc.schema [--traceable] FILE...`.
- `schema/examples/`: 5 records, one per data level and platform; all valid at Level 2. Feature values
  are null by design until HD4 - they are not filled with plausible numbers.
- 51 tests pass, ruff clean.

### Decisions taken while the PI was away
- Added `jsonschema==4.26.0` (MIT; already present transitively via sbi) so the standard is
  machine-checkable rather than prose. Pinned in pyproject.
- `culture_age_days` flattens organoid months (converted at 30 d/month) and DIV into one field; the
  conversion is recorded in `provenance.transform`. Listed in STANDARD.md section 8 as a known gap.
- The schema has no compartmentalized-chip representation yet (per-compartment units + directed
  connectivity). That is the main planned extension and is needed for the digital twin.

## 2026-09-16: Phase 1, HD4

### Done
- `src/neuromps/features/mea.py`: the single feature extractor (ETL and twin both call it).
  Returns exactly the schema's `features` block; undefined features are `None`, never 0.0.
  `FeatureParams.as_method_record()` emits the method names the schema requires.
- 17 tests on synthetic trains with known statistics: exact burst counts/durations, ISI CV 0 for
  regular and ~1 for Poisson, STTC 1.0 for identical trains and ~0 for independent ones, activity
  threshold, no-active-electrode nulls, determinism, input not mutated, unsorted input, and a test
  that feature keys match `schema/neuromps_schema.json` exactly (prevents schema/code drift).
- **Assumption found and fixed:** a fixed participation threshold for network bursts counts chance
  coincidences - 16 independent 5 Hz electrodes produced ~50 spurious network bursts/min. The
  threshold is now chance-controlled (smallest participation whose expected false positives over the
  recording stay < 0.05, electrodes modelled as independent Poisson). Two tests pin both directions:
  independent electrodes give 0, and real synchrony on a 10 Hz background is still detected.
- Definition made explicit in code, tests and STANDARD.md: the network-burst detector is
  coincidence-based (Axion "network spike"-like), not rate-elevation; synchronous tonic firing counts.

### Exploratory (not a report number; n=1 well pair, no script under src/ yet)
- Smoke check on the real rat pharmacology plate, gabazine well A8 baseline -> exposure:
  mfr 1.76 -> 2.67 Hz, burst rate 1.70 -> 9.90 /min, spikes in bursts 11.6 -> 58.2 %,
  network bursts 0.0 -> 7.27 /min, STTC 0.045 -> 0.33; control well A3 flat (mfr 4.87 -> 4.64).
  Direction matches disinhibition. ~0.25 s per well (16 electrodes, 1800 s).
  Aggregate across wells belongs in HD5 ETL, where it becomes a real, script-produced number.

## 2026-09-16: Phase 1, HD5

### Done
- `etl/kapucu.py`: pharmacology plates -> schema records (one per well per phase). Normalizes the
  `D-AP5`/`DAP5` spellings and `30uM`-style doses, maps compounds to the mechanism vocabulary, turns
  noisy-electrode lists and upstream `ExcludedWell` marks into QC flags.
- `etl/toxcast_mea.py`: `build_curve_records()` for `fitted_curve` records.
- `etl/build.py`: collect -> validate (schema **and** conformance level 2) -> write
  `data/processed/{neuromps.parquet, neuromps_records.jsonl, provenance.json}` + metrics.
  Nothing is written unless every record passes.
- Wired into `run_all.py --stage data`.
- **Gate met:** `python run_all.py --quick --stage data` rebuilds the asset from raw in ~44 s with
  zero manual steps; 91 tests pass, ruff clean.

### The asset (data/processed/provenance.json)
- 20,663 records: 288 `spike_times` (Kapucu, 6 plates x 48 wells) + 20,375 `fitted_curve` (ToxCast).
- Species: 144 human, 20,519 rat. Usable: 20,238; flagged unusable: 425.
- QC flags in use: `excluded_well_upstream` (90), `noisy_electrodes_present`,
  `too_few_active_electrodes`, `incomplete_dose_series`.

### Three data problems the build caught (all fixed, with tests)
1. 272 ToxCast rows are single-concentration fits. The schema now allows `n_concentrations` >= 1 and
   these carry `incomplete_dose_series` + `usable: false` rather than being dropped.
2. 309 ToxCast rows have no `chnm`: they are the assay's controls, named in `spid` - DMSO, ethanol,
   acetic acid, HCl (vehicles) and **bicuculline** (reference GABA-A antagonist, which links to the
   `screenneuropharm_bic` source). Now named and typed by role.
3. Excluded wells have no spike rows at all, so 90 of them were silently absent from the asset,
   contradicting STANDARD.md 5. Records are now built from the union of log and spike wells.
- Also fixed: `check_traceable()` indexed by hash, so two files with identical content (per-phase
  noisy-electrode lists) produced a false mismatch. It now compares by path first.

## 2026-09-20: Phase 1, HD6

### Done
- `qc/flags.py`: cross-record QC applied over the whole asset - `replicate_divergence` /
  `outlier_feature_value` (robust z > 3.5 against the well's replicate group, groups < 3 skipped) and
  `missing_baseline_pair`. Flags are informational and never change `qc.usable`; `qc.note` names the
  diverging feature and its z-score.
- `qc/data_card.py` -> `outputs/DATA_CARD.md`, generated from the asset and MANIFEST: per-source
  records/levels/species/licenses, feature coverage, flag counts, excluded sources, limitations.
- Both wired into `run_all.py --stage data`. 103 tests pass, ruff clean.

### Two fixes found by testing
- Robust z-scores collapsed when more than half a replicate group shared a value exactly (MAD = 0),
  hiding the one divergent well. Falls back to the mean absolute deviation as the scale.
- The flag did not say which feature diverged, so a well at the group median for firing rate looked
  wrongly flagged (it had diverged on synchrony). `qc.note` now records feature and z-score.

### Flags on the real asset (outputs/DATA_CARD.md)
- kapucu2022: 153 `too_few_active_electrodes` (incl. the 90 excluded wells), 90
  `excluded_well_upstream`, 35 `replicate_divergence`, 11 `noisy_electrodes_present`.
- toxcast_mea: 272 `incomplete_dose_series`. No `missing_baseline_pair`: every exposure plate pairs
  with its baseline.
- Spot check: 23 of the 35 divergences are TTX wells still firing after the blocker - e.g. rat well
  C8 goes 4.32 -> 5.24 Hz, a failed application. The flag is doing its job.

### Next first task (HD7)
- ETL for `sharf2022` / `screenneuropharm_bic` / `fofie2025` (fetched, not yet in the asset), then
  Phase 2 baselines.

## 2026-09-20: Phase 1, HD7

### Done
- `etl/screenneuropharm.py`: 203 records. Rat E18 cortical + hippocampal, 23 arrays, MCS 60-electrode,
  8 min at 10 kHz, basal + bicuculline 100 nM - 300 uM. Species confirmed from the paper's methods.
- `etl/sharf.py`: 29 records. Human iPSC organoid slices on MaxWell HD-CMOS, ~180 s at 20 kHz;
  diazepam control/3/10/30/50 uM across 4 organoids, six 7-month organoids, one at 5-8 months.
  Sorted units merged onto their peak electrode, coincidences within 1 ms collapsed.
- Feature caches under `data/interim/` (first build 272 s / 29 s; cached rebuild ~55 s total).
- Data card now lists sources fetched but not in the asset.
- Asset: **20,895 records across 4 sources**. 119 tests pass, ruff clean.

### Dose-responses now in the asset (both normalized to each unit's own control)
- Rat, bicuculline (GABA-A antagonist), n=23 arrays: network bursts 0.31x, burst rate 0.57x,
  STTC 1.17x at 300 uM, firing rate flat.
- Human organoid, diazepam (GABA-A positive modulator), n=4: firing 0.57x at 50 uM, network bursts
  ~1.7x. Opposite modulation of the same receptor, in two species - a good spine for Phase 2.

### Decision: `fofie2025` fetched but not converted
The workbook has no dose units anywhere (they are only in the paper), and the sheet named
`Curves TTX-HWTX-Lidocaine` contains only TTX-labelled blocks, so compounds cannot be mapped from the
file alone. Converting it would mean guessing a unit or a compound. Rationale and a route to convert
it later are in data/SOURCES.md. It is also a clean illustration of why the standard requires
`dose_value` + `dose_unit` together.

### Open for Phase 3
- The ScreenNeuroPharm archive holds **16 two-compartment devices** (CxCx, CxHp, HpHp; "Joint"
  recordings with Gray/Green/Red electrode groups). These are real two-compartment MEA recordings
  with a bicuculline dose-response - the natural validation set for the twin - and are skipped only
  because the schema has no per-compartment unit or directed connectivity yet. Extending the schema
  for compartments is now the highest-value schema work.

## 2026-09-20: compartment schema extension (done before Phase 2, at PI request)

### Done
- Schema: `device` block (device_id, layout, compartment_id, compartment_role, microchannel
  geometry) and `coupling` block (peak_correlation, lag_ms, asymmetry, method). A compartment_id
  requires a device_id. `cell_source.system` accepts `unknown` **only with a note**.
- `features/coupling.py`: directed coupling from cross-correlated population rates, with two guards
  found by testing - direction is null below a 5-sigma noise floor, and the correlation is FFT-based
  (a per-lag loop made one test run take 28 s; now 2 s).
- ETL: 377 compartment records from 16 devices (5 CxCx, 5 HpHp, 6 CxHp), full bicuculline series
  each; compartment extraction is cached under `data/interim/`.
- STANDARD.md 5.2 documents all of it, including that coupling is descriptive, not causal.
- Asset: **21,272 records**. 140 tests pass, ruff clean. `--full` rebuild ~13 min, cached ~50 s.

### The result that makes this worth having
Per-device median asymmetry red->green (exploratory, n=6 devices, no statistical test yet):
- `CxCx` (same tissue both chambers): -0.05 at **0 ms** lag
- `HpHp`: -0.09 at -5 ms
- `CxHp` (cortical + hippocampal): **+0.10 at +10 ms**, 4 of 6 devices leading the same way,
  1 opposite.
So heterogeneous devices show directional coupling where homogeneous ones do not. This is the real
counterpart of the twin's asymmetric connectivity, in a dataset we acquired for another reason.
**Do not report this without a proper test** (per-device bootstrap or rank test, script under src/).

### Open
- `CxHp` colour-to-region mapping is unknown, so those records carry `system: unknown` + note. If the
  authors confirm the mapping, the labels can be filled in and the direction becomes interpretable as
  cortical->hippocampal or the reverse.

## 2026-09-20: Phase 2, HD8 - mechanism baselines

### Done
- `models/dataset.py`: 583 pairs (exposure vs that unit's own baseline), log2 feature ratios,
  `group` = source:species:unit so a culture never straddles a CV split.
- `models/mechanism.py`: three tasks + figures, wired into `run_all.py --stage models`.
- 160 tests pass, ruff clean.

### Results (all in outputs/metrics.json, figures in outputs/figures/)
1. **Within-platform multiclass** (Kapucu, 73 pairs / 50 groups, 6 classes, chance 0.167):
   logistic regression **0.537 +/- 0.102** balanced accuracy, gradient boosting 0.398,
   dummy 0.167. Real but modest; n is small and classes are unbalanced (4-26 per class).
2. **External detection** (fit on Kapucu, applied to other labs): bicuculline detected in only
   **37%** of pairs, rising 0.18 -> 0.48 with dose. Diazepam (opposite mechanism) detected in 8%,
   probability falling with dose (rho -0.80) - the negative control behaves. Specificity is not
   reported: those datasets have no vehicle controls, so it is undefined.
3. **Platform contrast** (same mechanism, >=30 uM): 5 of 9 features agree in sign across platforms,
   4 disagree - including mean firing rate and network burst rate. Gabazine *raises* network bursts
   on 16-electrode Axion wells; bicuculline *lowers* them on 60-electrode MCS arrays.

### What result 3 means
Feature deltas do not transfer unchanged across platforms, so a classifier trained on one lab's MEA
does not simply work on another's. This is the harmonization limit the standard warns about, now
measured rather than asserted, and it is the strongest argument for the twin: a mechanistic model can
bridge platforms where raw feature deltas cannot. Candidate causes to separate in Phase 3: electrode
count and pitch (16 @ 200 um vs 60 @ 200 um vs HD-CMOS), recording duration (1800 s vs 480 s),
culture age, and the chance-corrected network-burst threshold, which itself scales with electrode
count.

## 2026-09-20: Phase 2, HD9 - species transfer + compartment lag

### Done
- `models/transfer.py` and `models/asymmetry.py`, both in `run_all.py --stage models` (8.4 s).
  179 tests pass, ruff clean. Figures: `species_shift.png`, `compartment_lag_by_device_type.png`.

### Rat -> human transfer (Kapucu only: both species, one platform, one lab)
| | balanced accuracy |
|---|---|
| rat -> human | **0.550** [0.400, 0.600], permutation p = 0.0002 |
| human within-species (5x4 grouped CV) | 0.646 +/- 0.139 |
| rat within-species (5x2 grouped CV) | 0.543 +/- 0.067 |
| chance (5 shared classes) | 0.200 |

- Training on rat costs little relative to what the rat data itself supports; human data is somewhat
  more learnable. Transfer is real but modest, on 37 train / 32 test pairs.
- **Domain adaptation hurts:** per-species standardization 0.408, CORAL 0.442. With 32 target
  samples both estimate scale/covariance from noise. Reported as found, per PLAN.
- Species-invariant features: `mfr_hz`, `pct_spikes_in_bursts`, `burst_rate_per_min`.
  Most shifted: `synchrony_index`, `isi_cv`, `network_burst_rate_per_min` - **the same features that
  failed across platforms in HD8**. Rate and burst-fraction measures are portable; synchrony and
  network-burst measures are not, across either species or platform.

### Compartment devices: the discriminator is lag, not asymmetry
The pooled heterogeneous-vs-homogeneous split (p = 0.042) is an artefact of pooling - HpHp overlaps
CxHp. Per device type, one value per device, magnitudes only:
- median |lag|: **CxHp 11.25 ms vs CxCx 5.0 and HpHp 5.0**; Kruskal p = 0.0082; Holm pairwise
  CxHp vs HpHp 0.026, CxHp vs CxCx 0.041, **CxCx vs HpHp 0.48** (internal control: the two
  same-tissue types are indistinguishable).
- peak coupling falls with heterogeneity: CxCx 0.96, HpHp 0.91, CxHp 0.82.
- |asymmetry| separates CxHp from CxCx (Holm 0.04) but not from HpHp (0.33).

n = 5-6 devices per group. This supports "these device types differ in lag", **not** a quotable
effect size, and the twin should be asked to reproduce a lag difference of this order.

## 2026-09-20: Phase 2 tail - uncertainty and ablations (Phase 2 complete)

### Conformal prediction sets (Kapucu, 73 pairs, 6 classes)
| alpha | target | empirical coverage | mean set size | singletons |
|---|---|---|---|---|
| 0.10 | 0.90 | 0.918 | **4.56 of 6** | 0.00 |
| 0.20 | 0.80 | 0.822 | 3.01 | 0.07 |
| 0.30 | 0.70 | 0.712 | 2.32 | 0.15 |

Coverage tracks the target, so the method is working. **The set size is the result:** to be right
90% of the time the model must name 4.6 of the 6 mechanisms and never commits to one. Per-class
coverage at alpha=0.1: 1.0 for control and both GABA-A classes, 0.50 for glutamatergic agonists
(n=6). This is what should be reported, not the 0.55 balanced accuracy on its own.

### Feature ablations (same grouped CV)
- `rate_only` (mfr_hz + active_electrodes, 2 features) **0.530 +/- 0.040** - as good as all nine
  (0.514). `species_invariant` 0.516. `burst_only` 0.411, `network_only` 0.417.
- Leave-one-out: only `mfr_hz` clearly matters (-0.050 when dropped). Dropping
  `pct_spikes_in_bursts` (+0.010) or `network_burst_rate_per_min` (+0.007) slightly *helps* - they
  are noise at this sample size.
- Consistent with HD8/HD9: the portable signal is rate, not synchrony or network bursts.

### State
- `run_all.py --stage models` runs all four analyses in 8.7 s; 5 figures; metrics.json has 7 sections.
- 198 tests pass, ruff clean. **Phase 2 is complete.**

## 2026-09-20: Phase 3, HD11 - two-compartment network

### Done
- `twin/network.py`: conductance-based LIF with adaptation, 80/20 E/I, two compartments, recurrent
  within and a directed pathway between them (`forward_ratio` = the axon diode). Drug knobs
  `gabaa_scale` / `ampa_scale` / `threshold_scale` wired to conductances and threshold.
- `twin/validate.py` -> `outputs/twin/hd11_validation_simulated.json` + figure, both labelled
  SIMULATED. 210 tests pass, ruff clean.

### Spontaneous activity (simulated, 250 neurons/compartment, same features/mea.py as real data)
| feature | simulated | real range | in range |
|---|---|---|---|
| mfr_hz | 2.26 | 1.0 - 6.5 | yes |
| burst_rate_per_min | 12.7 | 1.5 - 15 | yes |
| network_burst_rate_per_min | 27.0 | 0 - 25 | slightly high |
| synchrony_index | 0.84 | 0.04 - 0.8 | slightly high |

Bursts appear at plausible rates, so the HD11 gate is met; the network is a little too synchronous,
which is what HD12 calibration is for.

### Two modelling errors found while validating
1. Connectivity was set by probability, so in-degree grew with network size and a calibration at 150
   neurons did not hold at 250. Now set by **in-degree**, which makes the model size-invariant.
2. The placeholder electrode mapping pooled every neuron into 16 electrodes, so per-electrode firing
   rate scaled with network size. Each electrode now listens to a fixed number of neurons.

### The caveat this phase produced (regression-tested)
**Coupling asymmetry tracks the compartments' relative firing rate at least as much as the pathway.**
At `forward_ratio` = 16, measured asymmetry runs from -0.07 (target appears to lead) to +0.83 (source
leads) purely by changing the target's background drive - because a directed pathway also excites its
target and makes it burst sooner.

Checked against the real devices, where it does **not** bite:
- chamber rate imbalance does not differ by device type (Kruskal p = 0.96)
- rate imbalance does not correlate with measured |lag| (rho = -0.05, p = 0.85)
- signed rate ratio does not predict signed asymmetry (rho = +0.04, p = 0.73)

So the HD9 lag result stands, but **any inference of direction from coupling must control for firing
rate**. That is now a requirement for HD14-16, not an afterthought.

## 2026-09-20: Phase 3, HD12 - MEA simulator and calibration

### Done
- `twin/mea.py`: neurons and electrodes placed in space. Each chamber is 1.6 x 1.2 mm with a 6x8
  electrode grid at 200 um pitch (48 per chamber, matching the real device); an electrode hears
  somata within `detection_radius_um`. The interface electrodes over the microchannel region are
  **not modelled** - their signals are axonal, not somatic - and are excluded rather than faked.
- `twin/calibrate.py`: targets computed from the asset at run time (ScreenNeuroPharm chamber
  baselines, n=48), compared **per compartment** because that is how a real chamber record is
  defined. Seeded random search over 7 parameters, last 40% of the budget refining around the best
  draw; all 45 trials kept in `outputs/twin/hd12_calibration_simulated.json`.
- `run_all.py --stage twin` runs validation then calibration. 241 tests pass, ruff clean.

### Calibration result (`metrics.twin_hd12_calibration`)
Best score **0.076**, from 2.31 at the worst draw.

| feature | simulated | real band | inside |
|---|---|---|---|
| mfr_hz | 6.50 | 6.79 - 21.01 | 4% under |
| burst_rate_per_min | 22.48 | 12.63 - 29.73 | yes |
| network_burst_rate_per_min | 29.25 | 11.69 - 22.70 | 29% over |
| synchrony_index | 0.91 | 0.68 - 0.84 | 8% over |
| pct_spikes_in_bursts | 90.46 | 84.66 - 93.69 | yes |

**Tuned:** background_weight, excitatory_weight, inhibitory_weight, adaptation_increment,
threshold_spread, intra_in_degree, detection_radius_um. Everything else at module default. The
searched ranges are recorded next to the result.

### What the residual mismatch means
The twin reproduces burst-level statistics and **over-synchronises**. The two features still outside
the band - network burst rate and synchrony - are exactly the two that HD8 found do not transfer
across platforms and HD9 found shift most between species. Three independent routes now point at the
same conclusion: rate and burst-fraction measures are portable and reproducible; synchrony and
network-burst measures are neither.

A model change was made while calibrating: **per-neuron threshold heterogeneity**, because identical
neurons synchronise almost perfectly. Notably the search then chose a near-zero spread (0.016), so
heterogeneity alone does not fix the over-synchrony; candidates for HD13+ are distance-dependent
connectivity, short-term synaptic depression, and correlated background input.

## 2026-09-20: calibration and intervals (PLAN gaps closed, before HD13)

### Done (`4cc93cf`)
- `models/calibration.py`: ECE + reliability diagram from out-of-fold predictions, same grouped CV.
- `models/bootstrap.py`: cluster bootstrap resampling **cultures, not rows** - two conditions of one
  well share a baseline, so row resampling would give intervals that are too narrow.
- Every headline metric now carries `*_ci95` and its resample count: Task 1 balanced accuracy per
  model, per-feature species shift, feature subsets, leave-one-out, and the ECE itself.
- 1000 resamples, 200 in `--quick`, recorded in metrics.json either way.
- `tests/test_metrics_contract.py` fails if any headline metric loses its interval - at the function
  level and against the written metrics.json. 260 tests pass.

### Task 1 calibration (`metrics.mechanism_baselines.task1_within_platform.calibration`)
**ECE = 0.111, 95% CI [0.077, 0.256]** over 73 out-of-fold predictions, 10 bins.
The model is over-confident at the top: the 4 predictions above 0.9 confidence were right 25% of the
time. Figure: `outputs/figures/task1_reliability_real.png`.

### Intervals that change how a result should be read
| metric | point | 95% CI |
|---|---|---|
| Task 1 logistic regression | 0.537 | [0.422, 0.685] |
| Task 1 gradient boosting | 0.398 | [0.276, 0.518] |
| rate_only subset | 0.530 | [0.396, 0.624] |
| all features | 0.514 | [0.398, 0.662] |

**The subset comparison does not survive its intervals.** `rate_only` vs `all` overlap almost
entirely, so "two features are as good as nine" is a suggestion, not a finding. Of the nine
species-shift features, **only `network_burst_rate_per_min` has an interval excluding zero**
(-4.91, CI [-9.79, -0.34]); `synchrony_index` at -4.18 [-5.74, +0.19] does not quite.

### A pipeline bug the end-to-end run exposed
A `--quick` fetch marked the 582 MB ScreenNeuroPharm archive `skipped`, which **erased its files from
MANIFEST.json**; every record built from them then failed the conformance-level-2 hash check and the
data stage aborted. The quick tier now keeps what an earlier `--full` run left on disk and re-hashes
("adopts") files present on disk but missing from the manifest. Provenance heals instead of
degrading. `run_all.py --quick` end to end: **280 s**.

## 2026-09-20: Phase 3, HD13 - drug knobs validated against real pharmacology

### Done (`66e650e`)
- `twin/perturbation.py`: each knob's simulated feature change vs the **real** direction for that
  mechanism, read from the asset (median log2 change of real exposure/baseline pairs, cluster
  bootstrap over cultures). A real direction counts only if its interval excludes zero; otherwise the
  feature is "not established" and is neither credited nor penalised.
- `--stage twin` now runs validation, calibration and the knob table. 270 tests pass, ruff clean.

### Qualitative validation table (`metrics.twin_hd13_perturbation`)
| knob | mechanism | real pairs | scored | agree |
|---|---|---|---|---|
| gabaa_block (gabaa_scale 0.2) | gabaa_antagonist | 507 | 5 | **4** |
| gabaa_potentiate (gabaa_scale 2.0) | gabaa_agonist_or_pam | 17 | 4 | **3** |
| ampa_block (ampa_scale 0.4) | glutamatergic_antagonist | 17 | 4 | **4** |
| sodium_block (threshold_scale 1.6) | na_channel_blocker | 26 | 3 | **3** |

Figure: `outputs/twin/hd13_drug_knob_validation_simulated.png`.

### The one substantive disagreement
Under disinhibition the twin **raises per-electrode firing sharply (+2.87 log2)** while the real
bicuculline/gabazine data show it **slightly falling (-0.34)**. Everything else about the
reorganisation agrees: burst rate, network burst rate and burst participation all move the same way.
So the twin captures the pattern and overstates the rate response - the same over-excitability that
appeared as over-synchrony in HD12. Candidate causes: no short-term depression, no synaptic
saturation, and a background drive that does not adapt.

### Convention added
A knob that silences the network now yields rate-like features of **zero, not undefined**, so the
ratio floors at log2(0.05) = -4.32, exactly as the real ETL already does for an abolished well.
Without it the sodium block scored 0/0, because every simulated feature came back missing.

## 2026-09-20: Phase 3, HD14 - simulation-based inference

### Done (`efb9142`)
- `twin/inference.py`: 5-parameter prior, simulator summarised exactly like a real recording,
  parallel sweep with an on-disk cache (**4342 simulations**, ~0.5 s each on 16 processes), NPE
  training, and posterior checks (rank statistics + prior-to-posterior contraction).
- Summary vector carries `log_rate_ratio` by design, per the HD11 caveat.
- `--stage twin` now also trains the posterior. 280 tests pass, ruff clean.

### Two diagnostics that changed the design
1. **Excitability cliff.** The first prior let `threshold_scale` reach 1.5, where the network is
   silent in 98% of draws; **64% of the first simulation budget was degenerate**. Bound tightened to
   1.15, cached draws outside it reused by rejection rather than rerun. Silent fraction 64% -> 21%
   and every contraction improved.
2. **P-values are not comparable across evaluation sets.** Growing the holdout from 134 to 434 made
   the rank-test p-values look worse for unchanged posteriors, because the test gained power. The KS
   **statistic** is now reported beside the p-value and is what to compare.

### Result (`metrics.twin_hd14_inference`) - the honest part
| parameter | contraction | KS statistic | KS p |
|---|---|---|---|
| threshold_scale | **0.786** | 0.091 | 0.0013 |
| ampa_scale | 0.423 | 0.065 | 0.0482 |
| gabaa_scale | 0.338 | 0.093 | 0.0009 |
| log10_forward_ratio | **0.281** | 0.076 | 0.0131 |
| inter_in_degree | 0.273 | 0.081 | 0.0062 |

Only excitability is well recovered. **The connectivity parameters - including the
forward/backward ratio the twin exists to infer - are weak, and no parameter passes the
rank-uniformity test at 434 holdout points.** A learning curve at 300/600/1200 training simulations
(`outputs/twin/hd14_learning_curve_simulated.json`) shows connectivity contraction still rising with
sample size, but tripling the simulations moved it only 0.28 -> 0.30. More simulations alone will
not fix this.

### What to try in HD15/HD16
- **Better summaries, not more simulations.** The rate cross-correlation is the measure HD11 showed
  is confounded by excitability; candidates are burst-onset lag per compartment, first-spike order
  within network events, and per-event participation ratios.
- Longer simulated recordings (10 s now; real chamber recordings are 480 s).
- If directionality stays unrecoverable, that is itself the Phase 3 result: it would say MEA summary
  features cannot identify chip connectivity at this recording length, which the report should state
  plainly rather than bury.

### Next first task (HD15)
- Simulation-based inference with `sbi`: prior over forward/backward ratio, inter-compartment
  strength and the three drug knobs; simulate, train a posterior estimator, cache to disk.
  **Carry the HD11 caveat into the design**: coupling asymmetry tracks relative firing rate, so the
  posterior must be conditioned on rate-controlled summaries or it will infer direction from
  excitability.

### Superseded plan note (kept for history)
- Brian2 two-compartment network with directional inter-compartment projections. Concrete targets
  from the real data: inter-compartment lag ~5 ms for same-population chambers and ~11 ms for
  different ones; peak coupling 0.96 / 0.91 / 0.82 by device type; and bicuculline-like
  disinhibition should reproduce the rat dose-response (network bursts 0.31x, burst rate 0.57x,
  STTC 1.17x at 300 uM).

## 2026-09-20: Phase 3, HD15 - the posterior meets real recordings

### Done
- `src/neuromps/twin/real_inference.py` + 17 tests: build simulator-shaped summary vectors from the
  real two-chamber recordings (red -> c0, green -> c1), apply the HD14/HD15 posterior, and test
  whether the inferred `gabaa_scale` tracks bicuculline dose. Wired into `run_all.py --stage twin`.
- **Geometry fix.** The inference simulator used a 4x6 grid (24 electrodes per compartment) while the
  real chambers carry 48 with 43-44 active, and HD12 calibrated against 48. Re-simulated 3500 draws
  on the calibrated layout (20 min wall, `outputs/logs/hd15_resimulate.log`) and retrained.
  `check_cache_matches_code` now refuses a cache whose electrode count differs from the current
  layout, so summary names *and* geometry are both guarded.
- Retrained posterior (`metrics.twin_hd14_inference`): every contraction improved slightly and
  `threshold_scale` now passes rank uniformity (KS 0.045, p = 0.46) where it failed in HD14.
  Connectivity stays weak: `log10_forward_ratio` 0.319, `inter_in_degree` 0.305.

### Result (`metrics.twin_hd15_real_inference`) - negative, and the useful part is *why*
*Numbers regenerated 2026-09-20 under the HD16 coupling fix (`a627a8c`); the conclusion is unchanged.*
| check | value |
|---|---|
| real values inside the simulated range | 99.2% (no dimension outside) |
| dimensions with <5% simulated draws in the real IQR | burst rate and active electrodes, both chambers |
| inferred `gabaa_scale` vs dose | rho = 0.083, 95% CI [-0.228, +0.431] over 16 devices |
| inferred `gabaa_scale` for **untreated** recordings | **0.415, CI [0.369, 0.469]** - excludes 1.0 |
| posterior draws landing inside the prior | median 67%, 1.8% of recordings below 1% |

**The twin's posterior does not recover drug-knob direction from real recordings.** The baseline
check says why: drug-free chips infer a GABA-A scale of 0.41, so the knob is absorbing general model
mismatch rather than tracking the compound. A dose trend measured on top of that offset is measured
on a shifted baseline, and there is no trend to measure - see
`outputs/twin/hd15_real_dose_response.png`.

### Three things the first pass got wrong, all fixed in code
1. **Pseudo-replication.** 113 recordings come from 16 devices; the first pass reported rho = -0.197,
   p = 0.036 across recordings as if independent. `dose_trend` now takes `groups` and reports a
   cluster bootstrap over devices. That interval included zero even for the original number.
2. **Range containment is not density containment.** Once the layout matched, every dimension passed
   the 1-99 percentile test - while the simulated electrode counts were bimodal (0 when silent, 48
   when saturated) and almost no simulated draw landed near the real 43-44. `distribution_check` now
   also reports `simulated_support_in_real_iqr` and flags such dimensions as `unsupported`.
3. **A naive p-value on the figure.** The plot printed `p = None` from a key that never existed. The
   caption now quotes the clustered interval and the device count, which is what a reader should see.

### Blocked / open
- `--quick` caps the stage at 30 recordings (74 s standalone) to stay inside the `run_all` budget;
  the full stage is 113 recordings and ~5.5 min. Both record `n_recordings` and `subsampled`.
- **Budget: `run_all.py --quick` now takes 523 s of the 600 s limit** (`outputs/logs/hd15_run_all_quick.log`),
  up from 280 s at HD12. The largest items are the HD11 twin validation (135 s), the source
  license re-checks (111 s of network calls) and posterior training (74 s); HD15 itself adds ~15 s.
  The next stage that needs quick-mode time has to trim one of those first, not just add to the end.
- A `--quick` run overwrites `outputs/` with quick-mode values, so the timing run above was reverted
  with `git checkout -- outputs/` to keep the committed artefacts at full-run values (limitation 23).
- The four unsupported dimensions are the twin's known weak spot (burst structure), the same place
  HD12's residual mismatch sits. Fixing the simulator's burst statistics is the prerequisite for any
  real-data inference claim, not more simulations.

### Next first task (HD16)
- Directionality experiment on simulated data, where ground truth exists: can inference distinguish
  symmetric from asymmetric connectivity, and how many electrodes and how long a recording does it
  need? HD15 says not to attempt the same on real data until the burst statistics match.

## 2026-09-20: Phase 3, HD16 - can features tell direction at all?

### A defect found first, in the twin's own summaries
The twin measured inter-compartment coupling from **every neuron** in a compartment, while the real
ETL measures it from the **per-electrode** spike trains a device detects. That gave the simulator a
cleaner population rate than any MEA can record, and made coupling independent of electrode count -
the one variable HD16 exists to vary. `twin/mea.electrode_coupling` now applies the real pipeline's
input contract; `inference` and `validate` both call it, and the HD14 simulation cache was rebuilt
(3500 draws, 20 min, `outputs/logs/hd16_resimulate.log`).

No existing guard caught this: summary names and electrode count were both unchanged.
`summary_pipeline_fingerprint()` hashes the source of every function that decides what a summary
means, and the cache now refuses to load across a change in any of them.

### The experiment (`metrics.twin_hd16_directionality`)
120 networks per arm, ground-truth wiring known: symmetric (forward ratio 1) vs asymmetric (3, 10).
Electrode count is varied by subsampling an evenly spaced sub-grid of one fixed 6x8 array, and
recording length by truncating one simulation, so neither varies the culture.

**The free-rate arm is confounded, exactly as HD11 predicted.** `all` reaches AUC 0.92-0.99, but
`rate_only` - three firing-rate features - reaches 0.86-0.98 on its own and beats `coupling_only`.
An asymmetric pathway adds excitation to the chamber it targets, so the classes differ in rate
before they differ in anything directional.

**Matching the drive did not fix it, twice.** Matching the rate *ratio* left the absolute rates
shifted, which is just as readable (every cell went to AUC 1.00). Matching both chambers' absolute
rates over a finer grid overcorrected instead: the ratio-3 imbalance flipped from -0.28 to +0.31 and
the worst relative rate gap stayed at 0.21-0.26. Both arms are reported with their gaps
(`matching_quality`) so neither is read as clean.

**Residualising the features against both firing rates is the control that works**, and needs no
intervention in the simulation. The validity check is built in: `rate_only` after residualising
scores 0.41-0.57 and is never better than chance, in either arm.

| result | AUC | 95% CI |
|---|---|---|
| direction from residualised features, 10 s, **6 electrodes** (rate-matched arm) | 0.978 | [0.948, 0.998] |
| direction from residualised features, 10 s, 24 electrodes (free-rate arm) | 0.936 | [0.893, 0.974] |
| validity check: rate features after residualising | 0.41-0.57 | never above chance |

**Direction is recoverable from MEA summaries beyond firing rate, and cheaply - 10 s and 6
electrodes per compartment suffice in simulation.** This is the first clean positive in Phase 3, and
it is a statement about identifiability in the twin's own model, not about real devices: HD15 showed
these same summaries do not transfer to real recordings.

### HD14 and HD16 disagree only in appearance
Regenerating the posterior under the corrected coupling left connectivity as weak as before
(`log10_forward_ratio` contraction 0.295, `inter_in_degree` 0.320, `threshold_scale` 0.816, the only
parameter passing rank uniformity at KS 0.045, p = 0.46). So the features separate two *well-spaced
wiring classes* at AUC 0.98 while the posterior cannot *estimate the ratio* across the prior range.
Both are true: direction is detectable as a contrast and poorly identified as a continuous
parameter, and the report should say exactly that rather than lead with either number alone.

### Blocked / open
- The residualised AUC is unstable across cells in the free-rate arm (0.56-0.97, non-monotonic in
  both duration and electrode count). The cheapest-configuration figure is an order of magnitude,
  not a specification; `cheapest_configuration` therefore requires the CI lower bound to clear the
  target, not the point estimate.
- Residualising removes *linear* rate dependence only.
- Only ratios 3 and 10 against 1 were tested; the weak-asymmetry regime is untested.
- The same simulator generated both classes, so this measures identifiability under a correct model.

### Gate check (Phase 3)
- `run_all.py --stage twin` full: **1956 s** (`outputs/logs/hd16_twin_full.log`), every twin metric
  regenerated under the corrected coupling.
- `run_all.py --quick`: **463 s** of the 600 s limit (`outputs/logs/hd16_run_all_quick.log`), with
  HD16 adding about 26 s. The HD15 measurement of 523 s was inflated by slower license re-checks on
  the network (111 s then, 50 s now), so the budget headroom moves by about a minute between runs
  for reasons outside the code. The HD11 twin validation remains the largest single item at 134 s.
- **Phase 3 gate met.**

### Next first task (Phase 4, HD17)
- Streamlit app: browse the asset with QC flags and provenance; mechanism prediction with conformal
  set; twin sliders. The twin tab should show the HD16 identifiability result and the HD15 negative
  side by side, since together they are the honest story.

## 2026-09-20: Phase 4, HD18 - the technical report (HD17 skipped for now)

### Done
- `report/TECHNICAL_REPORT.md` (11 sections per PLAN): problem, data standard, data asset, methods,
  model results, twin results, limitations, reproducibility, AI-assistance disclosure, sources.
  109 citations, 106 verified against live metrics, 3 pointers to structured fields.
- `report/KAGGLE_WRITEUP.md`: the competition summary, 13 citations, all verified.
- `src/neuromps/report/verify.py` + 20 tests: every number in a report cites the
  `outputs/metrics.json` key that produced it, and the checker re-reads each citation and fails on
  disagreement. This is the enforcement of the "no hand-written numbers" rule.
- `src/neuromps/report/attribution.py`: `CITATIONS.md` generated from `data/raw/MANIFEST.json` and
  `THIRD_PARTY.md` from installed package metadata, with a drift test. `LICENSE` (MIT) added, noting
  that downloaded data keep their own licenses.
- README now indexes the documents and how to check them.

### What the checker caught immediately
- A **stale asset timestamp** in `RESULTS_DRAFT.md` (16:17:51 against the live 17:02:34) - exactly
  the drift the rule exists to prevent.
- **Three citations anchored to the wrong number**, where prose put a different figure closest to
  the comment, and four where a magnitude was written for a negative delta.
- **Two transcription slips of my own** while writing the report (0.3872 for 0.3869).

The checker needed fixing too, each time against a real document: typographic minus signs read as
positive, a number separated from its citation by words skipped *silently*, and structured metric
values crashing it. It now separates verified citations from pointers, so an unchecked claim is
visible rather than assumed fine.

### Open
- **HD17 (Streamlit app) is not built.** The report and writeup say so explicitly rather than
  describing it in the present tense. The app skeleton in `app/` is untouched.
- The demo video (HD19) and the Kaggle submission links (HD20) remain.
- `RESULTS_DRAFT.md` overlaps the technical report and should either be folded in or marked as
  superseded before submission.

### Next first task
- HD17: the Streamlit app, with the twin tab showing HD16's positive and HD15's negative together.

## 2026-09-20: Phase 4, HD17 - the Streamlit demo

### Done
- `app/app.py` plus `app/loaders.py` and one module per tab; `app/README.md` documents how to run it,
  what each tab shows and the measured timings. `?tab=N` opens one tab on its own, which is what
  makes a result linkable and the screenshots reproducible.
- `run_all.py --stage demo` (`src/neuromps/demo/precompute.py`): a 5x5x3 grid over
  forward/backward ratio, GABA-A scale and AMPA scale spanning the prior, each point storing its
  simulated MEA features (through the same `features/mea.py`), a 2 s raster per compartment and
  posterior samples; posterior samples for all 113 real two-chamber recordings; and a mechanism
  classifier with its conformal threshold fitted once. Full run 86.7 s, `--quick` (3x3x2) 18.5 s,
  sizes and timings in `metrics.demo_precompute`.
- 12 `AppTest` smoke tests: every tab renders without raising, the widgets each tab is built around
  exist, the banner is present, and an interaction stays inside the one-second budget.
- Screenshots `outputs/figures/app_tab{1,2,3}.png`, captured from the running app.

### Measured interaction timings (`AppTest`, this machine)
| action | time |
|---|---|
| cold load | 1.9 s |
| warm rerun | 0.65 s |
| twin slider | 0.64-0.85 s |
| real-recording selector | 0.84 s |
| feature slider to new prediction | 0.62 s |
| live Brian2 run (button) | 4.5 s, then cached |

### Four bugs the build found
1. **Brian2 cannot be imported in Streamlit's thread.** It registers a SIGINT handler, and
   `signal.signal` only works in a main thread; a spawned multiprocessing child re-imports the app
   script and hangs. The live run now shells out to `python -m neuromps.demo.simulate_one`.
2. **QC badges counted characters.** `qc.flags` is a semicolon-separated string, so iterating it
   produced "e: 1275, r: 566". Found by reading the first screenshot.
3. **Three delta features had NaN slider bounds**, because `np.percentile` propagates NaN from the
   sources that publish no burst statistics; the slider then had equal ends and Streamlit raised.
4. **Query parameters arrive as a list in `AppTest` and a string in the server**, so the deep link
   worked in neither until both were handled.

### Budget
- `run_all.py --quick` hit **668 s**, over the 600 s limit, and the demo stage was not the cause:
  posterior *evaluation* had grown to 192 s because the reseeded estimator has lower acceptance and
  rejection sampling slowed. Quick mode now evaluates 60 holdout points instead of 350 (44.8 s), and
  records how many it saw.

### Next first task
- Work through `report/REVIEW.md`.

## 2026-09-20: responding to the independent review (`report/REVIEW.md`)

An external reviewer scored the report 6.8/10, confirmed all 115 citations resolved and matched, and
raised 17 findings. I checked the three critical ones against `metrics.json` myself before changing
anything; **all three hold**, and all three were cases of reporting the weaker of two analyses that
were already in `outputs/`.

### Accepted and fixed

1. **F1 (critical) — the directionality headline came from the arm that fails its own check.**
   `metrics.json` defines the criterion: in the rate-matched arm, rate-only features must fall
   towards chance. They run 0.8587-0.9859 and every cell beats chance, so by our own rule that arm
   is uninterpretable — yet the headline quoted its AUC 0.9784 at 6 electrodes while citing the
   *free-rate* arm's validity check. The headline is now the free-rate residualised result, AUC
   0.9359 [0.893, 0.9739] at 24 electrodes, and §7.6 reports the matched arm as inconclusive with
   the evidence that it failed.
2. **F2/F3/F9/F12 (critical) — headline 1 was contradicted by the only platform analysis we have,
   which was never reported.** `task3_platform_contrast` shows `mfr_hz` reversing sign across
   platforms (+0.348 against -0.684) while `synchrony_index` agrees. A new §6.4 reports Task 3
   properly, the headline now says *species* rather than *platforms*, and Figure 4's caption is
   corrected. The zero-valued `active_electrodes` "disagreement" is called out as degenerate.
3. **F5/F6 (major) — §6.6 reported the pooled test the lab notebook had abandoned.** Replaced with
   the per-device-type analysis: lag 11.25 ms for CxHp against 5.0 ms for both same-tissue types,
   Kruskal p = 0.0082, with the internal control behaving (CxCx vs HpHp Holm 0.4785). The pooled
   test is mentioned as deliberately not the headline, with the reason.
4. **F4 (major) — `--quick` does not regenerate the §7.6 numbers.** True: every
   `cheapest_configuration` is `null` at quick settings. §9 now says so. Stage timings are written
   to `metrics.run_all_timings_{quick,full}` by `run_all.py`, so §9.1 is no longer the one uncited
   table.
5. **F7/F8/F10/F11/F13/F14/F16/F17 — figures and over-claims.** `species_shift.png` gained the
   difference panel with the intervals its caption promised (only `network_burst_rate_per_min`
   excludes zero); the HD11 title no longer claims a sign flip the regenerated data does not
   contain; the HD16 figure is four larger panels per arm including the validity-check panel;
   titles that truncated are wrapped. Prose: "beats gradient boosting" removed, the calibration
   caveat extended above 0.8 confidence, `isi_cv` hedged like `synchrony_index`, "never better than
   chance" weakened to what the key licenses, and the §7.2 sweep numbers now carry citations.

### Partially disagreed

- **F15 (naming rule).** The reviewer is right that `hd15_real_dose_response.png` breaks the rule
  Appendix A states, but renaming it would be wrong: it plots *real* recordings under a
  simulation-trained posterior, and calling it `simulated` would misdescribe it in the other
  direction. Appendix A now states the rule as it actually is, and says which file is the hybrid.

### Still open
- The PDF is at **22 pages of the 14-22 range**, so the next addition will fail the CI check.
- CI has still not been observed to pass.

## 2026-09-21: `--quick` gets its own run root

Two problems, found by running `.venv/bin/python run_all.py --quick` on a clean `data/`.

### 1. A quick run crashed on the asset it had just built

`--quick` fetches only the small-tier sources, so the ScreenNeuroPharm two-chamber archive (582 MB,
`--full` only) is absent and the asset has no compartment records. Because the flattened table only
grows a column when some record carries the block, `device.compartment_id` was *missing*, not null,
and four modules indexed it unconditionally. `models/asymmetry.py` raised `KeyError`, ending the run
before the twin and demo stages.

`src/neuromps/asset.py` is now the single reader: `compartment_records(df)` returns the compartment
rows, or an empty **column-complete** frame, so callers keep their own filters and skip instead of
raising. `asymmetry`, `calibrate`, `real_inference`, `demo/precompute` and `qc/data_card` all go
through it, and each logs which `--full` source it is missing. `twin/calibrate.py` gained a real
skip: with zero chamber baselines it would not have crashed, it would have fit against NaN bands and
overwritten `hd12_calibration_simulated.json` with the result.

### 2. A quick run overwrote the full run it cannot reproduce

The deeper problem: `--quick` wrote to `outputs/` and `data/processed/`, so the CI path replaced the
artifacts the report cites (`data_asset.n_records` 21,272 → 288) and left `metrics.json` describing
an asset no longer on disk.

`src/neuromps/paths.py` now resolves every path **at call time** from `NEUROMPS_RUN_MODE`:

| run | results | data asset | manifest | sbi cache |
|---|---|---|---|---|
| `--full`, and a bare `run_all.py` | `outputs/` | `data/processed/` | `data/raw/MANIFEST.json` | `data/interim/` |
| `--quick` | `outputs/quick/` | `data/processed/quick/` | `data/raw/MANIFEST.quick.json` | `data/interim/quick/` |

`run_all.py` is the only caller of `set_run_mode` — a module setting it from its own `--quick` flag
would change where every later import writes, and in a test session would outlive the call. Reads
fall back: a quick stage that needs an artifact it has not produced (the posterior, for
`--stage demo --quick`) takes the full run's copy and logs that it did. The sbi simulation cache is
deliberately **outside** that fallback: 150 quick draws left where a full run looks for 3000 would
train the next posterior on them with nothing in the output saying so.

The app shows the full run and falls back per path to the quick one, naming on the page what fell
back. CI now asserts `git diff --exit-code` after the smoke run, so an import-time path constant
fails the build instead of quietly clobbering a tracked figure.

### Verified

- Full `--quick` from a clean state: **exit 0**, and `git status` byte-identical before and after
  (checked with md5sum over `metrics.json`, `DATA_CARD.md`, `MANIFEST.json`, every tracked figure).
- `pytest`: the 7 report-number and schema failures that the earlier clobbering caused are gone.
  Still failing, both pre-existing and unrelated: 10 `test_app_smoke.py` (circular import between
  `app/app.py` and the `app` package — the demo does not start) and `test_report_build.py`
  (missing LaTeX `caption` package).

### Next
- The app circular import: `streamlit run app/app.py` is broken, which blocks HD19.
- `data/processed/neuromps.parquet` on this machine is a leftover 288-record quick asset sitting in
  the full root. Only a `--full` run replaces it correctly.

## 2026-09-21: `mateus2024` unblocked — step 1, the licence record

### Done
- **Rejection reversed.** Zenodo 14525182 (Mateus & Aguiar, asymmetric-microchannel MEA) was
  excluded on 2026-09-16 for a licence conflict: metadata `cc-by-4.0`, README CC BY-NC-ND. The
  corresponding author confirmed by email on 2026-09-21 that the metadata governs. Verified
  independently of the email: the Zenodo API returns `cc-by-4.0`, open access.
- `src/neuromps/fetch/mateus2024.py` + 13 tests. It blocks if the metadata licence moves, reads the
  README on every fetch, and records its contradicting claim verbatim in the manifest rather than
  preferring the convenient reading. Quick tier fetches the README only; full adds the 109 MB spike
  archive. The 66 MB image archive is deliberately not fetched.
- `data/SOURCES.md` keeps the original rejection struck through with the reversal beside it, and
  states plainly that the resolution rests on correspondence a judge cannot check — with the request
  to the authors to make it public recorded as the open item.
- `h5py==3.16.0` pinned (approved). `CITATIONS.md` and `THIRD_PARTY.md` regenerate to match.

### What the dataset contains (read over HTTP range requests; no bulk download yet)
57 spike-data recordings, 12 experiments, 10 min each, MEA2100-256, hippocampal, DIV 12-21.
Conditions: `Control` 15, `Arrows` 13, `Rams` 11, `Tesla`/`Tesla_v2` 18. Median file 2.3 MB, so
these are spikes, not raw traces. An `Experiments breakdown.xlsx` accompanies them.

### Why it matters
HD16 asked whether MEA features can tell symmetric from asymmetric connectivity and could only
answer in simulation, where we set the ground truth ourselves. This deposit has the geometry
fabricated into the device, so the same question can be asked of real recordings.

### Next first task (step 2)
- Fetch the archive at full tier and map one recording per condition: whether electrodes can be
  assigned to the two sides of the microchannels, and what the Excel breakdown gives per chip.
  Report what is there before writing any ETL.

### Open
- The licence resolution is not yet publicly verifiable. Asked the authors for a line in the Zenodo
  record description.

## 2026-09-21: `mateus2024` step 2 — what is actually in the deposit

### Fetched
109.2 MB `1_MEA_Recordings.zip` at full tier, SHA256 and timestamp in `MANIFEST.json`. The 66 MB
image archive is still not fetched.

### Format
MCS HDF5 (`McsHdf5ProtocolType=RawData`, protocol 3) written by Multi Channel DataManager, but the
payload is a **TimeStampStream**: spike times per channel, not raw traces. Per recording:
`Data/Recording_0/TimeStampStream/Stream_0/TimeStampEntity_<id>`, one `(1, n_spikes)` int64 array of
**microseconds**, with `InfoTimeStamp` mapping entity id to electrode label, unit and exponent.
Layout `MEA_256MEA_MEA2100_252_6`: 252 electrodes on a 16x16 grid, columns A-H and J-R (no I),
rows 1-16. Duration 600.1 s. Reading it needs only `h5py`; the spike dict feeds `features/mea.py`
directly, so the shared feature extractor applies unchanged.

### Design, from the deposit's own methods
Action potentials were detected at 5x SD with a 3 ms dead time. "Forward" (Source-to-Target) and
"backward" (Target-to-Source) propagation along a microchannel is identified from spike times, and a
propagating event must be **detected on all 5 electrodes of the microchannel with <=1 ms between
consecutive pairs** (>= 0.2 m/s). So electrodes sit *inside* the microchannels, five to a channel.

### The obstacle: there is no electrode-to-compartment map in the deposit
The Excel breakdown gives chips per condition per experiment and nothing about geometry, and spike
density is spread evenly across columns, so the compartments cannot be read off the file metadata.
The mapping has to be recovered from the data.

**It is recoverable.** Counting coincidences within 1 ms between grid-adjacent electrodes on one
`Control` and one `Rams` chip, the strong pairs are all **vertical runs in rows 7-10** on columns
F, G, H and N, O, P — microchannels running along columns in a middle band, chambers above and
below, which is what "5 electrodes per channel" predicts. 47 and 74 adjacent pairs respectively
clear 500 coincidences within 1 ms, against a few thousand spikes per electrode.

### A confound to state now, before any analysis
**Condition is confounded with experiment.** From the breakdown: `Control` chips appear only in
Exp9-11, and the asymmetric designs only in Exp1-8 and Exp12. Any Control-against-asymmetric
comparison is therefore also a comparison between culture batches, and no amount of statistics
inside this deposit separates the two. This has to be said wherever the comparison is reported.

### Next first task (step 3)
Implement the deposit's own propagation criterion - ordered runs of 5 in-channel electrodes with
<= 1 ms steps - to count forward and backward events per chip, then ask whether the forward/backward
ratio separates `Control` from `Arrows`/`Rams`/`Tesla`, with firing rate residualised out as in
HD16, and the batch confound reported beside it.

## 2026-09-21: `mateus2024` step 3 — the real-data directionality test

### Built
- `src/neuromps/etl/mateus2024.py`: the deposit's own criterion — an event on all five electrodes
  of a microchannel with <=1 ms between consecutive electrodes — implemented so the geometry falls
  out of the data rather than being assumed. Runs of five electrodes carrying such events *are* the
  microchannels. 17 tests, on trains whose direction is known by construction.
- `src/neuromps/models/microchannels.py`: per-chip aggregation, the condition comparison, and the
  checks that decide whether it means anything. 9 tests. Wired into the models stage, where it skips
  itself if the 109 MB archive has not been fetched (the quick tier).
- `outputs/figures/microchannel_asymmetry_real.png`, one point per chip.

### What the detector found
57 recordings, 54 with detectable microchannels, median 10-15 channels per chip in a row band of
5-12 — microchannels running along columns with a chamber at each end, as the five-electrode
criterion predicts. Two channel datasets in one control recording fail to decompress; they are
skipped and counted rather than losing the recording's other 250 channels.

### The result: no geometry effect at the chip level
| unit | n | control | arrows | rams | tesla | tesla_v2 | Kruskal p |
|---|---|---|---|---|---|---|---|
| recording (wrong) | 54 | 0.339 | 0.609 | 0.854 | 0.675 | 0.590 | 0.251 |
| **chip (correct)** | **26** | **0.617** | **0.595** | **0.875** | **0.744** | **0.607** | **0.640** |

Median |propagation asymmetry|. Per chip, every Holm-adjusted comparison against control is p = 1.0.
A chip recorded at DIV 12, 17 and 21 is one device and three rows; counting the rows is what makes a
geometry effect appear. This is the HD15 lesson a third time, now on a different dataset.

**And the statistic is confounded with sampling**: |asymmetry| falls as propagation events rise,
Spearman -0.494, p = 0.010 over 26 chips — a chip with few events looks one-sided by chance. The
better-sampled half gives Kruskal p = 0.086 on 13 chips, which is not a result either.

### What this does and does not say
It does **not** contradict the deposit's own paper. Mateus et al. had the electrode map, analysed
propagation velocities per channel, and asked a different question; we recovered an approximate
geometry from spike times and compared one aggregate per chip. What it says is narrower: **on this
statistic, with chips as the unit, the asymmetric geometries are not distinguishable from straight
channels** — and the naive version that says otherwise is an artefact of the analysis unit.

Two further limits: control chips appear only in experiments 9-11, so condition is confounded with
batch by the deposit's design; and 26 chips across five groups (3-8 each) would miss a moderate
effect.

### For the twin
HD16 showed direction is recoverable from simulated features once rate is residualised out. This is
the first attempt to put that question to real recordings, and it does not reproduce. Given the
sample and the confounds, that is weak evidence against the simulated result rather than a
refutation of it — and it is the honest counterweight to a simulation-only claim.

### Next
Report, writeup and metrics updates now that the numbers exist.

## 2026-09-21: `mateus2024` step 4 — into the asset, and the standard it needed

### Done
- `src/neuromps/etl/mateus2024_records.py` + 11 tests: two records per recording, one per chamber,
  split at the microchannel band recovered in step 3. Features from `features/mea.py` and coupling
  from `features/coupling.py`, the same functions every other source uses. **108 records** from 54
  recordings (3 had no detectable band and yield nothing rather than an invented split), 25 devices,
  93 usable.
- The asset is now **21,380 records** across five sources, `spike_times` up from 897 to 1,005 - a
  12% increase in the records the nine-feature pipeline can actually be computed on.

### The schema had to be extended, deliberately
The build refused all 108 records on first attempt, which is the standard working. Four things were
missing and each was added rather than worked around:
- `mateus2024` in the source enum.
- `device.microchannel_geometry`, `device.asymmetric_microchannels`, `device.n_in_channel_electrodes`
  — the geometry a directional claim rests on, and the electrodes that belong to neither chamber.
- `qc.flags: unreadable_channels` — a distinct condition (datasets in the deposit that fail to
  decode) that would otherwise have been mislabelled as `noisy_electrodes_present`.
- The required `feature_method` and `provenance` fields. `spike_detection` records that the spikes
  are **the depositors'**, detected before publication, since the deposit carries no raw traces.

### Two silent-change hazards, one real
- **`models/asymmetry.py` took every compartment record with no source filter.** Adding a second
  two-chamber family would have pooled cortical/hippocampal devices with microchannel chips into a
  comparison whose conclusion is about device type. It is now scoped to `screenneuropharm_bic` with
  a test; §6.7's numbers are unchanged (16 devices, p = 0.042, lag p = 0.0082).
- **HD15's real-data inference was safe by luck**: it filters on chamber names (`red`, `green`), and
  the new records use `source`/`target`. A future source naming chambers red/green would have joined
  the analysis silently. A test now pins that scope.

### Also fixed
- `run_all.py` stage timings replaced the whole map, so `--stage models` erased the demo timing the
  report cites. They merge now.
- A bulk number-refresh script of mine rewrote "92.4%" as "0.924%". The citation verifier caught it
  in the same run that introduced it.

### Report
Five sources in §4, the new counts throughout, and §4.2 now states the records are in the asset.
22 pages, 169 citations verified across the three documents.

## 2026-09-21: HD19 — fresh-environment reproducibility

### The check
Cloned the pushed repository into a clean directory, created a new Python 3.11 venv, installed with
`uv pip install -e ".[dev]"`, ran `python run_all.py --quick`, then launched the app. **No errors at
any step.** The app loads in 1.5 s with zero exceptions and shows the full committed results, because
`outputs/` is tracked; where a component comes from the quick run instead, the app says so.

### What it found, and what was fixed
**The quick path took 914 s on a cold clone against a 600 s budget.** The warm tree had hidden it:
downloads were cached and the SBI simulation cache lives in gitignored `data/interim/`. Two fixes,
both measured in the fresh clone:
- **Downloads run on a six-thread pool.** One source publishes 169 separate files and fetching them
  serially dominated the cold run: 349 s to 133 s, same files, same hashes, request order preserved.
- **The quick tier simulates less**: 70 posterior draws rather than 150, one HD11 seed at 12 s
  rather than two at 20 s, 40 posterior samples over 8 recordings in the demo precomputation. None
  of these numbers is ever reported.

**A fully cold clone now runs in 401 s** (fetch 52, data 41, models 11, twin 265, demo 33). Fetch is
the variable part: the same clone measured 52 s and 133 s on two runs.

### A claim that was wrong, now corrected
§9 said `--quick` reproduces every figure and metric except the HD16 sweep. Measured, it produces
**8 of 14 figures** and omits six `metrics.json` entries — `toxcast_mea_inventory`,
`twin_hd12_calibration`, `twin_hd15_real_inference`, `compartment_asymmetry`,
`microchannel_directionality`, and the full-run timings. The quick tier fetches small-tier sources
only, so the asset it builds is 288 records rather than 21,380, and the analyses needing full-tier
data or compartment records skip themselves with a logged reason. The report now says this, and says
that reproducing its numbers takes the full path.

### Still open
- The demo video (below).
- `RESULTS_DRAFT.md` overlaps the report and should be retired or folded in.

## 2026-09-21: the demo video (option B — captioned, no narration)

### Built
- `src/neuromps/video/shots.py` — the shot list, with every quoted number read from
  `outputs/metrics.json` at build time, so a caption cannot disagree with the report.
- `src/neuromps/video/build.py` — renders each shot to a 1920x1080 frame (screenshot or card,
  caption bar, section label) and encodes with ffmpeg. `make video`.
- 9 tests: the running time fits the five-minute limit, every still exists and every crop lies
  inside its image, the plan's five sections are covered, the research-use framing is on screen, and
  building against metrics that lack a cited key raises rather than inventing a value.

### The video
**21 shots, 276 s (4:36), 1920x1080 H.264, 3.4 MB.** Structure follows PLAN's budget: problem 35 s,
data asset and standard 55 s, transfer 60 s, twin and inference 90 s, reproducibility 36 s.

The app shots are real captures of the running app, taken over the DevTools protocol — including a
genuine click on the live-simulation button, so the frame shows **"Simulated 6 s in 3.82 s of wall
clock"** with the raster it produced. Nothing in the video is mocked.

### What it is not
There is no narration, and the captions carry the argument instead. This is the fallback artefact;
when we record option A, the same shot list is the script, and the app should be driven live rather
than shown as stills.

## 2026-09-22 — Dataset expansion review, and the human cohort

**Done.**
- Reviewed six proposed datasets against the report's three named gaps. Four pointers failed
  verification (a 404 DOI, an author with no deposit in Zenodo/DataverseNO/DataCite, an identifier
  belonging to a different group's spike-sorting benchmark, and a search phrase returning
  cardiomyocytes). Two more were rejected on the project's own licence policy: an NC deposit, and
  Wagenaar 2006, whose site grants use on condition of citation but applies no licence. All
  reasons are in `data/SOURCES.md` under "Dataset expansion review (2026-09-21)".
- **Added `trujillo2019`** (Zenodo 4751759, CC BY 4.0): human hPSC cortical organoids on Axion
  Maestro, 23 sessions x 8 organoids = 184 records. Human records 173 -> 357; asset 21,564.
- `fetch/http_random.py`: a block-caching reader over HTTP ranges, so a 1.3 GB MATLAB v7.3 file
  gives up its spike times for ~5 MB of transfer. 229 MB read instead of 37.6 GB for the set.
- `models/baselines.py`: the analysis the new records can support. Adding them changed no modelling
  number, because every model here consumes exposure-minus-baseline pairs and these are unpaired
  baselines. At matched platform and preparation, 7 of 9 features separate human from rat at rest -
  which qualifies the portability claim in S6.5 (that one is about drug-response deltas).
- Report: new S6.8, asset table and counts updated, two stale claims corrected (CI is green and the
  video is recorded; the rat count had been stale since mateus2024 for want of a citation comment).
  22 pages, 172 citations, 0 failing.

**Blocked / not done.**
- The pharmacology gap is *not* closed. Trujillo's three drug sessions carry no well-to-compound
  mapping, so those 24 records are flagged and excluded from pairing rather than labelled.
- Brewer 4679064 (rat hippocampal subregions across instrumented microtunnels, CC BY, 31 MB) is
  verified and downloaded but not integrated: it is the best candidate for the S7.7 geometry gap.
  Note before using it: only 4 of its 40 files carry an internal subregion label and 2 of those 4
  disagree with the filename, so compartments should be identified by electrode coordinates and the
  anatomical names treated as unreliable.

**Then done in the same session.**
- **Added `brewer2021`** (Zenodo 4679064, CC BY 4.0, 17 MB): ten rat hippocampal cultures in a
  four-chamber device whose tunnels are over electrodes. 40 compartment records with measured
  directed coupling to each other chamber. Asset now 21,604 records from seven sources. Anatomical
  labels are carried but not trusted, and every record's QC note says so.
- **`models/tunnels.py`**: the test this source makes possible. S7.2's rate confound had only been
  shown in the twin; across 46 chamber pairs in 9 real cultures, asymmetry has no rank relationship
  with rate imbalance (Spearman -0.0527, CI [-0.599, 0.467]) over a thirty-fold range. Written into
  S7.7 as a companion null, explicitly *not* a refutation: those pairs differ in wiring and rate at
  once, while S7.2 moved only rate.
- Quick tier trimmed again after CI timed out at 900 s: a private repo gets a **2-vCPU** runner, so
  `worker_processes` correctly returns 2 and the twin has half the parallelism assumed earlier.
  One Trujillo session, two Brewer cultures, 30 draws, 3 directionality repeats.

**CI is green** at `1a28703`, both jobs. The quick pipeline takes 1,126 s on the two-vCPU runner a
private repository gets (fetch 73, data 114, models 44, twin 773, demo 123), against 401 s here,
inside the 1500 s budget with about a quarter to spare. Five distinct causes were fixed to get
there: missing LaTeX packages and an unpinned pandoc, an app that hid its dataset card without the
gitignored asset, `app/app.py` shadowing the `app` package under pytest's path order, a process
count that halved a small machine, and gin.g-node.org rate-limiting CI addresses with 403.

**Whole pipeline regenerated from scratch** (`run_all.py --full`, 3,117 s: fetch 20, data 1037,
models 19, twin 1974, demo 66) as a reproducibility check after all of the above. Of 4,445 metric
keys, **zero substantive values changed** - only timestamps and stage timings. The seeded twin
reproduces exactly and the asset build is deterministic, which is the strongest evidence this
repository has that its numbers are not accidents of a particular run.

**Pharmacology gap closed (2026-09-22).** `Logfile_EPAv1.xls` inside the EPA archive turned out to
be the dose-timing table, so `vassallo2017` is integrated: 283 records, 38 same-chip baselines and
245 exposures, six compounds, spike times at 25 kHz, U.S. Public Domain. The modelling table goes
from **73 pairs in six mechanism classes to 754 in nine**. Asset: 21,887 records, eight sources.
New finding in S6.1: the mechanism classifier reaches 0.537 (CI 0.422-0.685, chance 0.167) on the
set it was developed on and 0.333 (CI 0.232-0.420, chance 0.25) on the independent platform - an
interval that contains chance.

**Second review cycle answered by running it (2026-09-22).** Five of the six modelling suggestions
were implemented rather than argued about, and three of those produced results.

- **Axonal ground truth (HD19).** `brewer2021` instruments its microtunnels - 40 of 44 electrodes
  sit inside them, two per tunnel - so the sign of a crossing's delay is the direction it physically
  travelled. The traffic is strongly directed (mean |flow index| 0.363 against 0.0093 for a shifted
  control) and the chamber coupling measure agrees with it on **13 of 31 pairs, a coin flip**, at
  every window from 0.5 to 5 ms. The project's headline negative result now rests on a real-device
  measurement instead of a simulation. Two file defects had to be handled first - 8 of 50 spike
  files store point-reflected coordinates, and the device was mounted two ways across the ten
  cultures - either of which silently inverts the answer.
- **Pre-whitening (S7.2).** Filtering each compartment's burst structure out before correlating
  makes the drive confound *stronger*, not weaker: drive span 0.671 raw against 0.817 pre-whitened,
  beside wiring spans of 0.109 and 0.173. The confound is in the network, not the estimator.
- **Paired inference (HD17).** Training on `p(theta_drug | x_treated - x_baseline, x_baseline)` from
  1,500 paired simulations moves untreated chips from 0.3781 to **1.0289, CI [0.9847, 1.056]** -
  containing the neutral value that S7.5's interval excluded - and contraction from 0.323 to 0.3894.
  The dose trend still does not appear. Baseline heterogeneity was a real defect and was not what
  was hiding it. Needed `NetworkParams.noise_seed` so a simulated pair shares wiring but not noise.
- **Short-term depression (HD18).** Tsodyks-Markram depression moves both calibration-gap features
  the predicted way (synchrony penalty 0.1314 -> 0.1233, burst fraction 0.0211 -> 0.0149) and only
  with slow recovery, but closes about a twentieth of the gap while pushing the three features
  already inside the band back out. It has to enter the calibration search, not be bolted on after
  it: the twin's next version. `depression_u = 0` keeps every earlier result reproducible.
- **Pooling the 754 pairs** stays declined, with a measured reason rather than an opinion: label and
  source are 97.35% the same variable in this asset and six of nine mechanism classes come from a
  single source, so a pooled classifier would be scored on recognising the laboratory.

Report is 22 pages with 216 citations verified. S12 folded into S9 and S7.7 was cut back to make
room; the per-chip microchannel scatter moved to `outputs/figures/`.

**Competition-rules pass and four additions (2026-09-22).** Read the rules properly and found the
submission out of compliance in ways no amount of science would have fixed.

- **Eligibility.** The Writeup now declares its category (End-to-End System) and follows the
  required structure: video, repository link, 274-word summary, report, demo, team. The report
  declares category and team too, since the cross-disciplinary bonus is awarded on what the report
  says. **Two team tables and the public video link are placeholders and must be filled before
  submitting**, and the repository has to be public.
- **The page gate was aimed wrong.** The rules say 15-20 pages *excluding appendices*; CI required
  14-22 of everything, which is why three sessions went into shaving words. `count_main_matter`
  now finds where the appendices start, eleven appendices hold the supporting detail, and main
  matter is 20 pages.
- **Five wrong numbers**, all unguarded by the citation checker: a stale record count, two wrong
  priors, two parameters described as calibrated that were held fixed, and a stale figure count.
  A private email quoted in four places is now paraphrased; the README is held to the checker.

Then four additions, aimed at impact and presentation rather than more validation.

- **The extractor, measured (S3.5).** Kapucu deposits its own meaRtools output for seven of our nine
  features. One reproduces on both ordering and scale; three rank identically on a different scale;
  three burst measures disagree about *which* recordings burst more, "bursts per minute" by a factor
  of two. The standard's justification, measured rather than asserted.
- **No chamber readout recovers axonal direction (S7.7).** All four together reach 0.3854 against
  chance of one half, p = 0.79, at a sample that could have seen 0.6438.
- **The dose signal is not in the features either (S7.5).** Zero of seven testable features move
  monotonically with bicuculline dose, absolute or baseline-subtracted. That moves S7.5's failure off
  simulation-based inference: the twin was inverting a measurement that does not contain the answer.
- **Two things to pick up.** `python -m neuromps.design` turns the S7.6 grid into a lookup, with the
  simulated-upper-bound caveat on every answer; the architecture figure is drawn by a script.

**Explainers, and a tidy-up (2026-09-24).** Nothing in the science moved; this was about making the
work readable and the repository honest about its own state.

- **Three plain-language companions**, written for readers without an ML or modelling background and
  held to the same citation checker as the report: `MODELLING_FOR_BIOLOGISTS.md` (63 citations) on
  the classifiers - deltas rather than states, why splits group by culture, balanced accuracy,
  cluster bootstraps, calibration, conformal sets; `DIGITAL_TWIN_FOR_BIOLOGISTS.md` (61) on the
  simulation - the neuron model in words, what calibration is for a simulation, simulation-based
  inference from scratch, contraction and rank uniformity; and `APP_WALKTHROUGH.md` (10), a guided
  tour of the demo. All three are gitignored with their PDFs, at the author's request.
- **`build_report.sh` takes `--source`, `--title`, `--subtitle`.** A second build script would have
  meant reimplementing the font selection, the metrics-comment stripping and the engine fallback.
  Two things hardcoded to the technical report had to generalise: the header strip and the
  text-layer check. The report builds unchanged at 20 pages of main matter. `make explainers`
  builds all three companions and skips cleanly on a clone that lacks them.
- **App screenshots regenerated.** They predated `vassallo2017` and showed 21,604 records from 7
  sources against the asset's 21,887 from 8, while the writeup and report offer them to reviewers
  who would rather not run the app. Two traps for whoever does this next: a one-shot
  `chrome --headless --screenshot` captures mid-render, because Streamlit paints over a websocket
  and `--virtual-time-budget` does not advance a real socket; and `body.scrollHeight` is always the
  viewport height, because Streamlit scrolls `[data-testid="stMain"]` rather than the document.
- **CI, twice.** Two new build tests shelled out to `build_report.sh` without the pandoc skip guard
  their neighbour already carried - the `test` job does not install pandoc, only `report` does.
  Reproduced by putting an empty directory ahead of pandoc on the PATH, which is the habit that was
  missing both times CI broke this week.
- **Six stale background tasks cleared.** All `pytest` wait-loops that could never finish, because
  `pgrep -f "[p]ytest -q"` matches the loop's own wrapped command line: each was watching itself.
  Same defect as the `pkill` that killed its own shell this session and the 8.5-hour runaway in the
  notebook. Two leftover servers were also stopped - a Streamlit and a headless Chrome left running
  for two and a half and three and a half days, holding 3.8 GB.

**Next first task.** Unchanged, and all four are the author's: fill the team tables in the report and
the Writeup, publish the demo video and paste its public link, make the repository public, and submit
the registration form. Nothing in the code or the documents is waiting on anything.

## 2026-09-25 — Calibrating the twin's synchrony, and what was actually wrong

The session started as "calibrate `synchrony_index` better" and ended by finding that the twin's
culture was ten times too sparse and an earlier calibration had hidden it inside the electrode.

**The diagnosis came from the stored trials, before any new simulation.** Re-scoring §7.3's 40 saved
draws showed the loss was the first problem: `score_features` used a log-ratio for every feature,
which is right for a rate and wrong for a bounded index. Synchrony at 0.964 against a band ending at
0.843 scored 0.134 - less than a rate off by 15% - so the search could buy rate accuracy with
synchrony error, and did. The seven draws that *did* reach the synchrony band all ran
`network_burst_rate_per_min` at 79-185 against a real band of 11.7-22.7, so the twin only looked
synchronised by firing network-wide continuously.

**Four mechanisms were tried. Three did nothing.**

- *Short-term depression* (HD1.8, already run): 0.011 across the whole grid against a 0.118 gap.
- *Per-synapse delay heterogeneity* (new `intra_delay_sd_ms`): 0.005, inside the seed noise floor of
  0.007, and the sign reverses at a smaller electrode. A clean negative for a mechanism that looked
  obvious - the recurrent delay had been one scalar for every synapse.
- *Distance-dependent connectivity* (new `connection_sigma_um`): no effect where drive dominates, and
  the **wrong sign** where recurrence does (0.850 to 0.900 as the range narrows). Concentrating the
  same in-degree locally makes local clusters tighter, not looser.

**What was actually wrong was geometry and density.** The chamber held 250 neurons in 1.92 mm² - 130
neurons/mm² against roughly 1000-3000 in a real culture - and `detection_radius_um` was searched up
to 300 µm against a 200 µm electrode pitch. The oversized electrode compensated twice over: it
pooled about thirty neurons to manufacture a per-electrode firing rate, and it fed each neuron's
spikes to 5.7 electrodes on average, inflating pairwise STTC by construction. Capping the radius at
half the pitch and putting `n_per_compartment` into the search moved them to 92.5 µm and 967 neurons,
and `synchrony_index` landed **inside the real band for the first time** at 0.8387, with
`burst_rate_per_min` too; the other three miss by 6%, 12% and 1% of their band edges.

**The headline score is optimistic and now says so.** Best-of-160 chosen on two seeds scores 0.0756;
re-run on five seeds the search never saw it scores 0.1994, an optimism of 0.1238. Synchrony holds up
better than the total (0.7732 ± 0.0685, in band on 60% of those seeds). `HELD_OUT_SEEDS` and
`summarise_held_out` make that a recorded metric rather than a thing someone checked once.

**Other calibration changes.** Draws are scored on more than one network seed; the search runs in
parallel with refinement in waves; `boundary_hits` reports any value pinned to an edge (the previous
calibration had two and nothing said so; this one has none). A bounded feature at or below zero is
now scored rather than treated as a crashed run - STTC spans -1 to 1, so an asynchronous network had
been scoring identically to a simulation that produced nothing.

**One pattern, four times: a stage trusting an artefact from the stage before it.**

- The sbi simulation cache recorded summaries, geometry and a code fingerprint but **not the
  calibration**, so a recalibration left no trace and 3,500 stale draws would have been reused
  silently. Now keyed on the calibration.
- `real_inference` read that cache without validating it at all - and its whole purpose is asking
  whether real recordings fall inside the simulated range, which is meaningless against a superseded
  twin and looks entirely normal.
- `perturbation` **reported the wrong network**: it recorded its own argument (250) while the
  dict-merge meant the calibrated 967 actually ran. The only one of the four that produced a
  confidently false provenance record rather than a crash or a silent reuse.
- `inference` crashed on the same merge, which is how the other three came to light.

**The inference density gap, closed.** Inference had deliberately run 150 neurons for tractability. I
estimated that closing it would cost ~6x from the neuron count; measured, it is **1.55x** (15.72 s
against 10.16 s per simulation), because most of the cost is fixed overhead. `inference_calibration()`
now includes density and `params_from_vector` defaults to it, so nothing in the pipeline silently
runs a sparser twin than §7.3 reports; an explicit size still wins, for tests and probes.

**Report.** §7.3 rewritten: a table where every simulated value, band edge and penalty carries its own
citation, the held-out optimism in the main text, and a plain statement that the previous version was
wrong. Appendix B updated - it still said "seven", `background_rate_hz` "held at 10 Hz, not searched"
and radius "100-300 µm". §7.4's GABA-A potentiation cell moved 2/4 to 1/4 after the perturbation
re-run. 310 citations, 303 verified, 0 failing; 20 pages of main matter, which needed real trimming
because the report was already at exactly 20.

**A figure was broken in the committed report and no test could see it.** An edit had moved the
species-comparison figure to Appendix E and split its caption ~600 lines: a stranded `![` in the
appendix, an orphaned `](path){width=51%}` in §6.8, both rendering as literal text in the PDF. Both
figure tests match a complete `![alt](path)`, so a *broken* figure was invisible to the checks meant
to find broken figures. Repaired from git rather than rewritten, with a detector that asks whether
another `![` opens before this one closes - a stranded opener otherwise borrows the next figure's
closing bracket and looks complete. Validated by running it against HEAD, where it finds both halves.

**Also fixed:** the citation verifier crashed on an empty collection. `boundary_hits: []` reached
`_trailing_numbers(window, 0)`, whose `found[-0:]` returns *every* number rather than none, and the
strict zip took the whole verification run down.

**HD1.9 retired** at the author's request - its 296 µm vs 100 µm premise was obsolete once the
calibration picked 92.5 µm unprompted. Module, tests, artefacts and metrics key removed; all were
untracked, so no history was lost. Retiring it made Appendix B's claim of "three synaptic mechanisms
tried" unsupported (only depression has a reproducible artefact; the connectivity result lived only
in a scratchpad), so that text now claims only what the repository can regenerate.

### Blocked / open

- **A full downstream chain is mid-run** and its results are not in yet: perturbation, then
  `inference --rebuild` at the calibrated density, then `real_inference` and `delta_inference`
  (~2 hours; delta alone is 1,500 pairs = 3,000 simulations). Until it lands, `metrics.json` carries
  HD1.4/1.5/1.7 numbers from the old calibration and **§7.5 and its appendices are stale**. Expect
  citation failures there of the same kind §7.4 just had; the checker will name them.
- **`DIGITAL_TWIN_FOR_BIOLOGISTS.md` fails the citation check.** Its §3.2 and §3.3 describe the old
  seven-parameter search and tabulate synchrony as an unfixed gap. It needs the §7.3 treatment, not a
  numeral swap, and it also covers HD1.3/1.4 - so it is best done once, after the chain lands.
- **The citation checker still has a blind spot.** A count spelled as a word ("seven") is not a
  number, and a dict like `inside_real_band` returns *pointer*, so prose naming which features passed
  is never verified. That is exactly how §7.3's claim stayed wrong through many passing runs. The
  malformed-figure equivalent is now closed; this one is not.
- **A `pkill -f` killed a sibling job** whose wrapper command line contained the pattern. Bracket
  escaping stops *grep* matching itself and does nothing about this. The PID-based wait written ten
  minutes earlier is the habit that works.
- The email to Ort, Koch and Delev about the Zenodo 21283609 pickles is drafted and unsent; there is
  no mail transport on this machine.

**Next first task.** When the chain lands: read the citation checker's failures for §7.5 and fix
them, then regenerate `DIGITAL_TWIN_FOR_BIOLOGISTS.md`. The author's four are unchanged - team
tables, demo video link, make the repository public, submit the registration form.

### Session close, same day: the chain landed and the documents were reconciled

The downstream re-run at the calibrated density finished. `real_inference` failed the first time —
sbi aborted rejection sampling on low acceptance in the *sensitivity* variant, after the headline
inference had already succeeded — so that step now records why it could not be sampled instead of
taking the stage down with it. On the re-run it sampled fine, which makes the original failure a
non-deterministic timeout and the fallback worth having either way.

**Closing the inference density gap paid off.** Silent simulations fell from a fifth to 0, and three
of five parameters now pass the rank-uniformity check where none did. `gabaa_scale` still fails, and
it is the parameter §7.5 and §7.7 depend on, so that is stated where it bites.

**One conclusion changed, not just its numbers.** The paired estimator no longer recovers neutral:
0.9108 on [0.8583, 0.966], against 0.5219 for the absolute one. The report had said "the offset goes
away... contains the neutral value". It now says most of it goes and not all. The citation checker
flagged the numbers; the word "recovers" it could not see, which is the blind spot again.

**Two more frozen values found.** HD1.5's `caveat` was a hardcoded sentence still claiming
"contraction 0.34 and no parameter passing" - now derived from the live metrics. Appendix K's table
had drifted in every row and is now generated from them, with KS p-values and a pass/fail column.

**A third document for reviewers.** `TECH_REPORT_FOR_BIOLOGISTS.md` (19 pages, 134 citations) covers
the whole project for wet-lab reviewers: the data asset and the extractor-agreement measurement
upfront, all results, and the negative results collected in one section. Schema mechanics and
pipeline automation are in its appendices. `DIGITAL_TWIN_FOR_BIOLOGISTS.md` is **retired** - the new
document covers the twin in its §13-§17, and the old one described a seven-parameter search that no
longer exists. References in `MODELLING_FOR_BIOLOGISTS.md` were repointed before deleting it.

**Two build-script bugs, both found by reading the output rather than the tests.** A document whose
opening paragraph ran long had its disclaimer split by a fixed `1,10` strip window, so the PDF opened
on a sentence fragment; the bound is now the first heading. And the text-layer check searched for a
contiguous title, so any title long enough to wrap reported itself missing. Both have regression
tests, and the first was verified to fail against the old rule.

All nine documents now verify with 0 failing citations. The technical report is back inside the page
gate at 20 pages of main matter, which needed a historical paragraph moved into an appendix.

**Next first task.** Commit this work onto `twin-synchrony-calibration` and update
[the open PR](https://github.com/Vodolay/AI4S/pull/1), whose description predates the whole
downstream re-run. Then the author's four, unchanged: team tables, demo video link, make the
repository public, submit the registration form.

## 2026-10-07: Phase 4, HD18 (branch `word-report-and-demo-refresh`)

### Done

**The report builds in Word, and the PDF is a rendering of it.** A reviewer's Word export read
better than the LaTeX build - filled table headers, striped rows - but it had been edited by hand
and had drifted: §5 had lost its three closing paragraphs with the whole cross-laboratory test,
Appendix I had lost four of the nine searched parameters, Appendix D&E had lost the cache-meaning
invariant, and Appendix M quoted the dose series at 30 µM, which is neither its endpoint nor its
maximum. `neuromps.report.docx` keeps the design and throws away the fork: it renders the same
Markdown `report.verify` checks, into `report/reference.docx` (that export with its body emptied),
and `report/build_docx.sh` converts with LibreOffice. 22 pages, 16 of main matter, all 11 figures.
`make report` is now the Word path; the LaTeX one is `make report-latex` and writes to its own file,
because both used to target the same PDF.

**Three defects in that renderer, two of which failed silently.** `tblPr` has a mandatory element
order, and an out-of-sequence `tblBorders` made LibreOffice discard every computed column width - the
tables looked untouched through two builds. The template's `List Bullet` style carries no numbering,
so style-only bullets rendered as bare indents. And a citation ending a line was treated as a
multi-line comment, deleting the line break and welding the next line on ("0.48at 100 µM"). All
three have tests; the last was swept for across the whole rebuilt PDF.

**Timings and pinned versions have a metrics key.** They were the only numbers in a document that
promises every number has one that nothing could check. `run_all.py` now writes `harness_quick` /
`harness_full` into the canonical metrics root - the one deliberate exception to the quick run's
isolation, fenced by four tests including one that a quick run touches no other key, and one that a
partial `--stage` run cannot shrink the end-to-end figure. Appendix J gained the pooled Kruskal p
(0.63969, not the 0.6400 the fork carried) and the dropout event counts; Appendix M now quotes the
whole dose series rather than stopping at its rising midpoint.

**Three model diagrams, drawn from the metrics.** `neuromps.report.diagrams` emits the classifier
(paired wells → nine log-ratios → logistic regression → six classes, with the culture-grouped folds
drawn), the twin (two chambers, axon diode, electrode catchments against the pitch) and the
inversion (prior → twin → summaries → posterior estimator, with the rank-uniformity gate). Six tests
hold them to the metrics, including one that changes a value and asserts the picture changes.

**The app leads with the models.** Tabs are now Mechanism, Twin, Data asset; the data tab's schema
walkthrough and provenance narrative moved behind an expander. Deep links are slugs (`?tab=twin`)
because the numbers moved. A frozen caption was found and fixed: the posterior panel still claimed
"only `threshold_scale` is well identified (contraction 0.8)", and the density correction had made
both halves of that wrong.

**The video is recut and its stills regenerate.** 20 shots, 4:45. Data is down from 55 s to 26 s, the
Tukker/Mateus licence card is gone, and the three diagrams carry the middle. `neuromps.video.stills`
drives headless Chrome instead of the hand-capture that let the old stills survive a density fix, a
cross-laboratory result and an inference re-run while showing numbers the report had stopped making.
It warms up, retries, and checks how far down each page actually painted - which caught a blank
skeleton frame and a page that had rendered one column. The live Brian2 frame is reachable because
`?live=1` runs it on load, gated on a server-side opt-in so that following that link to a deployed
app still costs a visitor nothing.

### Not done
- CI gates the LaTeX build (`.github/workflows/ci.yml`), not the Word one that is now the
  deliverable. It passes at 20 main-matter pages, but switching it over means installing LibreOffice
  on the runner - a real cost on every push, and not a decision to make quietly.
- The `--quick` wall clock is not stable on the development machine: three runs of identical code
  gave 401.0 s, 544.7 s and 613.3 s, and the slowest crossed the ten minutes CLAUDE.md promises.
  Appendix K says so rather than quoting the best of the three. The twin stage is 63% of the
  runtime, so its `--quick` draw count is the lever if the promise is to be defended instead.

Merged as [#4](https://github.com/Vodolay/AI4S/pull/4). CI caught three defects the local suite did
not, two of them because the local environment was lying: `import run_all` resolved only under
`python -m pytest`, and `outputs/quick/metrics.json` still carried a calibration key from a run made
when the full data was present, so the local quick tier was not a quick tier. The third was a write
across the quick/full boundary that the smoke job's `git diff --exit-code` refused.

`report/OLD_TECHNICAL_REPORT.pdf` deleted; PLAN.md's video budget updated to the one that was built.

**Next first task.** The author's four, unchanged: team tables, demo video link, make the repository
public, submit the registration form.
