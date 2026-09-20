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
| check | value |
|---|---|
| real values inside the simulated range | 99% (no dimension outside) |
| dimensions with <5% simulated draws in the real IQR | burst rate and active electrodes, both chambers |
| inferred `gabaa_scale` vs dose | rho = 0.004, 95% CI [-0.290, +0.337] over 16 devices |
| inferred `gabaa_scale` for **untreated** recordings | **0.425, CI [0.360, 0.498]** - excludes 1.0 |
| posterior draws landing inside the prior | median 53%, 3.5% of recordings below 1% |

**The twin's posterior does not recover drug-knob direction from real recordings.** The baseline
check says why: drug-free chips infer a GABA-A scale of 0.42, so the knob is absorbing general model
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
- `--quick` caps the stage at 30 recordings (74 s) to stay inside the `run_all` budget; the full
  stage is 113 recordings and ~5.5 min. Both record `n_recordings` and `subsampled`.
- The four unsupported dimensions are the twin's known weak spot (burst structure), the same place
  HD12's residual mismatch sits. Fixing the simulator's burst statistics is the prerequisite for any
  real-data inference claim, not more simulations.

### Next first task (HD16)
- Directionality experiment on simulated data, where ground truth exists: can inference distinguish
  symmetric from asymmetric connectivity, and how many electrodes and how long a recording does it
  need? HD15 says not to attempt the same on real data until the burst statistics match.
