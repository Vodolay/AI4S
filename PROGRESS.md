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
