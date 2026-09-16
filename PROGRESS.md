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

### Blocked / open
- Dataset search for a CC-BY/CC0 hiPSC MEA pharmacology set was running at session end.

### Next first task (HD3)
- Write `schema/neuromps_schema.json` + `STANDARD.md` from `schema/FEATURE_SET.md`.
