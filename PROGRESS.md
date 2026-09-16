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

### Next first task (HD2)
- Filter ToxCast mc5-6 by `aenm` prefix `CCTE_Shafer_MEA_` (not the AEID range, which includes 11 non-MEA
  assays): 36 endpoints, 808 chemicals. Read the assay PDFs to pick the harmonized feature set.
