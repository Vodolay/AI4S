# PROGRESS

## 2026-09-16: Phase 0, HD1 (branch `phase0-setup`)

### Done
- Repo scaffold per PLAN.md; `pyproject.toml` pinned (Python 3.11 venv via `uv`; numpy held at 2.3.5
  because brian2 2.9.0 does not import with numpy 2.4).
- CI workflow `.github/workflows/ci.yml`: ruff, pytest, `run_all.py --quick` smoke (CPU-only torch).
  Not yet exercised on GitHub (no remote configured).
- `src/neuromps/fetch/`: one module per source (`kapucu2022`, `toxcast_mea`, `iras_mea`, `axon_diode`),
  shared downloader with hash cache, remote-zip member extraction, Europe PMC license check,
  `data/raw/MANIFEST.json` writer. 17 tests passing, ruff clean.
- Quick fetch ran: all 4 sources `ok` (216 files, ~210 MB). Rerun re-downloads nothing and hashes are
  stable. Log: `outputs/logs/fetch_quick.log`.
- `data/SOURCES.md` records the raw-vs-preprocessed decision (spike-time CSVs, not the 2.4 TB raw h5).
- **Gate (≥2 sources with real files and licenses recorded): met.**

### Blocked / needs PI decision
- `iras_mea` is CC BY-NC 4.0. Keep, or drop to avoid non-commercial restrictions?
- Kapucu pharmacology is single-concentration with 5 compounds; the mechanism classifier will be small.
  ToxCast mc5-6 gives scale, but only as curve fits / hit calls.

### Next first task (HD2)
- Check `outputs/logs/fetch_full.log` result; filter ToxCast mc5-6 to MEA AEIDs 2454–2541 and read the
  assay PDFs to pick the harmonized feature set.
