# Data sources

Machine-readable provenance (URL, license, SHA256, size, download time per file) lives in
`data/raw/MANIFEST.json`, written by `python -m neuromps.fetch.runner [--full]`.
This file records the human decisions behind each source.

| Source key | What | License | Access | Status (2026-09-16) |
|---|---|---|---|---|
| `kapucu2022` | Kapucu et al. 2022 *Sci Data* 9:120: hPSC-derived vs rat cortical networks on 48-well MEA; development (DIV 2–77) + one pharmacology plate per species | CC-BY-4.0 (data, `Data/LICENSE.txt`); article CC-BY-4.0 | G-Node GIN, plain HTTPS, pinned commit `0f0f135` | arrived (full tier: 239 files, 1.2 GB) |
| `toxcast_mea` | US EPA ToxCast invitrodb v4.3, `CCTE_Shafer_MEA_acute_*` (rat cortical, acute) and `CCTE_Shafer_MEA_dev_*` (network formation) | CC0-1.0 (figshare record 6062623 v14) | EPA Clowder API, anonymous | arrived (full tier: 42 files, 2.57 GB) |
| `iras_mea` | Tukker et al. 2020 *Toxicol Sci* 178:71–87: hiPSC co-cultures vs rat cortical MEA, seizure-liability compounds | **CC-BY-NC-4.0** | Europe PMC REST | **excluded** (license); recorded as `skipped` in MANIFEST |
| `axon_diode` | Asymmetric-microchannel chip papers (Renault 2015, Lassus 2018, Courte 2020): design parameters only | CC-BY-4.0 each (read from article XML) | Europe PMC REST | arrived |

## Decisions

- **Kapucu 2022, raw vs preprocessed.** The raw 12.5 kHz `.h5` recordings total ~2.4 TB (75 files, up to
  120 GB each). We use the authors' spike-time CSVs (`Channel,Time`) and meaRtools per-well outputs instead.
  Raw files are never fetched. Consequence: spike detection is the authors' (MATLAB code in the GIN `Codes/` folder + noise-electrode
  lists), not ours; this must be stated as a limitation.
  - quick tier (~191 MB): both pharmacology plates (Baseline / Pharma / TTX), all meaRtools feature
    CSVs, all expLogs and noisy-electrode lists, license, PCA table.
  - full tier (+~1.0 GB): development-series and PCA-plate spike-time CSVs.
- **Kapucu pharmacology design.** Single concentrations, not dose–response: kainic acid 5 µM, CNQX 50 µM,
  D-AP5 50 µM, GABA 10 µM, gabazine 30 µM, plus Control wells, then TTX 1 µM (from the expLogs). This
  limits the mechanism classifier to 5 compound classes × 2 species. Plan accordingly in Phase 2.
  ETL gotchas seen in the Pharma expLogs: `D-AP5` and `DAP5` both occur (normalize); dose sits in the
  `Dose` column as `50uM` with `Units` empty; `ExcludedWell` rows must be dropped.
- **ToxCast.** No per-assay download exists. The quick tier fetches assay annotations, cytotoxicity and
  target-mapping tables, the release note and the 36 MEA assay description PDFs (all that Clowder lists).
  The full tier extracts only `mc5-6_winning_model_fits-flags_invitrodbv4_3_AUG2024.csv` (2.55 GB,
  ~661 MB compressed transfer) from the 7.5 GB `INVITRODB_SUMMARY.zip` via HTTP range requests; the
  member's CRC32 is checked by `zipfile`; extracted size matches the zip listing (2,550,615,005 bytes).
  mc5-6 holds curve fits and hit calls, not per-concentration responses. The raw file is not modified.
  - **Filter MEA rows by name (`aenm` starts with `CCTE_Shafer_MEA_`), never by AEID range.** AEIDs
    2454–2541 also contain 11 unrelated assays (CCTE_Deisenroth_AIME_*, CCTE_GLTED_hDIO2/3). By name:
    36 endpoints (17 acute, 19 dev), 20,375 rows, 808 chemicals, out of 3,527,285 rows in total.
  - Clowder quirks (handled in `remote_zip.py`, with tests): suffix ranges return HTTP 500, and if
    gzip is offered Clowder ignores the range and streams the whole file gzipped. Ranges near the end of
    the 7.5 GB zip take minutes to start, so the full extraction took ~15 min.
- **IRAS / Westerink: excluded (PI decision, 2026-09-16).** Tukker et al. 2020 was fetchable and its
  license was verified from the article XML as CC BY-NC 4.0. We dropped it from the data asset anyway:
  - The submission competes for cash prizes and is released under an open license (MIT). A
    non-commercial term on an input dataset cannot be carried through to that license.
  - The competition rules (Kaggle Rules tab, read 2026-09-16) do **not** keep submissions under the
    entrant's own license. Ownership of submitted materials is held jointly by the team and the
    Organizing Committee, and third-party datasets keep their original licensing. Kaggle's general
    rules also require any open-source code used in the model to carry an OSI-approved license that
    does not limit commercial use. The revisit condition (rules clearly keep submissions under our
    own open license) is therefore **not met**.
  - Implementation: the fetcher module was removed; `src/neuromps/fetch/excluded.py` writes a
    `status="skipped"` record (no files, reason in `note`) on every fetch run, so the exclusion is
    part of the provenance record, not just a missing entry. Local copies were deleted.
  - Cost: we lose an independent hiPSC-vs-rat MEA pharmacology set with dose–response. The
    rat→hPSC transfer study relies on Kapucu 2022 alone, which must be stated as a limitation.
  - Vassallo et al. 2017 (rat cortical MEA, same lab) was not found as open access and was never fetched.
- **License policy (applies to every source).** Accept only licenses that allow commercial reuse and
  redistribution with attribution: CC0, CC BY, public domain, or OSI-approved software licenses. Exclude
  NC, ND and unlicensed material even if it is publicly downloadable. Record every exclusion in
  `excluded.py` and here.
- **ToxCast MEA inventory (HD2, `python -m neuromps.etl.toxcast_mea`).** 36 endpoints: 17 acute
  (508 chemicals, 532 samples/endpoint) and 19 dev (463 chemicals). Same platform as Kapucu 2022
  (Axion Maestro, 48-well, 16 electrodes, 12.5 kHz), but the acute assay is a 40-min baseline vs
  40-min exposure comparison and the dev assay is a 12-day continuous exposure summarized as an AUC
  over DIV 5/7/9/12. Numbers in `outputs/metrics.json`, per-endpoint table in
  `outputs/tables/toxcast_mea_endpoints.csv`.
- **OQ-1: ToxCast effect direction is unresolved.** tcpl normalizes each endpoint in its own fitting
  direction, so the sign of `top` cannot be interpreted globally; a single global flip inverts the
  cytotoxicity endpoints (LDH must rise when cells die). The ETL reports raw `top` plus hit counts by
  sign. Resolve per-AEID from the EPA assay annotations before any claim about the direction of a
  chemical's effect. See `schema/FEATURE_SET.md`.
- **No compound overlap between the two datasets.** None of kainic acid, CNQX, D-AP5, GABA, gabazine
  or TTX is among the ToxCast acute chemicals (TTX is its positive control). Cross-platform
  comparison must therefore be at the level of feature response patterns, not per-compound agreement.
- **Axon-diode lineage.** Peyrin et al. 2011 *Lab Chip* (doi:10.1039/c1lc20014c) and Courte et al. 2018
  *Methods Cell Biol* (doi:10.1016/bs.mcb.2018.07.002) are not open access: cited only, never
  downloaded. Design parameters are taken from the open-access papers.
- **Kapucu 2022 code** in the GIN `Codes/` folder is BSD-3 / GPL-3.0 (meaRtools, SpikeDetection-Toolbox).
  We do not fetch or vendor it. If any of it is ever used, check GPL obligations against our MIT license first.
