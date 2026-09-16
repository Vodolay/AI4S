# Data sources

Machine-readable provenance (URL, license, SHA256, size, download time per file) lives in
`data/raw/MANIFEST.json`, written by `python -m neuromps.fetch.runner [--full]`.
This file records the human decisions behind each source.

| Source key | What | License | Access | Status (2026-09-16) |
|---|---|---|---|---|
| `kapucu2022` | Kapucu et al. 2022 *Sci Data* 9:120: hPSC-derived vs rat cortical networks on 48-well MEA; development (DIV 2–77) + one pharmacology plate per species | CC-BY-4.0 (data, `Data/LICENSE.txt`); article CC-BY-4.0 | G-Node GIN, plain HTTPS, pinned commit `0f0f135` | arrived (full tier: 239 files, 1.2 GB) |
| `toxcast_mea` | US EPA ToxCast invitrodb v4.3, `CCTE_Shafer_MEA_acute_*` (rat cortical, acute) and `CCTE_Shafer_MEA_dev_*` (network formation) | CC0-1.0 (figshare record 6062623 v14) | EPA Clowder API, anonymous | arrived (full tier: 42 files, 2.57 GB) |
| `iras_mea` | Tukker et al. 2020 *Toxicol Sci* 178:71–87: hiPSC co-cultures vs rat cortical MEA, seizure-liability compounds | **CC-BY-NC-4.0** | Europe PMC REST | **excluded** (license); recorded as `skipped` in MANIFEST |
| `sharf2022` | Sharf et al.: human iPSC brain organoid slices, MaxWell MaxOne HD-MEA (26,400 electrodes); **diazepam dose–response** control/3/10/30/50 µM in 4 organoids; Kilosort2 spike output | CC0-1.0 | Zenodo record 6578989 | arrived (full: kilosort2.zip) |
| `fofie2025` | Fofie et al. 2025: hiPSC-derived **nociceptors** (RealDRG), Axion Maestro Pro 48/96-well; concentration–response for TTX, huwentoxin-IV, lidocaine, retigabine, DNQX, GABA and others | CC0-1.0 | figshare article 28755155 | arrived (1.25 MB xlsx) |
| `screenneuropharm_bic` | ScreenNeuroPharm: rat cortical + hippocampal cultures, 60-electrode MCS MEA, **bicuculline** concentration series; spike trains | **MIT** (source repo) | Zenodo record 7665283 | arrived (full tier only) |
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
  redistribution with attribution: CC0, CC BY, public domain, or permissive OSI-approved licenses
  (MIT, BSD, Apache-2.0) where a repository applies one to data. Exclude NC, ND and unlicensed
  material even if it is publicly downloadable. Where repository metadata and the dataset's own
  license file disagree, take the more restrictive one and record the conflict. Record every
  exclusion in `excluded.py` and here. Each fetcher re-checks its source's license at fetch time and
  blocks the source if it changed upstream.
- **ToxCast MEA inventory (HD2, `python -m neuromps.etl.toxcast_mea`).** 36 endpoints: 17 acute
  (508 chemicals, 532 samples/endpoint) and 19 dev (463 chemicals). Same platform as Kapucu 2022
  (Axion Maestro, 48-well, 16 electrodes, 12.5 kHz), but the acute assay is a 40-min baseline vs
  40-min exposure comparison and the dev assay is a 12-day continuous exposure summarized as an AUC
  over DIV 5/7/9/12. Numbers in `outputs/metrics.json`, per-endpoint table in
  `outputs/tables/toxcast_mea_endpoints.csv`.
- **OQ-1 (resolved): ToxCast effect direction is per assay.** acute `activity_change = -top`,
  dev `= +top`; positive means the measured quantity rose. Verified against cytotoxicity anchors and
  re-checked on every ETL run (`check_direction_convention`, agreement acute 92.4%, dev 99.1%).
  Details in `schema/FEATURE_SET.md`.
- **No compound overlap between the two datasets.** None of kainic acid, CNQX, D-AP5, GABA, gabazine
  or TTX is among the ToxCast acute chemicals (TTX is its positive control). Cross-platform
  comparison must therefore be at the level of feature response patterns, not per-compound agreement.
- **New sources added 2026-09-16 (PI decision), after the Tukker exclusion left a gap.**
  - `sharf2022` supplies the human dose–response arm. Caveats: brain organoid slices, not 2D culture;
    one organoid per array (no per-well replicates); HD-CMOS electrode geometry is not comparable to
    the 16-electrode Axion wells, so electrode counts and synchrony measures need separate treatment.
    Only `kilosort2.zip` (spike-sorted output) is fetched; the 36 raw .h5 files (~60 GB) are not.
  - `fofie2025` gives the widest compound overlap (TTX, GABA, lidocaine, DNQX, retigabine) at 1.25 MB.
    Caveats: **sensory neurons, not cortical**, so it supports mechanism-level comparison only; and the
    workbook holds **total spike counts per electrode/well only**, so firing-rate features can be
    derived but burst and network features cannot.
  - `screenneuropharm_bic` gives a rat GABA-A antagonist concentration series to pair with Kapucu's
    single gabazine condition. Licensed MIT in the source repository (Zenodo says only "other-open");
    MIT on data is unusual but permissive and allows commercial reuse. Fetched with `--full` only.
- **Verified in the downloaded archives (2026-09-16).**
  - `sharf2022/kilosort2.zip`: 46 entries; drug arm has **19 files, not 20** — organoid 2957 has no
    30 µM condition (control/3/10/50 µM only). Treat the diazepam series as complete for organoids
    2950, 2953, 2954 and incomplete for 2957.
  - `screenneuropharm_bic`: spike trains (`ptrain_*`) under `Data/<prep>/<Cx|Hp>/<array>/...`, for
    cortical **and** hippocampal cultures, at bicuculline 1, 3, 10, 30, 100, 300 µM (plus a `BIC500`
    label to check during ETL). A 6-point dose-response, far richer than Kapucu's single gabazine
    condition. The archive also ships the authors' MATLAB/Python analysis code (MIT).
- **Rejected in the same search (recorded for the report's data-selection story).**
  - Zenodo 14525182 (Mateus & Aguiar), compartmentalized asymmetric-microchannel MEA — the closest
    published analogue of our two-compartment chip: Zenodo metadata says CC BY 4.0 but the dataset's
    own README says CC BY-NC-ND. Excluded on the conflict rule above.
  - Carstens et al. 2025 (3D hiPSC BrainSphere HD-MEA) and a GABA-A seizure-liability study: data
    "available upon reasonable request" — gated, so out of scope.
  - Zenodo 14363732 (36 GB MEA recordings) and the IMI2-NeuroDeRisk comparison: CC BY-NC / CC BY-NC-ND.
  - DANDI 001872 and 001336: human organoid MEA in NWB under CC-BY, but no pharmacology.
- **Axon-diode lineage.** Peyrin et al. 2011 *Lab Chip* (doi:10.1039/c1lc20014c) and Courte et al. 2018
  *Methods Cell Biol* (doi:10.1016/bs.mcb.2018.07.002) are not open access: cited only, never
  downloaded. Design parameters are taken from the open-access papers.
- **Kapucu 2022 code** in the GIN `Codes/` folder is BSD-3 / GPL-3.0 (meaRtools, SpikeDetection-Toolbox).
  We do not fetch or vendor it. If any of it is ever used, check GPL obligations against our MIT license first.
