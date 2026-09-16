# Data sources

Machine-readable provenance (URL, license, SHA256, size, download time per file) lives in
`data/raw/MANIFEST.json`, written by `python -m neuromps.fetch.runner [--full]`.
This file records the human decisions behind each source.

| Source key | What | License | Access | Status (2026-09-16) |
|---|---|---|---|---|
| `kapucu2022` | Kapucu et al. 2022 *Sci Data* 9:120: hPSC-derived vs rat cortical networks on 48-well MEA; development (DIV 2–77) + one pharmacology plate per species | CC-BY-4.0 (data, `Data/LICENSE.txt`); article CC-BY-4.0 | G-Node GIN, plain HTTPS, pinned commit `0f0f135` | arrived (quick tier) |
| `toxcast_mea` | US EPA ToxCast invitrodb v4.3, `CCTE_Shafer_MEA_acute_*` (rat cortical, acute) and `CCTE_Shafer_MEA_dev_*` (network formation) | CC0-1.0 (figshare record 6062623 v14) | EPA Clowder API, anonymous | arrived (quick tier) |
| `iras_mea` | Tukker et al. 2020 *Toxicol Sci* 178:71–87: hiPSC co-cultures vs rat cortical MEA, seizure-liability compounds; supplementary docx | **CC-BY-NC-4.0** | Europe PMC REST | arrived; optional |
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
  member's CRC32 is checked by `zipfile`. Filtering to MEA AEIDs (2454–2541) happens in ETL; the raw
  file is not modified. mc5-6 holds curve fits and hit calls, not per-concentration responses.
- **IRAS / Westerink (optional).** Article and supplement are CC BY-NC 4.0: non-commercial research
  reuse is permitted. **Needs PI sign-off** before any derived table is redistributed, given the
  competition has cash prizes. Vassallo et al. 2017 was not found as open access and is excluded.
- **Axon-diode lineage.** Peyrin et al. 2011 *Lab Chip* (doi:10.1039/c1lc20014c) and Courte et al. 2018
  *Methods Cell Biol* (doi:10.1016/bs.mcb.2018.07.002) are not open access: cited only, never
  downloaded. Design parameters are taken from the open-access papers.
- **Europe PMC supplementary zips** are assembled on request; their SHA256 may change between downloads
  while contents are identical. If a hash changes, compare member lists before assuming upstream edits.
