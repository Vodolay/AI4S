# Data sources

Machine-readable provenance (URL, license, SHA256, size, download time per file) lives in
`data/raw/MANIFEST.json`, written by `python -m neuromps.fetch.runner [--full]`.
This file records the human decisions behind each source.

| Source key | What | License | Access | Status (2026-09-16) |
|---|---|---|---|---|
| `kapucu2022` | Kapucu et al. 2022 *Sci Data* 9:120: hPSC-derived vs rat cortical networks on 48-well MEA; development (DIV 2–77) + one pharmacology plate per species | CC-BY-4.0 (data, `Data/LICENSE.txt`); article CC-BY-4.0 | G-Node GIN, plain HTTPS, pinned commit `0f0f135`; quick tier also mirrored (below) | arrived (full tier: 239 files, 1.2 GB) |
| `toxcast_mea` | US EPA ToxCast invitrodb v4.3, `CCTE_Shafer_MEA_acute_*` (rat cortical, acute) and `CCTE_Shafer_MEA_dev_*` (network formation) | CC0-1.0 (figshare record 6062623 v14) | EPA Clowder API, anonymous | arrived (full tier: 42 files, 2.57 GB) |
| `iras_mea` | Tukker et al. 2020 *Toxicol Sci* 178:71–87: hiPSC co-cultures vs rat cortical MEA, seizure-liability compounds | **CC-BY-NC-4.0** | Europe PMC REST | **excluded** (license); recorded as `skipped` in MANIFEST |
| `sharf2022` | Sharf et al.: human iPSC brain organoid slices, MaxWell MaxOne HD-MEA (26,400 electrodes); **diazepam dose–response** control/3/10/30/50 µM in 4 organoids; Kilosort2 spike output | CC0-1.0 | Zenodo record 6578989 | arrived (full: kilosort2.zip) |
| `fofie2025` | Fofie et al. 2025: hiPSC-derived **nociceptors** (RealDRG), Axion Maestro Pro 48/96-well; concentration–response for TTX, huwentoxin-IV, lidocaine, retigabine, DNQX, GABA and others | CC0-1.0 | figshare article 28755155 | arrived (1.25 MB xlsx) |
| `screenneuropharm_bic` | ScreenNeuroPharm: rat cortical + hippocampal cultures, 60-electrode MCS MEA, **bicuculline** concentration series; spike trains | **MIT** (source repo) | Zenodo record 7665283 | arrived (full tier only) |
| `axon_diode` | Asymmetric-microchannel chip papers (Renault 2015, Lassus 2018, Courte 2020): design parameters only | CC-BY-4.0 each (read from article XML) | Europe PMC REST | arrived |

## Decisions

### Kapucu quick tier mirrored (2026-10-10)

GIN is Kapucu's only host, and every run needs Kapucu. During the first end-to-end full run from a
fresh clone, gin.g-node.org stopped accepting connections and the run could not proceed. The quick
tier's 169 files are therefore also published as a release asset of this repository,
[`data-mirror-kapucu2022-quick`](https://github.com/Vodolay/AI4S/releases/tag/data-mirror-kapucu2022-quick):
unmodified copies of GIN commit `0f0f135`, redistributed under CC BY 4.0 with the citation inside the
archive. The fetcher uses it only for the quick tier, only when GIN does not answer or fails part-way,
and only after the archive's SHA-256 matches the value pinned in `src/neuromps/fetch/kapucu2022.py`.
Each file's hash then matches the GIN copy recorded in `MANIFEST.json`, and the source's note says the
mirror was used. The full tier is not mirrored; a full run still needs GIN.

### Dataset expansion review (2026-09-21)

Six datasets were proposed to fix three named gaps in the report: the human/rat
imbalance (173 human records against 21,207 rat), the lack of a two-compartment
device with a known electrode map to test the §7.7 directionality null, and the
lack of multi-compound pharmacology with paired baselines for the twin's drug
knobs. Every pointer was checked before it was believed. Four of the six did not
survive the check, so the search was widened; what replaced them is below.

**Accepted.**

- `trujillo2019` - **Zenodo 4751759**, CC BY 4.0. Human hPSC-derived cortical
  organoids on 64-electrode planar MEAs, 12 organoids per session, recorded
  weekly over ten months: 42 sessions of 240.25 s at 12.5 kHz. Spike times are
  stored as sample indices in MATLAB v7.3 (HDF5) files of 1.0-1.5 GB each, whose
  bulk is LFP traces we do not use. Two sessions have a paired `_Drugs`
  counterpart recorded the same day. This is the single largest correction
  available to the human/rat imbalance, and it brings human pharmacology with a
  same-day baseline, which no source in the asset had.
  - *This is the dataset the proposal called `DANDI:000034`.* That identifier is
    somebody else's work (see rejections). The Trujillo organoid recordings are
    on Zenodo.
  - Fetched by partial read, not download: `fetch/http_random.py` reads the HDF5
    object headers and the spike references over HTTP range requests, so one
    session costs **5.2 MB of transfer instead of 1073 MB** and about 8 s.
    Zenodo serves this record at ~2.3 MB/s, so downloading all 42 files would
    have taken about eight hours; the whole set now takes minutes. Provenance
    follows the precedent already set by `remote_zip.py`: the extracted artifact
    is written locally, hashed, and recorded with the byte count actually read.
- `brewer2021` - **Zenodo 4679064**, CC BY 4.0, 31 MB. Rat hippocampal
  subregions (EC, DG, CA3, CA1) reconstructed in a four-compartment device over
  an electrode array, with axons crossing **microtunnels that are individually
  instrumented**, so the direction and speed of transmission are measured rather
  than inferred. `spike_trains_data.zip` is 17 MB. This is the known-geometry
  two-compartment recording the §7.7 null needs: the mateus2024 chips have
  asymmetric microchannels but no electrode map inside them, which is exactly
  the confound the report says it cannot rule out.

**Rejected, with reasons.**

- *van de Wijdeven et al., 4TU `10.4121/uuid:c3c3a4f6-8c46-4cb2-87c2-55f653457a41`.*
  The DOI returns 404. No deposit by these authors was found in DataCite, 4TU or
  Zenodo. The papers are real; a public dataset behind them was not found.
- *Winter-Hjelm et al., "Zenodo / NTNU Open Research Data".* No deposit exists
  under this author in Zenodo (creator query: 0 hits), DataverseNO (0), or
  DataCite (0 datasets). The proposal gave a search phrase rather than an
  identifier, and the search does not resolve. Note that the asymmetric
  microchannel *design parameters* this would have supplied are already in the
  asset as `axon_diode`, read from the article XML of Renault 2015, Lassus 2018
  and Courte 2020.
- *DANDI:000034, "Trujillo organoids".* `000034` is Buccino et al.,
  "SpikeInterface, a unified framework for spike sorting", 74 GB - a different
  dataset by different authors. The organoid sets that do exist on DANDI
  (001336, 001603, 001850, 001872) are all CC BY 4.0 but store **raw ecephys
  traces at 0.2-4.3 GB per session, 20-1570 GB per set**. Using them would mean
  spike-sorting them ourselves, which changes what the asset is: every record in
  it carries the depositors' spike detection, recorded as provenance. Rejected
  on method and on cost, not on licence.
- *D'Amico et al. / MaxWell HD-CMOS, "search Zenodo".* No identifier given and
  none found for human iPSC neurons. The MaxWell HD-MEA datasets that are on
  Zenodo under that description (13839556, 13839126) are iPSC-derived
  **cardiomyocytes**. Human MaxWell HD-MEA neural data is already in the asset
  as `sharf2022` (organoid slices, MaxOne, 26,400 electrodes, CC0).
- *HESI NeuTox multi-laboratory reference screen.* No open deposit found in
  Zenodo, figshare or the EPA Science Inventory; the consortium's outputs are
  published as papers. The EPA MEA data that is public is already in the asset
  as `toxcast_mea` (CC0, invitrodb v4.3).
- *Wagenaar, Pine & Potter 2006 (`mcc-1`).* The proposal's CRCNS URL 404s, but
  the data are genuinely public at potterlab.bme.gatech.edu. Rejected on the
  project's own licence policy, recorded above: the site states only
  *"Downloading from this site is subject to the condition that you acknowledge
  our work in any publications that use our data, by citing this article"*,
  which is an attribution condition rather than a licence grant, and the policy
  excludes unlicensed material even when it is publicly downloadable. Applying
  it here is the same rule that excluded Tukker 2020; applying it to one source
  and not the other would make the policy decorative. Secondary point: 963
  recordings of rat cortex would deepen the very imbalance this review was meant
  to fix.
- *Auslender, Zenodo 14363732 ("MEA recordings dataset", 36 GB).* CC BY-**NC**
  4.0. Excluded by licence policy, as Tukker 2020 was.
- *Monma, Zenodo 14346577 ("Directional intermodular coupling ...").* The title
  is close to what §7.7 needs, but the contents are spiking-network **simulation**
  output, not recordings. Ingesting it as data would break the rule that no
  synthetic data is presented as real.
- *Voskobiynyk et al., Zenodo 3998336 (BIN1, CC0).* Rat hippocampal neurons on
  Axion Maestro, but the recordings are Axion's proprietary `.spk` binaries and
  the manipulation is genetic rather than pharmacological. Neither gap is
  addressed at a cost worth paying.

### Found on a second pass: the EPA multi-laboratory MEA data (2026-09-22)

`10.23719/1517583`, "Data set for Vassallo et al", published by the **U.S. EPA Office of
Research and Development** and listed on data.gov. This is the EPA arm of the multi-laboratory
study the review above looked for under "HESI NeuTox" and did not find, and it contradicts a line
this file has carried since 2026-09-16 saying Vassallo 2017 was not open access. It was found by
querying DataCite, which indexes government repositories that Zenodo, Dryad, figshare, OSF and GIN
searches never reach.

What it is, verified by reading the archive over HTTP range requests rather than trusting the
description:

- **199 MB, 46 members**, of which 39 are spike files named `<date> <chip ids> <COMPOUND> -- <chip>.txt`.
- **Spike times, not summaries.** Each line is `sample_index<TAB>channel`. The archive's own
  conversion script fixes the rate: `samplingFrequency = 25000`. Ticks run to ~1.6e8, so a
  recording spans roughly 107 minutes.
- **Six compounds in a concentration-response design**: acetaminophen (ACE), dimethyl phthalate
  (DMP), glyphosate (GLY), deltamethrin (DELT), domoic acid (DA) and chlorpyrifos-oxon (COX,
  named in full in the companion multi-laboratory workbook).
- **Paired controls on the same chip.** `EPA Final Data.xlsx` carries one row per chip and
  concentration - `Chip`, `Chemical`, `Concentrations (nM)`, `Group` - including **0 nM** rows for
  the same chip that later receives the compound. That is the pairing every model in `models/`
  consumes and that no other multi-compound source in this asset provides.
- **Licence: U.S. Public Domain**, 17 U.S.C. section 105, via the EPA ScienceHub licence page. The
  policy above accepts public domain explicitly, and it is the same basis as `toxcast_mea`.

**Integrated on 2026-09-22 as `vassallo2017`, once the logfile was read.** The segmentation
problem below was real but not blocking: `Logfile_EPAv1.xls`, inside the same archive, gives the
start and end second of the baseline and of every concentration, on the same timebase as the
spikes. All 38 experiments matched their spike file 1:1 on (session date, chip id) - the full names
disagree because a logfile row lists every compound on the plate while the file names the one that
chip received - and the log's compound column agreed with the filename code in all 38. That yields
**283 records, 38 baselines and 245 exposures**, of which 171 survive QC as pairs.

What remains true from the first assessment: The acquisition originally wrote one file per concentration step
(`-- timestamps.txt`, `timestamps0001.txt` ... , visible in the archive's own `breakValues.py`),
but the published per-chip files are continuous. Reading one in full - `102010 15160 10309 DA --
10309.txt` - gives **1,005,054 spikes on 56 channels spanning 371 minutes**, and its twelve largest
inter-spike gaps are 92-305 s, all in the last hour, where the culture is falling quiet. There is no
gap structure that marks a dose boundary, so the segments cannot be recovered from the spike times.
The summary workbook gives a `Filter Length` of 600 s per row, which suggests ten-minute windows per
concentration, but 371 minutes is far more than six such windows, and a `dose_value` assigned by
guessing when the pipette went in would be fiction. Resolving that - from `Logfile_EPAv1.xls`, or by
asking Timothy Shafer, whose address is on the data.gov record - is the first task for whoever picks
this up. Two of its compounds map onto mechanism classes the asset already uses (domoic acid is a
glutamatergic agonist, deltamethrin a sodium-channel modulator); chlorpyrifos-oxon is an AChE
inhibitor, which the schema's enum has no value for.

**Still open after this review.** The human cohort remains thinner than the rat
one even with `trujillo2019`, and every human record still comes from
hPSC-derived tissue rather than primary human neurons - a limitation to keep in
the report rather than one this review can close.

The **multi-compound pharmacology gap is closed** by the EPA dataset above,
which two search passes over Zenodo had failed to find. The modelling table goes
from 73 pairs in six mechanism classes to 754 in nine, and for the first time
the asset holds a concentration-response series with its baseline on the same
chip, from a laboratory that had nothing to do with the rest of it. The near misses were in-vivo recordings,
imaging (Zenodo 15195636 is a *Fractal* image-processing benchmark despite
"multiwell" in its title), or cardiomyocytes. What the asset has instead is
kapucu2022 (5 compound classes x 2 species, paired, but single concentrations),
screenneuropharm_bic (one compound, dose series), sharf2022 (diazepam series in
human organoid slices) and fofie2025 (many compounds, but fitted curves rather
than spike times). Anyone re-attacking this should start from a published
*consortium* dataset rather than a single lab's deposit, and should expect to
ask the authors: the HESI NeuTox outputs exist as papers, not as a deposit.


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
  - **Well-level rows: the full MySQL dump** (`toxcast_mea/mysql/invitrodb_v4_3.sql.gz`, 17,651,807,605
    bytes, Clowder dataset "MySQL_Data"; `neuromps.fetch.toxcast_mysql`, outside the fetch tiers). Read
    as text, never loaded or executed: `python -I -m neuromps.etl.toxcast_wells` streams it once
    (~10 min) and writes `data/interim/toxcast_mea/` (schema, `mc0`-`mc3` parquet, `summary.json`).
    **Gap in the dump:** `mc0`, `mc1` and `mc2` lack whole assay components that `mc3` has (also
    outside MEA). For 8 of the 17 acute components (acids 2447-2449, 2455, 2457, 2459, 2461, 2495) the
    dump has `mc3` rows but no level-0 wells; their `mc3.m0id` values occur nowhere in `mc0`. The
    missing rows are spread over all 10 hash partitions of `mc0`, so this is not one lost partition.
    `summary.json` reports it per component (`n_mc3_rows_without_mc0`, `components_without_mc0`).
    **`apid` is not a physical plate** in these assays: it is the exposure date (e.g. `20150811`) and
    pools several 48-well MEA plates. The physical plate is named only in `srcf`
    (`TC_MW 1073-16_<culture date>_<exposure date>_...`); `summary.json` gives vehicle counts both per
    `apid` (tcpl's plate key, used for `bval`) and per `srcf` plate. `mc0.rval` for the 15
    electrophysiology endpoints is the per-well percent change from that well's own 40-min baseline
    (assay description PDFs, section 1.2); LDH and AB are plate-reader values.
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
  - ~~Vassallo et al. 2017 (rat cortical MEA, same lab) was not found as open access and was never
    fetched.~~ **Wrong, corrected 2026-09-22:** it is public. See the entry below.
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
- **`fofie2025` fetched but NOT converted (decision 2026-09-20).** The workbook is parseable in
  structure - 10-row blocks with a condition label in column A, electrodes as columns, timepoints as
  rows - but **the dose units appear nowhere in the file**; labels read `TTX 10`, `TTX 30`, and the
  units live only in the paper (TTX 30 nM, HWTX 20 nM, lidocaine 500 µM, per doi:10.1016/j.crmeth.2025.101051).
  Worse, the sheet named `Curves TTX-HWTX-Lidocaine` contains only TTX-labelled blocks: sheet titles
  and content disagree, so the other compounds cannot be mapped without reading the figures. Rather
  than guess a unit or a compound, the source stays fetched and out of the asset. It is a useful
  illustration for the report: a CC0 dataset can be unusable on its own when the units live in the
  paper, which is exactly what the standard's `dose_value` + `dose_unit` requirement prevents.
  To convert it later, transcribe the compound/dose map from the paper into a small CSV under
  `src/neuromps/etl/` and cite it per row.
- **Two/three-compartment devices converted (2026-09-20).** 16 devices in the ScreenNeuroPharm
  archive are multi-chamber: 5 `CxCx`, 5 `HpHp` and 6 `CxHp`, each with the full bicuculline series.
  Device geometry (28 microchannels, 10 µm wide x 250 µm long, on a 12x12 200 µm-pitch MEA) is from
  Brofiga et al. 2022, doi:10.3390/mi13081212. Electrode groups: Red = columns 1-5, Green = columns
  8-12, Gray = columns 6-7 over the connecting region (recorded as `compartment_role: interface`).
  - For `CxHp` the dataset and its code never say which colour holds cortical and which hippocampal
    cells, so `cell_source.system` is `unknown` **with a required note**, rather than guessed. Their
    asymmetry is still measurable, which is the point of keeping them.
  - Exploratory (n=6 devices, not a report number, no test yet): per-device median asymmetry
    red->green is +0.10 with a +10 ms lag for `CxHp` versus about -0.05 at 0 ms for `CxCx`; 4 of 6
    `CxHp` devices lead in the same direction, 1 runs the other way. A proper statistical comparison
    belongs in Phase 2/3, with the twin as the model of what asymmetry should look like.
- **Rejected in the same search (recorded for the report's data-selection story).**
  - ~~Zenodo 14525182 (Mateus & Aguiar), compartmentalized asymmetric-microchannel MEA.~~
    **Rejection reversed on 2026-09-21** — see `mateus2024` in the table above and the note below.
  - Carstens et al. 2025 (3D hiPSC BrainSphere HD-MEA) and a GABA-A seizure-liability study: data
    "available upon reasonable request" — gated, so out of scope.
  - Zenodo 14363732 (36 GB MEA recordings) and the IMI2-NeuroDeRisk comparison: CC BY-NC / CC BY-NC-ND.
  - DANDI 001872 and 001336: human organoid MEA in NWB under CC-BY, but no pharmacology.
- **The one licence conflict we resolved rather than avoided (`mateus2024`, 2026-09-21).**
  Zenodo 14525182 is the closest published analogue of the two-compartment chip the twin models:
  hippocampal cultures on straight (`Control`) against asymmetric (`Arrows`, `Rams`, `Tesla`)
  microchannels, 57 spike-data recordings, geometry known by design. It was excluded on 2026-09-16
  because the deposit contradicts itself: the Zenodo metadata licence field says `cc-by-4.0` and the
  record is open access, while the README inside says, in capitals, CC BY-NC-ND.

  On 2026-09-21 the deposit's corresponding author replied by email, copying the first author, and
  confirmed that the Zenodo metadata licence is the one they intend and that the README's
  CC BY-NC-ND line was an unintentional default they can no longer edit. On that basis the source is
  included under CC BY 4.0, with attribution as recorded in `CITATIONS.md`. Their wording is
  paraphrased rather than quoted here: they agreed to a licence reading, not to the publication of
  their message.

  What a reader should know about the footing this rests on, since it is weaker than every other
  source in the asset:

  - The **metadata licence is public and checkable** by anyone: `zenodo.org/api/records/14525182`
    returns `cc-by-4.0`, and `fetch/mateus2024.py` blocks the fetch if it ever moves.
  - The **README's contradicting claim is public too**, and is recorded verbatim in
    `data/raw/MANIFEST.json` on every fetch rather than quietly dropped. If the depositors correct
    it, the manifest note will say so instead.
  - The **resolution itself is private correspondence**, which a judge cannot verify. We have asked
    the authors to add a clarifying line to the Zenodo record description, which depositors can edit
    without cutting a new version; until they do, this dependency stands as stated here.

  The NoDerivatives term is the one that mattered: under CC BY-NC-ND we could not have published
  derived features at all, which is the whole of what this project does with a recording.

- **Axon-diode lineage.** Peyrin et al. 2011 *Lab Chip* (doi:10.1039/c1lc20014c) and Courte et al. 2018
  *Methods Cell Biol* (doi:10.1016/bs.mcb.2018.07.002) are not open access: cited only, never
  downloaded. Design parameters are taken from the open-access papers.
- **Kapucu 2022 code** in the GIN `Codes/` folder is BSD-3 / GPL-3.0 (meaRtools, SpikeDetection-Toolbox).
  We do not fetch or vendor it. If any of it is ever used, check GPL obligations against our MIT license first.

### Final dataset sweep for Task 1 pairs (2026-09-25)

Task 1 fits 73 pairs. The question asked was whether more public data exists, including
species beyond rat and human. Searched: Zenodo's API (170 + 148 deposits over fourteen
queries), DataCite across repositories, and the EPA ScienceHub catalogue. One candidate
worth having; the rest of the well is close to dry.

**The one candidate. Zenodo 21283609**, "Graph Theory Benchmarking MEA Data Set - Human
Brain Slice Cultures", CC BY 4.0, 593 MB, deposited 2026-07-24.

- **Adult human post-operative neocortical slice cultures** on a Multi Channel Systems
  256-electrode array, 252 recording channels. This is a preparation the asset does not
  have: its human side is iPSC-derived and organoid only.
- 37 recordings, each a pickled dict carrying MAD-detected **spike times** per channel,
  plus `slice_id`, `recording_date`, `tissue`, `medium` and a `drug` field taking
  `nodrug`, `8mMK` (8 mM potassium) and `5uMgabazine`.
- Read without downloading, using `fetch/remote_zip.HttpRangeFile` over the central
  directory and a 4 MB prefix of each member. Of the 17 recordings whose `slice_id` and
  `drug` both resolved in that prefix, **three slices carry a baseline and a drug
  condition**: `ID2310CT004` and `ID2411CT002` are gabazine/nodrug, `ID2405CT005` is
  8mMK/nodrug. Twelve recordings did not resolve in a 4 MB prefix, and the filenames
  carry roughly fourteen gabazine tags, so the true count is higher than three - but it
  is a handful, not a doubling of the 73.

**Why it is still worth having, and it is not the pair count.** `gabaa_antagonist`
currently comes from `kapucu2022` (gabazine, one condition) and `screenneuropharm_bic`
(bicuculline). Section 6.1 shows the real ceiling on Task 1 is not sample size but that a
pair's mechanism label predicts its source 97.35% of the time. A third source for that
class, on adult human cortex rather than a stem-cell model, is worth more to that
confound than the same number of pairs from a fourth compound in an existing source.

**The objection to weigh first.** The recordings are **Python pickles**. Ingesting them
means unpickling files fetched over the network, which executes arbitrary code during the
build. Every other source in the asset is HDF5, MATLAB, CSV, parquet or Excel, none of
which does. Reading the metadata above deliberately used `pickletools.genops` and a raw
byte scan rather than `pickle.load`, and the same restraint would be needed in an ETL -
or the deposit's authors asked for a plain export. This is a provenance property the
project has not had to have an opinion on before, and it should have one before ingesting.

**The EPA vein is exhausted.** DataCite lists twelve EPA MEA datasets. Seven were checked
by reading their file lists, and four ZIP archives were opened over range requests without
downloading. All of them ship summary endpoint tables or analysis output - `MEA 86
Compound data` is AUC and plate-normalised values; `NTP_TC_Analysis.zip` is 11,967 PDFs of
dose-response plots beside 80 CSVs; `Kosnik 2020` is 25 CSVs through `tcpl` with R
scripts; `Tipping point` is 273 PDFs. `vassallo2017`, already in the asset, was the single
deposit that shipped raw spike files. Two near-misses worth naming because they read well
in a search result and are not what they look like:

- `10.23719/1518437` (Saavedra 2024), "Comparison of Acute Effects of Neurotoxic Compounds
  on Network Activity in **Human and Rodent** Neural Cultures" - exactly the comparison
  section 6.4 lacks, but the deposit is one spreadsheet of per-figure normalised values.
- `10.23719/1374038`, "MEA 86 Compound data" - 86 compounds, one xlsx, no spike times.

**Nothing else passed.** Across 148 further Zenodo deposits from eight queries covering
mouse, zebrafish, organoid, multiwell and slice preparations, **none** had an open licence,
an MEA, spike times and a drug condition together. The field publishes endpoint tables.

**Already downloaded and still unconverted: `fofie2025`.** CC0, hiPSC-derived nociceptors,
and the widest compound overlap of anything here - TTX, huwentoxin-IV, lidocaine,
retigabine, DNQX, GABA. It was parked on 2026-09-20 because the dose units appear nowhere
in the workbook and because it holds spike *counts* per electrode, not spike times. The
first half of that is now a solved kind of problem: the EPA logfile precedent is to take
the value from the paper and record where it came from. The second half is not - counts
support rate features only, so it cannot supply the nine-feature delta vector Task 1
consumes. It could support a rate-only arm, which is a different claim.

### `fofie2025` converted after all, as a rate-only arm (2026-09-25)

The September decision to fetch this source and not convert it rested on two claims, one of
which was wrong.

- **"The dose units appear nowhere in the file" is false.** Four of the five data sheets label
  conditions with the unit: `GABA [10 µM]`, `TTX (5 nM)`, `DNQX (50 µM)`, `Capsazepine (30 µM)`.
  The one sheet that writes bare numbers, `Curves TTX-HWTX-Lidocaine`, is skipped rather than
  read against the paper - a dose is a number and a unit, and half of one is not a dose. The
  earlier note generalised from that sheet to the workbook.
- **"Spike counts, so firing-rate features cannot be derived" is half right, and the useful
  half is the other one.** An absolute rate needs the recording duration, which the deposit
  does not publish. A rate *change* does not: every model here consumes
  log2(exposure / baseline), and with both counts taken over equal-length recordings of the
  same well the duration cancels out of the ratio. That is the whole of what makes this source
  usable, and it is why its records would sit at `per_well_features` rather than `spike_times`.

**What it yields.** 291 pairs over 48 condition blocks, eight mechanism classes, each well its
own control - the strongest pairing in the asset, with no plate effect to model around.
`gabaa_agonist_or_pam` occupies a single block and is dropped, because a grouped split cannot
hold out a class that lives in one group.

**What validated the parse.** Not that it produced rows, but that untreated wells move by a
median of -0.17 log2 units while sodium-channel blockers move by -1.95. Three layouts had to be
read - rows-as-wells, blocks-stacked-down-the-sheet, and derived %-change columns sitting beside
the counts - and a misalignment in any of them would not produce that separation.

**Two defects the first parse had, both caught by asserting on the biology rather than the row
count.** Every `No Intervention` well was being dropped, because the condition name was reduced
to its first word and `no` then matched nothing in the vehicle list. And the grouping key was
the condition block, which in the rows-as-wells sheets holds several conditions at once - so
126 of 133 groups spanned more than one compound, which would have leaked the label across the
split. The group is now block *and* compound, and a test asserts no group spans two.

**The scope limit, which is not small.** These are hiPSC-derived **sensory neurons**. What they
share with the rest of the asset is the compound mechanism, not the preparation, the network or
the readout. §6.6b says so wherever it quotes a number.
