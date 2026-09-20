# NeuroMPS Data Standard (draft v0.1)

A minimal, checkable standard for reporting functional electrophysiology from neural
microphysiological systems (MEA plates, HD-CMOS arrays, compartmentalized neural chips).

**Status:** draft, produced for the AI4S Open Innovation challenge from five public datasets.
**Scope of use:** research decision-support only. Conformance carries no efficacy, safety, diagnostic
or clinical meaning.

Normative artifact: [`neuromps_schema.json`](neuromps_schema.json) (JSON Schema 2020-12). Where this
document and the schema disagree, the schema wins. Validate with:

```bash
python -m neuromps.qc.schema schema/examples/*.json
```

## 1. Scope

**In scope:** spike-derived functional measures from in vitro neural preparations, the experimental
context needed to compare them (cell source, platform, age, compound, dose, exposure, recording), the
methods that produced them, quality control, and provenance back to a hashed raw file.

**Out of scope:** raw voltage traces and spike sorting (referenced, not redefined), imaging and omics
readouts, chip fabrication, and any clinical interpretation.

## 2. Terminology

| Term | Meaning here |
|---|---|
| **Unit** | The smallest independently recorded preparation: a well, an array, or a chip compartment. `unit_id`. |
| **Record** | One unit under one condition at one data level. The unit of exchange. |
| **Data level** | How far the source processed the data before publishing: `spike_times`, `per_well_features`, or `fitted_curve`. |
| **Active electrode** | An electrode whose firing rate meets a stated threshold; the threshold is part of the record, not a convention. |
| **Burst** / **network burst** | Single-electrode and cross-electrode spike clusters, as defined by the *named* detector in `feature_method`. |
| **Phase** | `baseline`, `exposure` or `washout`. Acute designs pair a baseline and an exposure record for the same unit. |

## 3. The three data levels (central requirement)

Public datasets do not publish at the same level, and **records are comparable only within a level**.
Every record states its level; a consumer that mixes levels without an explicit modelling step is
misusing the asset.

| Level | What the source gives | What can be computed | Example |
|---|---|---|---|
| `spike_times` | spike or sorted-unit times | the full feature vector | Kapucu 2022, Sharf 2022, ScreenNeuroPharm |
| `per_well_features` | publisher's per-well summaries | only the features they published | Fofie 2025 (total spike counts only) |
| `fitted_curve` | concentration-response fit over many wells | potency and efficacy, no per-unit features | ToxCast (`mc5-6`) |

## 4. Technical requirements

Every record MUST carry:

1. **Identity and rights:** `record_id`, `source`, `license` (SPDX), `citation`. The schema admits only
   licenses permitting commercial reuse and redistribution: CC0, CC BY, MIT, BSD-3-Clause, Apache-2.0.
2. **Biology:** `cell_source.species` and `cell_source.system`; line, donor and protocol where known.
   Species and system are never inferred from a file name.
3. **Platform:** vendor, system, `electrodes_per_unit`, `sampling_rate_hz`; plate format and electrode
   pitch where known.
4. **Condition:** `compound.name` and `role`; a dose value MUST be accompanied by a unit
   (`dependentRequired`), and `mechanism_class` is a closed vocabulary.
5. **Provenance:** at least one source file with its SHA256 as recorded in `data/raw/MANIFEST.json`,
   the ordered `transform` chain, and who generated the record and when.

Conditional requirements:

- `spike_times` → `features`, `feature_method`, `recording` and `qc`, and `feature_method` MUST name
  `spike_detection` and `burst_method`.
- `per_well_features` → `features` and `feature_method`, and `feature_method.spike_detection`.
- `fitted_curve` → a `curve` block with endpoint, number of concentrations and hitcall.
- A non-null `features.synchrony_index` → `feature_method.synchrony_method`.

### 4.1 Core feature vector

`mfr_hz`, `burst_rate_per_min`, `burst_duration_s`, `spikes_per_burst`, `pct_spikes_in_bursts`,
`network_burst_rate_per_min`, `synchrony_index`, `isi_cv`, `active_electrodes`.

Every field is nullable. **A source that cannot support a feature records `null`, never a substitute.**
This is the rule that keeps the asset honest: Fofie 2025 publishes total spike counts, so its burst
fields stay null rather than being approximated.

### 4.2 Methods are part of the measurement

Feature names do not imply comparability. `synchrony_index` computed as STTC is **not** the same
quantity as Axion's synchrony index, and burst counts depend on the detector. Therefore:

- `feature_method.synchrony_method` uses a closed vocabulary (`sttc`, `axion_synchrony_index`,
  `correlation_coefficient`, `mutual_information`).
- `feature_method.computed_by` distinguishes our own extractor from publisher-supplied numbers.
- `spike_detection` records whose detector produced the spikes, including when it is not ours.

### 4.3 Platform comparability

Electrode count per unit spans 8 to 26,400 across the current sources. Electrode-count-dependent
quantities (`active_electrodes`, and any synchrony measure that averages over electrode pairs) MUST
NOT be compared across platforms without an explicit, recorded rescaling.

## 5. Quality control

`qc.usable` is a boolean decision; `qc.flags` is a closed vocabulary explaining it:
`too_few_active_electrodes`, `excluded_well_upstream`, `noisy_electrodes_present`, `short_recording`,
`missing_baseline_pair`, `replicate_divergence`, `outlier_feature_value`, `incomplete_dose_series`.

Defaults adopted from the EPA acute MEA assay, for comparability: an electrode is active at
≥ 5 spikes/min, and a well is usable with ≥ 10 active electrodes. Both are recorded per record, so a
consumer can re-apply their own thresholds.

Upstream exclusions are preserved, not silently dropped: Kapucu's `ExcludedWell` rows and
noisy-electrode lists become `excluded_well_upstream` and `noisy_electrodes_present`.

### 5.1 Cross-record flags

Three flags need context a single record lacks, so they are applied as a pass over the whole asset
(`neuromps.qc.flags`):

- `replicate_divergence` / `outlier_feature_value`: a well whose feature value is more than 3.5
  robust (median/MAD) z-scores from its replicate group — same source, plate, phase, compound and
  dose. Groups smaller than 3 are never flagged. Where more than half a group shares a value exactly
  the MAD is zero, so the mean absolute deviation is used as the scale instead; otherwise the single
  divergent well would be invisible.
- `missing_baseline_pair`: an exposure recording whose well has no baseline recording.

**Flags are informational and never change `qc.usable` by themselves.** `qc.note` names the diverging
feature and its z-score, because a well can sit at the group median for one feature and diverge on
another. In near-silent conditions (a TTX plate) the flag fires on wells that are still firing; that
is intended, since it identifies a failed drug application.

## 6. Provenance

Every record traces to hashed raw files: source URL, license, SHA256, size and download time live in
`data/raw/MANIFEST.json`; the record names the files it derives from and the transform chain applied.
Excluded sources stay in the manifest with `status: skipped` and a reason, so that a dataset that was
considered and rejected on license grounds remains visible.

## 7. Conformance

- **Level 1 (minimum):** validates against the schema.
- **Level 2 (traceable):** Level 1, and every `provenance.source_files[].sha256` matches an entry in
  the accompanying `MANIFEST.json`.
- **Level 3 (reproducible):** Level 2, and the transform chain runs end to end from the raw files
  (`python run_all.py`).

## 8. Known gaps in this draft

- No representation for compartmentalized chips yet (per-compartment units, directed connectivity
  between them). This is needed for the two-compartment digital twin and is the main planned
  extension.
- `culture_age_days` flattens organoid age (months, converted at 30 days/month) and days in vitro into
  one field. The conversion is recorded in `provenance.transform`, but the underlying concepts differ.
- No vocabulary for media, coating or seeding density, which affect network activity.
- Dose is single-value; a record cannot express a concentration series except at `fitted_curve` level.
