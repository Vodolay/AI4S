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

### 4.4 Named definition sets

A record's `features` block is computed under one definition set, `neuromps`: the defaults of
`FeatureParams` in `features/mea.py`, which its `feature_method` block states. Other laboratories
define the same features differently (report §3), so the asset also carries each spike-time record
under other **named definition sets**, without changing the record (decision D37).

- **The registry** (`src/neuromps/features/definitions.py`) names each set. A set is a bundle of
  options the one extractor already has (`active_min_spikes_per_min`, `active_strict`,
  `burst_rate_denominator`, `burst_detector`, …) plus one electrode policy: whether a deposit's
  listed noisy electrodes are dropped before extraction. A set is not a second extractor. Each
  set states where every choice comes from and has a status: `reference` (what the asset's records
  use), `validated` (reproduces another implementation's published output on real recordings; the
  evidence is named by its `outputs/metrics.json` key) or `experimental` (computed and shown, used
  for no claim). Current sets: `neuromps` (reference), `kapucu2022` (validated,
  `extractor_agreement.settings_ladder`) and `kapucu2022_logisi` (experimental,
  `extractor_agreement.logisi`).
- **The sidecar** `data/processed/features_by_definition.parquet` holds every spike-time record
  under every set, long-form: `record_id`, `source`, `definition`, `feature`, `value`. It joins
  the asset on `record_id`; an undefined value is NaN, as in the asset. Its `neuromps` rows MUST
  equal the asset's `features.*` exactly, which the pipeline checks on every run. The sidecar
  derives from the same sources as the asset and carries the same licences
  (`data/processed/NOTICE.md`). Each set's method fields are in `outputs/metrics.json` key
  `feature_definitions.registry`.
- **Translation** between two sets (`outputs/tables/feature_definitions.md`) is reported only where
  a depositor's own output exists to check it against (the Kapucu 2022 developmental plates), with
  a verdict per feature: same number, scale factor, or no conversion. A ratio there is a property of
  those recordings, so it is reported per plate as well as pooled; it is not a conversion constant
  for other cultures. A "same number" between two of our sets says nothing about the depositors:
  the table's last columns say how close the target set itself comes to their published values.
  Data with another laboratory's definitions should be recomputed under a named set from spike
  times where possible, not converted.

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

## 5.2 Multi-compartment devices

A compartmentalized chip (two or three chambers joined by microchannels) is described as **one record
per compartment**. Records of the same chip share a `device.device_id`; `device.compartment_id` names
the compartment and `device.compartment_role` distinguishes a `chamber` from the `interface` strip of
electrodes over the connecting region, whose signals belong to neither chamber exclusively. Channel
geometry (`microchannel_count`, `microchannel_length_um`, `microchannel_width_um`) is recorded because
it is the physical parameter a connectivity model needs.

Measured coupling between compartments lives in `coupling`, one entry per other compartment:

| Field | Meaning |
|---|---|
| `peak_correlation` | strength of co-variation at the best lag |
| `lag_ms` | lag of that peak; **positive means this compartment leads** |
| `asymmetry` | +1 all influence outward, −1 inward, 0 symmetric |
| `method` | the named algorithm, as everywhere else in this standard |

Two rules keep this honest:

1. **Coupling is descriptive, not causal.** Common drive to both compartments produces correlation
   with no projection between them, and the measure cannot separate a direct projection from a shared
   input that reaches one compartment first.
2. **No direction without evidence.** `lag_ms` and `asymmetry` are null unless the peak correlation
   clears a noise floor of 5 standard errors; otherwise uncorrelated compartments would yield a
   confident direction computed from noise.

### `unknown` cell systems

`cell_source.system` accepts `unknown`, but **only with a `cell_source.note` explaining why**. It
exists so an unlabelled preparation is recorded as unlabelled rather than guessed. The current case:
cortical–hippocampal devices whose dataset never records which electrode group holds which
population. Their compartment asymmetry is still measurable, so the records are kept and the label is
not invented.

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
