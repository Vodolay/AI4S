# Draft feature set (HD2 decision, input to `schema/neuromps_schema.json` in HD3)

## The two sources sit at different levels of aggregation

| | Kapucu 2022 | ToxCast CCTE_Shafer MEA |
|---|---|---|
| Unit of observation | well × recording (spike times available) | chemical × endpoint (curve fit over 7 concentrations) |
| What we can compute | our own features, via `features/mea.py` | nothing: only fitted `top`, `ac50`, `hitcall` |
| Platform | Axion Maestro, 48-well, 16 electrodes/well, 12.5 kHz | identical platform and sampling rate |
| Exposure | acute application, recorded before/after | acute: 40 min baseline vs 40 min exposure; dev: continuous DIV 0–12, AUC over DIV 5/7/9/12 |

Consequence: **ToxCast cannot enter the same per-well feature table.** It stays a separate
platform/level in the asset, as the PLAN risk register anticipated. It provides chemical-space scale
(508 chemicals, acute) and an external check on which features respond to neuroactive compounds; it
cannot provide per-well feature vectors.

The **acute** assay is the analogue of Kapucu's pharmacology plates. The **dev** assay measures network
formation under 12-day exposure and is not comparable to an acute drug application; keep it for the
maturation narrative only.

## Core feature vector (computed by `features/mea.py` from spike times)

Chosen for two reasons: each is computable from spike times alone, and each has a named counterpart in
the ToxCast acute endpoint list, so the standard maps onto an existing regulatory battery.

| Our feature | Kapucu meaRtools output | ToxCast acute endpoint |
|---|---|---|
| `mfr_hz` mean firing rate per active electrode | `meanfiringrate_by_active_electordes` [sic] | `firing_rate_mean` |
| `burst_rate_per_min` | `bursts_per_min` | `burst_number` (per recording) |
| `burst_duration_s` | `mean_dur` | `burst_duration_mean` |
| `spikes_per_burst` | `mean_spikes_in_burst` | `per_burst_spike_number_mean` |
| `pct_spikes_in_bursts` | `per_spikes_in_burst` | `burst_percentage_mean` |
| `active_electrodes` | `nae` | `active_electrodes_number` (dev only) |
| `network_burst_rate_per_min` | derived (not in meaRtools output) | `network_burst_percentage` (related, not identical) |
| `synchrony_index` | `STTC` (spike time tiling coefficient) | `synchrony_index` (Axion's own measure) |
| `isi_cv` | derived | no counterpart |

**Not interchangeable:** `STTC` and Axion's `synchrony_index` are different algorithms, as are our
network-burst detector and Axion's. Report them as separate fields with the method recorded; never
merge them into one column. This is exactly the harmonization gap the data standard should name.

## The other three sources

| Source | Level | What the core vector can use |
|---|---|---|
| `sharf2022` (human organoid, HD-MEA) | spike times per sorted unit | all features, but electrode geometry differs (26,400 CMOS electrodes vs 16 per Axion well): `active_electrodes` and synchrony are not comparable across platforms without an explicit rescaling, which the standard must record |
| `fofie2025` (hiPSC nociceptors, Axion) | total spike counts per electrode and per well | `mfr_hz` only. No spike times, so no burst, network or ISI features |
| `screenneuropharm_bic` (rat cortical/hippocampal, MCS 60-electrode) | spike trains | all features; 60-electrode single-well layout, 8-min recordings |

This is the second harmonization gap worth naming in the standard: sources differ not only in features
but in the **level** at which data is published (spike times → per-well features → fitted curves). A
record in the asset must state its level, and comparisons are only valid within a level.

## Cross-compartment features (added for multi-compartment devices)

`features/coupling.py` adds, per ordered compartment pair: `peak_correlation`, `lag_ms` and
`asymmetry`, from the cross-correlation of binned population rates (5 ms bins, ±200 ms). These are
device-level quantities, stored in the record's `coupling` block rather than the per-unit feature
vector, and they are the quantities the two-compartment twin must reproduce.

## Definitions to pin down in HD3/HD4
- Burst detector: Kapucu used meaRtools (MaxInterval variants); ToxCast used Axion AxIS. We compute
  our own from spike times, so the standard must record `burst_method` per record.
- Active electrode: ToxCast acute uses ≥ 5 spikes/min, and a well is usable with ≥ 10 active
  electrodes. Adopt that as the default QC threshold for comparability, and record it.
- Spike detection: not ours in either source (AxIS adaptive detector at 8× RMS for ToxCast; the
  authors' MATLAB code for Kapucu). Record `spike_detection` as provenance, and treat it as a
  limitation of the asset.

## Open questions
- **OQ-1 (resolved 2026-09-16).** The sign of ToxCast `top` means opposite things in the two assays,
  so `etl/toxcast_mea.py` maps it per assay into `activity_change` (positive = the measured quantity
  rose): acute uses `-top` (normalized against a positive control of -100, so +100 = activity
  abolished), dev uses `+top` (percent of control on AUC metrics, so -100 = all metrics collapse to
  zero). Derived from cytotoxicity anchors — chemicals that kill cells must reduce activity — over 16
  acute and 221 dev chemicals; `check_direction_convention()` re-derives it on every run and raises
  below 90% agreement (current: acute 92.4%, dev 99.1%, in `outputs/metrics.json`). The annotation
  field `signal_direction` alone does not settle this: all 32 activity endpoints are `bidirectional`.
  The 4 viability endpoints keep their annotated direction and get no `activity_change`.
  Note for interval endpoints (interburst interval and similar): a rise means *less* activity.
- **No compound overlap.** None of Kapucu's five drugs (kainic acid, CNQX, D-AP5, GABA, gabazine) nor
  TTX appear among the 508 acute-assay chemicals: TTX is the assay's positive control, not a test
  chemical. There is no shared compound for a direct cross-dataset check, so cross-platform
  comparison must be made at the level of feature response patterns, not per-compound agreement.
