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

## Definitions to pin down in HD3/HD4
- Burst detector: Kapucu used meaRtools (MaxInterval variants); ToxCast used Axion AxIS. We compute
  our own from spike times, so the standard must record `burst_method` per record.
- Active electrode: ToxCast acute uses ≥ 5 spikes/min, and a well is usable with ≥ 10 active
  electrodes. Adopt that as the default QC threshold for comparability, and record it.
- Spike detection: not ours in either source (AxIS adaptive detector at 8× RMS for ToxCast; the
  authors' MATLAB code for Kapucu). Record `spike_detection` as provenance, and treat it as a
  limitation of the asset.

## Open questions
- **OQ-1 (blocking for any ToxCast direction claim):** the physical direction of ToxCast `top` is
  per-endpoint, not global. tcpl normalizes each endpoint in its own fitting direction, so the sign
  cannot be read as "activity up/down" without the per-endpoint direction from the EPA assay
  annotations. A global flip is demonstrably wrong: it inverts the cytotoxicity endpoints (LDH must
  rise when cells die). `etl/toxcast_mea.py` therefore reports the raw `top` and hit counts by sign
  only. Resolve by reading `analysis_direction` / normalization method per AEID from
  `assay_annotations_invitrodb_v4_3_AUG2025.xlsx` (needs an xlsx reader, see PROGRESS.md).
- **No compound overlap.** None of Kapucu's five drugs (kainic acid, CNQX, D-AP5, GABA, gabazine) nor
  TTX appear among the 508 acute-assay chemicals: TTX is the assay's positive control, not a test
  chemical. There is no shared compound for a direct cross-dataset check, so cross-platform
  comparison must be made at the level of feature response patterns, not per-compound agreement.
