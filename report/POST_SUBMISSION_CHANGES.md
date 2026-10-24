# What changed after the submission

*A running record of every change to NeuroMPS after the round-1 deadline (2026-10-10 15:59:59
UTC; the submitted state is commit `464eb76`). Each entry opens with the one sentence a slide
would carry, then says what changed, why, what it did to the results, and where the evidence
is. Numbers cite `outputs/metrics.json` as the report does, so the citation check holds this
document to the same standard. Newest entries are at the end; planned work is in the queue at
the bottom.*

| # | Date | Change | Effect on results | Commits |
|---|---|---|---|---|
| 1 | 2026-10-10 | Unused Kaggle notebook removed; seven figures redrawn at print size | none (no number or claim changed) | `e5ed404`, `bbd75f9` |
| 2 | 2026-10-11 | Bicuculline doses put on one unit scale | the dose conclusion reverses | `886d8d3` |
| 3 | 2026-10-22 | Demo app runs as a hosted page from the repository alone | none; access only | `c4a9004` |
| 4 | 2026-10-22/23 | From chip to animal: the drug knobs on a whole fly brain | new results, pre-registered | `c83eb14` … `aba184c` |
| 5 | 2026-10-24 | Report page limit lifted; the fly study joins the report | none; structure only | this branch |

---

## 1. Housekeeping pushed just after the deadline

**Slide message:** none; listed for completeness.

Two commits reached GitHub after the deadline: one removed a Kaggle notebook nothing used
(`e5ed404`), the other redrew seven figures at the size they print at (`bbd75f9`). They were
reverted during round-1 judging so that reviewers saw the submitted state (`2ecca42`), and
restored after it (`a05b013`).

## 2. A unit error hid a dose trend, and another team's result exposed it

**Slide message:** *Burst rate falls with bicuculline dose after all; we had ranked
nanomolar doses as if they were micromolar, and found the bug by taking a competitor's
disagreement seriously.*

- **What changed.** ScreenNeuroPharm records its two lowest bicuculline doses in nM and the
  rest in µM. Three analyses ranked the dose value without its unit, so 0.1 µM sat among the
  highest doses. One function, `models/dataset.doses_in_um`, now converts in all of them, and a
  test (`tests/test_dose_units.py`) checks each.
- **Why.** NeuroChip Twin, another entry, reported a bicuculline dose trend that our report
  said did not exist. Tracing the disagreement found the bug.
- **Effect.** The feature dose test moved from no feature tracking dose to
  2 <!-- metrics.json: feature_dose_response.absolute.n_features_moving --> of
  7 <!-- metrics.json: feature_dose_response.absolute.n_features_tested -->: burst rate
  (Spearman −0.5813 <!-- metrics.json: feature_dose_response.absolute.by_feature.burst_rate_per_min.spearman_rho -->)
  and network-burst rate
  (−0.604 <!-- metrics.json: feature_dose_response.absolute.by_feature.network_burst_rate_per_min.spearman_rho -->)
  fall with dose across
  113 <!-- metrics.json: feature_dose_response.n_recordings --> recordings. The twin's inferred
  inhibition still does not follow dose, so the conclusion became: the measurement carries the
  trend, and the inverse model is what fails.
- **Evidence.** Report §18 (negative results) and Appendix M; `src/neuromps/models/dataset.py`.

## 3. The demo runs as a hosted web page

**Slide message:** *Judges can open the demo in a browser; nothing to install.*

- **What changed.** Streamlit Community Cloud installs a requirements file and runs
  `app/app.py` without installing the package. The app now finds `neuromps` in `src/`, and
  `app/requirements.txt` pins only what the page needs, at the project's versions (a test keeps
  them equal). The one live-simulation button says it needs the full local install instead of
  failing.
- **Effect.** None on results. Checked in a clean environment with only the hosted
  requirements: every tab renders.
- **Evidence.** `tests/test_app_hosted.py`.

## 4. From chip to animal: the chip's drug knobs on a whole fly brain

**Slide message:** *On a simulated whole fly brain, the chip's firing rate ranks drugs
correctly but raises the alarm late: in every drug family, behaviour failed first.*

- **What changed.** A new package, `src/neuromps/fly/`, runs the connectome-constrained model
  of Shiu et al. (2024) on FlyWire with
  138,639 <!-- metrics.json: fly_models.flywire783.neurons --> neurons, applies the twin's drug
  knobs by transmitter class, and reads the brain both as a chip (a virtual electrode array
  scored by `features/mea.py`) and as an animal (the neurons the published model validated
  against feeding, aversive gating and grooming). It is replicated on the male CNS connectome
  (165,122 <!-- metrics.json: fly_models.malecns.neurons --> neurons). New data sources, all
  CC-BY or MIT, enter the manifest as optional; `run_all.py --stage fly` runs it, and the app has
  a fourth tab.
- **Why.** Round 1 scored us lowest on practical value. A chip assay is used to predict what a
  drug does to a nervous system, which no culture can check; a mapped, simulated brain can.
- **How it was kept honest.** Three pre-registrations, each committed before the simulations it
  scores; the release of every table checked against the model's (the published neuron lists
  come from an older release); every failed prediction reported.
- **Effect.**
  - The model is the published one: sugar drives the feeding motor neuron at
    76.75 <!-- metrics.json: fly_chip_to_animal.predictions.R1_replication.value.mn9_783_hz --> Hz
    against
    77.375 <!-- metrics.json: fly_chip_to_animal.predictions.R1_replication.value.mn9_630_hz --> Hz
    on the release the paper used.
  - The chip's firing-rate change tracks behavioural disruption (Spearman
    0.863 <!-- metrics.json: fly_chip_to_animal.predictions.C1_chip_tracks_disruption.value -->;
    0.878 <!-- metrics.json: fly_malecns_replication.predictions.M_C1.value --> on the male CNS).
  - Registered prediction failed: the chip, at the registered threshold, flags the drug after
    behaviour has failed in all five drug families.
  - At behavioural onset, GABA block has moved firing rate by
    0.149 <!-- metrics.json: fly_confirm_network_bursts.precision.abs_log2_mfr_change_at_behavioural_onset.flywire_sweep.gaba_block -->
    log2, about the untreated variation of the best real chips in the asset
    (0.156 <!-- metrics.json: fly_confirm_network_bursts.precision.real_chip_noise.kapucu2022_control.q95_abs_log2 -->).
  - A candidate early-warning feature (network-burst rate) proved to be noise: untreated
    recordings already cross its threshold in a fraction
    0.382 <!-- metrics.json: fly_checks.false_alarm_rate_untreated.by_feature.network_burst_rate_per_min -->
    of splits.
  - Spontaneous activity recovers the direction of wiring between brain regions at chance
    (0.4946 <!-- metrics.json: fly_chip_to_animal.predictions.D1_spontaneous_direction_is_near_chance.value -->).
- **Evidence.** Report Act VII (§23–§26); `preregistration/`; figures
  `outputs/figures/fly_*_SIMULATED.png`; app tab 4.

## 5. The report has no page limit; the fly study joins it

**Slide message:** none; structure only.

- **What changed.** The 15-to-20-page gate on the report's main matter is removed from the
  build and from CI, now that the round-1 page limit no longer applies; a lower bound remains so
  a broken build that drops sections is still caught. The fly study moves from a companion note
  into the report as Act VII (§23–§26), after §22 so that no earlier section number changes;
  the short Appendix P that pointed to the note is gone.

---

## Work queue (decided 2026-10-24, from the post-round-1 review)

| Item | Status | Registered before running |
|---|---|---|
| Synaptic depression under GABA-A block: does it make the twin's firing rate fall, as real chips do? | planned | yes |
| ToxCast potency bridge: do spike-level features rank compounds as ToxCast's fitted potencies do? | data check first | yes |
| Output-tract electrode: does recording descending neurons flag drugs earlier than a planar array? | planned | yes |
| Brain and nerve cord as a two-chamber chip (male CNS): chamber-level directionality against the wiring | planned | exploratory |
| Twin posteriors as classifier features | planned, lower priority | yes |
| Random-graph-to-connectome blend for organoids; embodied fly | dropped | — |

Why the last two were dropped: grafting fly wiring onto a model of a mammalian cortical culture
has no biological basis, and organoid–monolayer differences have many confounds, so a match
would prove nothing; an embodied (physics-simulated) fly is weeks of work for a question the
behaviour neurons already answer.
