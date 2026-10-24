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
| 5 | 2026-10-24 | Report page limit lifted; the fly study joins the report | none; structure only | `1c7d94f` |
| 6 | 2026-10-24 | EPA screen cutoffs set beside the fly's precision needs | new comparison (descriptive) | `cd074b3` |
| 7 | 2026-10-24 | Vesicle depletion tested as the cause of the GABA-A reversal | registered prediction failed; new negative result | `3071ab4`, `8fb5065` |
| 8 | 2026-10-24 | The twin as a source of labelled data, and as a prior for the classifier | 1 of 3 registered predictions pass | `761068e`, `93bb4f6` |
| 9 | 2026-10-24 | Electrodes on the fly brain's output tract against a planar array | 4 of 4 registered predictions pass | `cd074b3`, `2c66067` |
| 10 | 2026-10-24 | Behaviour read at the motor neurons of the male CNS | 6 of 7 registered predictions pass | `a1c5f27`, `a2f4ee2` |
| 11 | 2026-10-24 | The motor-level ties resolved on a finer dose grid | 1 of 2 registered predictions pass; one tie resolves | `69db4e6`, this merge |

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

## 6. A regulatory screen's hit cutoff, beside what the fly says a chip must see

**Slide message:** *EPA's MEA screen calls a firing-rate hit only beyond a change about six
times larger than the one at which the fly's behaviour had already failed.*

- **What changed.** `src/neuromps/fly/screen_cutoffs.py` reads the hit cutoff of the ToxCast MEA
  firing-rate endpoints from the asset's 20,375 ToxCast records and converts it to log2. The
  acute screen normalises activity to a positive control at −100 (activity abolished); its
  cutoff of
  70.97 <!-- metrics.json: fly_screen_cutoffs.acute_firing_rate.cutoff_percent -->
  means suppression to about 29% of control, a change of
  −1.784 <!-- metrics.json: fly_screen_cutoffs.acute_firing_rate.cutoff_log2.down --> log2,
  over
  515 <!-- metrics.json: fly_screen_cutoffs.acute_firing_rate.chemicals --> screened chemicals.
- **Why.** The review pointed out that most of the asset's records sat outside the modelling.
  This puts them to a use the fly study makes possible: does the screen's own threshold leave
  room to catch the mechanisms in time?
- **Effect.** The largest firing-rate change at which the fly's behaviour had already failed is
  0.304 <!-- metrics.json: fly_screen_cutoffs.largest_fly_change_at_behavioural_onset_log2 -->
  log2 (acetylcholine block), so the screen's cutoff is
  5.87 <!-- metrics.json: fly_screen_cutoffs.acute_cutoff_over_largest_fly_change --> times
  larger. Caveat: the cutoff applies to a fitted curve's maximum response across the tested range, not
  to one dose, so this compares magnitudes, not decisions. The fly side is simulated; the
  cutoffs are real.
- **Considered and dropped.** A compound-by-compound potency bridge between our spike-level
  sources and ToxCast: only four compounds appear in both (bicuculline, deltamethrin,
  acetaminophen, glyphosate), too few to rank.

## 7. Vesicle depletion does not explain why real chips slow down under GABA-A block

**Slide message:** *We tested the reviewer's best explanation for our twin's one wrong
direction, registered in advance, and it does not hold: depletion shrinks the firing-rate rise
but never turns it into the fall real chips show.*

- **What changed.** `src/neuromps/twin/depression_drugs.py` runs the four drug knobs and a
  five-step GABA-A dose series under each depression setting (Tsodyks–Markram depletion at the
  recurrent excitatory synapses), and scores a rule registered before the runs
  (`preregistration/twin_depression_under_block.json`).
- **Why.** The post-round-1 review proposed depletion as the missing mechanism behind three
  weak points at once: the twin's wrong firing-rate direction under GABA-A block (§14), its
  missed bicuculline dose trend (§18) and the cross-platform reversal (§11).
- **Effect.** Under strong GABA-A block the twin's firing rate rises by
  2.389 <!-- metrics.json: twin_depression_under_block.summary.control_mfr_delta_at_block -->
  log2 without depletion and by at least
  0.8687 <!-- metrics.json: twin_depression_under_block.summary.smallest_mfr_delta_at_block_with_depression -->
  with it; real GABA-A antagonist pairs change by
  −0.335 <!-- metrics.json: twin_depression_under_block.summary.real_gabaa_antagonist_mfr_delta -->.
  None of the
  7 <!-- metrics.json: twin_depression_under_block.summary.settings_tested --> settings qualifies,
  and burst rates do not fall with dose under any. Registered outcome: V1 fail, V2 pass, V3 not
  applicable. The weights were not recalibrated with depletion in place, so the mechanism is not
  ruled out, only depletion added to the present twin.
- **Evidence.** Report §18, row 9 and figure; `outputs/twin/depression_under_block_simulated.png`.

## 8. Trained on the twin alone, the classifier recognises real mechanisms

**Slide message:** *A classifier that never saw a real label recognises real drug mechanisms at
two and a half times chance; the same simulations do not improve a classifier trained on real
labels.*

- **What changed.** `src/neuromps/models/twin_prior.py` simulates 30 labelled pairs for each of
  the five mechanism classes the twin has a knob for, and tests two uses of them, registered
  before simulating (`preregistration/twin_prior_classifier.json`): as the centre of the
  classifier's penalty (the review's Fix 2), and as the only training data.
- **Effect.**
  - Trained on simulated pairs alone and applied to the 73 real pairs: balanced accuracy
    0.41 <!-- metrics.json: twin_prior_classifier.twin_alone.balanced_accuracy -->, interval
    0.2808 <!-- metrics.json: twin_prior_classifier.twin_alone.balanced_accuracy_ci95.0 --> to
    0.5072 <!-- metrics.json: twin_prior_classifier.twin_alone.balanced_accuracy_ci95.1 -->,
    against one in six by chance (pass).
  - As a prior: balanced accuracy
    0.53 <!-- metrics.json: twin_prior_classifier.twin_prior.balanced_accuracy_mean --> against
    the baseline's
    0.537 <!-- metrics.json: twin_prior_classifier.baseline.balanced_accuracy_mean -->, and
    conformal sets unchanged (two fails).
  - A diagnosis for free: real sodium-channel blockers are never recognised, because the
    twin's threshold knob moves firing by
    −3.273 <!-- metrics.json: twin_prior_classifier.twin_alone.na_channel_blocker_median_deltas.simulated.mfr_hz -->
    log2 where the real blockers move it by
    −0.795 <!-- metrics.json: twin_prior_classifier.twin_alone.na_channel_blocker_median_deltas.real.mfr_hz -->:
    right direction, wrong size.
- **Evidence.** Report §14, last paragraph.

## 9. Record the tract, not the tissue around it

**Slide message:** *On the simulated fly brain, electrodes on the output tract (the brain's own
microchannels) catch GABA block before behaviour fails; a planar array of the same size does
not.*

- **What changed.** `src/neuromps/fly/tract.py` reads the parent sweep's spontaneous recordings
  through three arrays: the full planar array, 64 of its electrodes, and 64 single-unit
  electrodes on descending neurons. The test, at one threshold set by real-chip noise, was
  registered before recording (`preregistration/fly_output_tract_electrode.json`).
- **Why.** The review's Investigation B: does instrumenting the connection between compartments
  reveal failure a flat array misses? Our real devices (§15) had already said that only
  electrodes inside microchannels see direction.
- **Effect.** All four registered predictions pass. Under GABA block, at behavioural onset, the
  tract's firing rate has moved by
  0.361 <!-- metrics.json: fly_output_tract.abs_delta_mfr_at_onset.tract_64.gaba_block --> log2
  against
  0.044 <!-- metrics.json: fly_output_tract.abs_delta_mfr_at_onset.planar_64.gaba_block --> for
  the equal-count planar array; the tract flags
  3 <!-- metrics.json: fly_output_tract.families_flagged_in_time.tract_64 --> families in time
  against
  2 <!-- metrics.json: fly_output_tract.families_flagged_in_time.planar_64 -->, with no false
  alarms. Checks added after the pass: over ten random draws of each array the tract is ahead
  in
  7 <!-- metrics.json: fly_output_tract_robustness.registered_index.draws_meeting_t1 --> of
  10 <!-- metrics.json: fly_output_tract_robustness.registered_index.draws -->, and the
  descending-neuron behaviour readout is not what drives it.
- **Evidence.** Report §24 ("Where to put the electrode") and figure; §20, fifth area.

## 10. One step closer to behaviour: the motor neurons

**Slide message:** *Read at the motor neurons that move the fly's mouthparts and legs, the chip
is never ahead of behaviour: it is late in three drug families and level in two.*

- **What changed.** `src/neuromps/fly/motor.py` repeats the male-CNS replication's behaviour
  trials (same seeds) and reads them at the proboscis motor pool (feeding and its bitter gate),
  the front-leg motor pool (antennal grooming) and all motor neurons (recruitment). Registered
  before any drug condition was run (`preregistration/fly_motor_readout.json`).
- **Why.** The behaviour readouts were firing rates of interneurons; the male connectome includes
  the nerve cord, so behaviour can be read where it leaves the nervous system. This is the
  first step toward a closed-loop, embodied fly, which needs a body and sensory feedback on top.
- **Effect.** Six of seven predictions pass. The interneuron readouts reproduce exactly; sugar
  drives the proboscis pool at
  31.405 <!-- metrics.json: fly_motor_readout.predictions.MR1_the_motor_pools_carry_the_behaviours.value.proboscis_sugar -->
  Hz and bitter cuts it to
  5.625 <!-- metrics.json: fly_motor_readout.predictions.MR1_the_motor_pools_carry_the_behaviours.value.proboscis_sugar_bitter -->;
  the chip tracks motor disruption (Spearman
  0.8922 <!-- metrics.json: fly_motor_readout.predictions.MR4_chip_tracks_motor_disruption.value -->).
  The chip-is-late prediction failed narrowly: late in
  3 <!-- metrics.json: fly_motor_readout.summary.families_chip_later_than_motor -->, tied in
  2 <!-- metrics.json: fly_motor_readout.summary.families_chip_tied_with_motor -->, earlier in
  0 <!-- metrics.json: fly_motor_readout.summary.families_chip_earlier_than_motor --> (four late
  were needed). Motor output fails no later than the interneurons in every family.
- **Evidence.** Report §25 ("One step closer to behaviour") and figure;
  `outputs/fly/malecns_motor_SIMULATED.csv`.

## 11. Resolving the two ties, without re-scoring the failed test

**Slide message:** *We did not move the goalposts on the failed test; we ran a new registered
one. With the ties resolved, the chip is late in four drug families, level in one, and never
ahead.*

- **What changed.** `src/neuromps/fly/tie_break.py` runs four finer dose levels inside each of
  the two tied steps of entry 10, with the same seeds and untreated references, registered
  before running (`preregistration/fly_motor_tie_break.json`).
- **Why.** Entry 10's chip-is-late test failed because two families tied, both onsets inside one
  coarse dose step. Changing the test after the fact would have undone the registration; a new,
  registered test with a finer grid answers the open question honestly.
- **Effect.** The re-run tied levels reproduce exactly. The raised threshold resolves to chip
  later (motor at
  0.32 <!-- metrics.json: fly_motor_tie_break.resolutions.excitability_block.motor_onset -->,
  chip at
  0.4 <!-- metrics.json: fly_motor_tie_break.resolutions.excitability_block.chip_onset -->);
  GABA potentiation stays tied. The follow-up predicted both would resolve, so it failed.
  Entry 10's test stays failed; combined, the chip is later in
  4 <!-- metrics.json: fly_motor_tie_break.combined_with_parent.chip_later -->, level in
  1 <!-- metrics.json: fly_motor_tie_break.combined_with_parent.tied -->, earlier in
  0 <!-- metrics.json: fly_motor_tie_break.combined_with_parent.chip_earlier -->.
- **Evidence.** Report §25 and figure; `outputs/fly/malecns_motor_tie_break_SIMULATED.csv`.

---

## Work queue (decided 2026-10-24, from the post-round-1 review)

| Item | Status | Registered before running |
|---|---|---|
| Synaptic depression under GABA-A block | done, entry 7 (failed as registered) | yes |
| ToxCast potency bridge | replaced by entry 6 (only four shared compounds) | — |
| Output-tract electrode | done, entry 9 (4 of 4 pass; 7 of 10 draws) | yes |
| Brain and nerve cord as a two-chamber chip (male CNS) | scoped out: one chamber pair is an anecdote, and FlyWire already tested 698 <!-- metrics.json: fly_chip_to_animal.direction.n_pairs --> region pairs (§25) | — |
| Twin posteriors as classifier features | done as a twin-fitted prior, entry 8 (the delta posterior infers one knob, outside Task 1's domain) | yes |
| Random-graph-to-connectome blend for organoids | dropped | — |
| Embodied, closed-loop fly (body, nerve cord to muscles, sensory feedback) | roadmap after the final; its first step, motor-neuron readouts, is entry 10 | — |

Why the blend was dropped: grafting fly wiring onto a model of a mammalian cortical culture has
no biological basis, and organoid–monolayer differences have many confounds, so a match would
prove nothing. The embodied fly is not dropped but scheduled: it adds new information only with
the loop closed (nerve cord to muscles, body, sensory feedback), which is a research project
for after the final; reading the motor neurons (entry 10) is its first step.
