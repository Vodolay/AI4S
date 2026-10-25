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
| 11 | 2026-10-24 | The motor-level ties resolved on a finer dose grid | 1 of 2 registered predictions pass; one tie resolves | `69db4e6`, `796a33b` |
| 12 | 2026-10-24 | Roadmap to an embodied, closed-loop fly | plan, no new results | `42ee520` |
| 13 | 2026-10-24 | Embodied stage 1: the male CNS drives a muscle-driven leg | both registered tests fail; finding the cause changes the plan | `cbdd6d6` … this branch |
| 14 | 2026-10-24 | Embodied stage 1c: the male nerve cord from descending neurons, brain removed | 3 of 5 registered predictions pass | `739f6f0` … `e576cd9` |
| 15 | 2026-10-24 | Embodied stage 1d: a held reach against a rhythmic swing, fresh seeds | 4 of 4 registered predictions pass; D2 (the swing) fails under the audit's corrected measure (entry 22) | `66502eb` … `19ab0ae` |
| 16 | 2026-10-24 | Embodied stages 1e and 1f: the FlyWire brain drives the male nerve cord and the leg | 1e uninterpretable as registered; 1f 5 of 5 pass on fresh seeds | `f2f5ad3` … this branch |
| 17 | 2026-10-24 | Embodied stage 2a: fictive walking in the male nerve cord | 1 of 5 pass; no walking rhythm | `91c5b9e` … `ff43970` |
| 18 | 2026-10-24 | Embodied stage 2b: sugar extends the flybody proboscis, bitter holds it back | 6 of 6 registered predictions pass | `76f90e8` … `4b586c0` |
| 19 | 2026-10-24 | Embodied stage 2c: a registered adaptation sweep | stopped by its rule; no cell walks | `c4c29a1` … `5610ae8` |
| 20 | 2026-10-24 | Embodied stage 2d: the glutamate sign rule | 1 of 7 at the selected cell | `edbf611` … `a0ccff7` |
| 21 | 2026-10-24 | Embodied stage 2e: the Pugliese et al. rate model; a flaw in our rhythm measure | 2 of 8 (one vacuous); the size stand-in starves DNg100 | `21df803` … `7a13eba` |
| 22 | 2026-10-24 | The registered rhythm-measure audit | one earlier pass (1d D2) withdrawn; all else stands; originals reproduce | `76611c0` … this branch |
| 23 | 2026-10-24 | Embodied stage 2f: command neurons driven alike, two size rules | precondition fails in both; the command is silenced by the activity it starts | `7e36cfb` … this branch |
| 24 | 2026-10-24 | Embodied stage 2g: the published front-leg network, rebuilt from the male CNS | precondition fails narrowly (12 and 15 of 32); exploratory: a regular motor rhythm in the feasible draws | `00cefa1` … this branch |
| 25 | 2026-10-24 | Embodied stage 2h: the front-leg network with a fine input search | 4 of 7 (synapse-count sizes) and 5 of 7 (no size scaling) pass, one of them vacuously in each; a leg rhythm under DNg100; circuit-dependent without size scaling; no alternation | `c6d7874` … this branch |
| 26 | 2026-10-24 | Embodied stage 3a: the first closed loop (claw-neuron feedback, grooming leg) | 3 of 5; go/no-go fails: feedback shifts motor pools toward resistance but barely moves a held leg, and a sham does the same | `5cdeba9` … this branch |
| 27 | 2026-10-24 | Embodied stage 3b: a push on the tibia during the grooming hold | 2 of 5; the loop does not speed the return; the front leg's proprioceptors are thinly traced | `cd0a711` … this branch |
| 28 | 2026-10-24 | Embodied stage 4a: the drug design with movement as the readout | 3 of 6; the chip flags after movement fails in all five families; the tract no longer leads | `b1a49df` … this branch |
| 29 | 2026-10-24 | Embodied stage 5: simulated directions against published fly findings | no qualifying pharmacology found (both predictions uninterpretable); identified-neuron studies agree in 1 of 3 | `1723e0c` … this branch |
| 30 | 2026-10-24/25 | Release: one command reproduces every fly number; report Act VIII, paper draft, decision log | reproduction identical; precision at movement onset (exploratory); the stage-2d corrected cell fails on fresh seeds (2 of 7) | `f053de3` … this branch |
| 31 | 2026-10-25 | Embodied stage 5b: the identified 13A/13B leg neurons, manipulated | 2 of 5; silencing 13A shortens the reach as published; 13B and activation do not | `c20fa08` … this branch |
| 32 | 2026-10-25 | Embodied stage 5c: the 13A/13B manipulations read at the joint | 0 of 4; the model's inhibitory leg neurons brake rather than steer; stage 5b's agreement came by another route | `04a2387` … this branch |

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
  - A finding for free: real sodium-channel blockers are never recognised, because the
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

## 12. A roadmap, not a rushed demo: the embodied, closed-loop fly

**Slide message:** *Next, the simulated fly gets a body and its sensory feedback, so that "does
the chip flag a drug before behaviour fails?" can be asked of movement itself; six stages, each
with a registered go/no-go test.*

- **What changed.** `report/ROADMAP_EMBODIED_FLY.md` (Word and PDF beside it) sets out the plan:
  brain and nerve cord, motor neurons to muscles, muscles to joints in a physics body
  (NeuroMechFly v2 on MuJoCo), and joints back to proprioceptive neurons. A small script,
  `src/neuromps/fly/roadmap_inventory.py`, counts what the male connectome already provides: of
  381 <!-- metrics.json: fly_roadmap_inventory.leg_motor_neurons_total --> leg motor neurons,
  260 <!-- metrics.json: fly_roadmap_inventory.leg_motor_neurons_typed_by_muscle --> are typed
  by the muscle they drive, and
  1,030 <!-- metrics.json: fly_roadmap_inventory.vnc_proprioceptive_sensory_neurons -->
  proprioceptive sensory neurons can carry the feedback.
- **Why.** Building it properly takes weeks; showing a half-working body would cost credibility.
  A staged plan with registered tests shows where the project goes and how it will be checked.
- **Evidence.** `report/ROADMAP_EMBODIED_FLY.md`.

## 13. The leg moves, but not specifically: what stage 1 found

**Slide message:** *Wired to a muscle-driven leg, the male fly brain moves it toward the head
under antennal drive, but under sugar drive too; the cause is that the published model's
parameters do not carry over to the second connectome, where activity avalanches.*

- **What changed.** MuJoCo (an optional dependency) runs the FlyMimic leg from FlyGym 2;
  `src/neuromps/fly/embodied_stage1.py` maps male-CNS motor neurons onto its muscles by name and
  drives it open loop. Two registered tests: stage 1 and stage 1b (weight rescaled by the
  connectomes' input-synapse ratio).
- **Effect.** Stage 1 fails on specificity (sugar brings the claw closer than antennal drive);
  stage 1b's feeding check fails (sugar drives the feeding motor neuron at
  0.0 <!-- metrics.json: embodied_stage1b.predictions.B0_feeding_still_works.value.mn9_sugar_hz -->
  Hz), so its leg results are uninterpretable. The cause: a sugar drive activates
  251 <!-- metrics.json: fly_spread.flywire783.active_total --> neurons in the published FlyWire
  model and
  14,263 <!-- metrics.json: fly_spread.malecns_published.active_total --> in the male CNS.
- **What it changes.** The next step drives the male nerve cord from its descending neurons
  alone, then joins the validated FlyWire brain to it (roadmap §7).
- **Evidence.** `report/ROADMAP_EMBODIED_FLY.md` §7; `outputs/fly/embodied_stage1*_SIMULATED.*`.

## 14. The grooming neurons move the right leg

**Slide message:** *Driven from the descending neurons that start antennal grooming, the male
nerve cord moves the front leg, and only the front leg, to the head and holds it there; walking
neurons swing it instead.*

- **What changed.** `src/neuromps/fly/embodied_stage1c.py` cuts the male CNS to its nerve cord
  (brain removed), drives named descending neurons at 150 Hz, and moves the FlyMimic leg as in
  stage 1. Registered before running.
- **Effect.** Three of five predictions pass: silence without drive; the left front-leg motor
  pool at
  15.846 <!-- metrics.json: embodied_stage1c.predictions.C2_the_front_leg_is_the_one_that_moves.value.front -->
  Hz under grooming drive against
  2.72 <!-- metrics.json: embodied_stage1c.predictions.C2_the_front_leg_is_the_one_that_moves.value.middle -->
  and
  1.697 <!-- metrics.json: embodied_stage1c.predictions.C2_the_front_leg_is_the_one_that_moves.value.hind -->
  for the other legs; no other descending neuron recruits the front leg as strongly. The
  kinematic prediction failed: forward-walking drive dips nearly as close to the head
  (1.154 <!-- metrics.json: embodied_stage1c.predictions.C1_grooming_neurons_bring_the_claw_to_the_head.value.forward_walking -->
  mm against
  1.1345 <!-- metrics.json: embodied_stage1c.predictions.C1_grooming_neurons_bring_the_claw_to_the_head.value.grooming -->)
  at the bottom of a rhythmic swing. A sustained reach and a swing need a different readout,
  to be registered as stage 1d.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage1c_SIMULATED.*`.

## 15. The fly's grooming neurons hold its leg at its head

**Slide message:** *On fresh seeds and with readouts registered in advance: driving the fly's
antennal-grooming neurons brings the simulated front leg to the head and holds it there for
nearly the whole second; escape neurons extend it. (That walking neurons "swing" it did not
survive the rhythm-measure audit; see entry 22.)*

- **What changed.** `src/neuromps/fly/kinematics.py` adds two readouts (time near the head;
  rhythmicity), `src/neuromps/fly/embodied_stage1d.py` reruns stage 1c's design on fresh seeds.
  The readouts were suggested by stage 1c's traces, so they were registered before this run
  (`preregistration/embodied_stage1d_reach_versus_swing.json`) and tested out of sample.
- **Effect.** All four predictions pass: time near the head
  0.9665 <!-- metrics.json: embodied_stage1d.predictions.D1_grooming_holds_the_claw_at_the_head.value.grooming -->
  under grooming drive against at most
  0.368 <!-- metrics.json: embodied_stage1d.predictions.D1_grooming_holds_the_claw_at_the_head.value.forward_walking -->
  under any other; forward walking more rhythmic than grooming; the same at every activation
  scale; the front-leg specificity of stage 1c replicates. Stage 1c's failed predictions stay
  failed. *Found later (entries 21, 22): the rhythmicity measure had no peak requirement; under
  the registered audit's corrected measure D2 fails (forward walking
  0.0956 <!-- metrics.json: rhythm_audit.embodied_stage1d.predictions.D2_forward_walking_swings_the_leg.audited.value.forward_walking -->); the hold at the head (D1) does not depend on it.*
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage1d_SIMULATED.*`.

## 16. From an antennal touch in the brain to a leg at the head

**Slide message:** *Antennal input to the validated FlyWire brain, relayed through matched
descending neurons into the male nerve cord, moves a muscle-driven leg to the head; sugar input
does not, and removing four grooming neurons abolishes the movement.*

- **What changed.** `src/neuromps/fly/model.py` can replay recorded spike trains into chosen
  neurons; `src/neuromps/fly/relay.py` pairs descending neurons across the two connectomes by
  cell type and side and joins the FlyWire brain to the male nerve cord; stage 1e and stage 1f
  were each registered before running.
- **Effect.** Stage 1e's absolute precondition failed (grooming descending neurons at
  8.531 <!-- metrics.json: embodied_stage1e.predictions.E0_the_grooming_neurons_are_reached.value -->
  Hz against a 10 Hz line), so its leg results are uninterpretable. Stage 1f, on fresh seeds,
  passes all five predictions: claw to
  0.5369 <!-- metrics.json: embodied_stage1f.predictions.F1_antennal_input_reaches_deepest.value.jon_ce -->
  mm under antennal input against
  0.9377 <!-- metrics.json: embodied_stage1f.predictions.F1_antennal_input_reaches_deepest.value.sugar -->
  under sugar; back to
  1.4793 <!-- metrics.json: embodied_stage1f.predictions.F4_the_reach_runs_through_the_grooming_neurons.value.lesioned -->
  mm when aDN1 and aDN2 are removed.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage1f_SIMULATED.*`.

---

## 17. The nerve cord alone does not walk

**Slide message:** *Driven by its walking command, the connectome-only nerve cord fires the leg
extensors tonically and leaves the flexors silent: no rhythm. Rhythm needs intrinsic currents
the connectome does not record; adding them is the next registered step.*

- **What changed.** `src/neuromps/fly/fictive.py` runs stage 2a (registered before running):
  the six legs' tibia flexor and extensor pools under walking, grooming and backward-walking
  commands, phases measured between pools. `src/neuromps/fetch/flybody.py` fetches the flybody
  whole-body model (Apache-2.0) for the feeding and walking body, each file checked against its
  git hash.
- **Effect.** Silent without drive (W0 passes); under the walking command the left middle-leg
  extensors fire at 54.12 <!-- metrics.json: embodied_stage2a.pool_rates_hz.forward_walking.L2_extensor --> Hz and the left
  front flexors at 0.0 <!-- metrics.json: embodied_stage2a.pool_rates_hz.forward_walking.L1_flexor --> Hz, so there is no
  rhythm and no phase (W1 to W4 fail). The roadmap listed this risk and the response to it.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage2a_SIMULATED.*`.

---

## 18. Sugar extends the proboscis, bitter holds it back

**Slide message:** *Sugar input to the validated FlyWire brain drives the motor neurons of each
proboscis muscle the literature names, and a simulated proboscis extends fully; bitter input
holds it back. Six of six registered predictions pass.*

- **What changed.** `src/neuromps/fly/proboscis.py` drives the flybody proboscis (body fixed,
  0.01 ms step, checked against a step half that size); `src/neuromps/fly/embodied_stage2b.py`
  turns the proboscis motor pools (MN9, MN1/MN2, MN4, MN3, named by FlyWire cell type after
  McKellar et al. 2020) into rostrum and haustellum commands. Registered before running.
- **Effect.** Full extension under sugar
  (0.3431 <!-- metrics.json: embodied_stage2b.predictions.G1_sugar_extends_the_proboscis.value.sugar --> mm of
  0.3428 <!-- metrics.json: embodied_stage2b.predictions.G1_sugar_extends_the_proboscis.value.e_full -->); none under bitter,
  antennal input or none; time-averaged extension from
  0.3054 <!-- metrics.json: embodied_stage2b.reported.sugar.mean_extension_mm --> to
  0.0488 <!-- metrics.json: embodied_stage2b.reported.sugar_bitter.mean_extension_mm --> mm when bitter is added. Sugar
  recruits only feeding motor neurons (exploratory).
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage2b_SIMULATED.*`.

---

## 19. A registered sweep that found nothing, and why

**Slide message:** *We swept the one cell property the roadmap named, reported every setting, and
none made the nerve cord walk. The wiring explains it: the walking command's active
neurons inhibit the flexors about twice as much as they excite them.*

- **What changed.** `src/neuromps/fly/model.py` gains spike-frequency adaptation (off by default;
  with it off the spikes are identical to the published model's). `connectome.rewire` builds a
  degree-preserving control. `src/neuromps/fly/embodied_stage2c.py` runs the registered sweep,
  selection rule and confirmation. `src/neuromps/fly/flexor_diagnostic.py` is the exploratory
  follow-up.
- **Effect.** 0 <!-- metrics.json: embodied_stage2c.sweep.cells_passing.W1_the_walking_command_makes_a_rhythm -->
  of 36 <!-- metrics.json: embodied_stage2c.sweep.grid_size --> cells produce a rhythm; the stage stops as
  registered. Flexors fire at 151.341 <!-- metrics.json: embodied_stage2c_diagnostic.pool_rates_hz.no_inhibition.flexor --> Hz once
  inhibition is removed; from the command's active neurons they receive
  274.73 <!-- metrics.json: embodied_stage2c_diagnostic.flexor_inputs.from_active.inhibitory_per_neuron --> inhibitory against
  130.78 <!-- metrics.json: embodied_stage2c_diagnostic.flexor_inputs.from_active.excitatory_per_neuron --> excitatory synapses each
  (exploratory).
- **Why it matters for the interview.** It shows the discipline holding when the result is null:
  the sweep was fixed before running, reported whole, and followed by a mechanism, not by
  re-tuning until it walked.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage2c_*`.

---

## 20. The glutamate sign is not the missing piece

**Slide message:** *Making glutamate excitatory switches the silent flexors on, but the cord then
fires as one slow wave, not as alternating legs. Three registered attempts agree that this
wiring, in this neuron model, has no walking rhythm generator. That negative result is
reportable.*

- **What changed.** `connectome.make_excitatory`; `src/neuromps/fly/embodied_stage2d.py` builds
  two registered bounds on where glutamate excites (onto motor neurons; everywhere), each
  without adaptation and over stage 2c's grid, through stage 2c's code (a rerun of stage 2c is
  byte-identical).
- **Effect.** Onto motor neurons: flexors 1.76 <!-- metrics.json: embodied_stage2d.sweep.cells.0.flexor_hz --> Hz, no cell passes a
  walking test. Everywhere: flexors 40.4 <!-- metrics.json: embodied_stage2d.sweep.cells.37.flexor_hz --> Hz with
  7,118 <!-- metrics.json: embodied_stage2d.sweep.cells.37.active_neurons --> active neurons, rhythm in
  5 <!-- metrics.json: embodied_stage2d.sweep.cells_passing.glut_all.W1_the_walking_command_makes_a_rhythm --> cells, alternation
  in none; the selected cell passes only silence without drive on fresh seeds.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage2d_*`.

---

## 21. The walking flies online are trained, not wired; and our rhythm measure was flawed

**Slide message:** *The walking flies circulating online are driven by trained controllers, not by
connectome wiring. The one wiring-based walking rhythm (Pugliese et al. 2025) uses a firing-rate
model. Our reimplementation did not test it fairly: our stand-in for neuron size starves the
walking command. Checking the result exposed a flaw in our own rhythm measure, now being
audited.*

- **What changed.** `src/neuromps/fly/rate_model.py` reimplements Pugliese et al.'s rate model
  from the paper and its public configuration, batched over parameter draws.
  `src/neuromps/fly/embodied_stage2e.py` runs the registered stage 2e.
- **Effect.** Under DNg100 drive, 11 <!-- metrics.json: embodied_stage2e.reported.forward_walking.active_neurons --> neurons
  become active; DNg100 itself reaches 6.54 <!-- metrics.json: embodied_stage2e.exploratory_driven_neurons.forward_walking.drive_only_rate_hz.0 --> Hz,
  because size (here synapse count) sets its threshold. Rhythm predictions fail; the lesion
  prediction passes vacuously. DNb08's "rhythm" (0.7367 <!-- metrics.json: embodied_stage2e.reported.dnb08.rhythmicity_L1_flexor -->)
  is one burst and a plateau. The registered rhythmicity measure has no peak requirement; the
  registered audit (entry 22) re-examined every stage that used it.
- **Why it matters for the interview.** We publish the flaw ourselves, say which earlier claims
  it touches, and fix it with a registered audit, not a quiet re-score.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage2e_*`.

---

## 22. The rhythm-measure audit: one registered pass did not survive

**Slide message:** *We found a flaw in our own rhythm measure, registered an audit before
re-scoring anything, and re-ran every affected stage. One earlier pass (the walking leg's
"swing") does not survive. Every other registered outcome stands, and the original numbers
reproduce exactly.*

- **What changed.** `kinematics.rhythm` requires a genuine autocorrelation peak; `fictive.phase`
  and the period measures take a method switch (the registered measure stays the default);
  `src/neuromps/fly/rhythm_audit.py` re-simulates stages 1d, 2a, 2c, 2d and 2e on their own seeds
  and measures every trial both ways. Registered in `preregistration/rhythm_measure_audit.json`
  before any corrected value was computed on stage data.
- **Effect.** The registered measure reproduces every recorded outcome. Under the corrected one,
  stage 1d's D2 fails (forward walking 0.0956 <!-- metrics.json: rhythm_audit.embodied_stage1d.predictions.D2_forward_walking_swings_the_leg.audited.value.forward_walking -->, grooming
  0.2374 <!-- metrics.json: rhythm_audit.embodied_stage1d.predictions.D2_forward_walking_swings_the_leg.audited.value.grooming -->); stage 2d's rhythm cells drop from
  5 <!-- metrics.json: rhythm_audit.embodied_stage2d.registered.cells_passing.glut_all.W1_the_walking_command_makes_a_rhythm --> to
  1 <!-- metrics.json: rhythm_audit.embodied_stage2d.audited.cells_passing.glut_all.W1_the_walking_command_makes_a_rhythm -->;
  DNb08's "rhythm" in stage 2e falls to
  0.0 <!-- metrics.json: rhythm_audit.embodied_stage2e.reported.audited.dnb08.rhythmicity_L1_flexor -->. Every other registered
  outcome is unchanged.
- **Why it matters for the interview.** A worked example of the discipline under pressure: the
  flaw found by ourselves, the fix registered before use, the earlier claim withdrawn in public.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/rhythm_audit_SIMULATED.json`.

---

## 23. The walking command is switched off by the activity it starts

**Slide message:** *Driven fairly, the walking command neuron starts a wave of activity that grows
until it silences the command itself; the whole nerve cord then runs on its own. The published
model avoids this by simulating only the front-leg network and capping recruitment at 500
neurons. That is what we test next.*

- **What changed.** `rate_model.drive_input` gives each driven neuron the input that alone
  brings it to a target rate; `src/neuromps/fly/embodied_stage2f.py` runs the registered stage
  2f (two size rules, corrected rhythm measure, a precondition that the command is on, a lesion
  test that counts only if a rhythm exists).
- **Effect.** The precondition fails in both size rules (DNg100
  0.18 <!-- metrics.json: embodied_stage2f.synapse_count.predictions.S0_the_command_is_on.value --> and
  0.0 <!-- metrics.json: embodied_stage2f.none.predictions.S0_the_command_is_on.value --> Hz), so the other predictions are
  uninterpretable. Exploratory trace: DNg100 at
  42.67 <!-- metrics.json: embodied_stage2f_diagnostic.synapse_count.t_50ms.dng100_hz --> Hz after 50 ms, silenced by 0.25 s under
  -4452.5 <!-- metrics.json: embodied_stage2f_diagnostic.synapse_count.t_250ms.inhibitory_input --> inhibitory input;
  5,247 <!-- metrics.json: embodied_stage2f_diagnostic.synapse_count.t_1850ms.active_neurons --> neurons stay active without it.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage2f_*`.

---

## 24. The published front-leg network, rebuilt from the male CNS

**Slide message:** *Rebuilt as published, the male fly's front-leg network does produce a regular
motor rhythm, but too few runs found a working input for the registered test to count. The
network is nearly all-or-none, so the published input rule overshoots it. A finer input search
is next.*

- **What changed.** `src/neuromps/fly/embodied_stage2g.py` builds Pugliese et al.'s front-leg
  network from the male nerve cord (2,133 <!-- metrics.json: embodied_stage2g.network.neurons --> neurons) and applies their
  recruitment rule, with the corrected rhythm measure.
- **Effect.** Precondition failed narrowly
  (12 <!-- metrics.json: embodied_stage2g.synapse_count.predictions.P0_the_rule_finds_a_working_input.value --> and
  15 <!-- metrics.json: embodied_stage2g.none.predictions.P0_the_rule_finds_a_working_input.value --> of 32 draws; 16 needed), so
  the predictions are uninterpretable. Exploratory: in feasible draws, a regular low-rate
  oscillation in one or two motor-neuron pools (score
  0.8747 <!-- metrics.json: embodied_stage2g.none.reported.forward_walking.median_score -->, period
  154.0 <!-- metrics.json: embodied_stage2g.none.reported.forward_walking.median_period_ms --> ms without size scaling), removed
  by silencing the three-neuron circuit in that size rule only.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage2g_SIMULATED.*`.

---

## 25. The published walking rhythm, reproduced on the male CNS, with its limits

**Slide message:** *In our own reimplementation on a different fly's connectome, the walking
command drives a front-leg motor rhythm at a walking fly's tempo. Without size scaling it needs
the published three-neuron circuit, and not just as a relay. But it still does not alternate
antagonist muscles.*

- **What changed.** `src/neuromps/fly/embodied_stage2h.py`: stage 2g's network with a geometric
  bisection of the input; `--check` adds the exploratory silencing variants.
- **Effect.** A working input in 32 <!-- metrics.json: embodied_stage2h.synapse_count.predictions.H0_a_working_input_exists.value --> and
  18 <!-- metrics.json: embodied_stage2h.none.predictions.H0_a_working_input_exists.value --> of 32 draws; a leg rhythm in both size
  rules (0.8886 <!-- metrics.json: embodied_stage2h.none.predictions.H1_the_walking_command_makes_a_leg_rhythm.value --> without
  size scaling, period 196.5 <!-- metrics.json: embodied_stage2h.none.predictions.H6_a_fly_like_step_period.value --> ms), removed
  by silencing the circuit only without size scaling. Exploratory: keeping the relay and
  silencing the other two neurons also removes it
  (0.0483 <!-- metrics.json: embodied_stage2h_check.none.keep_relay_silence_INXXX466_IN16B036.median_score -->). No reliable
  promotor-remotor offset, no tibia alternation; the grooming comparison passes vacuously.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage2h_*`.

---

## 26. The first closed loop: brain-free nerve cord, leg and leg sensors

**Slide message:** *The simulated nerve cord now moves a leg and feels it move. Feedback from the
leg's position sensors shifts the motor neurons toward resisting the movement, as insect
reflexes do. In a held posture that barely moves the leg, and blind input does the same. A
perturbation test is next.*

- **What changed.** `body.Session` steps the leg; `model.simulate` takes an `extend` hook;
  `src/neuromps/fly/closed_loop.py` couples nerve cord and leg every millisecond and drives
  sensory neurons from a joint angle; `src/neuromps/fly/embodied_stage3a.py` runs the registered
  arms; `--check` the exploratory activation-scale check.
- **Effect.** Go/no-go failed: tibia shift
  0.0054 <!-- metrics.json: embodied_stage3a.predictions.L1_feedback_resists_the_extension.value.shift --> rad (0.1 needed);
  3 of 5 pass (silence, no resistance from the opposite assignment, the reach survives).
  Motor level: extensor
  43.25 <!-- metrics.json: embodied_stage3a.reported.open.grooming.tibia_extensor_hz --> to
  31.75 <!-- metrics.json: embodied_stage3a.reported.resistance.grooming.tibia_extensor_hz --> Hz under resistance, matched by the sham.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage3a_*`.

---

## 27. A push on the leg: the loop does not push back

**Slide message:** *We pushed the simulated leg and timed its return. The position sensors'
feedback did not bring it back faster: the muscles alone set the return. The front leg's sensors
are thinly traced in this connectome, about one fifth of the hind leg's. That is the limit to
state, and to fix with a better-traced leg.*

- **What changed.** `body.Session.step` accepts joint torques; `closed_loop.Loop` takes a
  perturbation window; `src/neuromps/fly/embodied_stage3b.py` runs the registered perturbation
  test; `--check` counts proprioceptors per leg.
- **Effect.** Push flexes the tibia 0.628 <!-- metrics.json: embodied_stage3b.predictions.M0_the_push_moves_the_leg.value --> rad;
  return 32.0 <!-- metrics.json: embodied_stage3b.predictions.M1_feedback_speeds_the_return.value.resistance --> ms with
  resistance feedback against 30.0 <!-- metrics.json: embodied_stage3b.predictions.M1_feedback_speeds_the_return.value.open -->
  open loop (2 of 5 pass; contingency uninterpretable). Left front leg: 4 <!-- metrics.json: embodied_stage3b_check.chordotonal_by_leg.front_L.claw -->
  claw neurons; left hind leg: 22 <!-- metrics.json: embodied_stage3b_check.chordotonal_by_leg.hind_L.claw -->.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage3b_*`.

---

## 28. Drugs in the loop: the chip is still late when the readout is movement

**Slide message:** *We reran the drug study with the simulated fly's own movements as the
behaviour. In every drug family, feeding and grooming change at a lower dose than the chip first
notices. The chip is late at every level we have looked at: neurons, motor neurons, and now
movement.*

- **What changed.** The stage-2b and relay trials take a drug knob (stage 2b reruns
  byte-identical); `src/neuromps/fly/embodied_stage4a.py` runs the parent's 22 conditions on the
  proboscis and the grooming reach and compares each family's movement onset with the recorded
  chip and tract detections.
- **Effect.** 3 of 6 pass: the movements are there under vehicle, acetylcholine block stops them,
  and movement never fails before the neural readouts. The chip flags after movement fails in all
  five families (GABA-A block: movement 0.2 <!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_block.movement -->,
  chip 0.6 <!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_block.chip -->); the tract no longer leads (4 families each); the bitter
  gate's release is not monotonic.
- **Evidence.** Roadmap §7 and figure; `outputs/fly/embodied_stage4a_*`.

---

## 29. Checking against real flies: the experiments have not been done

**Slide message:** *We registered how we would search the literature before searching, so we
could not pick the papers that agree. No published study gives flies these drugs and scores
feeding or grooming movements. The model makes predictions that real experiments have yet to
test.*

- **What changed.** `preregistration/embodied_stage5_real_fly_directions.json` (queries,
  inclusion, mapping, comparison rule); `src/neuromps/fly/real_fly_literature.json` (every
  candidate with its classification); `src/neuromps/fly/embodied_stage5.py` scores the
  comparison against the stage-4a medians.
- **Effect.** 0 <!-- metrics.json: embodied_stage5.predictions.V1_the_pharmacology_agrees.value.comparisons --> pharmacological
  or receptor-genetic comparisons, so both predictions are uninterpretable. Identified-neuron
  studies: the simulation agrees in 1 <!-- metrics.json: embodied_stage5.transmitter_neuron.agree --> of
  3 <!-- metrics.json: embodied_stage5.transmitter_neuron.comparisons --> (agreement for silencing GABAergic grooming neurons;
  mismatches explained by class-wide knobs and the absence of satiety).
- **Evidence.** Roadmap §7; `outputs/fly/embodied_stage5_SIMULATED.json`.

---

## 30. One command reproduces every fly number; the report, a paper draft and a decision log

**Slide message:** *Every fly result, from the first brain simulation to the closed loop, now
reruns with one command and comes back identical. The technical report carries the embodied fly,
a paper draft exists, and every choice of path is logged so other branches can be explored.*

- **What changed.** `src/neuromps/fly/pipeline.py` and the `embodied` stage of `run_all.py`;
  Act VIII of `report/TECHNICAL_REPORT.md`; `report/PAPER_DRAFT.md`; `report/DECISION_LOG.md`;
  the README's embodied-fly section.
- **Effect.** A from-scratch run took
  10884.5 <!-- metrics.json: run_all_timings_full.by_stage_s.embodied --> s and reproduced every recorded value.
  Exploratory: at movement onset the chip has moved
  0.149 <!-- metrics.json: embodied_stage4a_precision.abs_log2_mfr_change_at_movement_onset.gaba_block --> to
  0.452 <!-- metrics.json: embodied_stage4a_precision.abs_log2_mfr_change_at_movement_onset.gaba_potentiate --> log2. The stage-2d cell
  the corrected measure selects fails on fresh seeds (rhythm
  0.2035 <!-- metrics.json: embodied_stage2d_confirm.predictions.E1_the_walking_command_makes_a_rhythm.value.forward_walking -->).
- **Evidence.** Roadmap §7; `outputs/fly/embodied_stage4a_precision_SIMULATED.json`,
  `outputs/fly/embodied_stage2d_confirm_SIMULATED.json`.

---

## 31. Manipulating the same neurons as the experiment: one of four directions reproduced

**Slide message:** *Silencing the fly's GABAergic 13A leg neurons in the simulation shortens the
grooming reach, as silencing them reduces grooming in real flies. The other three published
manipulations do not show in the model's reach, which points to what the model still lacks.*

- **What changed.** `src/neuromps/fly/embodied_stage5b.py` silences (outputs removed) or
  activates (150 Hz) the left front leg's 13A or 13B neurons in the grooming relay, registered in
  `preregistration/embodied_stage5b_identified_neurons.json` before running.
- **Effect.** 2 of 5 pass: the reach is there; silencing 13A shortens it to
  0.3806 <!-- metrics.json: embodied_stage5b.predictions.I1_silencing_13A_reduces_grooming.value.depth_mm --> mm from
  0.9927 <!-- metrics.json: embodied_stage5b.predictions.I0_the_reach_is_there.value -->; 13B leaves it unchanged; activating 13A shortens it
  only to 0.7309 <!-- metrics.json: embodied_stage5b.predictions.I3_activating_13A_reduces_grooming.value.depth_mm --> mm.
- **Evidence.** Roadmap §7; `outputs/fly/embodied_stage5b_SIMULATED.*`.

---

## 32. At the joint, the model's leg neurons brake rather than steer

**Slide message:** *Read at the knee, none of the published leg postures appears when the
simulated 13A/13B neurons are silenced or activated, and the one earlier agreement came by
another route. The model's inhibitory leg neurons act as broad brakes, not joint-specific
controllers: a stated limit of the connectome-only nerve cord.*

- **What changed.** `src/neuromps/fly/embodied_stage5c.py` reads stage 5b's manipulations at
  every front-leg joint, registered in
  `preregistration/embodied_stage5c_identified_neurons_joints.json` before running.
- **Effect.** 0 of 4: silencing 13A extends the tibia
  (-0.1488 <!-- metrics.json: embodied_stage5c.predictions.J1_silencing_13A_flexes_the_tibia.value.shift_rad --> rad) instead of flexing it;
  activating 13A relaxes it toward rest
  (+0.8325 <!-- metrics.json: embodied_stage5c.predictions.J3_activating_13A_extends_the_tibia.value.shift_rad --> rad) instead of
  over-extending it.
- **Evidence.** Roadmap §7; `outputs/fly/embodied_stage5c_SIMULATED.*`.

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
| Embodied, closed-loop fly (body, nerve cord to muscles, sensory feedback) | roadmap after the final (`report/ROADMAP_EMBODIED_FLY.md`, entry 12); its first step, motor-neuron readouts, is entry 10 | — |

Why the blend was dropped: grafting fly wiring onto a model of a mammalian cortical culture has
no biological basis, and organoid–monolayer differences have many confounds, so a match would
prove nothing. The embodied fly is not dropped but scheduled: it adds new information only with
the loop closed (nerve cord to muscles, body, sensory feedback), which is a research project
for after the final; reading the motor neurons (entry 10) is its first step.
