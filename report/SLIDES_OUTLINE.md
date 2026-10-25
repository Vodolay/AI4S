# Slide outline: from chip to animal (post-submission work)

*Draft for the team's deck (Yi Nuo Zhao, Ivan T. Ivanov), 2026-10-25. One slide per row of the
story. Every number cites the `outputs/metrics.json` key that produced it, and every figure is a
script output labelled SIMULATED. Slide messages come from `report/POST_SUBMISSION_CHANGES.md`;
the caveat line on each slide is meant to be said aloud, not cut. All results are simulated; this
is research decision support, and nothing here is a claim about any compound in any animal or
person.*

## The arc in one sentence

A simulated fly lets the same network be read as a chip and as an animal, and at every level we
could build (named neurons, motor neurons, movement) the chip ranks drugs well but flags them
after behaviour has already changed.

## Slides

### 1. The question
- **Message:** a neural chip reports firing, not behaviour; how late is it?
- **Show:** the two readouts of one simulated brain, chip array and behaviour (schematic, no
  numbers).
- **Say:** the fly connectome makes the question answerable in simulation.

### 2. The chip ranks drugs correctly
- **Message:** the chip's firing-rate change tracks behavioural disruption (Spearman
  0.863 <!-- metrics.json: fly_chip_to_animal.predictions.C1_chip_tracks_disruption.value -->).
- **Figure:** `outputs/figures/fly_chip_to_animal_dose_SIMULATED.png`.
- **Caveat:** the drugs are transmitter-class knobs, not compounds.

### 3. ...but raises the alarm late
- **Message:** in every family behaviour fails at a lower dose than the chip first flags,
  replicated on fresh seeds.
- **Figure:** `outputs/figures/fly_sentinel_ladder_SIMULATED.png`.
- **Caveat:** the lateness is relative to the registered 0.5 log2 detection line.

### 4. Where to record
- **Message:** electrodes on the brain's output tract catch more in time than a planar array.
- **Figure:** `outputs/figures/fly_output_tract_SIMULATED.png`.
- **Caveat:** this advantage disappears at the movement level (slide 8).

### 5. Motor neurons
- **Message:** read at the motor neurons, the chip is later in
  4 <!-- metrics.json: fly_motor_tie_break.combined_with_parent.chip_later --> families and never
  earlier.
- **Figure:** `outputs/figures/fly_motor_onsets_SIMULATED.png`.

### 6. A body
- **Message:** an antennal touch in the brain brings the leg to the head through a second
  connectome (claw within
  0.5369 <!-- metrics.json: embodied_stage1f.predictions.F1_antennal_input_reaches_deepest.value.jon_ce -->
  mm of the antenna); sugar extends the proboscis, bitter holds it back
  (0.3054 <!-- metrics.json: embodied_stage2b.reported.sugar.mean_extension_mm --> against
  0.0488 <!-- metrics.json: embodied_stage2b.reported.sugar_bitter.mean_extension_mm --> mm).
- **Figure:** `outputs/figures/embodied_stage1f_depth_SIMULATED.png`,
  `outputs/figures/embodied_stage2b_proboscis_SIMULATED.png`.
- **Caveat:** one muscle-driven leg; the proboscis is servoed.

### 7. Drugs in the loop: the headline
- **Message:** with movement as the readout the chip is still late in every family: GABA-A
  block changes movement at
  0.2 <!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_block.movement -->
  of the knob's range and is flagged at
  0.6 <!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_block.chip -->.
- **Figure:** `outputs/figures/embodied_stage4a_drugs_movement_SIMULATED.png`.
- **Say:** at movement onset the chip has moved only
  0.149 <!-- metrics.json: embodied_stage4a_precision.abs_log2_mfr_change_at_movement_onset.gaba_block -->
  to 0.452 <!-- metrics.json: embodied_stage4a_precision.abs_log2_mfr_change_at_movement_onset.gaba_potentiate -->
  log2, within real-chip noise (exploratory).

### 8. We tried to break the headline
- **Message:** fresh seeds, the chip's own replication, each movement left out, a finer dose
  grid: the chip stays late in every family with the full index. In GABA-A potentiation the
  lateness rests on the bitter-gated proboscis; without it the two tie at
  0.6 <!-- metrics.json: embodied_stage4c.predictions.F2_without_the_bitter_gate_chip_later.value.movement_onset -->.
- **Figure:** a table of onsets per family from `outputs/fly/embodied_stage4b_SIMULATED.json`
  and `embodied_stage4c_SIMULATED.json` (to draw).
- **Caveat:** say the failed leave-one-out prediction out loud; it is the honest limit.

### 9. Walking: negatives and a reproduction
- **Message:** the connectome alone, as integrate-and-fire neurons, does not walk (extensors
  tonic at 54.12 <!-- metrics.json: embodied_stage2a.pool_rates_hz.forward_walking.L2_extensor -->
  Hz, flexors silent); a published rate model, with its own size rule (neuron volume), drives a
  front-leg rhythm on the male CNS
  (0.9446 <!-- metrics.json: embodied_stage2i.predictions.V1_the_walking_command_makes_a_leg_rhythm.value -->)
  that needs the published three-neuron circuit (silenced:
  0.0115 <!-- metrics.json: embodied_stage2i.predictions.V5_the_rhythm_needs_the_three_neuron_circuit.value.lesion -->).
- **Figure:** `outputs/figures/embodied_stage2i_front_leg_SIMULATED.png`.
- **Say:** shuffle the same volumes across neurons and the rhythm is gone
  (0.0021 <!-- metrics.json: embodied_stage2j.arms.volume_shuffled.scores.P1_the_walking_command_makes_a_leg_rhythm.value -->),
  so the identified neurons' sizes matter.
- **Caveat:** no flexor-extensor alternation; a coxa offset seen once did not replicate.

### 10. We audited our own measure
- **Message:** our rhythm measure scored smooth signals as rhythmic; a registered audit withdrew
  one earlier pass (forward walking
  0.0956 <!-- metrics.json: rhythm_audit.embodied_stage1d.predictions.D2_forward_walking_swings_the_leg.audited.value.forward_walking -->
  under the corrected measure) and left the rest standing.
- **Figure:** `outputs/figures/rhythm_audit_SIMULATED.png`.

### 11. Closing the loop: a documented negative
- **Message:** leg-sensor feedback changes the motor neurons but does not speed a pushed leg's
  return, on the front leg or the well-traced hind leg (resistance sooner than sham in
  14 <!-- metrics.json: embodied_stage3d.predictions.Q1_feedback_that_follows_the_push_speeds_the_return.value.seeds_favouring_resistance -->
  of 32 seeds).
- **Figure:** `outputs/figures/embodied_stage3c_hind_leg_SIMULATED.png`.
- **Say:** what would change it is a muscle-driven, well-traced leg, which no model at hand has.

### 12. Real flies
- **Message:** a registered literature search found
  0 <!-- metrics.json: embodied_stage5.predictions.V1_the_pharmacology_agrees.value.comparisons -->
  fly pharmacology studies on these movements; manipulating the identified 13A neurons as an
  experiment did shortens the simulated reach
  (0.3806 <!-- metrics.json: embodied_stage5b.predictions.I1_silencing_13A_reduces_grooming.value.depth_mm -->
  from 0.9927 <!-- metrics.json: embodied_stage5b.predictions.I0_the_reach_is_there.value --> mm),
  but read at the joint the posture does not match.
- **Say:** the gap lists the experiments that would test the model.

### 13. How it was done
- **Message:** every test registered before it ran, every outcome reported, every number
  produced by a script and checked against the text; one command reproduces the fly studies
  (embodied stage
  10884.5 <!-- metrics.json: run_all_timings_full.by_stage_s.embodied --> s, every value identical).
- **Show:** `preregistration/`, `report/DECISION_LOG.md`, the verifier's output.

## Not for the slides

Anything that reads as a statement about a compound, an animal or a person. The knobs are
transmitter-class scalings in a simulated fly.
