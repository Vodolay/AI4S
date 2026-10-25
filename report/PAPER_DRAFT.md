# Neural organ-on-a-chip readouts flag drugs after behaviour fails: evidence from a connectome-constrained, embodied fly

*Draft, 2026-10-24. Authors and author order to be decided by the team (Yi Nuo Zhao, Ivan T.
Ivanov). Every number cites the `outputs/metrics.json` key that produced it and is checked by
`python -m neuromps.report.verify`; every study was registered before it ran
(`preregistration/`). All results are from simulations; this is research decision support, and
nothing here is a claim about any compound in any animal or person.*

## Abstract

Neural microelectrode-array (MEA) chips are used to screen compounds for neuroactivity, but a
chip cannot report what its readout would mean for a behaving animal. We ask that question in
a simulated fly, where the same network can be read as a chip and as an animal. Transmitter-class
drug knobs were applied to a connectome-constrained whole-brain model (FlyWire), read by a
virtual MEA through the same feature extractor used for real recordings, and compared with the
fly's behaviour read first as named neurons, then as motor neurons, and finally as movements of a
simulated body driven through a second connectome (the male CNS). The chip's firing rate ranks
drugs by how much they disrupt behaviour (Spearman
0.863 <!-- metrics.json: fly_chip_to_animal.predictions.C1_chip_tracks_disruption.value -->),
but at every level the chip first flags a drug at a higher dose than behaviour fails. With
movement as the readout, GABA-A block changes the simulated fly's feeding and grooming at
0.2 <!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_block.movement -->
of the knob's range, and the chip flags it at 0.6
<!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_block.chip -->.
Recording the brain's output tract rather than the tissue helps at the neural level and not at
the movement level. Along the way, a published walking rhythm generator was reproduced on a second
connectome, a flaw in our own rhythm measure was found and audited in the open, and a registered
literature search found no fly pharmacology on these behaviours to check the model against.

## 1. Introduction

MEA chips record the firing of cultured neural networks and are used to rank compounds for
neuroactivity and to flag hazards. Their readouts (mean firing rate, bursts, synchrony) are
features of a network, not of behaviour, and how a change in a chip feature relates to a change
in what an animal does is usually unknown: the culture has no body, and the animal is not
recorded the way the chip is. Whole-brain connectomes of *Drosophila* and models built on them
(Shiu et al. 2024) make the question answerable in simulation: the same network can be recorded
by a virtual array and read out as behaviour, under the same perturbation. We use that to ask
whether a chip-like readout flags a drug before, or after, the behaviour it stands for has
changed, and we follow the question from named neurons to motor neurons to the movements of a
simulated body. Every test was registered before it ran, failures are reported beside passes, and
every number in this paper is produced by a script and checked against the text.

## 2. Results

### 2.1 The chip ranks drugs correctly and raises the alarm late

Five transmitter-class knobs (GABA-A block and potentiation, GluCl block, acetylcholine block and
an excitability block), each at registered doses, were applied to the published leaky
integrate-and-fire model on FlyWire. A virtual array recorded the network as a chip and was
scored by the project's single MEA feature extractor; behaviour was read from the neurons the
published model validated against feeding and grooming. The chip's change in firing rate tracked
behavioural disruption across conditions (Spearman
0.863 <!-- metrics.json: fly_chip_to_animal.predictions.C1_chip_tracks_disruption.value -->),
and a second connectome, the male CNS (165,122 <!-- metrics.json: fly_models.malecns.neurons -->
neurons), replicated the ranking
(0.878 <!-- metrics.json: fly_malecns_replication.predictions.M_C1.value -->). But in every
family behaviour failed at a lower dose than the registered chip threshold detected: at
behavioural onset, GABA-A block had moved the chip's firing rate by only
0.149 <!-- metrics.json: fly_confirm_network_bursts.precision.abs_log2_mfr_change_at_behavioural_onset.flywire_sweep.gaba_block -->
log2, below the noise of a real chip's control recordings
(0.156 <!-- metrics.json: fly_confirm_network_bursts.precision.real_chip_noise.kapucu2022_control.q95_abs_log2 -->).

### 2.2 Where to record

Electrodes placed on the brain's output tract (its descending neurons) move more at behavioural
onset than a planar array of the same size (GABA-A block:
0.361 <!-- metrics.json: fly_output_tract.abs_delta_mfr_at_onset.tract_64.gaba_block --> against
0.044 <!-- metrics.json: fly_output_tract.abs_delta_mfr_at_onset.planar_64.gaba_block --> log2)
and flag in time in more families
(3 <!-- metrics.json: fly_output_tract.families_flagged_in_time.tract_64 --> against
2 <!-- metrics.json: fly_output_tract.families_flagged_in_time.planar_64 -->).

### 2.3 Motor neurons

Read at the male CNS's motor neurons, the chip is later than motor output in
4 <!-- metrics.json: fly_motor_tie_break.combined_with_parent.chip_later --> families, level in
1 <!-- metrics.json: fly_motor_tie_break.combined_with_parent.tied -->, and never earlier
(0 <!-- metrics.json: fly_motor_tie_break.combined_with_parent.chip_earlier --> families).

### 2.4 A body

The male nerve cord drives a muscle-driven front leg (FlyMimic) through a motor-neuron-to-muscle
map built from published names, and the FlyWire brain drives the proboscis of a whole-fly model
(flybody) through the motor neurons of each proboscis muscle. Joining the two connectomes by
replaying the FlyWire brain's descending spikes into the male nerve cord, antennal input brings
the claw to
0.5369 <!-- metrics.json: embodied_stage1f.predictions.F1_antennal_input_reaches_deepest.value.jon_ce -->
mm of the antenna, and removing two descending neuron types (`DNg62`, `DNge078`) abolishes the
reach (1.4793 <!-- metrics.json: embodied_stage1f.predictions.F4_the_reach_runs_through_the_grooming_neurons.value.lesioned -->
mm). Sugar input extends the proboscis fully
(0.3431 <!-- metrics.json: embodied_stage2b.predictions.G1_sugar_extends_the_proboscis.value.sugar -->
mm) and bitter input holds it back
(time-averaged 0.0488 <!-- metrics.json: embodied_stage2b.reported.sugar_bitter.mean_extension_mm -->
against 0.3054 <!-- metrics.json: embodied_stage2b.reported.sugar.mean_extension_mm --> mm).

### 2.5 With movement as the readout, the chip is still late

Rerunning the drug design with these movements as the behaviour gives the same answer as the
neural and motor readouts: in every family the chip first flags a drug at a higher dose than
movement fails (GABA-A potentiation: movement at
0.5 <!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_potentiate.movement -->,
chip at 1.0 <!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_potentiate.chip -->).
Movement never fails before the neural readouts; the body buffers small neural changes but not
enough for the chip to catch up. At the dose where movement first fails, the chip has moved by
only 0.149 <!-- metrics.json: embodied_stage4a_precision.abs_log2_mfr_change_at_movement_onset.gaba_block --> to
0.452 <!-- metrics.json: embodied_stage4a_precision.abs_log2_mfr_change_at_movement_onset.gaba_potentiate --> log2 (exploratory), within
the noise of real chips. At this level the output-tract electrode no longer leads
(4 <!-- metrics.json: embodied_stage4a.predictions.N4_the_output_tract_still_leads.value.tract_64 -->
families in time against
4 <!-- metrics.json: embodied_stage4a.predictions.N4_the_output_tract_still_leads.value.planar_64 -->
for the planar array).

### 2.6 Walking: a negative result and a reproduction

The male nerve cord, run as leaky integrate-and-fire neurons, does not turn its walking command
(`DNg100`) into a rhythm: not with the published parameters, not with spike-frequency adaptation
over a registered grid, and not with either of two rules for where glutamate excites. A
firing-rate model published for this question (Pugliese et al. 2025), reimplemented from its
description and built on the male CNS as its authors build it, does: the walking command drives
a front-leg motor rhythm (corrected score
0.8886 <!-- metrics.json: embodied_stage2h.none.predictions.H1_the_walking_command_makes_a_leg_rhythm.value -->,
period 196.5 <!-- metrics.json: embodied_stage2h.none.predictions.H6_a_fly_like_step_period.value -->
ms), and silencing its three-neuron circuit removes it
(0.0 <!-- metrics.json: embodied_stage2h.none.predictions.H5_the_rhythm_needs_the_three_neuron_circuit.value.lesion -->),
including when the neuron that relays the command is spared
(0.0483 <!-- metrics.json: embodied_stage2h_check.none.keep_relay_silence_INXXX466_IN16B036.median_score -->,
exploratory). The result depends on how neuron size is treated, and neither model alternates
antagonist muscles.

### 2.7 Closing the loop

Feedback from the front leg's position sensors, closed through the simulated leg every
millisecond, shifts the tibia motor pools toward resisting the movement but does not return a
pushed leg faster
(32.0 <!-- metrics.json: embodied_stage3b.predictions.M1_feedback_speeds_the_return.value.resistance -->
against 30.0 <!-- metrics.json: embodied_stage3b.predictions.M1_feedback_speeds_the_return.value.open -->
ms). The male CNS traces few proprioceptors for the front leg
(4 <!-- metrics.json: embodied_stage3b_check.chordotonal_by_leg.front_L.claw --> claw neurons,
against 22 <!-- metrics.json: embodied_stage3b_check.chordotonal_by_leg.hind_L.claw --> on the
hind leg).

### 2.8 Real flies

A literature comparison registered before its search found
0 <!-- metrics.json: embodied_stage5.predictions.V1_the_pharmacology_agrees.value.comparisons -->
studies that give adult flies these drugs and score proboscis extension or grooming; studies of
identified GABAergic neurons agree with the simulated direction in
1 <!-- metrics.json: embodied_stage5.transmitter_neuron.agree --> of 3
<!-- metrics.json: embodied_stage5.transmitter_neuron.comparisons -->. Manipulating the identified neurons as the experiment did (stage 5b), silencing
the GABAergic 13A leg neurons shortens the simulated grooming reach
(0.3806 <!-- metrics.json: embodied_stage5b.predictions.I1_silencing_13A_reduces_grooming.value.depth_mm --> mm from
0.9927 <!-- metrics.json: embodied_stage5b.predictions.I0_the_reach_is_there.value -->), as silencing them reduces grooming in flies; the
13B neurons and activation do not show in the reach. Read at the femur-tibia joint (stage 5c),
none of the published postures appears: silencing 13A extends the tibia
(-0.1488 <!-- metrics.json: embodied_stage5c.predictions.J1_silencing_13A_flexes_the_tibia.value.shift_rad --> rad) rather than locking it in
flexion, so the reach agreement came by another route. In this model the inhibitory leg neurons
act as broad brakes, not as joint-specific controllers.

## 3. Discussion

Across three levels of readout (named neurons, motor neurons and movement) and two connectomes,
a chip-like readout of a simulated fly's brain ranks drugs by how much they disrupt behaviour but
first flags them after behaviour has changed. For assay design this suggests three things, all
testable in real preparations: report the precision a chip needs at behavioural onset rather
than a fixed threshold; record where the network's output leaves it, which helps at the neural
level; and do not read a chip's silence as behavioural safety at low doses. The embodied stages
add that a body does not rescue the chip, and that the model's walking claims depend on the
neuron model: a connectome alone, in a leaky integrate-and-fire network, did not contain a
working rhythm generator, while a published firing-rate model did. The negative results, the
registered failures and the withdrawn claim are part of the evidence: they bound what the
positive results can be taken to mean.

## 4. Methods (summary)

- **Models.** Shiu et al. (2024) leaky integrate-and-fire model on FlyWire 783 (published
  constants) and on the male CNS (consensus transmitter signs); Pugliese et al. (2025)
  firing-rate model, reimplemented. Drug knobs scale whole transmitter classes.
- **Chip.** A virtual array scored by the project's single MEA feature extractor; detection at
  the registered threshold; false-alarm rates from untreated splits.
- **Bodies.** FlyMimic left front leg (Hill-type muscles) and flybody proboscis in MuJoCo; motor
  neurons mapped to muscles by published names; activation = filtered pool rate over a full rate.
- **Closed loop.** Nerve cord and leg advanced together every millisecond; sensory neurons driven
  by a joint angle.
- **Registration.** Every test was committed to `preregistration/` before it ran, with fresh
  seeds after any result had been seen; failures and uninterpretable outcomes are reported; one
  measurement flaw was audited under its own registration.
- **Reproduction.** `python run_all.py --full` regenerates every number and figure; the fly
  studies run with `python -m neuromps.fly.pipeline`.

## 5. What this work does not show

It is simulation throughout: transmitter-class knobs are not compounds, the bodies are partial
(one muscle-driven leg, a servoed proboscis), the loop is closed through four sensory neurons,
and the walking results are a negative for one neuron model and a reproduction of another group's
model. Nothing here says what any compound does to a fly, a culture or a person.

## Figures (planned; each exists as a script output)

| Figure | What it shows | File |
|---|---|---|
| 1 | The chip and the animal under the drug knobs: chip change against behavioural disruption, per dose | `outputs/figures/fly_chip_to_animal_dose_SIMULATED.png` |
| 2 | When the chip flags each family against when behaviour fails (neural level) | `outputs/figures/fly_sentinel_ladder_SIMULATED.png` |
| 3 | Output-tract electrode against a planar array | `outputs/figures/fly_output_tract_SIMULATED.png` |
| 4 | Motor-level onsets | `outputs/figures/fly_motor_onsets_SIMULATED.png`, `fly_motor_tie_break_SIMULATED.png` |
| 5 | The body: the grooming reach through the brain-to-nerve-cord relay, and the proboscis | `outputs/figures/embodied_stage1f_depth_SIMULATED.png`, `embodied_stage2b_proboscis_SIMULATED.png` |
| 6 | Drugs in the loop: movements against dose | `outputs/figures/embodied_stage4a_drugs_movement_SIMULATED.png` |
| 7 | Walking: the published rate model's front-leg rhythm, and the registered negatives | `outputs/figures/embodied_stage2h_front_leg_SIMULATED.png`, `embodied_stage2c_sweep_SIMULATED.png` |
| 8 | The closed loop and the perturbation test | `outputs/figures/embodied_stage3b_perturbation_SIMULATED.png` |
| S1 | The rhythm-measure audit | `outputs/figures/rhythm_audit_SIMULATED.png` |

## Data and code

Code, registrations and outputs: https://github.com/Vodolay/AI4S. Data sources, licences and
checksums: `data/raw/MANIFEST.json`, `CITATIONS.md`.

## References (to complete)

- Shiu PK et al. A Drosophila computational brain model reveals sensorimotor processing. Nature
  (2024).
- Pugliese et al. Connectome simulations identify a central pattern generator circuit for fly
  walking. bioRxiv (2025), doi:10.1101/2025.09.12.675944.
- Berg S et al. Sexual dimorphism in the complete connectome of the Drosophila male central
  nervous system. Nature (2026); male CNS v1.0.
- McKellar CE et al. Controlling motor neurons of every muscle for fly proboscis reaching. eLife
  (2020).
- Mamiya A, Gurung P, Tuthill JC. Neural coding of leg proprioception in Drosophila. Neuron
  (2018).
- Mendes CS et al. Quantification of gait parameters in freely walking wild type and sensory
  deprived Drosophila melanogaster. eLife (2013).
- Syed DS, Ravbar P, Simpson JH. Inhibitory circuits control leg movements during Drosophila
  grooming. eLife (2026; preprint 2024).
- Pool A-H et al. Four GABAergic interneurons impose feeding restraint in Drosophila. Neuron
  (2014).
- Vaxenburg R et al. Whole-body physics simulation of fruit fly locomotion. Nature (2025)
  (flybody).
- Dorkenwald S et al. Neuronal wiring diagram of an adult brain. Nature (2024); Schlegel P et al.
  Whole-brain annotation and multi-connectome cell typing of Drosophila. Nature (2024) (FlyWire).
- FlyMimic (FlyGym 2), the MEA data sources and every other dataset: see `CITATIONS.md`.
