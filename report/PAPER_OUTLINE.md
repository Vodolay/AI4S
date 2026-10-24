# Paper outline: from chip to animal

*Working outline for a research paper built on the NeuroMPS fly studies. It is a plan: results
quoted here cite `outputs/metrics.json`; everything still to be done is marked as such. All fly
results are from **simulation**, for research decision support.*

## Working title

**Does a neural chip raise the alarm before behaviour fails? Pre-registered tests on
connectome-constrained fly brains, from virtual electrodes to a muscle-driven leg.**

## The claim, in one paragraph

Neural organ-on-a-chip assays read drug effects from a dish of neurons and are used to predict
effects on what a nervous system does, which no culture can check. Connectome-constrained models
of a whole fly brain and nerve cord let both be read at once: through a virtual electrode array
scored exactly as real chips are, and through the neurons, then the muscles, that produce
behaviour. Across pre-registered tests on two independent connectomes, the chip's firing rate
ranks drugs by how much they disrupt behaviour, but it is never ahead of behavioural failure; how
far it lags sets a precision a chip assay must reach, and where the electrode sits matters as
much as how precise it is.

## Why it is publishable

- **Question.** Whether an in-vitro electrophysiology readout is an early enough sentinel has not
  been testable, because the behavioural ground truth is missing; a mapped brain supplies it.
- **Rigour.** Every confirmatory test was registered in a committed file before its runs, failed
  predictions are reported as failed, and every number in the text is checked against the
  script output that produced it.
- **Two connectomes.** FlyWire (female brain) and the male CNS (brain and nerve cord, independently
  reconstructed) give a built-in replication.
- **Useful negatives.** An early-warning feature that turned out to be noise; vesicle depletion
  that does not explain a real-chip paradox; spontaneous activity that cannot recover wiring
  direction.

## Structure and status

| Section | Content | Status | Source |
|---|---|---|---|
| 1. Introduction | MEA assays, the missing behavioural ground truth, connectome models | to write | — |
| 2. Methods: models | Shiu et al. LIF on FlyWire 783; male CNS with matched neuron sets; release checks; drug knobs by transmitter class | done | report §23, §25 |
| 2. Methods: chip readout | virtual array, `features/mea.py`, false-alarm calibration | done | §24 |
| 2. Methods: pre-registration | eight registrations, the rules, deviations | done | `preregistration/` |
| 3.1 Behaviour and chip | ranking (Spearman 0.863 <!-- metrics.json: fly_chip_to_animal.predictions.C1_chip_tracks_disruption.value -->), late in every family, replicated | done | §24, §25 |
| 3.2 Precision a chip needs | per-mechanism firing-rate change at behavioural onset against real-chip noise and the EPA screen cutoff | done (exploratory, labelled) | §20, §24 |
| 3.3 Where to record | output tract against planar array, with robustness | done | §24 |
| 3.4 Motor neurons | the chip later than motor output in 4 <!-- metrics.json: fly_motor_tie_break.combined_with_parent.chip_later --> families, level in 1, never earlier | done | §25 |
| 3.5 A muscle-driven leg | stage 1 failed on specificity; stage 1b's feeding check failed; cause: the published parameters put the male brain past a propagation threshold (251 <!-- metrics.json: fly_spread.flywire783.active_total --> against 14,263 <!-- metrics.json: fly_spread.malecns_published.active_total --> active neurons); next, the nerve cord from descending neurons, then the FlyWire brain joined to it | in progress | `report/ROADMAP_EMBODIED_FLY.md` §7 |
| 3.5a The nerve cord from descending neurons | grooming neurons drive the front-leg pool far above the other legs and above every control descending neuron (stage 1c, 3 of 5 pass); the claw reaches the head and holds; the registered kinematic readout failed to separate it from a walking swing | done; stage 1d (fresh seeds) passes 4 of 4 as registered (its D2, the walking swing, fails under the audited rhythm measure): the claw is held at the head for 0.9665 <!-- metrics.json: embodied_stage1d.predictions.D1_grooming_holds_the_claw_at_the_head.value.grooming --> of the second under grooming drive | roadmap §7 |
| 3.5c Brain to leg, end to end | the FlyWire brain joined to the male nerve cord by matched descending neurons: antennal input reaches the head (0.5369 <!-- metrics.json: embodied_stage1f.predictions.F1_antennal_input_reaches_deepest.value.jon_ce --> mm), sugar does not; removing aDN1 and aDN2 abolishes it (stage 1f, 5 of 5 on fresh seeds; stage 1e uninterpretable as registered) | done | roadmap §7 |
| 3.5d Fictive walking | under its walking command the connectome-only nerve cord fires the tibia extensors tonically (left middle leg 54.12 <!-- metrics.json: embodied_stage2a.pool_rates_hz.forward_walking.L2_extensor --> Hz) and leaves the flexors silent: no rhythm (stage 2a, 1 of 5) | failed; next, adaptation currents as a registered model change | roadmap §7 |
| 3.5e Feeding, brain to proboscis | sugar input to the FlyWire brain drives MN9 and MN4 over their antagonists and extends the flybody proboscis fully; bitter holds it back (time-averaged extension 0.3054 <!-- metrics.json: embodied_stage2b.reported.sugar.mean_extension_mm --> to 0.0488 <!-- metrics.json: embodied_stage2b.reported.sugar_bitter.mean_extension_mm --> mm); stage 2b, 6 of 6 | done | roadmap §7 |
| 3.5f Adaptation sweep | no cell of a registered 6 x 6 adaptation sweep walks (stage 2c, stopped by its rule); exploratory: the command's active neurons inhibit the tibia flexors (274.73 <!-- metrics.json: embodied_stage2c_diagnostic.flexor_inputs.from_active.inhibitory_per_neuron --> inhibitory against 130.78 <!-- metrics.json: embodied_stage2c_diagnostic.flexor_inputs.from_active.excitatory_per_neuron --> excitatory synapses per neuron) | failed; next step to decide (glutamate sign, feedback, rebound) | roadmap §7 |
| 3.5g Glutamate sign | glutamate excitatory onto motor neurons leaves flexors weak; everywhere turns them on (40.4 <!-- metrics.json: embodied_stage2d.sweep.cells.37.flexor_hz --> Hz) but the cord fires in slow population waves; no alternation in any cell (stage 2d, 1 of 7). With 2a and 2c: no walking rhythm generator in the connectome under LIF | failed; a negative result for the paper | roadmap §7 |
| 3.5h Rate model (Pugliese et al.) | reimplemented on the male nerve cord; the size stand-in starves DNg100 (6.54 <!-- metrics.json: embodied_stage2e.exploratory_driven_neurons.forward_walking.drive_only_rate_hz.0 --> Hz), so their result is not yet tested (stage 2e, 2 of 8, one vacuous) | failed as run; follow-up with per-neuron drive | roadmap §7 |
| Measure audit | the registered rhythmicity measure scored smooth non-rhythmic signals high; a registered audit with a peak-requiring measure re-ran stages 1d-2e: stage 1d's D2 fails (forward walking 0.0956 <!-- metrics.json: rhythm_audit.embodied_stage1d.predictions.D2_forward_walking_swings_the_leg.audited.value.forward_walking -->); all other registered outcomes stand; originals reproduce exactly | done | roadmap §7 |
| 3.5i Whole cord, command driven alike | DNg100 starts a self-sustaining wave that silences it (stage 2f precondition fails in both size rules: 0.18 <!-- metrics.json: embodied_stage2f.synapse_count.predictions.S0_the_command_is_on.value --> Hz); the published model runs only the front-leg network with recruitment capped at 500 | failed; next, their construction on the male CNS (stage 2g) | roadmap §7 |
| 3.5j Front-leg network (Pugliese et al. construction) | rebuilt from the male CNS; the network is nearly all-or-none, so their input rule finds a working input in too few draws (stage 2g, precondition failed narrowly); exploratory: a regular rhythm in 1-2 motor-neuron pools, removed by silencing the circuit without size scaling | to rerun with a finer input search (stage 2h) | roadmap §7 |
| 3.5k Front-leg rhythm reproduced | with a fine input search, DNg100 drives a front-leg motor rhythm on the male CNS (0.8886 <!-- metrics.json: embodied_stage2h.none.predictions.H1_the_walking_command_makes_a_leg_rhythm.value -->, no size scaling), which needs INXXX466 and IN16B036, not only the relay (exploratory); depends on the size rule; no antagonist alternation (stage 2h) | done; a headline for the paper, with its limits | roadmap §7 |
| 3.5b Model transfer between connectomes | the published model's parameters do not carry over to a second connectome: a result in its own right | done (diagnostic) | `fly/spread.py`, `fly/excitability.py` |
| 3.6 Closed loop | stage 3a: nerve cord, leg and claw-neuron feedback run together; feedback shifts the tibia pools toward resistance (extensor 43.25 <!-- metrics.json: embodied_stage3a.reported.open.grooming.tibia_extensor_hz --> to 31.75 <!-- metrics.json: embodied_stage3a.reported.resistance.grooming.tibia_extensor_hz --> Hz) but a held leg barely moves and a sham does the same (go/no-go failed) | stage 3b: a push on the tibia returns no faster with feedback (32.0 <!-- metrics.json: embodied_stage3b.predictions.M1_feedback_speeds_the_return.value.resistance --> against 30.0 <!-- metrics.json: embodied_stage3b.predictions.M1_feedback_speeds_the_return.value.open --> ms); front-leg proprioceptors thinly traced (4 <!-- metrics.json: embodied_stage3b_check.chordotonal_by_leg.front_L.claw --> claw neurons against 22 <!-- metrics.json: embodied_stage3b_check.chordotonal_by_leg.hind_L.claw --> on the hind leg) | failed twice; a limit to state | roadmap §7 |
| 3.6b Drugs in the loop | with movement as the readout the chip flags after movement fails in all five families (GABA-A block: movement 0.2 <!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_block.movement -->, chip 0.6 <!-- metrics.json: embodied_stage4a.predictions.N3_the_chip_flags_before_movement_fails.value.gaba_block.chip -->); movement never fails before the neural readouts; the tract no longer leads (stage 4a, 3 of 6) | done; a headline for the paper | roadmap §7 |
| 3.7 Negative results | network-burst noise, depletion, direction from spontaneous activity | done | §18, §24, §25 |
| 4. Discussion | what this means for assay design; limits of LIF models; what real experiments would test it | to write | — |
| Code and data | one-command reproduction; licences; manifest | done | README |

## What a reviewer will ask, and the answer we need

| Question | Answer, or what is still needed |
|---|---|
| Is an LIF model with one free weight enough? | it reproduces the published circuit results on both connectomes; the limits are stated (§26); stage 3 adds feedback |
| Is the "chip is late" result an artefact of the disruption index? | leave-one-readout-out and motor-level readouts (done); embodied readouts (planned) |
| Are the knobs pharmacology? | no: transmitter-class scalings; stated as such; stage 5 compares directions with published fly pharmacology |
| Why trust the male-CNS runs at a rescaled weight? | the rescaling rule comes from connectome statistics alone and was registered before use; both weights are reported |
| Does any of it transfer to real chips? | the chip readout is the same extractor applied to real recordings; the precision table is a testable prediction for real assays |

## Target

A preprint first (bioRxiv), then a computational-biology journal; venue to be decided with the
team. Authorship and author order are for the team to decide.
