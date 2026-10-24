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
| 3.5a The nerve cord from descending neurons | grooming neurons drive the front-leg pool far above the other legs and above every control descending neuron (stage 1c, 3 of 5 pass); the claw reaches the head and holds; the registered kinematic readout failed to separate it from a walking swing | done; stage 1d (fresh seeds) passes 4 of 4: the claw is held at the head for 0.9665 <!-- metrics.json: embodied_stage1d.predictions.D1_grooming_holds_the_claw_at_the_head.value.grooming --> of the second under grooming drive | roadmap §7 |
| 3.5c Brain to leg, end to end | the FlyWire brain joined to the male nerve cord by matched descending neurons: antennal input reaches the head (0.5369 <!-- metrics.json: embodied_stage1f.predictions.F1_antennal_input_reaches_deepest.value.jon_ce --> mm), sugar does not; removing aDN1 and aDN2 abolishes it (stage 1f, 5 of 5 on fresh seeds; stage 1e uninterpretable as registered) | done | roadmap §7 |
| 3.5d Fictive walking | under its walking command the connectome-only nerve cord fires the tibia extensors tonically (left middle leg 54.12 <!-- metrics.json: embodied_stage2a.pool_rates_hz.forward_walking.L2_extensor --> Hz) and leaves the flexors silent: no rhythm (stage 2a, 1 of 5) | failed; next, adaptation currents as a registered model change | roadmap §7 |
| 3.5e Feeding, brain to proboscis | sugar input to the FlyWire brain drives MN9 and MN4 over their antagonists and extends the flybody proboscis fully; bitter holds it back (time-averaged extension 0.3054 <!-- metrics.json: embodied_stage2b.reported.sugar.mean_extension_mm --> to 0.0488 <!-- metrics.json: embodied_stage2b.reported.sugar_bitter.mean_extension_mm --> mm); stage 2b, 6 of 6 | done | roadmap §7 |
| 3.5b Model transfer between connectomes | the published model's parameters do not carry over to a second connectome: a result in its own right | done (diagnostic) | `fly/spread.py`, `fly/excitability.py` |
| 3.6 Closed loop | proprioceptive feedback; does the conclusion survive it? | planned (stage 3) | roadmap |
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
