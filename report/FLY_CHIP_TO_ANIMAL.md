# From chip to animal: the chip's drug knobs on a whole fly brain

*Companion note to the technical report, written after round 1. Every result here is from
**simulation**, for research decision support; nothing is a claim about any compound's effect
in any animal or person. Each number cites the `outputs/metrics.json` key a script under
`src/neuromps/fly/` wrote it to.*

## The question

A neural chip reports what a drug does to a dish of neurons. The question a chip assay is
used to answer is different: will the drug disturb what a nervous system *does*? Neither the
chip nor any culture can answer that directly. A connectome can let us ask it, because
an adult fly brain has now been mapped synapse by synapse and run as a working model.

We took the digital twin's drug knobs and applied them to a simulated whole fly brain:
the connectome-constrained model of Shiu et al. (2024) on FlyWire, with
138,639 <!-- metrics.json: fly_models.flywire783.neurons --> neurons and
15,091,983 <!-- metrics.json: fly_models.flywire783.connections --> connections. We then
read that same brain in two ways at once:

- **as a chip**: a virtual electrode array laid over the brain's regions, its recordings
  scored by `features/mea.py`, the same extractor every real recording in the asset goes
  through;
- **as an animal**: the neurons the published model validated against behaviour: MN9, the
  proboscis motor neuron, under sugar drive (feeding), and again with bitter drive added
  (aversive gating); aBN1 under antennal mechanosensory drive (grooming); and how many
  descending neurons the sugar drive recruits (selectivity).

Five drug families were each run at four or five levels: GABA block, glutamate-gated chloride
(GluCl) block, GABA potentiation, acetylcholine block and a raised spike threshold. In the fly,
GABA and glutamate are the fast inhibitory transmitters and acetylcholine the fast excitatory
one, so these are the fly's counterparts of the twin's GABA-A, AMPA and sodium-channel knobs.

## Registered before it was run

The design, ten predictions and their pass/fail tests were committed before the sweep was
run (`preregistration/fly_chip_to_animal.json`; commit and SHA-256 in
`fly_chip_to_animal.registration`). The file also lists every exploratory result we had
seen before registering. Two addenda followed the same rule: a replication on a second,
independently reconstructed brain, and a confirmation of an exploratory finding. Every
prediction is reported below, whichever way it fell.

## What we found

**The model is the published one.** Sugar drive gives MN9
76.75 <!-- metrics.json: fly_chip_to_animal.predictions.R1_replication.value.mn9_783_hz --> Hz
on the FlyWire release we use, against
77.375 <!-- metrics.json: fly_chip_to_animal.predictions.R1_replication.value.mn9_630_hz --> Hz
on the release the paper used, with the same code and seeds. Only neurons whose identity
is unchanged between the two releases are used, and the code refuses any table that
matches fewer than 99% of the model's neurons, which is what a table from another release
would do.

**The behaviour predictions hold.** Blocking GABA releases feeding from bitter suppression
(rank correlation with dose
1.0 <!-- metrics.json: fly_chip_to_animal.predictions.B1_disinhibition_releases_feeding_from_bitter.value -->)
and recruits ever more off-target descending neurons
(1.0 <!-- metrics.json: fly_chip_to_animal.predictions.B3_disinhibition_recruits_off_target_descending_neurons.value -->).
Blocking acetylcholine silences the motor and grooming outputs.

**The chip ranks drugs correctly.** Across the 21 drug conditions, the change in the
array's firing rate tracks how much behaviour is disrupted (Spearman
0.863 <!-- metrics.json: fly_chip_to_animal.predictions.C1_chip_tracks_disruption.value -->).
The fly array also agrees with real chips about the direction each mechanism moves each
feature as often as the chip twin does:
12 <!-- metrics.json: fly_chip_to_animal.predictions.A1_direction_agreement_with_real_chips.value.fly -->
against
12 <!-- metrics.json: fly_chip_to_animal.predictions.A1_direction_agreement_with_real_chips.value.twin -->
agreeing features.

**But the chip is late: our registered prediction failed in every family.** We predicted
that the chip would flag each drug at a dose no higher than the one that impairs behaviour.
In all five families, behaviour failed first. At the lowest GABA-block level, aversive
gating of feeding has already broken while the array's firing rate has moved by about 10%.
Dropping any one behaviour readout from the disruption index leaves the chip late in at
least four of the five families (the raised threshold flips when the aversive-gating readout
is dropped), and the result replicates on fresh seeds, where firing rate was late in every
family.

**What a chip would need to catch it.** The change in firing rate at the first dose that
impairs behaviour is the precision a chip must reach to flag the drug in time:

| Mechanism | Firing-rate change at behavioural onset (log2) |
|---|---|
| GABA block | 0.149 <!-- metrics.json: fly_confirm_network_bursts.precision.abs_log2_mfr_change_at_behavioural_onset.flywire_sweep.gaba_block --> |
| Threshold raised | 0.124 <!-- metrics.json: fly_confirm_network_bursts.precision.abs_log2_mfr_change_at_behavioural_onset.flywire_sweep.excitability_block --> |
| GABA potentiation | 0.199 <!-- metrics.json: fly_confirm_network_bursts.precision.abs_log2_mfr_change_at_behavioural_onset.flywire_sweep.gaba_potentiate --> |
| GluCl block | 0.263 <!-- metrics.json: fly_confirm_network_bursts.precision.abs_log2_mfr_change_at_behavioural_onset.flywire_sweep.glucl_block --> |
| ACh block | 0.304 <!-- metrics.json: fly_confirm_network_bursts.precision.abs_log2_mfr_change_at_behavioural_onset.flywire_sweep.ach_block --> |

Real chips in our asset vary between untreated recordings by up to
0.156 <!-- metrics.json: fly_confirm_network_bursts.precision.real_chip_noise.kapucu2022_control.q95_abs_log2 -->
(Kapucu's vehicle controls, 95th percentile) and
1.752 <!-- metrics.json: fly_confirm_network_bursts.precision.real_chip_noise.vassallo2017_inactive.q95_abs_log2 -->
(Vassallo's inactive compounds). At the better of the two, a chip could flag acetylcholine
block, GluCl block and GABA potentiation in time, but not GABA block or a raised threshold;
at the worse, none of the five. The registered 0.5 log2 threshold was simply set above
what these mechanisms produce at behavioural onset. GABA block was the hardest mechanism to
catch in the second brain as well. This table is exploratory: it was computed after the
registered test had failed.

**A tempting fix that is noise.** In the first sweep, network-burst rate crossed the
threshold at or before behaviour in every family. We registered a confirmation, on fresh
seeds and on the second brain, before seeing either. Out of sample it crossed early in
8 <!-- metrics.json: fly_confirm_network_bursts.pooled_out_of_sample.network_bursts_at_or_before_behaviour -->
of
10 <!-- metrics.json: fly_confirm_network_bursts.pooled_out_of_sample.tests -->
mechanism tests, against
3 <!-- metrics.json: fly_confirm_network_bursts.pooled_out_of_sample.firing_rate_at_or_before_behaviour -->
for firing rate. Two of the four registered confirmation tests passed. Then we checked what
the registered analysis did not: two groups of *untreated* recordings already differ by the
threshold in a fraction
0.382 <!-- metrics.json: fly_checks.false_alarm_rate_untreated.by_feature.network_burst_rate_per_min -->
of random splits, while firing rate never does (untreated 95th percentile
0.028 <!-- metrics.json: fly_checks.false_alarm_rate_untreated.by_feature.q95_abs_log2.mfr_hz --> log2).
Trains shifted in time, which keeps every rate and breaks every coincidence, still produce
network bursts. Network-burst rate looked early because it is noisy, not because it sees
more. We report it as a negative result.

**Direction between brain regions is not recovered from spontaneous activity.** For region
pairs joined by many synapses, the connectome gives the true direction. The chip's
cross-correlation measure, on spontaneous activity, gets the sign right on strongly
asymmetric pairs at chance
(0.4946 <!-- metrics.json: fly_chip_to_animal.predictions.D1_spontaneous_direction_is_near_chance.value -->).
Pulsed stimulation of one region, read 5 to 30 ms after each pulse, does better
(0.6222 <!-- metrics.json: fly_chip_to_animal.predictions.D2_pulsed_stimulation_recovers_direction.value.evoked -->)
but misses the registered 0.70; read only through the array's electrodes it returns to
chance
(0.4889 <!-- metrics.json: fly_chip_to_animal.predictions.D3_an_electrode_array_sees_it_too.value -->).
This is the report's own §17 caution, now tested on a brain where the answer is known.

**A second brain agrees.** On the male central nervous system connectome (MaleCNS, brain
and nerve cord,
165,122 <!-- metrics.json: fly_models.malecns.neurons --> neurons,
25,563,096 <!-- metrics.json: fly_models.malecns.connections --> connections), with the
published neuron sets matched by cell type and side, all five registered replication
tests pass. Sugar drives MN9 at
31.625 <!-- metrics.json: fly_malecns_replication.predictions.M0_feeding_circuit_exists.value.mn9_sugar_hz --> Hz,
and the chip's firing-rate change again tracks behavioural disruption (Spearman
0.878 <!-- metrics.json: fly_malecns_replication.predictions.M_C1.value -->).

## What this does not show

- It is a model. The fly model is a leaky integrate-and-fire network with one free weight,
  no adaptation and no neuromodulation; it fires asynchronously and does not burst the way a
  culture does. The behaviour readouts are firing rates of named neurons, not behaviour.
- The drug knobs scale whole transmitter classes. A real compound acts on particular receptor
  subunits, with its own potency and off-target effects. Nothing here says what any compound
  does to a fly, a culture or a person.
- The precision table depends on our disruption index and its 0.5 threshold, and was
  computed after the registered test failed. It is a reason to design chip assays for
  precision, not a specification.
- The real-chip noise floors come from two sources with very different spreads, from few
  recordings in one of them.

## How to reproduce

`python run_all.py --stage fly` fetches the licensed files (Shiu et al. model files, MIT;
FlyWire connectivity and annotations, CC-BY-4.0) and runs the registered study.
`python -m neuromps.fly.replication`, `python -m neuromps.fly.confirm` and
`python -m neuromps.fly.checks` run the two addenda and the noise checks. The male CNS
files (CC-BY) are fetched with `python -m neuromps.fetch.runner --full --only malecns`.
The demo app's tab "From chip to animal" shows the sweep with a dose slider.
