# NeuroMPS — a provenance-tracked MEA data asset and a calibrated two-compartment digital twin

**Technical report.** Entry for the AI4S Open Innovation: AI + Organ-on-a-Chip challenge
(5th Pazhou Algorithm Competition).

> **Research decision-support only.** Nothing here is an efficacy, safety, diagnostic or clinical
> claim, and nothing here should be used to make one.

**Submission category: End-to-End System.** A complete path from public recordings to actionable
output — provenance-enforced ingestion, a standardized feature layer, models and a calibrated
digital twin, and a harness that reports which measurements the data can support — with an
interactive demo.

**Team.**
<!-- FILL IN BEFORE SUBMITTING. The competition gives a +0.5 bonus to teams covering both AI/CS and
     biology/bioengineering/clinical expertise, and asks that composition be declared here as well as
     in the Kaggle Writeup. Replace the row below with the real roster and name the relevant
     background explicitly. -->

| member | role | background |
|---|---|---|
| **[NAME]** (team leader) | **[ROLE]** | **[AI/CS and/or biology/bioengineering/clinical]** |

Every computed number below carries an HTML comment naming the `outputs/metrics.json` key it came
from. `python -m neuromps.report.verify` re-reads each citation and fails if the document and the
live numbers disagree, so a figure that goes stale when a stage is regenerated cannot survive in the
prose. The decision record, including what was tried and abandoned, is in `report/LAB_NOTEBOOK.md`.

**Structure.** §1–3 are what was built and why it can be trusted. §4–12 are what the data says.
§13–17 are the simulation. §18–21 are what can and cannot be concluded from it.
---

## 1. What this is, and what it is not

Microelectrode arrays record spikes from a culture on a grid of electrodes. People use them to ask
whether a compound changes network activity, and if so how. The obstacles are familiar: every
laboratory uses a different plate, a different spike sorter and a different definition of "burst",
so numbers from two papers are rarely comparable, and most published data is a table of summary
values rather than the spikes themselves.

This project has two layers.

**Layer A is a dataset.** Eight public sources, assembled into one table with one definition of
every feature, every record carrying where it came from, under what licence, and a checksum of the
file it was computed from. The point is not volume. The point is that a number from Kapucu's plate
and a number from ScreenNeuroPharm's chamber were computed by the *same code from the same kind of
input*, so comparing them means something.

**Layer B is a simulation** — a "digital twin" — of a two-compartment neural chip: two populations
of model neurons wired to each other through a directional pathway, with virtual electrodes laid
over them. It is calibrated against the real recordings in Layer A, and it is scored through the
same feature code, so the twin cannot flatter itself by measuring differently.

**What it is not.** It is not a screening platform, it does not predict whether a compound is safe
or effective, and several of its most useful results are negative — measurements that do not contain
the signal people assume they contain. Those are in §18, and we think they are the most valuable
part of the work.

---

## 2. The data, and the case that tested the rules

The asset holds 21,887 records
<!-- metrics.json: data_asset.n_records --> from eight sources. They are not equal in kind, and the
difference matters more than the count:

| what the record holds | how many | what you can do with it |
|---|---|---|
| spike times | 1,512 <!-- metrics.json: data_asset.counts.by_data_level.spike_times --> | recompute any feature from scratch |
| fitted dose-response curves | 20,375 <!-- metrics.json: data_asset.counts.by_data_level.fitted_curve --> | read a potency estimate someone else fitted |

Every record declares which it is. This sounds like bookkeeping and is not: nearly all of the
modelling in §5–§9 can only use the 1,512 spike-time records, because only those can be run through
one common feature definition. A table that mixed the two would look twenty times larger and would
be twenty times less useful.

By species: 21,530 rat <!-- metrics.json: data_asset.counts.by_species.rat --> and 357 human
<!-- metrics.json: data_asset.counts.by_species.human -->. That imbalance is the single biggest
limitation of the whole project and it is not something we can fix by being clever; it reflects what
is publicly available under a licence that permits reuse.

**A rejection.** One dataset (Tukker 2020) was a good fit scientifically and was excluded, because
it is CC BY-NC and the competition's rules assign joint ownership of entries to the organising
committee — which is not something a non-commercial licence permits us to grant. The exclusion is
recorded in the manifest alongside the accepted sources, with the reason, rather than the dataset
simply being absent.

We raise this because a reviewer should be able to check that the licence discipline is real rather
than asserted. Every source re-checks its own licence at fetch time and the build refuses to emit an
asset if any file's checksum disagrees with the manifest.

---

## 3. Are two laboratories' numbers actually comparable?

This is the question the whole of Layer A rests on, so we tested it rather than assuming it.

All features are computed by one module, `features/mea.py`, from spike times. Nine features: mean
firing rate, burst rate, burst duration, spikes per burst, the percentage of spikes inside bursts,
network burst rate, a synchrony index, the coefficient of variation of inter-spike intervals, and
the number of active electrodes. Burst detection is a stated rule (spikes closer than 100 ms, at
least five in a row) rather than a vendor default, and synchrony is the spike time tiling
coefficient — a pairwise measure that, unlike a plain correlation, does not reward two electrodes
simply for both being busy.

**The check.** Kapucu and colleagues deposited their own feature tables alongside their spike files,
computed with `meaRtools`. That gives an independent implementation to test against: we ran our code
over their spikes and compared, across 336 recordings
<!-- metrics.json: extractor_agreement.n_recordings --> on 3 plates
<!-- metrics.json: extractor_agreement.n_plates -->, 2,298 paired values
<!-- metrics.json: extractor_agreement.n_paired_values --> in total.

The answer is worth stating plainly, because it is the most important methodological result in the
report:

| feature | rank agreement (ρ) | our value ÷ theirs |
|---|---|---|
| synchrony index | 0.9839 <!-- metrics.json: extractor_agreement.by_feature.synchrony_index.spearman_rho --> | 0.736 <!-- metrics.json: extractor_agreement.by_feature.synchrony_index.median_ratio_ours_over_theirs --> |
| mean firing rate | 0.9831 <!-- metrics.json: extractor_agreement.by_feature.mfr_hz.spearman_rho --> | 0.7539 <!-- metrics.json: extractor_agreement.by_feature.mfr_hz.median_ratio_ours_over_theirs --> |
| active electrodes | 0.9763 <!-- metrics.json: extractor_agreement.by_feature.active_electrodes.spearman_rho --> | 1.3624 <!-- metrics.json: extractor_agreement.by_feature.active_electrodes.median_ratio_ours_over_theirs --> |
| % spikes in bursts | 0.9703 <!-- metrics.json: extractor_agreement.by_feature.pct_spikes_in_bursts.spearman_rho --> | 0.9864 <!-- metrics.json: extractor_agreement.by_feature.pct_spikes_in_bursts.median_ratio_ours_over_theirs --> |
| spikes per burst | 0.9167 <!-- metrics.json: extractor_agreement.by_feature.spikes_per_burst.spearman_rho --> | 0.9418 <!-- metrics.json: extractor_agreement.by_feature.spikes_per_burst.median_ratio_ours_over_theirs --> |
| burst rate | 0.8907 <!-- metrics.json: extractor_agreement.by_feature.burst_rate_per_min.spearman_rho --> | 0.4423 <!-- metrics.json: extractor_agreement.by_feature.burst_rate_per_min.median_ratio_ours_over_theirs --> |
| burst duration | 0.7926 <!-- metrics.json: extractor_agreement.by_feature.burst_duration_s.spearman_rho --> | 0.8189 <!-- metrics.json: extractor_agreement.by_feature.burst_duration_s.median_ratio_ours_over_theirs --> |

**Read the two columns separately.** The left column asks "do we rank recordings the same way?" The
right asks "do we get the same number?"

One feature reproduces on both: the percentage of spikes in bursts. Three rank almost identically
but sit on a different scale — our synchrony is consistently about three-quarters of theirs, our
firing rate about three-quarters, our active-electrode count about a third higher. And the burst
measures disagree about *which recordings burst more*: `burst_rate_per_min` differs by more than a
factor of two, and burst duration ranks at ρ = 0.79.

**What this means for you.** Within this dataset, comparisons are safe, because one code path
produced every number. Across papers, they are not: "bursts per minute" from two laboratories can
differ twofold from the definition alone, before any biology. This is exactly why the project insists
on recomputing from spikes rather than pooling published summary tables — and it is measured here
rather than asserted.

![Our extractor against the depositors' own meaRtools output, REAL recordings. Rank agreement on the horizontal axis, ratio of values on the vertical. A feature can rank identically and still be on a different scale.](../outputs/figures/extractor_agreement_real.png){width=62%}

---

## 4. The three questions we asked the data

Every model in this report works on **changes, not states**. A record is a treated well paired with
its own baseline, and the model sees the log-ratio of each feature between the two. This is the
single most consequential design decision in Layer B of the analysis, and it is made for the reason
an experimenter would make it: cultures differ enormously from each other, so "this well fires at
4 Hz" carries mostly the identity of the culture, while "this well halved its firing rate" carries
the treatment.

The three questions:

1. **Mechanism.** Given the change in the nine features, can you say which class of compound caused
   it — GABA-A antagonist, sodium-channel blocker, glutamatergic antagonist, and so on?
2. **Detection.** Can a model trained on one laboratory's data flag a known compound in a different
   laboratory's recordings?
3. **Platform contrast.** Does the *same mechanism* look the same on two different platforms?

**Two rules apply throughout, and they are why the numbers here are lower than you may expect from
the literature.**

*Splits group by culture.* If two wells come from the same culture, they never appear on opposite
sides of a train/test split. Cultures are the unit that repeats, not wells; splitting by well lets a
model recognise the culture instead of the treatment and inflates every score.

*Uncertainty is resampled over cultures, not rows.* All intervals here are cluster bootstraps that
resample whole cultures. With 73 pairs from 50 cultures, resampling rows would produce intervals
perhaps half as wide and entirely fictitious.

---

## 5. Can you read a mechanism from the features?

**Within one platform**, on 73 treated/baseline pairs
<!-- metrics.json: mechanism_baselines.task1_within_platform.n_pairs --> from 50 cultures
<!-- metrics.json: mechanism_baselines.task1_within_platform.n_groups --> across six mechanism
classes:

| model | balanced accuracy | 95% interval |
|---|---|---|
| always guess the commonest class | 0.167 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.dummy_most_frequent.balanced_accuracy_mean --> | — |
| gradient boosting | 0.398 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_mean --> | 0.2762 to 0.5178 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_ci95.1 --> |
| logistic regression | **0.537** <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean --> | 0.4216 to 0.6845 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_ci95.1 --> |

Balanced accuracy averages the per-class recall, so a model that ignores the rare classes cannot
win by getting the common one right. Chance here is 0.167
<!-- metrics.json: mechanism_baselines.task1_within_platform.chance_balanced_accuracy -->, one in six.

**Three things to take from this table.** First, there is real signal: roughly half the mechanisms
are called correctly where chance is one in six, and the interval clears chance. Second, the
interval is wide — 0.42 to 0.68 — because 73 pairs from 50 cultures is a small study, and we would
rather show you that than a point estimate. Third, **the simpler model wins**, decisively. Gradient
boosting has far more capacity and uses it to memorise 73 examples. When a flexible model loses to a
linear one by that margin, the honest reading is that the dataset is small, not that boosting is bad.

Errors are not random. They concentrate between the two GABA-A classes — antagonists and
positive modulators — which is chemically sensible, and visible in the confusion matrix below.

![Task 1 confusion matrix, REAL data: out-of-fold predictions over 73 treated/baseline pairs. Errors concentrate between the two GABA-A classes.](../outputs/figures/mechanism_confusion.png){width=52%}

**On a platform the model has never seen**, the picture changes. Tested on `vassallo2017` — a
different laboratory, different vendor, six compounds Kapucu never used, 171 pairs
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.n_pairs --> over 26 chips
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.n_groups --> in four classes —
logistic regression reaches 0.333
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.models.logistic_regression.balanced_accuracy_mean -->
against a chance of 0.25
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.chance_balanced_accuracy -->.

That is close to nothing. It is also, we think, the more informative of the two numbers, and it is
reported rather than buried: **within-platform performance does not transfer.**

**Why we never pooled the platforms into one big training set.** It is tempting, and it would have
produced a better-looking number. It would also have been meaningless: across the full pool, 97.35%
of pairs <!-- metrics.json: mechanism_baselines.label_source_confounding.source_determined_by_label -->
have their source determined by their label — 6
<!-- metrics.json: mechanism_baselines.label_source_confounding.n_labels_from_one_source_only -->
of the 9 mechanism classes
<!-- metrics.json: mechanism_baselines.label_source_confounding.n_labels -->
appear in one source only. A pooled classifier would score highly by learning which laboratory a
recording came from. We measured that confound instead of walking into it.

---

## 6. Detection, and the negative control that matters more

Can a model trained on Kapucu's plate flag bicuculline in ScreenNeuroPharm's chambers — a different
laboratory, a different compound in the same mechanism class?

Across 497 recordings <!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.n -->
it calls GABA-A antagonist 36.6% of the time
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.detection_rate -->.

Two ways to read that, and only one is right. It is far above the rate at which that class appears
by chance, so something transfers. But it is not a detector; two-thirds of the time it says something
else.

**The part that makes the first number trustworthy is the negative control.** On Sharf's diazepam
recordings — a GABA-A *positive modulator*, the opposite pharmacology — the same model calls
antagonist 7.7% of the time
<!-- metrics.json: mechanism_baselines.task2_external_detection.sharf_diazepam_negative_control.detection_rate -->.
A model that simply shouted "GABA-A antagonist" at anything unusual would have fired on both. It
does not.

**And one thing it cannot do.** Within the bicuculline series the detection probability has no
relationship with dose: Spearman ρ = 0.014
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman -->,
p = 0.751
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman_p -->.
Mechanism and dose are different questions, and this model answers only the first. §18 returns to
this, because it turned out to be a property of the measurement rather than of the model.

---

## 7. Rat to human, which is the point of the exercise

The ambition behind organ-on-chip work is that a human preparation tells you something a rodent one
does not. So: train on rat, test on human, using only the five mechanism classes present in both.

| | balanced accuracy | chance |
|---|---|---|
| rat, tested within rat | 0.543 <!-- metrics.json: species_transfer.transfer.none.rat_within.balanced_accuracy --> | 0.2 |
| human, tested within human | 0.646 <!-- metrics.json: species_transfer.transfer.none.human_within.balanced_accuracy --> | 0.2 |
| **rat → human, no adaptation** | **0.55** <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy --> | 0.2 <!-- metrics.json: species_transfer.transfer.none.chance_balanced_accuracy --> |
| rat → human, CORAL adaptation | 0.442 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy --> | 0.2 |
| rat → human, standardised | 0.408 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy --> | 0.2 |

Transfer is real and it is not luck — a permutation test on the unadapted transfer gives p = 0.0002
<!-- metrics.json: species_transfer.transfer.none.rat_to_human.permutation_p -->.

**Two things here are worth a reviewer's attention.**

The first is that rat → human (0.55) is *not worse* than rat tested within rat (0.543). Mechanism
information carries across the species boundary about as well as it carries across cultures within
rat. We would not push that far — the intervals are wide and the human sample is 32 pairs — but it
is the opposite of the degradation we expected to find and report.

The second is that **both domain-adaptation methods made it worse**, CORAL to 0.442 and
standardisation to 0.408. These are the standard tools for exactly this problem. Applied here they
removed signal rather than distribution shift, which is what happens when a method designed for
thousands of samples meets dozens. We report the unadapted number as the result and the adapted ones
beside it, rather than reporting whichever was highest.

The honest caveat is the sample: 37 rat pairs
<!-- metrics.json: species_transfer.transfer.none.rat_to_human.n_train --> training, 32 human pairs
<!-- metrics.json: species_transfer.transfer.none.rat_to_human.n_test --> testing. These are small
numbers and the interval is correspondingly wide. What we would not claim is that 0.442 is *the*
rat-to-human transfer rate for MEA pharmacology. What the experiment supports is narrower and still
useful: a model trained on rat carries some mechanism information into human recordings, and loses a
substantial part of it on the way.

---

## 8. Which features are actually doing the work

If the nine-feature vector is doing nine things, dropping most of it should hurt. It does not.

| feature subset | balanced accuracy |
|---|---|
| all nine | 0.514 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.all.balanced_accuracy_mean --> |
| the two rate features only | **0.53** <!-- metrics.json: uncertainty_and_ablations.feature_subsets.rate_only.balanced_accuracy_mean --> |
| species-invariant subset | 0.516 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.species_invariant.balanced_accuracy_mean --> |
| synchrony only | 0.455 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.synchrony_only.balanced_accuracy_mean --> |
| firing rate only | 0.442 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.mfr_only.balanced_accuracy_mean --> |
| network features only | 0.417 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.network_only.balanced_accuracy_mean --> |
| burst features only | 0.411 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.burst_only.balanced_accuracy_mean --> |

**Two rate features do at least as well as all nine.** The intervals overlap heavily, so we would
not claim rate-only is *better*; the defensible statement is that the other seven features are not
contributing anything we can demonstrate at this sample size.

This is a useful result for anyone planning experiments. It says that the elaborate burst and
synchrony machinery — the part that differs most between laboratories and that §3 shows is least
reproducible across implementations — is also the part carrying least mechanism information here.

**A test of that, on a preparation the model has never seen.** Because rate alone appeared
sufficient, we could use a source that reports firing rates only and no burst structure —
`fofie2025`, human iPSC-derived *sensory* neurons, 291 pairs
<!-- metrics.json: rate_only_arm.n_pairs --> from 48 wells <!-- metrics.json: rate_only_arm.n_groups -->.
Training on Kapucu's cortical rate change and testing on these sensory neurons gives 0.5012
<!-- metrics.json: rate_only_arm.across_sources.balanced_accuracy --> against a chance of 0.3333
<!-- metrics.json: rate_only_arm.across_sources.chance -->, on **one feature** and a preparation with
a different neuronal identity entirely.

The caveat is stated in the metrics file itself: what these two sources share is the compound's
mechanism, not the preparation or the network
<!-- metrics.json: rate_only_arm.caveat -->.

---

## 9. How confident is the model, really?

Two checks, because a probability that is not calibrated is worse than no probability.

**Calibration.** When the model says 70%, is it right 70% of the time? The expected calibration
error is 0.1114 <!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.ece --> on
73 points <!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.n -->, with an
interval reaching 0.2557
<!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.ece_ci95.1 -->. The upper end
is large. Our own reading: treat the probabilities as an *ordering*, not as risk — and not even as an
ordering above about 0.8 confidence, where observed accuracy falls rather than rises.

**Conformal sets.** Instead of one answer, return a set guaranteed to contain the truth a stated
fraction of the time. Asking for 95%, the empirical coverage is 0.918
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep.0.empirical_coverage --> and the average
set holds 4.562 <!-- metrics.json: uncertainty_and_ablations.conformal_sweep.0.mean_set_size -->
classes out of six.

A set of 4.6 out of 6 is honest and nearly useless — which is the correct answer for 73 calibration
points, and not a defect to tune away. We report it because a method that promises 95% coverage and
delivers narrow sets on this much data would be doing something wrong.

---

## 10. Two-compartment devices, and a result we did not want

Layer B of this project models an asymmetric two-compartment chip — the "axon diode" idea, where
microchannel geometry is designed to bias which way axons grow and therefore which way activity
flows. Before simulating it, we asked what the real devices show.

**Asymmetry is measurable, and it tracks tissue rather than geometry.** Across 16 devices
<!-- metrics.json: compartment_asymmetry.n_devices --> we measured the asymmetry of cross-compartment
coupling. Devices pairing two *different* tissues (cortex with hippocampus) are more asymmetric,
median 0.138 <!-- metrics.json: compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.median_heterogeneous -->,
than devices pairing the same tissue with itself, median 0.094
<!-- metrics.json: compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.median_homogeneous -->,
p = 0.042 <!-- metrics.json: compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.p_value -->.
The two same-tissue types are indistinguishable from each other.

This is a small effect on 16 devices and we would not build on it alone. It points the right way:
what the two chambers *contain* matters more than the fact that they are connected.

![Inter-compartment coupling lag by device type, REAL data, one value per device: five CxCx, five HpHp, six CxHp. The two same-tissue types are indistinguishable; the heterogeneous type sits apart.](../outputs/figures/compartment_lag_by_device_type.png){width=56%}

**Now the uncomfortable one.** In a deposit that recorded axonal crossings directly — actual
propagation events through the channels, on 10 cultures
<!-- metrics.json: axonal_directionality.n_cultures --> — we can ask the question the whole chamber
readout exists to answer: *if you only had the chamber recordings, could you tell which way the
axons were firing?*

Using all four chamber-level measures together — coupling asymmetry, peak correlation, lag and the
rate imbalance between chambers — a classifier reaches a balanced accuracy of 0.3854
<!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.balanced_accuracy -->
against a chance of 0.5
<!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.chance -->. Worse than
guessing.

And this is not simply a small-sample failure. On this many cultures the analysis could have detected
a balanced accuracy of 0.6438
<!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.detectable_balanced_accuracy -->
had one been there. It was not.

**What that means practically.** If you are running a compartmentalised device and inferring
directionality from chamber-level activity, this result says the inference is not supported — at
least on these cultures, with these four readouts. Direction was recovered here only by instrumenting
the channels themselves. We would rather hand a reviewer that finding than a modelling result built
on the assumption it contradicts.

![Microchannel propagation asymmetry by device geometry, REAL recordings.](../outputs/figures/microchannel_asymmetry_real.png){width=52%}

A related null: in 9
<!-- metrics.json: tunnel_rate_tracking.n_devices --> four-chamber cultures with instrumented tunnels, coupling asymmetry shows no relationship with
the rate imbalance between chambers (ρ = −0.0527
<!-- metrics.json: tunnel_rate_tracking.spearman_rho -->, p = 0.728
<!-- metrics.json: tunnel_rate_tracking.spearman_p -->) — so the asymmetry measure is at least not
merely restating which chamber is busier.

![Measured axonal flow against chamber-level readouts, REAL recordings. The chamber measures do not order the cultures by the direction their axons actually fired.](../outputs/figures/axonal_flow_vs_chamber.png){width=58%}

---

## 11. Untreated baselines: what differs before anyone adds a drug

Set the pharmacology aside. Take untreated recordings and ask what separates rat from human, and
dissociated cultures from organoids.

The features split into two groups, and the split is consistent across three separate analyses.
Rate-like features are comparatively stable across species. **Synchrony and burst-structure features
move most** — they are the features that differ most between rat and human baselines, they are the
features §8 shows carry least mechanism information, and they are the features §3 shows are least
reproducible between two implementations of the same definition.

That three-way agreement is, we think, the most actionable thing in this report for experimental
design: if you are choosing a small number of endpoints to compare across species or laboratories,
the burst and synchrony family is where you will have the most trouble, from three independent
directions at once.

One analysis disagrees, and we kept it. The platform contrast (§12) finds that for the *same*
mechanism on two platforms, mean firing rate reverses sign while synchrony agrees. Three analyses say
synchrony is the unreliable family; one says it is the reliable one. We do not resolve that, and the
disagreement is itself reported.

![Per-feature species shift, REAL data: rat against human baseline distributions with cluster-bootstrap intervals. Rate and burst-fraction features move least; synchrony moves most.](../outputs/figures/species_shift.png){width=50%}

![Untreated baseline activity, REAL recordings, by species and preparation. Burst rate is the one feature flat across all three groups.](../outputs/figures/baseline_species_comparison.png){width=58%}

---

## 12. The same mechanism on two platforms

GABA-A antagonists, at 30 µM
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.min_dose_uM --> and above, measured on
two platforms: Kapucu's gabazine and ScreenNeuroPharm's bicuculline.

Mean firing rate goes **up** on one platform and **down** on the other. Synchrony moves the same way
on both.

If you take one thing from this report about cross-laboratory comparison, take this. The mechanism is
the same and the direction of the most commonly reported endpoint is not. Anyone pooling MEA
pharmacology across platforms on the assumption that "GABA-A antagonists raise firing rate" would get
this backwards half the time.

![The same mechanism on two platforms, REAL data: per-feature log2 change for GABA-A antagonists. Mean firing rate reverses sign across platforms; synchrony agrees.](../outputs/figures/platform_contrast_gabaa_antagonist.png){width=58%}

---

## 13. The digital twin: what it is, and what it is for

A "digital twin" here is a simulated culture, not a statistical model. Two compartments of model
neurons — 80% excitatory, 20% inhibitory, each cell leaky and integrate-and-fire with
spike-frequency adaptation — wired densely within each compartment and sparsely to each other
through a directional pathway. Virtual electrodes sit over the sheet and "hear" the neurons within a
radius, and their spike trains go through **the same `features/mea.py`** the real recordings use.
That last point is a deliberate constraint: the twin cannot look good by being measured differently.

What a twin is *for*, in this project, is asking questions that are expensive to ask experimentally:
how many electrodes and how long a recording do you need to see an effect; what would this readout
look like if the wiring were twice as directional; is the quantity you are measuring even capable of
answering the question you are asking.

**Calibration** is the twin's version of training, and it means something different from fitting a
curve. We are not fitting the twin to reproduce one culture — that would be overfitting to one
culture. We are asking it to land inside the *interquartile band* of 48 real chamber baselines
<!-- metrics.json: twin_hd12_calibration.targets_real.n_records -->, i.e. to be an unremarkable
member of the population rather than a copy of a member.

---

## 14. Calibration, and the mistake that took longest to find

A random search over 9 parameters
<!-- metrics.json: twin_hd12_calibration.tuned_parameters -->, each candidate scored on 2 network
seeds <!-- metrics.json: twin_hd12_calibration.n_seeds_per_draw --> across 160 draws
<!-- metrics.json: twin_hd12_calibration.n_draws -->.

| feature | simulated | real band | inside? |
|---|---|---|---|
| synchrony index | 0.8387 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.synchrony_index --> | 0.678 to 0.843 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.synchrony_index.q75 --> | yes |
| burst rate / min | 22.9219 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.burst_rate_per_min --> | 12.627 to 29.726 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.burst_rate_per_min.q75 --> | yes |
| network burst rate / min | 24.0 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.network_burst_rate_per_min --> | 11.692 to 22.697 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.network_burst_rate_per_min.q75 --> | 6% over |
| mean firing rate | 5.9862 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.mfr_hz --> | 6.793 to 21.007 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.mfr_hz.q75 --> | 12% under |
| % spikes in bursts | 83.8409 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.pct_spikes_in_bursts --> | 84.657 to 93.685 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.pct_spikes_in_bursts.q75 --> | 1% under |

**The story behind that table is worth a reviewer's time, because it is a cautionary tale about
simulated cultures generally.**

For a long time the twin was far too synchronous — a synchrony index around 0.96 against a real band
topping out at 0.84 — and it looked like a missing biophysical mechanism. Three candidates were
tried and none of them worked: short-term synaptic depression (the textbook answer for what
terminates a population burst), heterogeneity in axonal conduction delays, and distance-dependent
wiring. The third actually moved synchrony the *wrong way*.

The real cause was not biophysics. It was **cell density and electrode geometry**. The simulated
chamber held 250 neurons in 1.92 mm² — about 130 neurons/mm², where a real dissociated culture runs
roughly 1000–3000 — and the electrode's listening radius had been allowed to grow larger than the
spacing between electrodes. An oversized electrode compensated for a sparse culture in two ways at
once: it pooled about thirty neurons to manufacture a plausible per-electrode firing rate, and
because neighbouring listening areas overlapped, one neuron's spikes were counted by several
electrodes — which inflates any pairwise synchrony measure by construction, with no biology involved.

Fixing the density and forbidding the electrodes to overlap put synchrony inside the real band for
the first time, at 967 neurons per compartment
<!-- metrics.json: twin_hd12_calibration.best.draw.n_per_compartment --> and a listening radius of
92.5135 µm <!-- metrics.json: twin_hd12_calibration.best.draw.detection_radius_um -->.

**Why this matters beyond our simulation.** Overlapping electrode catchment areas inflate measured
synchrony in exactly the same way in a real array. If two electrodes are close enough to pick up the
same units, part of their "synchrony" is the same neuron counted twice. Our fix was to forbid the
overlap; the equivalent question for a real array is whether electrode spacing and your units'
amplitude make that possible.

**How much to trust the calibrated numbers.** The winning parameter set was chosen as the best of
160 candidates on two random seeds, which flatters it. Re-run on 5 seeds
<!-- metrics.json: twin_hd12_calibration.held_out.n_seeds --> the search never saw, its score is
0.1994 <!-- metrics.json: twin_hd12_calibration.held_out.score_mean --> against 0.0756
<!-- metrics.json: twin_hd12_calibration.held_out.selection_score --> at selection. Synchrony holds
up better than the total, at 0.7732
<!-- metrics.json: twin_hd12_calibration.held_out.synchrony_index_mean --> ± 0.0685
<!-- metrics.json: twin_hd12_calibration.held_out.synchrony_index_sd -->, inside the band on 60% of
those seeds <!-- metrics.json: twin_hd12_calibration.held_out.synchrony_index_in_band_fraction -->.
So: a twin whose synchrony sits in the real range *on average*, with run-to-run variation wide
enough that any single simulated recording can fall outside it — much like a single culture.

![SIMULATED against REAL: the calibrated twin's baseline features against the real baseline band from rat two-chamber devices.](../outputs/twin/hd12_calibration_simulated.png){width=54%}

---

## 15. Drug knobs: does the twin respond like a culture?

The twin has three pharmacological knobs — scale inhibitory transmission, scale excitatory
transmission, raise the spike threshold — standing in for a GABA-A drug, a glutamatergic drug and a
sodium-channel blocker. The test is not whether the twin reproduces the *size* of a real drug effect;
it cannot, since the reference comes from a different platform. The test is whether it moves each
feature in the right *direction*, scored only on features whose real direction is established by a
cluster bootstrap.

| knob | features agreeing / scored |
|---|---|
| GABA-A block | 4 <!-- metrics.json: twin_hd13_perturbation.knobs.0.n_features_agreeing --> / 5 <!-- metrics.json: twin_hd13_perturbation.knobs.0.n_features_scored --> |
| GABA-A potentiation | 1 <!-- metrics.json: twin_hd13_perturbation.knobs.1.n_features_agreeing --> / 4 <!-- metrics.json: twin_hd13_perturbation.knobs.1.n_features_scored --> |
| AMPA block | 4 <!-- metrics.json: twin_hd13_perturbation.knobs.2.n_features_agreeing --> / 4 <!-- metrics.json: twin_hd13_perturbation.knobs.2.n_features_scored --> |
| sodium-channel block | 3 <!-- metrics.json: twin_hd13_perturbation.knobs.3.n_features_agreeing --> / 3 <!-- metrics.json: twin_hd13_perturbation.knobs.3.n_features_scored --> |

Blocking excitation and blocking sodium channels behave correctly in every scored feature. Blocking
inhibition is mostly right. **Potentiating inhibition is mostly wrong** — and that is the knob with
the weakest real reference behind it, 17 pairs from two sources.

![SIMULATED drug knobs against REAL pharmacology directions, per feature. Only features whose real direction is established by a cluster bootstrap are scored.](../outputs/twin/hd13_drug_knob_validation_simulated.png){width=52%}

We would read this as: the twin is usable for reasoning about *strong, well-evidenced*
perturbations and is not validated for subtle ones. Reporting one row of a four-row table as "mostly
wrong" is more useful to you than an average.

---

## 16. Running the model backwards, and what that cannot do

The more ambitious use of a twin is inverse: given a recording, infer which knob was turned. This
uses simulation-based inference — simulate thousands of chips with known settings, train a network
to map summary features back to settings, then apply it to real recordings.

**Two honesty checks come first**, and both are the sort of thing that is easy to skip.

*Does the posterior actually learn anything?* "Contraction" measures how much narrower the answer is
than the prior; 0 means nothing was learned. The spike threshold contracts by 0.661
<!-- metrics.json: twin_hd14_inference.contraction.threshold_scale.contraction -->, excitatory transmission by 0.599
<!-- metrics.json: twin_hd14_inference.contraction.ampa_scale.contraction -->, inhibitory transmission by only 0.3
<!-- metrics.json: twin_hd14_inference.contraction.gabaa_scale.contraction -->. Something real is learned about some
knobs and much less about others.

*Is it honest about its own uncertainty?* Rank uniformity asks whether the true value falls inside
the stated interval as often as promised — a model that is over-confident fails it. Three of the five
knobs now pass: the spike threshold at p = 0.6105
<!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.threshold_scale.ks_p -->, excitatory transmission at
0.3128 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.ampa_scale.ks_p --> and the pathway ratio at
0.1517 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.log10_forward_ratio.ks_p -->. Inhibitory
transmission still fails, at 0.0174
<!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.gabaa_scale.ks_p -->.

**That is a recent and instructive change.** Until the calibration described in §14 was corrected,
*no* parameter passed this check and 20% of simulations produced no spikes at all. Simulating at the
density the twin was actually calibrated at — rather than at a smaller, cheaper network — dropped the
silent fraction to 0
<!-- metrics.json: twin_hd14_inference.fraction_silent_simulations --> and moved three parameters from
over-confident to calibrated. The lesson is the same one as §14: a simulation run at the wrong cell
density is not a cheaper version of the right one, it is a different model.

Inhibitory transmission remaining over-confident matters, because it is the knob the pharmacology
sections care about most. Read what follows accordingly.

**Applied to real recordings**, the result was initially alarming. Untreated chips — no drug at all —
came back with an inferred inhibition scale of 0.5219
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median -->, where 1.0 means untreated, on an interval
of 0.2732 to 0.8503
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median_ci95.1 --> that does not reach it. The twin was
reporting that undrugged cultures were substantially disinhibited.

That is not a discovery about cultures. It is the model saying "real recordings do not look like my
simulations", and saying it in the only vocabulary it has.

**The partial fix, and the lesson.** Instead of inferring from a recording's absolute features, infer
from the *change* between a recording and its own baseline — the same paired design §4 uses for the
classifiers, for the same reason. On that footing the untreated chips come back at 0.9108
<!-- metrics.json: twin_hd17_delta_inference.real_data.gabaa_scale_at_baseline.median -->, on an interval of 0.8583 to
0.966 <!-- metrics.json: twin_hd17_delta_inference.real_data.gabaa_scale_at_baseline.median_ci95.1 -->.

Most of the bias is gone and **not all of it**: 0.91 is much closer to neutral than 0.52, but the
interval still stops short of 1.0. We would rather say that than claim the problem is solved. The
paired posterior is also better identified than the absolute one — it contracts by 0.5375
<!-- metrics.json: twin_hd17_delta_inference.held_out_checks.contraction --> against 0.3
<!-- metrics.json: twin_hd14_inference.contraction.gabaa_scale.contraction --> — while still failing
its own rank-uniformity check.

The general lesson is an experimental one: **compare a well to its own
baseline, not to a model of what a well should look like.** It took a digital twin and a failed
inference to rediscover it, and even then it only got most of the way.

**And for some recordings the model simply cannot answer.** Asked for 200
<!-- metrics.json: demo_precompute.posterior_draws.requested --> draws per recording, the posterior failed to produce them for 9
<!-- metrics.json: demo_precompute.posterior_draws.n_short_of_requested --> of 113
<!-- metrics.json: demo_precompute.posterior_draws.n_recordings --> — the worst managing 1
<!-- metrics.json: demo_precompute.posterior_draws.fewest_draws --> draw before giving up, though the number failing completely
was 0 <!-- metrics.json: demo_precompute.posterior_draws.n_unsampled -->. This is not a computing problem. The method works by
proposing parameter settings and keeping the ones that could plausibly have produced the recording
in front of it; when almost nothing it proposes is plausible, there is nothing to keep. The model is
saying, in the only way it can, that those recordings do not look like anything it knows how to
make.

**The uncomfortable part is when this started.** It did not happen before — and what changed was
that the twin got *better*. Fixing the cell density (§14) moved the simulator's baselines into the
real range, and a model fitted to a sharper simulator is a less forgiving one: a recording it cannot
reproduce now sits further outside what it will accept. So an improvement in the forward model made
the backward model harder to apply, which sounds contradictory and is not. It is worth knowing
before anyone reads a twin's inferred parameter as a measurement.

There is a setting that makes all nine failures disappear, by keeping proposals the method would
otherwise reject. We have not used it. The numbers it returns would plot perfectly well and would
mean less than the refusal does.

## 17. What it takes to see directionality at all

One thing a twin is genuinely good for: telling you how much experiment you need. Here the ground
truth is known, because we built the network, so we can ask what it takes to distinguish symmetric
from asymmetric wiring, over 120 simulated networks
<!-- metrics.json: twin_hd16_directionality.n_networks -->.

The result is encouraging and comes with an important catch. With all features and a linear
classifier, the cheapest configuration reaching the target reads asymmetry at an AUC of 0.9219
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_all_features.auc -->
using only 6 electrodes per compartment
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_all_features.electrodes_per_compartment -->
and 10 s of recording
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_all_features.duration_s -->.

**The catch is that most of that is firing rate, not direction.** A directional pathway also delivers
more excitation to its target chamber, so the target simply fires more — and a classifier can read
the asymmetry off the rate difference without knowing anything about direction. When we equalise the
firing rates between chambers and re-run, the discriminability drops substantially.

So the honest version of the design advice is: 6 electrodes and 10 seconds will distinguish these two
wiring patterns, but if you want to claim you measured *direction* rather than *which chamber is
busier*, you need the rate-matched comparison, and it is a much harder measurement.

---

## 18. The negative results, collected

Scattered through a long report, negative results are easy to miss. Collected, they are the part of
this work we would most like challenged — and the part most likely to save someone else an
experiment.

**1. The dose signal is not in the summary features.** This is the most consequential one. §6 showed
the mechanism classifier had no dose relationship, and §16 showed the twin's inverse model could not
recover dose either. It would be natural to blame the model. So we asked directly: across 113
bicuculline recordings <!-- metrics.json: feature_dose_response.n_recordings --> on 16 devices
<!-- metrics.json: feature_dose_response.n_devices -->, does *any* of the nine features move
monotonically with dose?

Using the raw values, 0
<!-- metrics.json: feature_dose_response.verdict.n_features_moving_absolute --> features do; after subtracting each device's
own baseline, 0
<!-- metrics.json: feature_dose_response.verdict.n_features_moving_baseline_subtracted --> features do — in both cases after correcting for testing nine features at
once.

The models were not failing to extract a signal. The signal was not there. If you are planning to
read dose from this feature set on this kind of preparation, that is worth knowing before you run it.

**2. Chamber-level readouts do not recover axonal direction** (§10). Balanced accuracy 0.3854
against 0.5 chance, where 0.6438 would have been detectable.

**3. Mechanism classification does not transfer across platforms** (§5). 0.333 against 0.25 chance
on an unseen platform, from 0.537 within platform.

**4. Domain adaptation made species transfer worse** (§7), both methods, on the problem they are
designed for.

**5. Three plausible biophysical mechanisms did not fix the twin's synchrony** (§14), and the actual
cause was the simulated culture's density and the electrode geometry.

**6. The inverse model is over-confident about inhibition** (§16) — the one knob the pharmacology
sections most depend on, and the only one of five still failing the rank-uniformity check after the
calibration was corrected.

**7. A twin trained on absolute features reported untreated cultures as disinhibited** (§16), which
was a statement about the model, not the cultures.

**8. The posterior could not produce an answer at all for 9
<!-- metrics.json: demo_precompute.posterior_draws.n_short_of_requested --> of 113 real recordings** (§16), and this began only
once the simulator was improved.

---

## 19. What you cannot conclude from this work

- **No clinical claims, and no efficacy or safety claims.** Not hedging: the design cannot support
  them and no analysis here was built to.
- **Not a screening method.** Within-platform mechanism accuracy around 0.54 with an interval from
  0.42 to 0.68, and near-chance performance on an unseen platform, is not a decision tool.
- **Not a general statement about rat-to-human transfer.** 37 training and 32 test pairs support the
  direction, not the magnitude.
- **Not a validated simulator.** The twin lands inside the real band on two of five features, is
  honest about its optimism on unseen seeds, and gets one of four drug knobs mostly wrong.
- **Not a dose model.** See §18.
- **Species coverage is severely unbalanced** — 21,530 rat records against 357 human — and this is a
  property of what is publicly licensed for reuse, not a choice.
- **Most classification rests on 73 pairs from 50 cultures.** Everything downstream inherits that.

---

## 20. How to check any of this

Every number in this document carries a pointer to the key in `outputs/metrics.json` that produced
it, and a test fails if any of them drifts from the file. Nothing here was typed in by hand.

- `python run_all.py --quick` reproduces every figure and metric in under ten minutes on a laptop
  CPU; `python run_all.py` is the full path.
- `data/raw/MANIFEST.json` records for every source file its URL, licence, SHA-256 and download time.
  The build refuses to emit the dataset if any checksum disagrees.
- A Streamlit app lets you look at the records and run the twin interactively; `APP_WALKTHROUGH.pdf`
  is a guided tour.
- `MODELLING_FOR_BIOLOGISTS.pdf` goes further into the classifiers — how a split is built, what a
  cluster bootstrap resamples, how a conformal set is constructed — for readers who want it.

## 21. AI assistance disclosure

This work was produced in a paired workflow with Anthropic's Claude (Claude Opus 5) via Claude Code.
The model wrote most of the code, tests and prose here under step-by-step human direction: the human
set the plan and phase gates, chose the scope, approved each step, and directed the science. Three
practices kept that honest — numbers are emitted into `metrics.json` and re-checked by
`neuromps.report.verify` rather than typed; results are reported as observed, including the ones
that failed; and `report/LAB_NOTEBOOK.md` records what was abandoned.

---

## Appendix A — The record model, data levels and quality control

Every record declares a **data level**: `spike_times`, `per_well_features`, or `fitted_curve`. The
level determines what may be computed from it, and the build enforces that rather than trusting it —
a feature that requires spikes cannot be derived from a record that only holds a fitted curve.

Records are validated against a JSON Schema (`schema/neuromps_schema.json`) at three conformance
levels, so a partially specified source can enter the asset while declaring what it is missing.

**Quality control flags are recorded and never silently applied.** A record can be marked unusable —
537 <!-- metrics.json: data_asset.counts.by_usable.False --> of 21,887
<!-- metrics.json: data_asset.n_records --> are — and the flag travels
with it, with the reason. Analyses choose whether to honour the flag and say which they did. The
alternative, dropping flagged records inside the pipeline, makes the drop invisible to the reader.

Each record also carries the method string that produced its features — the exact burst rule, the
synchrony method and the active-electrode threshold — so two records computed under different
settings can never be compared by accident.

## Appendix B — How the dataset is assembled

One module per source under `src/neuromps/fetch/`, each responsible for locating its files,
re-checking the licence at fetch time, and refusing to proceed if the licence has changed upstream.
A shared downloader hashes every file and caches by hash, so a re-run downloads nothing and any
change is detected.

Two things worth knowing, because both cost real time:

Large archives are read by HTTP range request rather than downloaded — the central directory of a
remote ZIP is fetched, the wanted member located, and only that member pulled. One EPA server
returns HTTP 500 for suffix ranges and, if offered gzip encoding, replies with a plausible
`Content-Range` while streaming the whole 7.5 GB file from byte zero. Both are handled and both have
regression tests.

The ETL produces one record per well per recording, with provenance attached. `run_all.py`
orchestrates fetch, ETL, features, models and twin; `--quick` runs a reduced tier that exercises
every path in under ten minutes.

## Appendix C — Sources and licences

Eight sources, all under licences permitting reuse: `kapucu2022`, `toxcast_mea`, `trujillo2019`,
`sharf2022`, `fofie2025`, `screenneuropharm_bic`, `brewer2021`, `mateus2024`, `vassallo2017`.
Per-source record counts are in `outputs/metrics.json` under `data_asset.counts.by_source`.

*Tukker 2020* was excluded: CC BY-NC, and the competition assigns joint ownership of entries, which
a non-commercial licence does not permit us to grant. Recorded as a skipped source with its reason,
not silently omitted.

## Appendix D — The nine-feature vector, as computed

Referenced from §3. Every value in this report comes from `features/mea.py`, which the twin is also
required to call.

| feature | definition as computed |
|---|---|
| `mfr_hz` | spikes per second per active electrode |
| `burst_rate_per_min` | bursts per minute per active electrode |
| `burst_duration_s` | mean burst duration |
| `spikes_per_burst` | mean spikes in a burst |
| `pct_spikes_in_bursts` | share of spikes falling inside bursts |
| `network_burst_rate_per_min` | network bursts per minute, chance-corrected participation |
| `synchrony_index` | spike-time tiling coefficient, mean over electrode pairs |
| `isi_cv` | coefficient of variation of inter-spike intervals |
| `active_electrodes` | electrodes clearing the activity threshold |

A burst is a run of at least 5 spikes with consecutive intervals under 100 ms. A network burst is a
contiguous set of time bins in which more electrodes fire together than chance allows. Both are
stated rules rather than vendor defaults, and both travel with each record as a method string, so two
records computed under different settings cannot be compared by accident.

## Appendix E — What the repository enforces, rather than documents

Referenced from §20. Each of these is a test that fails, not a convention that is followed.

- **Provenance.** The asset build fails if any record's provenance hash disagrees with the manifest.
- **One feature extractor.** The twin computes features through the same `features/mea.py` as real
  recordings, and — after a defect that survived several phases — through the same coupling measure
  on the same per-electrode input.
- **Interval discipline.** A contract test fails if any headline metric loses its confidence
  interval, and the resample count sits next to every estimate.
- **Cache meaning.** The simulation cache stores its summary names, electrode count, a hash of the
  code defining every summary, and the calibration it was generated under; it refuses to load across
  a change in any of them.
- **Report numbers.** `neuromps.report.verify` re-reads every citation in this document and fails on
  disagreement.
- **No clinical claims.** A test scans the reports for efficacy, diagnostic and treatment language
  and fails unless the line is a disclaimer ruling it out.

## Appendix F — Why coupling asymmetry cannot be read as direction

Referenced from §17. The measure the twin exists to use reports relative firing rate as much as
pathway direction, and the cleanest demonstration holds the wiring **fixed** and changes only the
drive. Lowering the receiving chamber's background drive moves asymmetry from 0.109
<!-- metrics.json: twin_hd11_validation.directionality_sweep.0.asymmetry_mean --> to 0.681
<!-- metrics.json: twin_hd11_validation.directionality_sweep.2.asymmetry_mean --> with the pathway unchanged. Across
the whole sweep drive moves the measure further than wiring does: a span of 0.671
<!-- metrics.json: twin_hd11_validation.confound_summary.raw.drive_span --> against 0.109
<!-- metrics.json: twin_hd11_validation.confound_summary.raw.wiring_span -->.

That is why §17 reports the rate-matched arm separately, and why §10's null on real devices is the
result it is: a chamber-level asymmetry is a statement about which chamber is busier until something
rules that out.

## Appendix G — Posterior calibration, per parameter

Referenced from §16. Contraction is how much narrower the posterior is than the prior; the KS
p-value tests whether the true value falls inside the stated interval as often as promised.

| parameter | contraction | KS p | rank uniformity |
|---|---|---|---|
| `threshold_scale` | 0.661 <!-- metrics.json: twin_hd14_inference.contraction.threshold_scale.contraction --> | 0.6105 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.threshold_scale.ks_p --> | passes |
| `ampa_scale` | 0.599 <!-- metrics.json: twin_hd14_inference.contraction.ampa_scale.contraction --> | 0.3128 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.ampa_scale.ks_p --> | passes |
| `log10_forward_ratio` | 0.438 <!-- metrics.json: twin_hd14_inference.contraction.log10_forward_ratio.contraction --> | 0.1517 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.log10_forward_ratio.ks_p --> | passes |
| `gabaa_scale` | 0.3 <!-- metrics.json: twin_hd14_inference.contraction.gabaa_scale.contraction --> | 0.0174 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.gabaa_scale.ks_p --> | **fails** |
| `inter_in_degree` | 0.269 <!-- metrics.json: twin_hd14_inference.contraction.inter_in_degree.contraction --> | 0.0121 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.inter_in_degree.ks_p --> | **fails** |

Trained on 3,000 simulations <!-- metrics.json: twin_hd14_inference.n_simulations --> at the calibrated cell density,
of which 0 <!-- metrics.json: twin_hd14_inference.fraction_silent_simulations --> produced no spikes. The two that
fail are the connectivity parameter and the inhibitory knob — the latter being the one §16 depends
on, which is why its intervals are not quoted as calibrated.

## Appendix H — Quality-control flags

Referenced from §3 and Appendix A. QC is recorded, never applied silently: a flagged record stays in the asset
with the reason attached.

| flag | records | what it means |
|----------------------------|-------|--------------------------------------------|
| `incomplete_dose_series` | 272 <!-- metrics.json: data_card.flag_counts.incomplete_dose_series --> | the compound's dose series is missing points |
| `too_few_active_electrodes` | 265 <!-- metrics.json: data_card.flag_counts.too_few_active_electrodes --> | under 10 electrodes cleared 5 spikes/min |
| `replicate_divergence` | 221 <!-- metrics.json: data_card.flag_counts.replicate_divergence --> | one replicate departs from its group |
| `outlier_feature_value` | 221 <!-- metrics.json: data_card.flag_counts.outlier_feature_value --> | a feature sits far outside its source's distribution |
| `excluded_well_upstream` | 90 <!-- metrics.json: data_card.flag_counts.excluded_well_upstream --> | the source's own log excluded the well |
| `noisy_electrodes_present` | 11 <!-- metrics.json: data_card.flag_counts.noisy_electrodes_present --> | the source marked electrodes as noisy |

## Appendix I — Digital twin parameters

Referenced from §13 and §14. The 9 marked *calibrated* are exactly the calibration search space
<!-- metrics.json: twin_hd12_calibration.tuned_parameters -->.

| parameter | role | how it was set |
|--------------------------------------|----------------------|----------------------|
| `intra_in_degree` | recurrent inputs per neuron | calibrated, searched over 10-35 |
| `inter_in_degree` | cross-compartment inputs per neuron | inferred, prior 2-20 |
| `forward_ratio` | forward against backward pathway strength | inferred, prior 1-20 in log space |
| `excitatory_weight`, `inhibitory_weight` | synaptic conductance steps | calibrated |
| `background_weight` | Poisson drive | calibrated; drive to the target chamber is a separate scale |
| `background_rate_hz` | Poisson drive rate | calibrated, searched over 1-40 Hz |
| `adaptation_increment` | spike-frequency adaptation step | calibrated |
| `adaptation_tau_ms` | adaptation time constant | **held at 380 ms**, not searched |
| `threshold_spread` | per-neuron excitability heterogeneity | calibrated, searched over 0-0.3 |
| `intra_delay_sd_ms` | spread of the recurrent conduction delay | **held at 0**, one delay for every synapse |
| `connection_sigma_um` | range of the within-compartment wiring rule | **held unset**, i.e. uniform random wiring |
| `detection_radius_um` | how far an electrode hears | calibrated, searched over 40-100 µm |
| `n_per_compartment` | neurons per chamber, i.e. cell density | calibrated, searched over 250-1500 |
| `gabaa_scale`, `ampa_scale`, `threshold_scale` | drug knobs, 1.0 is untreated | inferred |

**Why two of these moved out of the *held* column.** Both had been fixed rather than fitted, and
between them they let the twin look calibrated while being wrong. `n_per_compartment` at 250 is 130
neurons/mm² in a 1.92 mm² chamber, against roughly 1000 to 3000 in a dissociated culture.
`detection_radius_um` could be searched past the electrode pitch, and the search duly went there. A
radius above half the pitch makes neighbouring catchment discs overlap, so one neuron is heard by
several electrodes and contributes identical spike times to each: mean pairwise STTC is then
inflated by the measurement model rather than by anything the network does. The same oversized
electrode was pooling enough neurons to manufacture a per-electrode firing rate that a culture ten
times too sparse could not produce. Capping the radius at half the pitch removes the first artifact
and exposes the second, which is why the two had to change together. Short-term depression was tried
before the geometry was suspected, and §14 reports what it did. The model also carries a
per-synapse delay spread and a distance-dependent connection range; both are left at the neutral
values that reproduce a single conduction delay and uniform wiring, and neither is searched.

## Appendix J — Microchannel detector robustness

Referenced from §10.

**The unit decides the answer.** A chip recorded at DIV 12, 17 and 21 supplies three recordings and
one device. Counting recordings, median |asymmetry| runs from 0.339
<!-- metrics.json: microchannel_directionality.per_recording_naive.by_condition.control.median_abs_asymmetry -->
for control to 0.854
<!-- metrics.json: microchannel_directionality.per_recording_naive.by_condition.rams.median_abs_asymmetry -->
for rams, which looks like a geometry effect. Counting chips, it is 0.617
<!-- metrics.json: microchannel_directionality.comparison.by_condition.control.median_abs_asymmetry -->
against 0.875
<!-- metrics.json: microchannel_directionality.comparison.by_condition.rams.median_abs_asymmetry -->,
Kruskal p = 0.64 <!-- metrics.json: microchannel_directionality.comparison.kruskal_p -->, every
Holm-adjusted comparison against control at 1.0
<!-- metrics.json: microchannel_directionality.comparison.vs_reference_holm.rams -->. Both are in
`metrics.json`, the per-recording one labelled as the wrong unit, because the gap between them is
the result — the pseudo-replication of §16 a third time, on a different dataset.

**Two reasons not to read the null as a measurement.** |asymmetry| is a ratio of counts, so a chip
with few events looks one-sided by chance: across chips it correlates with event count at −0.494
<!-- metrics.json: microchannel_directionality.event_count_confound.spearman_rho -->, and the
better-sampled half leaves 13 chips
<!-- metrics.json: microchannel_directionality.well_sampled_half.n_chips --> at p = 0.086
<!-- metrics.json: microchannel_directionality.well_sampled_half.kruskal_p -->. And control chips
sit in experiments 9–11 by the deposit's design, confounding condition with culture batch. None of
this contradicts the deposit's own paper, whose authors had the electrode map.

**The null is not an artefact of the geometry assumptions.** The detector assumes the channels run
down the array's columns and that all five electrodes fired; the deposit settles neither. Under both
orientations, and again with one electrode allowed to miss, none of the four detectors separates the
conditions — worst case Kruskal p = 0.13074
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_one_missing.kruskal_p -->
beside the headline's 0.63969 — and tolerating a missing electrode triples the events to 1,101,233
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_one_missing.total_events -->
without changing it. The orientation is more than an assumption: down the columns the detector finds
a median of 14
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_strict.median_channels_detected -->
channels per recording against 1
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.row_strict.median_channels_detected -->
along the rows.

## Appendix K — Reproducibility detail

Referenced from §20.

- `--quick` completes in **401 s on a fresh clone** — install, download, build, model, twin and
  demo — against a 600 s budget, on a 20-core machine; the same pipeline takes 1,126 s on the
  two-vCPU runner CI provides, which is the number to plan against. Each run records its own
  per-stage timings (`metrics.run_all_timings_quick.by_stage_s`, `..._full`) rather than this report
  transcribing them, and logs them under `outputs/logs/`; the twin stage dominates, then
  downloading, then the asset build.
- **`--quick` is a path test, not a reproduction of the report.** It fetches the small-tier sources
  only, so it builds a much smaller asset, draws 16 of the 17 figures, and omits six `metrics.json`
  entries whose analyses need either a full-tier download or compartment records the small-tier
  asset lacks; each skips itself with a logged reason rather than failing. The §17 sweep also has
  too few simulations per cell to fit a discriminator, so its `cheapest_configuration` entries are
  written as `null`.
- **Reproducing the numbers in this report takes the full path**: `python run_all.py` after a
  full-tier fetch. `metrics.json` records which mode produced each value, and the two modes write to
  separate roots so neither can overwrite the other.
- Dependencies are pinned in `pyproject.toml`; numpy is held at 2.3.5 because brian2 2.9.0 does not
  import against 2.4.
- Tests run with `pytest` (feature extraction, QC flags, provenance hashing, the MEA simulator, and
  a contract test that fails if any headline metric loses its interval). Appendix D lists what the repository
  enforces rather than documents.

## Appendix L — The rate-only arm: parse and validation

Referenced from §8.

It is the only source in this project that publishes **spike counts per well** rather than spike
times — hiPSC-derived sensory neurons on Axion plates, a baseline and then several readings after a
compound. That makes the rate change the only feature it can supply, which is exactly the feature
under test. **The recording duration is not in the deposit and is not needed**: every model here
consumes log2(exposure / baseline), and with both counts taken over equal-length recordings of one
well the duration cancels out of the ratio. An absolute rate would need it and is not claimed.

The parse yields 291 pairs
<!-- metrics.json: rate_only_arm.n_pairs --> over 48 condition blocks
<!-- metrics.json: rate_only_arm.n_groups -->, each well its own control; `gabaa_agonist_or_pam`
<!-- metrics.json: rate_only_arm.dropped_single_block_classes --> occupies a single block and is
dropped, since a split cannot hold it out. The check that the sheets were read correctly is
biological rather than structural: untreated wells move by a median of −0.1743
<!-- metrics.json: rate_only_arm.by_mechanism.control.median_log2_change --> log2 units while
sodium-channel blockers move by −1.9529
<!-- metrics.json: rate_only_arm.by_mechanism.na_channel_blocker.median_log2_change -->, a
four-fold silencing. A misaligned column would not produce that.

## Appendix M — Task 2 detection by dose

Referenced from §6. The detection rate rises from 0.179 at 0.1 µM to 0.426 at 30 µM, which looks
like a trend until the rank correlation over all doses is computed and comes back at 0.014; the
per-dose counts are in `metrics.json` under
`mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose`.

## Appendix N — Species transfer, all settings

Referenced from §7.

| setting | balanced accuracy | 95% CI |
|---|---|---|
| human, within-species | 0.646 <!-- metrics.json: species_transfer.transfer.none.human_within.balanced_accuracy --> | — |
| rat, within-species | 0.543 <!-- metrics.json: species_transfer.transfer.none.rat_within.balanced_accuracy --> | — |
| rat → human, no adaptation | 0.55 <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy --> | [0.4, 0.6] <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy_ci95 --> |
| rat → human, CORAL | 0.442 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy --> | [0.302, 0.591] <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy_ci95 --> |
| rat → human, per-species standardisation | 0.408 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy --> | [0.267, 0.568] <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy_ci95 --> |
