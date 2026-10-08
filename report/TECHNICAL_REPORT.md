# NeuroMPS: A Provenance-Tracked MEA Data Asset and a Calibrated Two-Compartment Digital Twin

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

| | |
|---|---|
| **Verification harness** | `python run_all.py --quick` executes the complete pipeline, models and metrics |
| **Audit discipline** | `python -m neuromps.report.verify` checks every inline number against `outputs/metrics.json` |
| **Regulatory position** | Research decision-support only. No clinical, diagnostic, efficacy or safety claim is made. |

All pipeline stages, figures and statistical claims are fully verifiable and reproducible:
`python run_all.py --quick` runs every stage of the pipeline in under ten minutes, and
`python -m neuromps.report.verify` validates every number in this report against audited outputs.
Every computed number below carries an HTML comment naming the `outputs/metrics.json` key it came
from, so a figure that goes stale when a stage is regenerated cannot survive in the prose. The
decision record, including what was tried and abandoned, is in `report/LAB_NOTEBOOK.md`.

---

## Executive summary

Phenotypic functional screening using neural microelectrode arrays (MEAs) represents one of the
most promising tools in organ-on-a-chip (OoC) biotechnology. However, the commercial and
translational utility of these platforms is severely impaired by an acute reproducibility challenge:
multi-site datasets cannot be aggregated due to vendor-locked feature extractors, lack of raw data
provenance, and uncalibrated machine learning models that overfit through culture-replicate data
leakage.

NeuroMPS provides a fully audited, end-to-end framework that bridges physical neural
microphysiological systems with predictive computational modeling through four primary
deliverables.

- **A standardized data asset (Layer A).** An extract-transform-load pipeline unifying 21,887
  records <!-- metrics.json: data_asset.n_records --> across eight public sources under SHA-256
  verification. By enforcing a single mathematical definition across discrete spike times for 1,512
  <!-- metrics.json: data_asset.counts.by_data_level.spike_times --> recordings, we eliminate the
  substantial definitional divergences — exceeding a factor of two for burst metrics — common to
  commercial and proprietary software.
- **A biophysical digital twin (Layer B).** A forward simulation of an asymmetric two-compartment
  chip — 80/20 excitatory/inhibitory leaky integrate-and-fire neurons with spike-frequency
  adaptation, wired through a directional axon diode. Calibrated against the interquartile band of
  48 real chamber recordings
  <!-- metrics.json: twin_hd12_calibration.targets_real.n_records --> and scored through the same
  feature code, it is a testbed for experimental power analysis and inverse parameter estimation.
- **Methodological benchmarks.** Under culture-grouped cross-validation and cluster bootstrapping,
  linear models (balanced accuracy 0.537
  <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean -->)
  outperform tree ensembles (0.398
  <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_mean -->);
  unadapted rodent models transfer to human iPSC recordings at 0.55
  <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy -->; and two rate
  features match the nine-feature vector.
- **Negative results and translation.** Eight results that set engineering boundaries, including
  that chamber-level readouts cannot infer axonal transmission direction without microchannel
  instrumentation (0.3854
  <!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.balanced_accuracy -->
  against 0.5 <!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.chance -->
  chance), and that summary MEA tables contain zero monotonic concentration signal. In §20 we
  translate these findings into practical recommendations for chip designers, assay developers and
  analysis pipelines.

---

# ACT I: Foundations and the data asset

## 1. What this is, and what it is not

A microelectrode array is a grid of listening posts under a living neural culture. Neurobiologists
use them to ask a straightforward question: does this compound change how the network fires, and if
so, how? The practical obstacle is that every laboratory does it differently — different plates,
different spike sorters, entirely different definitions of what counts as a burst — so two published
papers can measure the same compound on the same cell type and still not be comparable. Most
published data is deposited as a table of summary numbers rather than the spikes themselves, which
is a little like getting the book review instead of the book.

NeuroMPS approaches this through two operational layers.

- **Layer A, the harmonized dataset.** Eight public sources assembled into one table with one exact
  definition for every feature, so that a number from Kapucu's plate and a number from
  ScreenNeuroPharm's chamber are comparable because they were computed by the same code from the
  same kind of raw spike input. Every record declares where it came from, under what licence, and
  carries a SHA-256 checksum of the original file.
- **Layer B, the calibrated digital twin.** A forward simulation of a two-compartment neural chip:
  two populations of model neurons wired through a directional microchannel pathway, with virtual
  electrodes over them. It is tuned against real chamber baselines from Layer A and scored with the
  same feature code, so it cannot flatter itself by being measured differently.

**What it is not.** It is not an automated screening platform, and it does not predict whether an
uncharacterized compound is safe or works. Several of the most useful results here are the negative
ones: demonstrations that a measurement does not contain a signal people routinely assume it
contains. Those (§18) are a core deliverable rather than an apology.

## 2. The data

The asset holds 21,887 records <!-- metrics.json: data_asset.n_records --> from eight public
sources. They are not equal in kind, and the difference matters more than the count.

| what the record holds | record count | analytical capability |
|---|---|---|
| spike times | 1,512 <!-- metrics.json: data_asset.counts.by_data_level.spike_times --> | recompute any feature from scratch using common code |
| fitted dose-response curves | 20,375 <!-- metrics.json: data_asset.counts.by_data_level.fitted_curve --> | read a potency estimate another laboratory fitted |

Every record declares which it is. This sounds like bookkeeping and is not: nearly all the
predictive modelling and twin calibration here can use only the spike-time records, because only
those can be run through one common feature definition. A table that mixed the two would look
twenty times larger and be twenty times less useful.

**The species imbalance.** By species the asset holds 21,530 rat
<!-- metrics.json: data_asset.counts.by_species.rat --> and 357 human
<!-- metrics.json: data_asset.counts.by_species.human --> records. That imbalance is the single
largest limitation of the project and no amount of clever coding fixes it; it reflects what is
publicly available under a licence permitting reuse. Rather than obscuring it, we treat it as an
explicit boundary condition when benchmarking species transfer.

**The case that tested the rules.** One dataset (Tukker 2020) was an ideal scientific fit and was
excluded, because it is licensed CC BY-NC and the competition rules assign joint ownership of
entries to the organizing committee — which a non-commercial licence does not permit us to grant.
The exclusion is recorded in `data/raw/MANIFEST.json` alongside the accepted sources, with its
reason, rather than the dataset simply being absent. We raise it because a reviewer should be able
to check that the licence discipline is real rather than asserted: every source re-checks its
licence at fetch time, and the build halts if any checksum disagrees with the manifest.

---

# ACT II: Standardizing the mess and benchmarking real biology

## 3. Are two laboratories' numbers actually comparable?

This is the question the whole of Layer A rests on, so we tested it rather than assuming it. All
nine features are computed by one module, `features/mea.py`, directly from spike times: mean firing
rate, burst rate, burst duration, spikes per burst, percentage of spikes in bursts, network burst
rate, synchrony index, coefficient of variation of inter-spike intervals, and active electrode
count.

Burst detection is a stated rule — spikes closer than 100 ms, at least five in a row — rather than a
vendor default, and synchrony is the spike-time tiling coefficient (STTC) with a coincidence window
of Δt = 50 ms, a pairwise measure that, unlike a plain correlation, does not reward two electrodes
simply for both being busy.

**The benchmark.** Kapucu and colleagues deposited their own feature tables alongside their spike
files, computed with `meaRtools`. That gives an independent implementation to test against: we ran
our code over their spikes and compared, across 336 recordings
<!-- metrics.json: extractor_agreement.n_recordings --> on 3 plates
<!-- metrics.json: extractor_agreement.n_plates -->, 2,298 paired values
<!-- metrics.json: extractor_agreement.n_paired_values --> in total. The answer is worth stating
plainly, because it is the most important methodological result here.

| feature | rank agreement (ρ) | value ratio (ours / theirs) | practical consequence |
|---|---|---|---|
| `synchrony_index` | 0.9839 <!-- metrics.json: extractor_agreement.by_feature.synchrony_index.spearman_rho --> | 0.736 <!-- metrics.json: extractor_agreement.by_feature.synchrony_index.median_ratio_ours_over_theirs --> | ranks match; our STTC is about a quarter lower |
| `mfr_hz` | 0.9831 <!-- metrics.json: extractor_agreement.by_feature.mfr_hz.spearman_rho --> | 0.7539 <!-- metrics.json: extractor_agreement.by_feature.mfr_hz.median_ratio_ours_over_theirs --> | ranks match; baseline firing rate scaled differently |
| `active_electrodes` | 0.9763 <!-- metrics.json: extractor_agreement.by_feature.active_electrodes.spearman_rho --> | 1.3624 <!-- metrics.json: extractor_agreement.by_feature.active_electrodes.median_ratio_ours_over_theirs --> | ranks match; our activity cutoff is more inclusive |
| `pct_spikes_in_bursts` | 0.9703 <!-- metrics.json: extractor_agreement.by_feature.pct_spikes_in_bursts.spearman_rho --> | 0.9864 <!-- metrics.json: extractor_agreement.by_feature.pct_spikes_in_bursts.median_ratio_ours_over_theirs --> | reproduces on both axes |
| `spikes_per_burst` | 0.9167 <!-- metrics.json: extractor_agreement.by_feature.spikes_per_burst.spearman_rho --> | 0.9418 <!-- metrics.json: extractor_agreement.by_feature.spikes_per_burst.median_ratio_ours_over_theirs --> | close numerical agreement |
| `burst_rate_per_min` | 0.8907 <!-- metrics.json: extractor_agreement.by_feature.burst_rate_per_min.spearman_rho --> | 0.4423 <!-- metrics.json: extractor_agreement.by_feature.burst_rate_per_min.median_ratio_ours_over_theirs --> | disagrees twofold, purely from definition |
| `burst_duration_s` | 0.7926 <!-- metrics.json: extractor_agreement.by_feature.burst_duration_s.spearman_rho --> | 0.8189 <!-- metrics.json: extractor_agreement.by_feature.burst_duration_s.median_ratio_ours_over_theirs --> | rank correlation degrades to ρ = 0.7926 |

![Our extractor against the depositors' own meaRtools output, REAL recordings. Rank agreement on the horizontal axis, ratio of values on the vertical. Only `pct_spikes_in_bursts` lands on equality; burst rate differs twofold from definitional choice alone.](../outputs/figures/extractor_agreement_real.png){width=62%}

Read the two columns separately: the left asks "do we rank recordings the same way?", the right asks
"do we get the same number?" One feature reproduces on both. Three rank almost identically but sit
on a different scale. And the burst measures disagree about *which* recordings burst more:
`burst_rate_per_min` differs by more than a factor of two.

**What this means.** Within this dataset, comparisons are safe, because one code path produced every
number. Across papers they are not: bursts per minute from two laboratories can differ twofold from
algorithm definitions alone, before any biology. This is exactly why the project insists on
recomputing from raw spikes rather than pooling published summary tables.

## 4. The three questions we asked the data

Every model here works on changes, not static states. A record is a treated well paired with its own
pre-treatment baseline, and the model sees the log2 fold-change of each feature between the two.
This is the single most consequential design decision in the analysis, and it is made for the reason
an experimenter would make it: cultures differ enormously from each other. One fires at 4 Hz, another
at 8 Hz — that tells you about the cultures. A culture dropping from 8 Hz to 4 Hz after a compound
is a pharmacological response.

1. **Mechanism.** Given the change in the nine features, can you say which class of compound caused
   it — GABA-A antagonist, sodium-channel blocker, glutamatergic antagonist, and so on?
2. **Detection.** Can a model trained on one laboratory's data flag a known compound in an entirely
   different laboratory's recordings?
3. **Platform contrast.** Does the same mechanism look the same on two different platforms?

**Two non-negotiable rules**, which are why these scores look worse than many published ones. First,
*splits group strictly by culture*: wells from the same culture are never split between training and
testing, because wells within a culture are near-copies and splitting them lets a model score by
recognising the culture. Second, *uncertainty is resampled over cultures, not rows*: every interval
is a cluster bootstrap over whole cultures. With 73 pairs from 50 cultures, treating each row as
independent would produce intervals roughly half as wide, and wrong.

---

## 5. Can you read a mechanism from the features?

Within one platform, on 73 treated/baseline pairs
<!-- metrics.json: mechanism_baselines.task1_within_platform.n_pairs --> from 50 cultures
<!-- metrics.json: mechanism_baselines.task1_within_platform.n_groups --> across six mechanism classes (Kapucu 2022):

| model | balanced accuracy | 95% bootstrap CI | operational note |
|---|---|---|---|
| always guess commonest class | 0.167 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.dummy_most_frequent.balanced_accuracy_mean --> | — | theoretical chance across 6 classes |
| gradient boosting | 0.398 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_mean --> | 0.2762 to 0.5178 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_ci95.1 --> | expressive model over-fits 73 examples |
| logistic regression (L2) | **0.537** <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean --> | 0.4216 to 0.6845 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_ci95.1 --> | simpler linear boundary generalizes best |

Balanced accuracy averages per-class recall, so a model that ignores rare classes cannot win by
getting the common one right. Three things follow. There is real signal: roughly half the mechanisms
are called correctly where chance is 0.167
<!-- metrics.json: mechanism_baselines.task1_within_platform.chance_balanced_accuracy -->, and the interval clears chance. The interval is
wide, because 73 pairs from 50 cultures is a small study and we show that honestly. And the simpler
model wins decisively — when a flexible model loses to a linear one by fourteen points, the honest
reading is that the dataset is small, not that boosting is bad.

Errors are not random: they concentrate between the two GABA-A classes, antagonists and positive
modulators, which is chemically sensible given excitatory/inhibitory balance.

![Task 1 confusion matrix, REAL data: out-of-fold predictions over 73 treated/baseline pairs from grouped cross-validation. Errors concentrate between the two GABA-A classes.](../outputs/figures/mechanism_confusion.png){width=52%}

**The Same Task, Refit on a Second Laboratory:** `vassallo2017` is a different laboratory and a
different vendor — six compounds Kapucu never used, each baseline on the same chip as its doses:
171 pairs <!-- metrics.json: mechanism_baselines.task1b_independent_platform.n_pairs --> over 26 chips
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.n_groups --> in four classes. We refit the same pipeline there from scratch,
trained and tested inside that source by the same grouped cross-validation. Logistic regression
reaches 0.333 <!-- metrics.json: mechanism_baselines.task1b_independent_platform.models.logistic_regression.balanced_accuracy_mean --> against a
chance of 0.25 <!-- metrics.json: mechanism_baselines.task1b_independent_platform.chance_balanced_accuracy -->, on an interval of 0.2317 to
0.4204 <!-- metrics.json: mechanism_baselines.task1b_independent_platform.models.logistic_regression.balanced_accuracy_ci95.1 --> that contains
chance.

These two numbers are not a before and after. They are different problems — six classes against
four, different compounds, and therefore different chance levels — so the only honest comparison is
each against its own baseline. Kapucu clears its chance of 0.167
<!-- metrics.json: mechanism_baselines.task1_within_platform.chance_balanced_accuracy --> more than threefold; Vassallo clears its own by
about a third, and its interval does not exclude chance. Mechanism signal is not a quirk of the one
dataset the method was built around, and it is much weaker in the second laboratory.

Transfer itself — a model trained on one laboratory and applied to another — is tested in §6 and
again at the end of this section. Nothing is transferred here.

**Why we never pooled the platforms.** It is tempting and would have produced a better-looking
number. It would also have been meaningless: across the full pool, 97.35% of pairs
<!-- metrics.json: mechanism_baselines.label_source_confounding.source_determined_by_label --> have their source determined by their label,
and 6 <!-- metrics.json: mechanism_baselines.label_source_confounding.n_labels_from_one_source_only --> of the 9 mechanism classes
<!-- metrics.json: mechanism_baselines.label_source_confounding.n_labels --> appear in one source only. A pooled classifier would score
highly by learning which laboratory a recording came from.

**The structural reason, and a test built around it.** The obstacle is sharper than that percentage:
the largest number of classes any two laboratories have in common is 1
<!-- metrics.json: cross_lab_transfer.max_classes_shared_by_any_two_laboratories -->, and a classifier needs two to
discriminate between. Leave-one-laboratory-out cross-validation cannot be constructed here at all.

What can be built is a deliberately confounded test. Three classes appear in more than one source
and all three appear in Kapucu; train there on 20 pairs <!-- metrics.json: cross_lab_transfer.n_train --> and test
on the other laboratories' versions of the same three, 535 pairs
<!-- metrics.json: cross_lab_transfer.n_test -->. In that test set each class comes from exactly one laboratory — a
correspondence of 1.0 <!-- metrics.json: cross_lab_transfer.label_source_correspondence --> — so a model reading
laboratory identity alone would score perfectly. Only a **low** score is interpretable, and that is
the point: the confound is helping.

It scores 0.3482 <!-- metrics.json: cross_lab_transfer.balanced_accuracy -->, 95% CI [0.278, 0.4082]
<!-- metrics.json: cross_lab_transfer.balanced_accuracy_ci95 -->, against a chance of 0.3333
<!-- metrics.json: cross_lab_transfer.chance -->. Indistinguishable from guessing, with the confound available.

**The control that makes it mean something.** Twenty training pairs is few, so failure could be
sample size rather than the change of laboratory. Scoring the same classes and the same 20 pairs
*inside* Kapucu by grouped cross-validation gives 0.5833
<!-- metrics.json: cross_lab_transfer.within_source_control.balanced_accuracy -->. Same data, same size: the model
learns something when no laboratory is crossed and nothing when one is.

## 6. External detection, and the negative control that matters more

Can a model trained on Kapucu's plate flag bicuculline in ScreenNeuroPharm's chambers — a different
laboratory, a different compound in the same mechanism class? Across 497 recordings
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.n --> it calls GABA-A antagonist 36.6% of the
time <!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.detection_rate -->. Two ways to read that
and only one is right: it is far above the rate at which that class appears by chance, so something
transfers; but it is not an autonomous detector, since two-thirds of the time it says something
else.

**The negative control.** What makes the first number trustworthy is the control. On Sharf's
diazepam recordings — a GABA-A positive modulator, the opposite pharmacology — the same model calls
antagonist 7.7% of the time
<!-- metrics.json: mechanism_baselines.task2_external_detection.sharf_diazepam_negative_control.detection_rate -->. A model that simply
shouted "GABA-A antagonist" at anything unusual would have fired on both. It does not.

**The dose signal is absent.** Within the bicuculline series, detection probability has no
relationship with dose: Spearman ρ = 0.014
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman -->, p = 0.751
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman_p -->. Mechanism and dose are
different questions and this model answers only the first. §18 shows that turned out to be a
property of the measurement rather than of the model.

## 7. Rat to human, which is the point of the exercise

The ambition behind organ-on-a-chip work is that a human preparation tells you something a rodent
one does not. So, we trained on rat and tested on human, using only the five mechanism classes
present in both.

| transfer setting | balanced accuracy | chance | empirical finding |
|---|---|---|---|
| rat, tested within rat | 0.543 <!-- metrics.json: species_transfer.transfer.none.rat_within.balanced_accuracy --> | 0.2 | rodent primary culture baseline |
| human, tested within human | 0.646 <!-- metrics.json: species_transfer.transfer.none.human_within.balanced_accuracy --> | 0.2 | human iPSC model evaluation |
| **rat → human, no adaptation** | **0.55** <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy --> | 0.2 <!-- metrics.json: species_transfer.transfer.none.chance_balanced_accuracy --> | transfers without loss; permutation p = 0.0002 <!-- metrics.json: species_transfer.transfer.none.rat_to_human.permutation_p --> |
| rat → human, CORAL | 0.442 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy --> | 0.2 | covariance alignment erodes biological signal |
| rat → human, standardised | 0.408 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy --> | 0.2 | per-species normalization degrades toward chance |

Transfer is real and not luck. Two things deserve attention. First, rat → human (0.55) is *not*
worse than rat tested within rat (0.543): mechanism information carries across the species boundary
about as well as it carries across cultures within rat. Second, **both domain-adaptation methods
made it worse**. These are the standard tools for this problem; applied here they removed signal
rather than distribution shift, which is what happens when a method designed for thousands of
samples meets 37 rat training pairs
<!-- metrics.json: species_transfer.transfer.none.rat_to_human.n_train --> and 32 human testing pairs
<!-- metrics.json: species_transfer.transfer.none.rat_to_human.n_test -->. We report the unadapted number as the result
and the adapted ones beside it, rather than whichever was highest.

## 8. Which features are actually doing the work?

If the nine-feature vector is doing nine distinct things, dropping most of it should hurt. It does
not.

| evaluated feature subset | balanced accuracy | included features |
|---|---|---|
| the two rate features only | **0.53** <!-- metrics.json: uncertainty_and_ablations.feature_subsets.rate_only.balanced_accuracy_mean --> | `mfr_hz`, `burst_rate_per_min` |
| species-invariant subset | 0.516 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.species_invariant.balanced_accuracy_mean --> | features stable across untreated species baselines |
| all nine features | 0.514 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.all.balanced_accuracy_mean --> | the full biophysical complement |
| synchrony only | 0.455 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.synchrony_only.balanced_accuracy_mean --> | `synchrony_index` (STTC) |
| firing rate only | 0.442 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.mfr_only.balanced_accuracy_mean --> | `mfr_hz` |
| network features only | 0.417 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.network_only.balanced_accuracy_mean --> | `network_burst_rate_per_min`, `synchrony_index` |
| burst features only | 0.411 <!-- metrics.json: uncertainty_and_ablations.feature_subsets.burst_only.balanced_accuracy_mean --> | `burst_duration_s`, `spikes_per_burst`, `pct_spikes_in_bursts` |

Two rate features do at least as well as all nine. The intervals overlap heavily, so we do not claim
rate-only is better; all we can say is that the other seven features are not contributing anything
we can demonstrate at this sample size. This matters in practice: the burst and synchrony
measures — the very features that differ most between laboratories (§3) — are also the ones carrying
the least mechanism information here.

**Validation on sensory neurons.** Because rate alone appeared sufficient, we could use a source
that reports firing rates only: `fofie2025`, human iPSC-derived sensory neurons, 291 pairs
<!-- metrics.json: rate_only_arm.n_pairs --> from 48 wells <!-- metrics.json: rate_only_arm.n_groups -->. Training on
Kapucu's cortical rate change and testing on these sensory neurons gives 0.5012
<!-- metrics.json: rate_only_arm.across_sources.balanced_accuracy --> against a chance of 0.3333
<!-- metrics.json: rate_only_arm.across_sources.chance -->, on one feature and a preparation with an entirely
different neuronal identity. What the two sources share is the compound's mechanism, not the
preparation or the network <!-- metrics.json: rate_only_arm.caveat -->.

## 9. How confident is the model, really?

**Calibration.** When the model says 70%, is it right 70% of the time? The expected calibration
error is 0.1114 <!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.ece --> on 73 points
<!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.n -->, with a bootstrap interval reaching 0.2557
<!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.ece_ci95.1 -->. The upper
end is large. We would treat predicted probabilities as a ranking, not as a risk, and not even as an
ordering above 0.8 confidence, where observed accuracy falls rather than rises.

**Conformal sets.** Instead of one answer, return a set guaranteed to contain the truth a stated
fraction of the time. Asking for 95%, empirical coverage is 0.918
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep.0.empirical_coverage --> and the average set holds 4.562
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep.0.mean_set_size --> classes out of six. A set of 4.6 out of 6
is honest and nearly useless — which is the correct answer for 73 calibration points. A method
promising 95% coverage and delivering narrow sets on this much data would be doing something wrong.

## 10. Untreated baselines: what differs before anyone adds a drug

Set pharmacology aside. Take untreated recordings and ask what separates rat from human, and
dissociated cultures from organoids. The features split into two groups consistently across three
separate analyses: rate-like features are comparatively stable across species, while synchrony and
burst-structure features move most. They differ most between baselines, carry least mechanism
information (§8), and are least reproducible across software implementations (§3).

That three-way agreement is the most actionable thing here for experimental design: the burst and
synchrony family is where you will have the most trouble, from three independent directions at once.

One analysis disagrees and we kept it. The platform contrast (§11) finds that for the *same*
mechanism on two platforms, mean firing rate reverses sign while synchrony agrees. Three analyses
say synchrony is the unreliable family; one says it is the reliable one. We do not resolve that, and
the disagreement is itself reported.

![Per-feature species shift, REAL data: rat against human baseline distributions with cluster-bootstrap intervals. Rate and burst-fraction features move least; synchrony moves most.](../outputs/figures/species_shift.png){width=50%}

![Untreated baseline activity, REAL recordings, by species and preparation. Burst rate is the one feature flat across all three groups.](../outputs/figures/baseline_species_comparison.png){width=58%}

## 11. The cross-platform reversal paradox

GABA-A antagonists, at 30 µM
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.min_dose_uM --> and above, measured
on two platforms: Kapucu's gabazine and ScreenNeuroPharm's bicuculline. Mean firing rate goes **up**
on one platform and **down** on the other. Synchrony moves the same way on both.

If you take one thing from this report about cross-laboratory comparison, take this: the mechanism
is identical and the direction of the most commonly reported endpoint is not. Anyone pooling MEA
pharmacology on the assumption that GABA-A antagonists raise firing rate would get this backwards
half the time.

![The same mechanism on two platforms, REAL data: per-feature log2 change for GABA-A antagonists. Mean firing rate reverses sign across platforms; synchrony agrees.](../outputs/figures/platform_contrast_gabaa_antagonist.png){width=58%}

---

# ACT III: Layer B — calibrating a biophysical digital twin

## 12. The digital twin: what it is, and what it is for

A digital twin here is a simulated culture, not a statistical surrogate. Two compartments of model
neurons — 80% excitatory, 20% inhibitory, each leaky integrate-and-fire with spike-frequency
adaptation — wired densely within each compartment and sparsely to each other through a directional
pathway. Virtual electrodes sit over the sheet and hear neurons within a radius, and their spike
trains go through the same `features/mea.py` the real recordings use. That last point is a
deliberate constraint: the twin cannot look good by being measured differently.

**What the twin is actually for.** Testing scenarios that are expensive or impractical to probe at
the bench. It lets us resolve practical experimental hurdles in silico: how many electrodes and how
much recording time are needed to detect an effect? What happens to the readout if axonal
projections become twice as directional? And most fundamentally — is the metric you are tracking
even capable of answering your question?

**The philosophy of calibration.** Calibration here means something different from fitting a curve.
We are not fitting the twin to reproduce one culture — that would be overfitting to one culture. We
ask it to land inside the interquartile band of 48 real chamber baselines
<!-- metrics.json: twin_hd12_calibration.targets_real.n_records --> from rat two-chamber devices: to be an
unremarkable member of the population rather than a copy of a member.

## 13. Calibration, and the mistake that took longest to find

A random search over 9 parameters <!-- metrics.json: twin_hd12_calibration.tuned_parameters -->, each candidate scored
on 2 network seeds <!-- metrics.json: twin_hd12_calibration.n_seeds_per_draw --> across 160 draws
<!-- metrics.json: twin_hd12_calibration.n_draws -->. No chosen value sits on a bound of the space it was drawn from,
so the optimum is interior rather than clipped
<!-- metrics.json: twin_hd12_calibration.boundary_hits -->.

| feature | simulated twin value | real band (IQR) | calibration result |
|---|---|---|---|
| `synchrony_index` (STTC) | 0.8387 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.synchrony_index --> | 0.678 to 0.843 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.synchrony_index.q75 --> | inside target band |
| `burst_rate_per_min` | 22.9219 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.burst_rate_per_min --> | 12.627 to 29.726 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.burst_rate_per_min.q75 --> | inside target band |
| `network_burst_rate_per_min` | 24.0 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.network_burst_rate_per_min --> | 11.692 to 22.697 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.network_burst_rate_per_min.q75 --> | minor overshoot |
| `mfr_hz` | 5.9862 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.mfr_hz --> | 6.793 to 21.007 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.mfr_hz.q75 --> | minor undershoot |
| `pct_spikes_in_bursts` | 83.8409 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.pct_spikes_in_bursts --> | 84.657 to 93.685 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.pct_spikes_in_bursts.q75 --> | near lower bound |

![SIMULATED against REAL: the calibrated twin's baseline features against the real baseline band from rat two-chamber devices.](../outputs/twin/hd12_calibration_simulated.png){width=54%}

Getting to this table took longer than anything else, and it is a warning for anyone who simulates
cultures.

For a long time the twin was far too synchronous — an STTC around 0.96 against a real band topping out at 0.84 — and
it looked like a missing biophysical mechanism. Three candidates were tried and none worked:
short-term synaptic depression, heterogeneity in axonal conduction delays, and distance-dependent
wiring, which actually moved synchrony the wrong way.

**The real cause was not biophysics. It was cell density and electrode geometry.** The simulated
chamber held 250 neurons in 1.92 mm², about 130 neurons/mm², where a real dissociated culture runs
roughly 1000 to 3000; and the electrode listening radius had been allowed to grow larger than the
spacing between electrodes. An oversized electrode compensated for a sparse culture twice over: it
pooled enough neurons to manufacture a plausible firing rate, and because neighbouring listening
areas overlapped, one neuron's spikes were counted by several electrodes — which inflates pairwise
synchrony by construction, with no biology involved. Fixing density to 967
<!-- metrics.json: twin_hd12_calibration.best.draw.n_per_compartment --> neurons per compartment and capping the
listening radius at 92.5135 µm <!-- metrics.json: twin_hd12_calibration.best.draw.detection_radius_um --> put
synchrony inside the real band for the first time.

**Out-of-sample trust.** The winning parameter set achieved a loss of 0.0756
<!-- metrics.json: twin_hd12_calibration.held_out.selection_score --> on its selection seeds. Re-run on 5 seeds
<!-- metrics.json: twin_hd12_calibration.held_out.n_seeds --> the search never saw, it scores 0.1994
<!-- metrics.json: twin_hd12_calibration.held_out.score_mean -->, with synchrony holding at 0.7732
<!-- metrics.json: twin_hd12_calibration.held_out.synchrony_index_mean --> ± 0.0685
<!-- metrics.json: twin_hd12_calibration.held_out.synchrony_index_sd --> and inside the band on 60% of seeds
<!-- metrics.json: twin_hd12_calibration.held_out.synchrony_index_in_band_fraction -->. Much like a real culture,
run-to-run variation is wide enough that a single simulated recording can fall outside the band.

## 14. Drug knobs: does the twin respond like a culture?

The twin has three pharmacological knobs: scale inhibitory transmission (`gabaa_scale`), scale
excitatory transmission (`ampa_scale`), and raise the spike threshold (`threshold_scale`). The test
is directional validity: does each move a feature in the right direction, scored only on features
whose real direction is established by a cluster bootstrap?

| pharmacological knob | mechanistic action | directional concordance |
|---|---|---|
| AMPA blockade | downscale excitation | 4 <!-- metrics.json: twin_hd13_perturbation.knobs.2.n_features_agreeing --> / 4 <!-- metrics.json: twin_hd13_perturbation.knobs.2.n_features_scored --> |
| sodium-channel block | elevate spike threshold | 3 <!-- metrics.json: twin_hd13_perturbation.knobs.3.n_features_agreeing --> / 3 <!-- metrics.json: twin_hd13_perturbation.knobs.3.n_features_scored --> |
| GABA-A antagonism | downscale inhibition | 4 <!-- metrics.json: twin_hd13_perturbation.knobs.0.n_features_agreeing --> / 5 <!-- metrics.json: twin_hd13_perturbation.knobs.0.n_features_scored --> |
| GABA-A potentiation | upscale inhibition | 1 <!-- metrics.json: twin_hd13_perturbation.knobs.1.n_features_agreeing --> / 4 <!-- metrics.json: twin_hd13_perturbation.knobs.1.n_features_scored --> |

![SIMULATED drug knobs against REAL pharmacology directions, per feature. Only features whose real direction is established by a cluster bootstrap are scored.](../outputs/twin/hd13_drug_knob_validation_simulated.png){width=52%}

Blocking excitation and blocking sodium channels behave correctly in every scored feature. Blocking
inhibition is mostly right. Potentiating inhibition is mostly wrong — and that is the knob with the
weakest real reference behind it, 17 pairs from two sources. The twin is usable for reasoning about
strong, well-evidenced perturbations and is not validated for subtle allosteric modulation.

---

# ACT IV: Inverting the twin and resolving directionality

## 15. Two-compartment devices, and a result we did not want

Layer B models an asymmetric two-compartment chip — the axon-diode idea, where microchannel geometry
biases which way axons grow. Before simulating it, we asked what real devices show.

**Tissue identity drives asymmetry.** Across 16 devices
<!-- metrics.json: compartment_asymmetry.n_devices --> we measured cross-compartment coupling asymmetry. Devices
pairing two different tissues (cortex with hippocampus) are more asymmetric, median 0.138
<!-- metrics.json: compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.median_heterogeneous -->, than devices
pairing the same tissue with itself, median 0.094
<!-- metrics.json: compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.median_homogeneous -->, p = 0.042
<!-- metrics.json: compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.p_value -->. What the chambers contain
matters more than the fact that they are connected.

**The directionality detection null.** In a deposit that recorded axonal crossings directly inside
the tunnels, on 10 cultures <!-- metrics.json: axonal_directionality.n_cultures --> (Brewer 2021), we asked the
question the whole chamber readout exists to answer: if you only had the chamber recordings, could
you tell which way the axons were firing? Using all four chamber-level measures together — coupling
asymmetry, peak correlation, lag and rate imbalance — a classifier reaches a balanced accuracy of
0.3854 <!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.balanced_accuracy --> against a
chance of 0.5 <!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.chance -->. Worse than
guessing. This is not merely a small-sample failure: on this many cultures the analysis could have
detected a balanced accuracy of 0.6438
<!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.detectable_balanced_accuracy --> had one
been there. Direction was recovered only by instrumenting the microchannels themselves.

![Measured axonal flow against chamber-level readouts, REAL recordings. The chamber measures do not order the cultures by the direction their axons actually fired.](../outputs/figures/axonal_flow_vs_chamber.png){width=58%}

![Inter-compartment coupling lag by device type, REAL data, one value per device: five CxCx, five HpHp, six CxHp. The two same-tissue types are indistinguishable; the heterogeneous type sits apart.](../outputs/figures/compartment_lag_by_device_type.png){width=56%}

![Microchannel propagation asymmetry by device geometry, REAL recordings.](../outputs/figures/microchannel_asymmetry_real.png){width=52%}

## 16. Running the model backwards, and what that cannot do

The more ambitious use of a twin is inverse parameter inference: given a recording, infer which knob
was turned. We implemented simulation-based inference using neural posterior estimation, trained on
3,000 simulations <!-- metrics.json: twin_hd14_inference.n_simulations --> at the calibrated cell density. Two
honesty checks come first.

| model parameter | biophysical role | posterior contraction | KS calibration (p) | uniformity status |
|---|---|---|---|---|
| `threshold_scale` | action-potential threshold | 0.661 <!-- metrics.json: twin_hd14_inference.contraction.threshold_scale.contraction --> | 0.6105 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.threshold_scale.ks_p --> | well calibrated |
| `ampa_scale` | excitatory synaptic weight | 0.599 <!-- metrics.json: twin_hd14_inference.contraction.ampa_scale.contraction --> | 0.3128 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.ampa_scale.ks_p --> | well calibrated |
| `log10_forward_ratio` | axonal pathway asymmetry | 0.438 <!-- metrics.json: twin_hd14_inference.contraction.log10_forward_ratio.contraction --> | 0.1517 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.log10_forward_ratio.ks_p --> | well calibrated |
| `gabaa_scale` | inhibitory synaptic weight | 0.3 <!-- metrics.json: twin_hd14_inference.contraction.gabaa_scale.contraction --> | 0.0174 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.gabaa_scale.ks_p --> | **over-confident** |
| `inter_in_degree` | cross-chamber connectivity | 0.269 <!-- metrics.json: twin_hd14_inference.contraction.inter_in_degree.contraction --> | 0.0121 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.inter_in_degree.ks_p --> | **over-confident** |

Contraction measures how much narrower the posterior is than the prior; 0 means nothing was learned.
Rank uniformity asks whether stated intervals are honest. Three knobs pass; inhibitory transmission
does not, and it is the knob the pharmacology sections depend on. **Intervals on the inhibitory knob
should not be quoted as calibrated.** Before the density correction (§13) no parameter passed and a
fifth of simulations were silent, against 0
<!-- metrics.json: twin_hd14_inference.fraction_silent_simulations --> now: simulating at the calibrated density
brought three knobs into calibration.

![SIMULATED posterior calibration: the rank of the true parameter among posterior samples. Flat would be calibrated; these are not.](../outputs/twin/hd14_posterior_ranks_simulated.png){width=54%}

**The false-disinhibition artifact.** Applied to real untreated chips, the model returned an
inferred inhibition scale of 0.5219
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median --> on an interval of 0.2732 to 0.8503
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median_ci95.1 --> — apparent disinhibition in wells
that had no drug. That is not biology; it is the model saying "real recordings do not look like my
simulations" in the only vocabulary it has. Reformulating inference around paired deltas shifted
untreated chips to 0.9108
<!-- metrics.json: twin_hd17_delta_inference.real_data.gabaa_scale_at_baseline.median --> on 0.8583 to 0.966
<!-- metrics.json: twin_hd17_delta_inference.real_data.gabaa_scale_at_baseline.median_ci95.1 --> and raised contraction to
0.5375 <!-- metrics.json: twin_hd17_delta_inference.held_out_checks.contraction -->. Most of the offset goes and not all
of it: the interval still stops short of neutral.

**Out-of-distribution rejection.** Asked for 200 draws <!-- metrics.json: demo_precompute.posterior_draws.requested --> per
recording across 113 <!-- metrics.json: demo_precompute.posterior_draws.n_recordings --> bicuculline trials, the posterior
could not produce them for 9 <!-- metrics.json: demo_precompute.posterior_draws.n_short_of_requested --> of them, the worst
yielding 1 <!-- metrics.json: demo_precompute.posterior_draws.fewest_draws --> draw; 0
<!-- metrics.json: demo_precompute.posterior_draws.n_unsampled --> failed outright. This appeared only once the twin moved to
its calibrated density: a posterior fitted to a sharper forward model is less forgiving of
observations that model cannot produce. In high-content screening an explicit refusal is more
valuable than a forced prediction.

## 17. What it takes to see directionality at all

One thing a twin is genuinely good for is telling you how much experiment you need. Across 120
simulated networks <!-- metrics.json: twin_hd16_directionality.n_networks --> with ground-truth wiring, the cheapest
configuration reaching the target distinguishes asymmetric from symmetric wiring at an AUC of 0.9219
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_all_features.auc --> using only 6 electrodes
per compartment
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_all_features.electrodes_per_compartment -->
and a 10-second recording
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_all_features.duration_s -->.

**The hidden catch.** Most of that is firing rate, not direction. A directional pathway delivers
more excitation to its target chamber, so the target simply fires more. Lowering the receiving
chamber's background drive moves measured coupling asymmetry from 0.109
<!-- metrics.json: twin_hd11_validation.directionality_sweep.0.asymmetry_mean --> to 0.681
<!-- metrics.json: twin_hd11_validation.directionality_sweep.2.asymmetry_mean --> with the wiring unchanged — across
the sweep, drive moves the measure further than wiring does, a span of 0.671
<!-- metrics.json: twin_hd11_validation.confound_summary.raw.drive_span --> against 0.109
<!-- metrics.json: twin_hd11_validation.confound_summary.raw.wiring_span -->. Six electrodes and ten seconds will
distinguish these cultures, but claiming it reflects directional communication requires explicit
rate-matching controls.

---

# ACT V: The negative results and scientific boundaries

## 18. The negative results, collected

Scattered through a long report, negative results are easy to miss. Collected, they are the part of
this work we would most like challenged, and the part most likely to save someone else an
experiment.

| # | finding | empirical metric | practical consequence |
|---|---|---|---|
| 1 | dose signal absent | 0 <!-- metrics.json: feature_dose_response.verdict.n_features_moving_absolute --> of 9 features move monotonically with dose (p = 0.751) | the dose signal is not in summary MEA tables |
| 2 | directionality null | chamber classifier 0.3854 <!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.balanced_accuracy --> against 0.5 chance | chamber readouts cannot infer axonal flow direction |
| 3 | no cross-laboratory transfer | 0.3482 <!-- metrics.json: cross_lab_transfer.balanced_accuracy --> against 0.3333 chance, with the confound helping | mechanism classifiers do not transfer between laboratories |
| 4 | domain adaptation failure | CORAL drops transfer to 0.442 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy -->, standardisation to 0.408 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy --> | high-dimensional adaptation erodes small-sample biology |
| 5 | biophysical artifact | depression and delay dispersion fail to fix synchrony | hypersynchrony was driven by cell density and radius overlap |
| 6 | posterior over-confidence | `gabaa_scale` fails rank uniformity at p = 0.0174 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.gabaa_scale.ks_p --> | the inverse model stays over-confident on inhibition |
| 7 | baseline hallucination | absolute features infer 0.5219 <!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median --> inhibition scale on untreated chips | the model misreads its own baseline offset as drug effect |
| 8 | support rejection | the calibrated twin cannot sample 9 <!-- metrics.json: demo_precompute.posterior_draws.n_short_of_requested --> of 113 <!-- metrics.json: demo_precompute.posterior_draws.n_recordings --> real recordings | a sharper forward model contracts support and refuses outliers |

**On the first, which is the most consequential.** §6 showed the mechanism classifier had no dose
relationship and §16 showed the inverse model could not recover dose either. It would be natural to
blame the models. So we asked directly: across 113 bicuculline recordings
<!-- metrics.json: feature_dose_response.n_recordings --> on 16 devices
<!-- metrics.json: feature_dose_response.n_devices -->, does *any* of the nine features move monotonically with dose?
Using raw values, 0 <!-- metrics.json: feature_dose_response.verdict.n_features_moving_absolute --> features do; after
subtracting each device's own baseline, 0
<!-- metrics.json: feature_dose_response.verdict.n_features_moving_baseline_subtracted --> do — in both cases after
correcting for testing nine features at once. The models were not failing to extract a signal. The
signal was not there.

## 19. What you cannot conclude from this work

- **No clinical or safety claims.** NeuroMPS is an exploratory analytical framework; no efficacy,
  diagnostic or clinical claims are made, and the design could not support them.
- **Not a screening tool.** Within-platform mechanism accuracy around 0.54 with an interval from
  0.42 to 0.68, and near-chance performance across laboratories, is not a decision instrument.
- **Narrow interspecies support.** Rat-to-human transfer rests on 37 rat training and 32 human
  testing pairs, supporting direction rather than magnitude.
- **Not a fully reconciled simulator.** The twin satisfies the real band on two of five features,
  exhibits seed variance, and gets GABA-A potentiation mostly wrong.
- **No monotonic dose tracking.** Feature deltas contain no monotonic concentration signal across
  the bicuculline series.
- **Severe species asymmetry.** 21,530 rat <!-- metrics.json: data_asset.counts.by_species.rat --> against
  357 human <!-- metrics.json: data_asset.counts.by_species.human --> records, a property of open-access
  licensing rather than a design choice.
- **Sample-size dependencies.** Primary classification rests on 73 paired recordings from 50
  independent cultures.

---

# ACT VI: Application value, translation and open verification

## 20. Application value and translation for neural organ-on-a-chip

The primary bottleneck in making neural organ-on-a-chip systems useful is not microfluidic
fabrication; it is the interpretability and reproducibility of functional electrophysiology.
Platforms from academic centres and commercial vendors alike generate multi-channel MEA recordings
that remain difficult to compare across sites. The results here translate into guidance in four
areas.

**1. Structural design guidance for hardware.** Developers often assume directional connectivity can
be validated by measuring cross-correlation latencies or coupling asymmetries from open somatic
chambers. §15 disproves that on real devices: chamber readouts reached 0.3854
<!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.balanced_accuracy -->
balanced accuracy where chance is 0.5, on a sample that could have detected 0.6438
<!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.detectable_balanced_accuracy -->.
Validating axonal rectification requires physical microelectrodes embedded directly within
microchannels, which saves time spent on measures that cannot answer the question.

**2. A standardization framework for contract research.** Screening protocols should abandon
proprietary vendor summary tables in favour of standardized raw-spike ETL. The feature ablation
(§8) shows two rate metrics match the predictive power of the full nine-feature vector, while
avoiding the cross-platform variance that afflicts burst-structure and synchrony metrics (§3, §11).
A smaller, more reliable feature vector is the more reproducible instrument.

**3. Cost-effective in-silico assay optimization.** Human iPSC plating is expensive and takes weeks.
The twin provides an in-silico testbed for assay pre-registration: the power sweep in §17
establishes a minimum electrode floor (6 electrodes, 10-second recordings) for distinguishing
wiring regimes, while the listening-radius finding in §13 is a diagnostic for multi-unit electrode
catchment overlap in physical recordings.

**4. Paired-delta normalization for bioinformatic workflows.** Primary cultures exhibit resting
rates spanning an order of magnitude. Reformulating screening features as paired log-ratios against
a well's own baseline stabilizes predictive baselines, narrows posterior intervals from 0.3
<!-- metrics.json: twin_hd14_inference.contraction.gabaa_scale.contraction --> to 0.5375
<!-- metrics.json: twin_hd17_delta_inference.held_out_checks.contraction -->, and removes most of
the false-disinhibition artifact of §16.

## 21. How to check any of this: the verification harness

Every computed metric in this report carries an automated pointer back to the key in
`outputs/metrics.json` that produced it, and a test fails if any number drifts from the file.

| harness component | command or file | operational scope |
|---|---|---|
| quick CI path | `python run_all.py --quick` | executes the pipeline end to end on a reduced tier |
| full reproduction | `python run_all.py` | full data tier, 3,000 <!-- metrics.json: twin_hd14_inference.n_simulations --> forward simulations, trains the density estimator, regenerates every figure |
| citation verification | `python -m neuromps.report.verify` | re-reads every citation in this document and fails on disagreement |
| provenance guard | `data/raw/MANIFEST.json` | source URLs, licences and SHA-256 hashes; the build halts on mismatch |
| interactive demo | `streamlit run app/app.py` | record inspection, QC flags, and twin sweeps over a precomputed grid |

## 22. AI assistance disclosure

This work was produced in a paired workflow with Anthropic's Claude Code. The model wrote most of
the code and tests here under step-by-step human direction: the human set the plan and phase gates,
chose the scope, approved each step, and directed the science. Three practices kept that honest —
numbers are emitted into `metrics.json` and re-checked by `neuromps.report.verify` rather than
typed; results are reported as observed, including the ones that failed; and
`report/LAB_NOTEBOOK.md` records what was abandoned.

---

# Appendices: consolidated technical reference

## Appendix A & B — Record architecture, data levels and ingestion ETL

Every record declares a data level: `spike_times`, `per_well_features` or `fitted_curve`. The level
determines what may be computed from it, and the build enforces that rather than trusting it — a
feature requiring spikes cannot be derived from a record holding only a fitted curve. Records
validate against `schema/neuromps_schema.json` at three conformance levels, so a partially specified
source can enter the asset while declaring what it is missing.

Quality-control flags are recorded and never silently applied: 537
<!-- metrics.json: data_asset.counts.by_usable.False --> of 21,887
<!-- metrics.json: data_asset.n_records --> records are marked unusable, with the diagnostic reason
attached, and the flag travels with the record. Analyses choose whether to honour it and say which
they did; dropping flagged records inside the pipeline would make the drop invisible to the reader.

Each record also carries the method string that produced its features — the exact burst rule, the
synchrony method and the active-electrode threshold — so two records computed under different
settings cannot be compared by accident.

Ingestion lives in `src/neuromps/fetch/` and uses HTTP range requests against remote ZIP archives to
pull only the required members, bypassing multi-gigabyte downloads. Two server behaviours needed
specific handling and carry regression tests: one EPA host returns HTTP 500 for suffix ranges, and
the same host, when offered gzip encoding, replies with a plausible `Content-Range` while streaming
the whole 7.5 GB file from byte zero.

## Appendix C — Sources and licences manifest

| dataset | platform format | species | experimental role | records | status |
|---|---|---|---|---|---|
| `toxcast_mea` | screening MEA | rat | high-throughput reference, fitted curves | 20,375 <!-- metrics.json: data_asset.counts.by_source.toxcast_mea --> | ingested |
| `screenneuropharm_bic` | two-compartment chip | rat | directionality and twin calibration | 580 <!-- metrics.json: data_asset.counts.by_source.screenneuropharm_bic --> | ingested |
| `kapucu2022` | planar 48-well MEA | rat / human | primary ML benchmark and transfer | 288 <!-- metrics.json: data_asset.counts.by_source.kapucu2022 --> | ingested |
| `vassallo2017` | planar MEA | rat | second within-platform benchmark | 283 <!-- metrics.json: data_asset.counts.by_source.vassallo2017 --> | ingested |
| `trujillo2019` | planar MEA | human organoid | baseline electrophysiology | 184 <!-- metrics.json: data_asset.counts.by_source.trujillo2019 --> | ingested |
| `mateus2024` | microchannel MEA | rat | microchannel propagation asymmetry | 108 <!-- metrics.json: data_asset.counts.by_source.mateus2024 --> | ingested |
| `brewer2021` | microchannel MEA | rat | axonal propagation ground truth | 40 <!-- metrics.json: data_asset.counts.by_source.brewer2021 --> | ingested |
| `sharf2022` | high-density MEA | human organoid | negative-control screening (diazepam) | 29 <!-- metrics.json: data_asset.counts.by_source.sharf2022 --> | ingested |
| `fofie2025` | multi-well MEA | human sensory | rate-only validation arm (§8) | 291 pairs <!-- metrics.json: rate_only_arm.n_pairs --> | rate-only, outside the record asset |
| `tukker2020` | planar MEA | rat / human | neurotoxicity screening | — | **excluded, CC BY-NC** |

`fofie2025` reports total spike counts rather than spike times, so it supports firing-rate features
only and is not part of the 21,887-record asset; it is used solely for the cross-source rate-only
arm. `tukker2020` is recorded as a skipped source with its reason rather than silently omitted.

## Appendix D & E — The nine-feature vector, and enforced repository invariants

| feature | formal definition as computed in `features/mea.py` |
|---|---|
| `mfr_hz` | spikes per second per active electrode |
| `burst_rate_per_min` | bursts per minute per active electrode (≥ 5 spikes, ISI < 100 ms) |
| `burst_duration_s` | mean duration from first to final spike across identified bursts |
| `spikes_per_burst` | mean count of spikes participating in a burst |
| `pct_spikes_in_bursts` | share of all spikes occurring inside bursts |
| `network_burst_rate_per_min` | synchronous population events per minute, clearing chance participation |
| `synchrony_index` | mean pairwise spike-time tiling coefficient, coincidence window Δt = 50 ms |
| `isi_cv` | coefficient of variation of inter-spike intervals |
| `active_electrodes` | electrodes clearing 5 spikes per minute |

Each of the following is a test that fails, not a convention that is followed.

- **Provenance.** The asset build fails if any record's provenance hash disagrees with the manifest.
- **One feature extractor.** The twin computes features through the same `features/mea.py` as real
  recordings, and — after a defect that survived several phases — through the same coupling measure
  on the same per-electrode input.
- **Interval discipline.** A contract test fails if any headline metric loses its confidence
  interval, and the resample count sits next to every estimate.
- **Cache meaning.** The simulation cache stores its summary names, electrode count, a hash of the
  code defining every summary, and the calibration it was generated under; it refuses to load across
  a change in any of them.
- **Report numbers.** `neuromps.report.verify` re-reads every citation here and fails on
  disagreement.
- **No clinical claims.** A test scans the reports for efficacy, diagnostic and treatment language
  and fails unless the line is a disclaimer ruling it out.

## Appendix F & G — Coupling asymmetry proof, and posterior calibration diagnostics

**Appendix F.** The measure the twin exists to use reports relative firing rate as much as pathway
direction, and the cleanest demonstration holds the wiring **fixed** and changes only the drive.
Lowering the receiving chamber's background drive moves asymmetry from 0.109
<!-- metrics.json: twin_hd11_validation.directionality_sweep.0.asymmetry_mean --> to 0.681
<!-- metrics.json: twin_hd11_validation.directionality_sweep.2.asymmetry_mean --> with the pathway unchanged. Across
the whole sweep, drive moves the measure further than wiring does: a span of 0.671
<!-- metrics.json: twin_hd11_validation.confound_summary.raw.drive_span --> against 0.109
<!-- metrics.json: twin_hd11_validation.confound_summary.raw.wiring_span -->. An open-chamber asymmetry is a
statement about which chamber is busier until something rules that out.

**Appendix G.** Trained on 3,000 simulations <!-- metrics.json: twin_hd14_inference.n_simulations --> at the
calibrated density, of which 0 <!-- metrics.json: twin_hd14_inference.fraction_silent_simulations --> produced no
spikes. The two parameters failing rank uniformity are the connectivity parameter and the inhibitory
knob — the latter being the one §16 depends on, which is why its intervals are not quoted as
calibrated.

## Appendix H — Quality-control flag registry

| QC flag | affected records | identification criteria | analytical handling |
|---|---|---|---|
| `incomplete_dose_series` | 272 <!-- metrics.json: data_card.flag_counts.incomplete_dose_series --> | concentration series missing intermediate dilution steps | flagged; excluded from monotonic dose sweeps |
| `too_few_active_electrodes` | 265 <!-- metrics.json: data_card.flag_counts.too_few_active_electrodes --> | fewer than 10 electrodes clearing 5 spikes/min | flagged; preserved in the data asset |
| `replicate_divergence` | 221 <!-- metrics.json: data_card.flag_counts.replicate_divergence --> | well deviates more than 3σ from its culture replicate cohort | flagged; logged in metadata |
| `outlier_feature_value` | 221 <!-- metrics.json: data_card.flag_counts.outlier_feature_value --> | feature value outside the acceptable biological distribution | flagged; excluded from density fitting |
| `excluded_well_upstream` | 90 <!-- metrics.json: data_card.flag_counts.excluded_well_upstream --> | source laboratory marked the well defective | flagged; excluded from benchmarks |
| `missing_baseline_pair` | 24 <!-- metrics.json: data_card.flag_counts.missing_baseline_pair --> | no baseline recording on the same well | flagged; cannot enter a paired-delta model |
| `noisy_electrodes_present` | 11 <!-- metrics.json: data_card.flag_counts.noisy_electrodes_present --> | significant mechanical or electrical artifact | flagged; isolated during ingestion |
| `unreadable_channels` | 2 <!-- metrics.json: data_card.flag_counts.unreadable_channels --> | channel could not be decoded from the deposit | flagged; channel dropped |

## Appendix I — Digital twin forward-model parameters

Calibrated values are the chosen draw from the §13 search; held values were fixed and not searched.

| parameter | role | calibrated value | how it was set |
|---|---|---|---|
| `n_per_compartment` | plating cell density per chamber | 967 <!-- metrics.json: twin_hd12_calibration.best.draw.n_per_compartment --> neurons | calibrated, searched over 250–1500 |
| `detection_radius_um` | virtual electrode listening catchment | 92.5135 <!-- metrics.json: twin_hd12_calibration.best.draw.detection_radius_um --> µm | calibrated, searched over 40–100 µm, capped at half the electrode pitch |
| `intra_in_degree` | recurrent local synaptic inputs | 18.4929 <!-- metrics.json: twin_hd12_calibration.best.draw.intra_in_degree --> per neuron | calibrated, searched over 10–35 |
| `background_rate_hz` | Poisson drive frequency | 3.3596 <!-- metrics.json: twin_hd12_calibration.best.draw.background_rate_hz --> Hz | calibrated, searched over 1–40 Hz |
| `background_weight` | Poisson drive conductance step | 0.0692 <!-- metrics.json: twin_hd12_calibration.best.draw.background_weight --> | calibrated, searched over 0.005–0.12 |
| `excitatory_weight` | excitatory conductance step | 0.386 <!-- metrics.json: twin_hd12_calibration.best.draw.excitatory_weight --> | calibrated, searched over 0.15–0.45 |
| `inhibitory_weight` | inhibitory conductance step | 5.4261 <!-- metrics.json: twin_hd12_calibration.best.draw.inhibitory_weight --> | calibrated, searched over 1–7 |
| `adaptation_increment` | spike-frequency adaptation step | 1.0093 <!-- metrics.json: twin_hd12_calibration.best.draw.adaptation_increment --> | calibrated, searched over 0.5–2.5 |
| `threshold_spread` | per-neuron excitability heterogeneity | 0.2258 <!-- metrics.json: twin_hd12_calibration.best.draw.threshold_spread --> | calibrated, searched over 0–0.3 |
| `adaptation_tau_ms` | adaptation decay time constant | 380 ms | **held**, literature standard |
| `intra_delay_sd_ms` | conduction delay dispersion | 0.0 ms | **held**, one delay for every synapse |
| `connection_sigma_um` | range of the within-compartment wiring rule | unset | **held**, uniform random wiring |
| `inter_in_degree`, `forward_ratio`, drug knobs | cross-chamber connectivity, pathway asymmetry, pharmacology | — | **inferred** by simulation-based inference (§16) |

All conductance steps are dimensionless in this model; they are not membrane voltages.

## Appendix J & K — Microchannel detector robustness, and timings

**Appendix J.** Counting recordings, median absolute asymmetry runs from 0.3392
<!-- metrics.json: microchannel_directionality.per_recording_naive.by_condition.control.median_abs_asymmetry --> for
control geometries to 0.8543
<!-- metrics.json: microchannel_directionality.per_recording_naive.by_condition.rams.median_abs_asymmetry --> for the
"rams" geometry. Counting biological chips instead, it is 0.6168
<!-- metrics.json: microchannel_directionality.comparison.by_condition.control.median_abs_asymmetry --> against 0.8745
<!-- metrics.json: microchannel_directionality.comparison.by_condition.rams.median_abs_asymmetry -->, a difference the
test does not resolve: Kruskal p across the geometries is 0.63969
<!-- metrics.json: microchannel_directionality.comparison.kruskal_p -->, and every Holm-adjusted comparison against the
control geometry is 1.0 <!-- metrics.json: microchannel_directionality.comparison.vs_reference_holm.rams -->. The null is
not an artifact of geometry assumptions either: it survives every detector variant tried, with a
worst-case Kruskal p of 0.13074
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_one_missing.kruskal_p -->. Tolerating
single-electrode dropouts triples what the detector has to work with, from 336,794
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_strict.total_events --> analysed
events to 1,101,233
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_one_missing.total_events -->, and
does not change the outcome. A counting
confound is measurable and reported: chips with fewer propagation events look more one-sided
(Spearman ρ = −0.494 <!-- metrics.json: microchannel_directionality.event_count_confound.spearman_rho -->, p = 0.01032
<!-- metrics.json: microchannel_directionality.event_count_confound.p_value -->), which is what a counting artifact looks
like. Condition is also confounded with culture batch by the deposit's own design
<!-- metrics.json: microchannel_directionality.batch_confound -->.

**Appendix K.** `--quick` completes end to end in 401.0 s
<!-- metrics.json: harness_quick.total_s --> on a workstation CPU with 20 cores
<!-- metrics.json: harness_quick.cpu_count -->, and in under twenty minutes on a
dual-vCPU CI runner; the two differ because `worker_processes` correctly asks for the cores it is
given. That figure is written by the run itself, and only by a run that executed every stage, so a
partial re-run cannot quietly reduce it. The longest stage is the twin at 259.6 s
<!-- metrics.json: harness_quick.by_stage_s.twin -->. Full reproduction runs through
`python run_all.py` after a full data-tier fetch. The environment is pinned in `pyproject.toml` and
recorded from what actually imported: Python `3.11.16` <!-- metrics.json: harness_quick.python -->,
NumPy `2.3.5` <!-- metrics.json: harness_quick.numpy -->, Brian2 `2.9.0`
<!-- metrics.json: harness_quick.brian2 -->, PyTorch `2.14.0` <!-- metrics.json: harness_quick.torch -->
and sbi `0.27.0` <!-- metrics.json: harness_quick.sbi -->. NumPy is held back because the next minor
removes `ndarray.ptp`, which Brian2 imports.

## Appendix L, M & N — Sensory validation, dose tracking and species transfer

**Appendix L, the rate-only arm.** Human sensory neurons across 291 pairs
<!-- metrics.json: rate_only_arm.n_pairs --> in `fofie2025` shift by −0.1743
<!-- metrics.json: rate_only_arm.by_mechanism.control.median_log2_change --> log2 units for vehicle controls,
against −1.9529 <!-- metrics.json: rate_only_arm.by_mechanism.na_channel_blocker.median_log2_change --> for
sodium-channel blockers, yielding 0.5012
<!-- metrics.json: rate_only_arm.across_sources.balanced_accuracy --> balanced accuracy against a chance of
0.3333 <!-- metrics.json: rate_only_arm.across_sources.chance -->.

**Appendix M, Task 2 dose tracking.** Detection rate rises from 0.179
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.0.detection_rate --> at 0.1 µM
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.0.dose_uM --> to 0.426
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.5.detection_rate --> at 30 µM
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.5.dose_uM -->. Quoted
there it would look like a dose response, so the rest of the series is quoted with it: the rate peaks
at 0.48 <!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.6.detection_rate -->
at 100 µM <!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.6.dose_uM -->
and falls to 0.091
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.8.detection_rate --> at the
top of it, 500 µM
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.8.dose_uM -->. The
overall rank correlation is non-significant (ρ = 0.014
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman -->, p = 0.751
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman_p -->).

**Appendix N, species transfer, all settings.**

| transfer setting | balanced accuracy | 95% bootstrap CI |
|---|---|---|
| human within-species | 0.646 <!-- metrics.json: species_transfer.transfer.none.human_within.balanced_accuracy --> | — |
| rat within-species | 0.543 <!-- metrics.json: species_transfer.transfer.none.rat_within.balanced_accuracy --> | — |
| rat → human, zero-shot | 0.55 <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy --> | 0.4 to 0.6 <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy_ci95.1 --> |
| rat → human, CORAL | 0.442 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy --> | 0.302 to 0.591 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy_ci95.1 --> |
| rat → human, standardised | 0.408 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy --> | 0.267 to 0.568 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy_ci95.1 --> |
