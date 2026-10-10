# NeuroMPS: A Provenance-Tracked MEA Data Asset and a Calibrated Two-Compartment Digital Twin

**Technical report.** Entry for the AI4S Open Innovation: AI + Organ-on-a-Chip challenge
(5th Pazhou Algorithm Competition).

> **Research decision-support only.** Nothing here is an efficacy, safety, diagnostic or clinical
> claim, and nothing here should be used to make one.

**Submission category: End-to-End System.** We built a complete path from public recordings to
actionable output: provenance-enforced ingestion, a standardized feature layer, models, a calibrated
digital twin, and a harness that tells us which measurements the data can and cannot support. An
interactive demo sits on top.

**Team.** We combine health science with mathematics and machine learning.

| member | role | background |
|---|---|---|
| **Yi Nuo Zhao** | Team Leader | Health science student |
| **Ivan T. Ivanov** | Machine Learning Specialist | Professor of Mathematics and ML/AI developer |

| | |
|---|---|
| **Code and demo** | https://github.com/Vodolay/AI4S; the interactive demo runs locally with one command (§21) |
| **Video** | https://www.youtube.com/watch?v=WnnD-oSFdcI, a walkthrough of the project and the demo |
| **Verification harness** | `python run_all.py --quick` runs the complete pipeline, models and metrics |
| **Audit discipline** | `python -m neuromps.report.verify` checks every number in this report against `outputs/metrics.json` |
| **Regulatory position** | Research decision-support only. We make no clinical, diagnostic, efficacy or safety claim. |

You can reproduce and check everything in this report. One command, `python run_all.py --quick`,
runs every stage of the pipeline in about nine minutes on our 20-core workstation (Appendix K), and
`python -m neuromps.report.verify` re-checks every number below against the output that produced
it. In the source of this report, each computed number carries a hidden pointer to the
`outputs/metrics.json` key it came from, so a stale figure cannot quietly survive a rerun. We kept a
record of everything we tried and abandoned in `report/LAB_NOTEBOOK.md`.

---

## Executive summary

Neural microelectrode arrays (MEAs) are one of the most useful readouts in organ-on-a-chip work: you
grow neurons on a grid of electrodes and watch how their firing changes when you add a compound. The
trouble starts when you try to compare results across laboratories. Every group computes its
features with its own software and its own definitions, and most deposit summary tables instead of
raw spikes, so a number in one paper rarely means the same thing as the "same" number in another.

We built NeuroMPS to tackle that problem directly, and to find out honestly how far the data can be
pushed. It has four parts.

- **A standardized data asset.** We assembled 21,887 records <!-- metrics.json: data_asset.n_records -->
  from eight public sources, each verified against a SHA-256 checksum of its original file. One
  feature extractor computes every number from the 1,512
  <!-- metrics.json: data_asset.counts.by_data_level.spike_times --> spike-time recordings, so
  comparisons inside the asset are exact. When we ran the depositors' own meaRtools pipeline on the
  same recordings, it disagreed with ours by more than a factor of two on burst rate (§3) — exactly
  the kind of divergence the asset exists to remove.
- **Benchmarks on real recordings.** We grouped every cross-validation split by culture and
  bootstrapped over cultures rather than rows. Under those rules, within one laboratory, a simple
  linear model (balanced accuracy 0.537
  <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean -->)
  beats gradient boosting (0.398
  <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_mean -->),
  and a model trained only on rat recordings still scores 0.55
  <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy --> on human iPSC
  recordings. Across laboratories, though, the same classifier falls to 0.3482
  <!-- metrics.json: cross_lab_transfer.balanced_accuracy --> against a chance of 0.3333
  <!-- metrics.json: cross_lab_transfer.chance -->: what it learned in one laboratory did not carry
  to the next. Two rate features alone match the full nine-feature vector.
- **A biophysical digital twin.** We simulate an asymmetric two-compartment chip in Brian2: leaky
  integrate-and-fire neurons, 80/20 excitatory/inhibitory, with spike-frequency adaptation, wired
  through a directional "axon diode". We calibrated it against the interquartile band of 48 real
  chamber recordings
  <!-- metrics.json: twin_hd12_calibration.targets_real.n_records --> and score it with the same
  feature code as the real data. We use it for experimental power analysis and as a generator of
  labelled training data: 3,000 <!-- metrics.json: twin_hd14_inference.n_simulations --> simulated
  recordings, each with known parameters, train a neural posterior estimator, a masked
  autoregressive flow, that returns a full posterior for a new recording in 0.31 s
  <!-- metrics.json: twin_hd14_inference.amortized_inference.median_seconds_per_posterior -->
  without running the simulator again.
- **Negative results and practical guidance.** Eight of our findings set firm boundaries, and §18
  pairs each with what we would do instead. For example, chamber-level readouts cannot tell which way axons are firing without electrodes inside
  the microchannels (0.3854
  <!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.balanced_accuracy -->
  against 0.5 <!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.chance -->
  chance), and no summary feature we could test tracks dose monotonically. In §20 we turn these
  findings into recommendations for chip designers, assay developers and analysis pipelines.

---

# ACT I: Foundations and the data asset

## 1. What this is, and what it is not

A microelectrode array is a grid of tiny listening posts under a living neural culture.
Neurobiologists use it to answer a simple question: does this compound change how the network fires,
and if so, how? In practice, every laboratory answers it differently. Labs use different plates,
different spike sorters and different definitions of what counts as a burst, so two papers can study
the same compound on the same cell type and still not be comparable. On top of that, most groups
deposit only a table of summary numbers rather than the spikes themselves — a bit like receiving the
book review instead of the book.

**Where this matters.** The people who feel this most are the ones who need to combine data they did
not record themselves: a lab checking whether a compound behaves on its own platform the way it did
in a published study, a contract research organization deciding which features to report, or a chip
designer trying to prove that a microchannel really makes signals flow one way. We built NeuroMPS
for exactly those situations.

We tackle the problem in three steps, and this report follows the same order.

- **We build one harmonized dataset.** We assembled eight public sources into one table, with one
  exact definition for every feature. A number from Kapucu's plate and a number from
  ScreenNeuroPharm's chamber are comparable because the same code computed both from the same kind
  of raw spike input. Every record says where it came from and under what license, and carries a
  checksum of its original file.
- **We ask what that dataset can support.** We run models on the real recordings to find out which
  conclusions survive a change of laboratory, of species and of platform — and which questions a
  chamber-level readout cannot answer at all.
- **We build a digital twin for the questions the recordings cannot answer.** It is a forward
  simulation of a two-compartment neural chip: two populations of model neurons wired through a
  directional microchannel pathway, with virtual electrodes over them. We tune it against real
  chamber baselines from the dataset and score it with the same feature code, so it cannot make
  itself look good by being measured differently.

![System architecture. REAL recordings and SIMULATED twin output pass through the same feature extractor, and provenance is enforced at the boundary of the data asset.](../outputs/figures/architecture.png){width=100%}

**What it is not.** NeuroMPS is not an automated screening platform, and it does not predict whether
an uncharacterized compound is safe or effective. Several of our most useful results are negative:
they show that a measurement does not contain a signal people routinely assume it does. We treat
those results (§18) as a core deliverable, not as an apology.

## 2. The data

Our asset holds 21,887 records <!-- metrics.json: data_asset.n_records --> from eight public
sources. They are not all the same kind of record, and that difference matters more than the count.

| what the record holds | record count | what you can do with it |
|---|---|---|
| spike times | 1,512 <!-- metrics.json: data_asset.counts.by_data_level.spike_times --> | recompute any feature from scratch with our common code |
| fitted dose-response curves | 20,375 <!-- metrics.json: data_asset.counts.by_data_level.fitted_curve --> | read a potency estimate that another laboratory fitted |

We require every record in our dataset to explicitly declare its data type. While that might sound
like routine bookkeeping, it directly protects our pipeline: virtually all of our predictive
modeling and twin calibration rely solely on spike-time recordings, because only those can pass
through our unified feature extractor. If we had simply lumped the two record types together into a
single table, the dataset would look twenty times larger — but it would be twenty times less useful
to us.

**The species imbalance.** By species, our asset holds 21,530 rat
<!-- metrics.json: data_asset.counts.by_species.rat --> and 357 human
<!-- metrics.json: data_asset.counts.by_species.human --> records. This imbalance is the single
biggest limitation of the project, and no amount of clever coding can fix it: it reflects what is
publicly available under licenses that permit reuse. Rather than hide it, we treat it as an explicit
boundary whenever we test species transfer.

**The case that tested our rules.** One dataset (Tukker 2020) was an ideal scientific fit, and we
still excluded it. It is licensed CC BY-NC, and the competition rules give the organizing committee
joint ownership of entries — a right that a non-commercial license does not allow us to grant. We
record the exclusion and its reason in `data/raw/MANIFEST.json`, next to the sources we accepted,
instead of letting the dataset quietly disappear. We mention it because you should be able to check
that our license discipline is real rather than claimed: every source re-checks its license at fetch
time, and the build stops if any checksum disagrees with the manifest.

---

# ACT II: Standardizing the mess and benchmarking real biology

## 3. Are two laboratories' numbers actually comparable?

The whole dataset rests on this question, so we tested it instead of assuming it. One module,
`features/mea.py`, computes all nine features directly from spike times: mean firing rate, burst
rate, burst duration, spikes per burst, percentage of spikes in bursts, network burst rate,
synchrony index, the coefficient of variation of inter-spike intervals, and the number of active
electrodes.

We detect bursts with a stated rule — spikes closer than 100 ms, at least five in a row — rather than
a vendor default. For synchrony we use the spike-time tiling coefficient (STTC) with a 50 ms
coincidence window. Unlike a plain correlation, STTC does not reward two electrodes just for both
being busy.

**The benchmark.** Kapucu and colleagues deposited their own feature tables alongside their spike
files, computed with `meaRtools`. That gave us an independent implementation to test against: we ran
our code over their spikes and compared the results across 336 recordings
<!-- metrics.json: extractor_agreement.n_recordings --> on 3 plates
<!-- metrics.json: extractor_agreement.n_plates -->, 2,298 paired values
<!-- metrics.json: extractor_agreement.n_paired_values --> in total. We consider this the most
important methodological result in the report, so here it is, plainly.

| feature | rank agreement (ρ) | value ratio (ours / theirs) | practical consequence |
|---|---|---|---|
| `synchrony_index` | 0.9839 <!-- metrics.json: extractor_agreement.by_feature.synchrony_index.spearman_rho --> | 0.736 <!-- metrics.json: extractor_agreement.by_feature.synchrony_index.median_ratio_ours_over_theirs --> | ranks match; our STTC is about a quarter lower |
| `mfr_hz` | 0.9831 <!-- metrics.json: extractor_agreement.by_feature.mfr_hz.spearman_rho --> | 0.7539 <!-- metrics.json: extractor_agreement.by_feature.mfr_hz.median_ratio_ours_over_theirs --> | ranks match; baseline firing rate scaled differently |
| `active_electrodes` | 0.9763 <!-- metrics.json: extractor_agreement.by_feature.active_electrodes.spearman_rho --> | 1.3624 <!-- metrics.json: extractor_agreement.by_feature.active_electrodes.median_ratio_ours_over_theirs --> | ranks match; our activity cutoff is more inclusive |
| `pct_spikes_in_bursts` | 0.9703 <!-- metrics.json: extractor_agreement.by_feature.pct_spikes_in_bursts.spearman_rho --> | 0.9864 <!-- metrics.json: extractor_agreement.by_feature.pct_spikes_in_bursts.median_ratio_ours_over_theirs --> | reproduces on both axes |
| `spikes_per_burst` | 0.9167 <!-- metrics.json: extractor_agreement.by_feature.spikes_per_burst.spearman_rho --> | 0.9418 <!-- metrics.json: extractor_agreement.by_feature.spikes_per_burst.median_ratio_ours_over_theirs --> | close numerical agreement |
| `burst_rate_per_min` | 0.8907 <!-- metrics.json: extractor_agreement.by_feature.burst_rate_per_min.spearman_rho --> | 0.4423 <!-- metrics.json: extractor_agreement.by_feature.burst_rate_per_min.median_ratio_ours_over_theirs --> | disagrees twofold, purely from definition |
| `burst_duration_s` | 0.7926 <!-- metrics.json: extractor_agreement.by_feature.burst_duration_s.spearman_rho --> | 0.8189 <!-- metrics.json: extractor_agreement.by_feature.burst_duration_s.median_ratio_ours_over_theirs --> | the weakest rank agreement of the seven |

![Our extractor against the depositors' own meaRtools output, REAL recordings. Rank agreement on the horizontal axis, ratio of values on the vertical. Only `pct_spikes_in_bursts` lands on equality; burst rate differs twofold from definitional choice alone.](../outputs/figures/extractor_agreement_real.png){width=100%}

Read the two columns separately. The left one asks "do we rank recordings the same way?", and the
right one asks "do we get the same number?" Only one feature passes both tests. Three rank
recordings almost identically but on a different scale. And the burst measures disagree about
*which* recordings burst more: `burst_rate_per_min` differs by more than a factor of two.

**What this means.** Inside our dataset, comparisons are safe, because one code path produced every
number. Across papers they are not: "bursts per minute" from two laboratories can differ twofold from
the algorithm definitions alone, before any biology enters the picture. That is exactly why we
recompute everything from raw spikes instead of pooling published summary tables.

## 4. The three questions we asked the data

Every model in this report works on changes, not on static states. We pair each treated well with
its own pre-treatment baseline, and the model sees the log2 fold-change of each feature between the
two. This is the most consequential design decision in our analysis, and we made it for the same
reason an experimenter would: cultures differ enormously from one another. If one culture fires at
4 Hz and another at 8 Hz, that tells you about the cultures. If a culture drops from 8 Hz to 4 Hz
after a compound goes in, that is a pharmacological response.

For each feature, measured in the same well before and during exposure, the model sees

::: equation
Δ = log~2~(*x*~exposure~ / *x*~baseline~)
:::

so every halving or doubling moves Δ by one unit, whatever the culture's starting rate. A well that
falls silent is floored at one twentieth of its baseline, so abolished activity stays a finite
number, and a zero baseline leaves the ratio undefined rather than inventing a value for it.

1. **Mechanism.** Given the change in the nine features, can we tell which class of compound caused
   it — GABA-A antagonist, sodium-channel blocker, glutamatergic antagonist, and so on?
2. **Detection.** Can a model trained on one laboratory's data flag a known compound in a different
   laboratory's recordings?
3. **Platform contrast.** Does the same mechanism look the same on two different platforms?

**Two rules we never break**, and they are why our scores look lower than many published ones.
First, *we split strictly by culture*: wells from the same culture never end up on both sides of a
training/testing split. Wells within a culture are near-copies of each other, so splitting them would
let a model score well just by recognizing the culture — like seeing the exam questions in advance.
Second, *we resample uncertainty over cultures, not rows*: every interval is a cluster bootstrap over
whole cultures. With 73 pairs from 50 cultures, treating each row as independent would produce
intervals about half as wide — and wrong.

![How a mechanism call is made, on REAL recordings: each culture's paired wells become a nine-feature log-ratio vector, one regularized logistic regression maps it to a mechanism class, and cross-validation keeps every culture inside a single fold.](../outputs/figures/mechanism_classifier_real.png){width=100%}

---

## 5. Can you read a mechanism from the features?

Within one platform, we worked with 73 treated/baseline pairs
<!-- metrics.json: mechanism_baselines.task1_within_platform.n_pairs --> from 50 cultures
<!-- metrics.json: mechanism_baselines.task1_within_platform.n_groups --> across six mechanism classes (Kapucu 2022):

| model | balanced accuracy | 95% bootstrap CI | operational note |
|---|---|---|---|
| always guess commonest class | 0.167 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.dummy_most_frequent.balanced_accuracy_mean --> | — | theoretical chance across 6 classes |
| gradient boosting | 0.398 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_mean --> | 0.2762 to 0.5178 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.gradient_boosting.balanced_accuracy_ci95.1 --> | expressive model over-fits 73 examples |
| logistic regression (L2) | **0.537** <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean --> | 0.4216 to 0.6845 <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_ci95.1 --> | simpler linear boundary generalizes best |

Balanced accuracy averages the recall of each class, so a model cannot win just by getting the most
common class right. Three things stand out. First, there is real signal: the model calls roughly half
the mechanisms correctly where chance is 0.167
<!-- metrics.json: mechanism_baselines.task1_within_platform.chance_balanced_accuracy -->, and the
interval clears chance. Second, the interval is wide, because 73 pairs from 50 cultures is a small
study, and we show that openly. Third, the simpler model wins decisively. When a flexible model loses
to a linear one by fourteen points, the honest reading is that our dataset is small, not that
boosting is bad.

The errors are not random either. They cluster between the two GABA-A classes, antagonists and
positive modulators, which makes chemical sense given how both act on the excitatory/inhibitory
balance.

![Task 1 confusion matrix, REAL data: out-of-fold predictions over 73 treated/baseline pairs from grouped cross-validation. Errors concentrate between the two GABA-A classes.](../outputs/figures/mechanism_confusion.png){width=72%}

**The same task, refit in a second laboratory.** `vassallo2017` comes from a different laboratory
and a different vendor, with six compounds Kapucu never used, and each baseline sits on the same chip
as its doses: 171 pairs <!-- metrics.json: mechanism_baselines.task1b_independent_platform.n_pairs --> over 26 chips
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.n_groups --> in four classes. We
refit the same pipeline there from scratch, training and testing inside that source with the same
grouped cross-validation. Logistic regression reaches 0.333
<!-- metrics.json: mechanism_baselines.task1b_independent_platform.models.logistic_regression.balanced_accuracy_mean --> against a
chance of 0.25 <!-- metrics.json: mechanism_baselines.task1b_independent_platform.chance_balanced_accuracy -->, on an interval of 0.2317 to
0.4204 <!-- metrics.json: mechanism_baselines.task1b_independent_platform.models.logistic_regression.balanced_accuracy_ci95.1 --> that includes
chance.

These two results are not a before and after. They are different problems — six classes against
four, different compounds, and therefore different chance levels — so the only fair comparison is
each against its own baseline. Kapucu clears its chance of 0.167
<!-- metrics.json: mechanism_baselines.task1_within_platform.chance_balanced_accuracy --> more than
threefold; Vassallo clears its own by about a third, and its interval does not exclude chance. So
mechanism signal is not a quirk of the one dataset we built the method around, but it is much weaker
in the second laboratory. Note that nothing is transferred here: both models are trained and tested
inside their own laboratory. We test transfer itself — a model trained in one laboratory and applied
to another — in §6 and at the end of this section.

**Why we never pooled the platforms.** Pooling is tempting, and it would have produced a
better-looking number. It would also have been meaningless. Across the full pool, the label alone
determines the source for 97.35% of pairs
<!-- metrics.json: mechanism_baselines.label_source_confounding.source_determined_by_label -->, and
6 <!-- metrics.json: mechanism_baselines.label_source_confounding.n_labels_from_one_source_only --> of the 9 mechanism classes
<!-- metrics.json: mechanism_baselines.label_source_confounding.n_labels --> appear in only one
source. A pooled classifier would score well simply by learning which laboratory a recording came
from.

**The structural reason, and a test built around it.** The obstacle is even sharper than that
percentage suggests: the largest number of classes that any two laboratories share is 1
<!-- metrics.json: cross_lab_transfer.max_classes_shared_by_any_two_laboratories -->, and a
classifier needs at least two classes to tell apart. That means we cannot run
leave-one-laboratory-out cross-validation here at all.

What we can build is a deliberately confounded test. Three classes appear in more than one source,
and all three appear in Kapucu. We train there on 20 pairs <!-- metrics.json: cross_lab_transfer.n_train --> and test
on the other laboratories' versions of the same three classes, 535 pairs
<!-- metrics.json: cross_lab_transfer.n_test -->. In that test set, each class comes from exactly one
laboratory — a correspondence of 1.0 <!-- metrics.json: cross_lab_transfer.label_source_correspondence --> — so in
this test, class and laboratory cannot be told apart. A high score would have been ambiguous: it
could mean the model read the pharmacology, or only that each laboratory's recordings look
different. A score at chance has one reading: neither the mechanism signal learned in Kapucu nor
any laboratory difference carried the classes across. That is why we built the test so that only a
**low** score is interpretable.

The model scores 0.3482 <!-- metrics.json: cross_lab_transfer.balanced_accuracy -->, 95% CI [0.278, 0.4082]
<!-- metrics.json: cross_lab_transfer.balanced_accuracy_ci95 -->, against a chance of 0.3333
<!-- metrics.json: cross_lab_transfer.chance -->. That is indistinguishable from guessing: what the
model learned in one laboratory did not carry to the others.

**The control that makes it mean something.** Twenty training pairs is not many, so the failure
could come from sample size rather than from the change of laboratory. To check, we scored the same
classes and the same 20 pairs *inside* Kapucu with grouped cross-validation, which gives 0.5833
<!-- metrics.json: cross_lab_transfer.within_source_control.balanced_accuracy -->. Same data, same
size: the model learns something when it stays in one laboratory and nothing when it crosses into
another.

## 6. External detection, and the negative control that matters more

Can a model trained on Kapucu's plate flag bicuculline in ScreenNeuroPharm's chambers — a different
laboratory, and a different compound from the same mechanism class? Across 497 recordings
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.n -->, it calls
"GABA-A antagonist" 36.6% of the time
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.detection_rate -->.
There are two ways to read that, and only one is right. It is far above the rate at which that class
would appear by chance, so something does transfer. But it is not an autonomous detector, because
two-thirds of the time it says something else.

**The negative control.** What makes that first number trustworthy is the control. On Sharf's
diazepam recordings — a GABA-A positive modulator, the opposite pharmacology — the same model calls
"antagonist" only 7.7% of the time
<!-- metrics.json: mechanism_baselines.task2_external_detection.sharf_diazepam_negative_control.detection_rate -->.
A model that simply shouted "GABA-A antagonist" at anything unusual would have fired on both. Ours
does not.

**The dose signal is absent.** Within the bicuculline series, detection probability shows no
relationship with dose: Spearman ρ = 0.014
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman -->, p = 0.751
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman_p -->.
Mechanism and dose are different questions, and this model only answers the first. As §18 shows,
this turned out to be a property of the measurement, not of the model.

## 7. Rat to human

The whole promise of organ-on-a-chip work is that a human preparation tells you something a rodent
one cannot. So we trained on rat and tested on human, using only the five mechanism classes present
in both.

| transfer setting | balanced accuracy | chance | empirical finding |
|---|---|---|---|
| rat, tested within rat | 0.543 <!-- metrics.json: species_transfer.transfer.none.rat_within.balanced_accuracy --> | 0.2 | rodent primary culture baseline |
| human, tested within human | 0.646 <!-- metrics.json: species_transfer.transfer.none.human_within.balanced_accuracy --> | 0.2 | human iPSC model evaluation |
| **rat → human, no adaptation** | **0.55** <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy --> | 0.2 <!-- metrics.json: species_transfer.transfer.none.chance_balanced_accuracy --> | transfers without loss; permutation p = 0.0002 <!-- metrics.json: species_transfer.transfer.none.rat_to_human.permutation_p --> |
| rat → human, CORAL | 0.442 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy --> | 0.2 | covariance alignment erodes biological signal |
| rat → human, standardized | 0.408 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy --> | 0.2 | per-species normalization degrades toward chance |

The transfer is real, not luck, and two things stand out. First, rat → human (0.55
<!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy -->) is *not* worse
than rat tested within rat (0.543
<!-- metrics.json: species_transfer.transfer.none.rat_within.balanced_accuracy -->): mechanism
information crosses the species boundary about as well as it crosses between cultures within rat.
Second, **both domain-adaptation methods made it worse.** These are the standard tools for this
problem, but here they removed real signal rather than distribution shift — which is what happens
when a method designed for thousands of samples meets 37 rat training pairs
<!-- metrics.json: species_transfer.transfer.none.rat_to_human.n_train --> and 32 human testing pairs
<!-- metrics.json: species_transfer.transfer.none.rat_to_human.n_test -->. We report the unadapted
number as our result, with the adapted ones beside it, rather than picking whichever was highest.

## 8. Which features are actually doing the work?

If the nine-feature vector were doing nine distinct jobs, dropping most of it should hurt. It does
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

The two rate features alone do at least as well as all nine. The intervals overlap heavily, so we do
not claim that rate-only is better; what we can say is that the other seven features add nothing we
can detect at this sample size. That matters in practice, because the burst and synchrony
measures — the very features that differ most between laboratories (§3) — are also the ones carrying
the least mechanism information here.

**Validation on sensory neurons.** Because rate alone looked sufficient, we could bring in a source
that reports only firing rates: `fofie2025`, human iPSC-derived sensory neurons, 291 pairs
<!-- metrics.json: rate_only_arm.n_pairs --> from 48 wells <!-- metrics.json: rate_only_arm.n_groups -->. Training on
Kapucu's cortical rate changes and testing on these sensory neurons gives 0.5012
<!-- metrics.json: rate_only_arm.across_sources.balanced_accuracy --> against a chance of 0.3333
<!-- metrics.json: rate_only_arm.across_sources.chance -->, using a single feature and a preparation
with a completely different neuronal identity. What the two sources share is the compound's
mechanism, not the preparation or the network <!-- metrics.json: rate_only_arm.caveat -->.

## 9. How confident is the model, really?

**Calibration.** When the model says 70%, is it right 70% of the time? Our expected calibration error
is 0.1114 <!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.ece --> on 73 points
<!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.n -->, with a bootstrap interval reaching 0.2557
<!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.ece_ci95.1 -->. That upper
end is large. We would treat the predicted probabilities as a ranking, not as a risk — and not even
as a reliable ranking above 0.8
<!-- metrics.json: mechanism_baselines.task1_within_platform.calibration.bins.8.lower --> confidence,
where observed accuracy actually falls.

**Conformal sets.** Instead of a single answer, a conformal predictor returns a set of classes that
is guaranteed to contain the truth a stated fraction of the time. When we ask for 95%, empirical
coverage is 0.918
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep.0.empirical_coverage --> and the average set holds 4.562
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep.0.mean_set_size --> of the six classes.
A set of four or five classes out of six is honest and nearly useless — which is the right answer for
73 calibration points. A method that promised 95%
<!-- metrics.json: uncertainty_and_ablations.conformal_sweep.0.target_coverage --> coverage with
narrow sets on this much data would
be doing something wrong.

## 10. Untreated baselines: what differs before a drug is added

Let's set pharmacology aside for a moment. If we take untreated recordings and ask what separates rat
from human, and dissociated cultures from organoids, the features consistently fall into two groups
across three separate analyses. Rate-like features stay fairly stable across species, while synchrony
and burst-structure features move the most. Those same features differ most between baselines, carry
the least mechanism information (§8), and are the least reproducible across software (§3).

That three-way agreement is the most actionable result here for experimental design: the burst and
synchrony family is where you will have the most trouble, from three independent directions at once.

One analysis disagrees, and we kept it. The platform contrast (§11) finds that for the *same*
mechanism on two platforms, mean firing rate reverses sign while synchrony agrees. So three analyses
say synchrony is the unreliable family, and one says it is the reliable one. We do not resolve that
tension; we report it.

![Per-feature species shift, REAL data: rat against human baseline distributions with cluster-bootstrap intervals. Rate and burst-fraction features move least; synchrony moves most.](../outputs/figures/species_shift.png){width=50%}

![Untreated baseline activity, REAL recordings, by species and preparation. Burst rate is the one feature flat across all three groups.](../outputs/figures/baseline_species_comparison.png){width=58%}

## 11. The cross-platform reversal paradox

We compared GABA-A antagonists at 30 µM
<!-- metrics.json: mechanism_baselines.task3_platform_contrast.min_dose_uM --> and above on two
platforms: Kapucu's gabazine and ScreenNeuroPharm's bicuculline. Mean firing rate goes **up** on one
platform and **down** on the other, while synchrony moves the same way on both.

If you take away one thing from this report about cross-laboratory comparison, make it this: the
mechanism is identical, yet the direction of the most commonly reported endpoint is not. Anyone
pooling MEA pharmacology on the assumption that "GABA-A antagonists raise firing rate" would get it
backwards half the time.

![The same mechanism on two platforms, REAL data: per-feature log2 change for GABA-A antagonists. Mean firing rate reverses sign across platforms; synchrony agrees.](../outputs/figures/platform_contrast_gabaa_antagonist.png){width=58%}

---

# ACT III: Calibrating a biophysical digital twin

## 12. The digital twin: what it is, and what it is for

Our digital twin is a simulated culture, not a statistical surrogate. It has two compartments of
model neurons — 80% excitatory, 20% inhibitory, each a leaky integrate-and-fire neuron with
spike-frequency adaptation — wired densely within each compartment and sparsely between them through
a directional pathway. Virtual electrodes sit over the sheet and "hear" neurons within a set radius,
and their spike trains go through the same `features/mea.py` that the real recordings use. That last
point is a deliberate constraint: the twin cannot look good by being measured differently.

![The SIMULATED two-compartment twin: two chambers of model neurons, a one-way axon diode between them, and virtual electrodes whose listening radius is capped so that no two can hear the same neuron.](../outputs/figures/twin_architecture_simulated.png){width=100%}

**What the twin is for.** It lets us test scenarios that are expensive or impractical to try at the
bench. How many electrodes, and how much recording time, do you need to detect an effect? What
happens to the readout if axonal projections become twice as directional? And, most fundamentally, is
the metric you are tracking even capable of answering your question?

**What calibration means here.** We are not fitting the twin to reproduce one culture — that would
simply be overfitting to that culture. Instead, we ask it to land inside the interquartile band of 48
real chamber baselines
<!-- metrics.json: twin_hd12_calibration.targets_real.n_records --> from rat two-chamber devices. In
other words, we want it to be an unremarkable member of the population, not a copy of one member.

## 13. Calibration

We ran a random search over 9 parameters <!-- metrics.json: twin_hd12_calibration.tuned_parameters -->,
scoring each candidate on 2 network seeds <!-- metrics.json: twin_hd12_calibration.n_seeds_per_draw --> across 160 draws
<!-- metrics.json: twin_hd12_calibration.n_draws -->. None of the chosen values sits on the edge of
the range it was drawn from, so the optimum is genuinely inside the search space rather than clipped
at a boundary
<!-- metrics.json: twin_hd12_calibration.boundary_hits -->.

| feature | simulated twin value | real band (IQR) | calibration result |
|---|---|---|---|
| `synchrony_index` (STTC) | 0.8387 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.synchrony_index --> | 0.678 to 0.843 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.synchrony_index.q75 --> | inside target band |
| `burst_rate_per_min` | 22.9219 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.burst_rate_per_min --> | 12.627 to 29.726 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.burst_rate_per_min.q75 --> | inside target band |
| `network_burst_rate_per_min` | 24.0 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.network_burst_rate_per_min --> | 11.692 to 22.697 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.network_burst_rate_per_min.q75 --> | minor overshoot |
| `mfr_hz` | 5.9862 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.mfr_hz --> | 6.793 to 21.007 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.mfr_hz.q75 --> | minor undershoot |
| `pct_spikes_in_bursts` | 83.8409 <!-- metrics.json: twin_hd12_calibration.best.features_simulated.pct_spikes_in_bursts --> | 84.657 to 93.685 <!-- metrics.json: twin_hd12_calibration.targets_real.by_feature.pct_spikes_in_bursts.q75 --> | near lower bound |

![SIMULATED against REAL: the calibrated twin's baseline features against the real baseline band from rat two-chamber devices.](../outputs/twin/hd12_calibration_simulated.png){width=54%}

Getting to this table took us longer than anything else, and the story is a warning for anyone who
simulates cultures.

For a long time the twin was far too synchronous — an STTC around 0.96 against a real band topping
out at 0.84 — and it looked as if we were missing a biophysical mechanism. We tried three candidates,
and none of them worked: short-term synaptic depression, varied axonal conduction delays, and
distance-dependent wiring, which actually pushed synchrony the wrong way.

**The real cause was not biophysics; it was cell density and electrode geometry.** Our simulated
chamber held 250 neurons in 1.92 mm², about 130 neurons/mm², whereas a real dissociated culture runs
roughly 1000 to 3000. On top of that, we had let the electrode's listening radius grow larger than
the spacing between electrodes. The oversized electrode compensated for the sparse culture in two
ways at once: it pooled enough neurons to produce a plausible firing rate, and, because neighboring
listening areas overlapped, it counted one neuron's spikes on several electrodes — which inflates
pairwise synchrony by construction, with no biology involved. Once we fixed the density at 967
<!-- metrics.json: twin_hd12_calibration.best.draw.n_per_compartment --> neurons per compartment and
imposed one geometric rule, synchrony landed inside the real band for the first time:

**An electrode's listening radius may not exceed half the electrode pitch, r ≤ p/2.**

On our 200 µm <!-- metrics.json: twin_hd13_perturbation.simulation.layout.pitch_um --> grid, that
rule guarantees that no neuron is heard by two electrodes. The calibrated radius settled at
92.5135 µm <!-- metrics.json: twin_hd12_calibration.best.draw.detection_radius_um -->, inside the
cap. The rule carries over to physical arrays: if an electrode's catchment exceeds half the pitch,
expect synchrony to read high for reasons that have nothing to do with the culture.

**Out-of-sample trust.** The winning parameter set reached a loss of 0.0756
<!-- metrics.json: twin_hd12_calibration.held_out.selection_score --> on its selection seeds. On 5 seeds
<!-- metrics.json: twin_hd12_calibration.held_out.n_seeds --> the search never saw, it scores 0.1994
<!-- metrics.json: twin_hd12_calibration.held_out.score_mean -->, with synchrony holding at 0.7732
<!-- metrics.json: twin_hd12_calibration.held_out.synchrony_index_mean --> ± 0.0685
<!-- metrics.json: twin_hd12_calibration.held_out.synchrony_index_sd --> and inside the band on 60% of seeds
<!-- metrics.json: twin_hd12_calibration.held_out.synchrony_index_in_band_fraction -->. Just like a
real culture, the twin varies enough from run to run that a single simulated recording can fall
outside the band.

## 14. Drug knobs: does the twin respond like a culture?

The twin has three pharmacological knobs: it can scale inhibitory transmission (`gabaa_scale`),
scale excitatory transmission (`ampa_scale`), or raise the spike threshold (`threshold_scale`). We
tested directional validity — does each knob move each feature in the right direction? — and we only
scored features whose real direction is established by a cluster bootstrap.

| pharmacological knob | mechanistic action | directional concordance |
|---|---|---|
| AMPA blockade | downscale excitation | 4 <!-- metrics.json: twin_hd13_perturbation.knobs.2.n_features_agreeing --> / 4 <!-- metrics.json: twin_hd13_perturbation.knobs.2.n_features_scored --> |
| sodium-channel block | elevate spike threshold | 3 <!-- metrics.json: twin_hd13_perturbation.knobs.3.n_features_agreeing --> / 3 <!-- metrics.json: twin_hd13_perturbation.knobs.3.n_features_scored --> |
| GABA-A antagonism | downscale inhibition | 4 <!-- metrics.json: twin_hd13_perturbation.knobs.0.n_features_agreeing --> / 5 <!-- metrics.json: twin_hd13_perturbation.knobs.0.n_features_scored --> |
| GABA-A potentiation | upscale inhibition | 1 <!-- metrics.json: twin_hd13_perturbation.knobs.1.n_features_agreeing --> / 4 <!-- metrics.json: twin_hd13_perturbation.knobs.1.n_features_scored --> |

![SIMULATED drug knobs against REAL pharmacology directions, per feature. Only features whose real direction is established by a cluster bootstrap are scored.](../outputs/twin/hd13_drug_knob_validation_simulated.png){width=52%}

Blocking excitation and blocking sodium channels behave correctly on every scored feature. Blocking
inhibition is mostly right. Potentiating inhibition is mostly wrong — and that is the knob with the
weakest real reference behind it, just 17 pairs from two sources. So the twin is usable for
reasoning about strong, well-evidenced perturbations, but we have not validated it for subtle
allosteric modulation.

---

# ACT IV: Inverting the twin and resolving directionality

## 15. Two-compartment devices, and a result we did not want

The twin models an asymmetric two-compartment chip — the "axon diode" idea, where microchannel
geometry biases which way axons grow. Before simulating it, we asked what real devices actually show.

**Tissue identity drives asymmetry.** Across 16 devices
<!-- metrics.json: compartment_asymmetry.n_devices -->, we measured how asymmetric the coupling
between compartments is. Devices pairing two different tissues (cortex with hippocampus) are more
asymmetric, with a median of 0.138
<!-- metrics.json: compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.median_heterogeneous -->,
than devices pairing a tissue with itself, median 0.094
<!-- metrics.json: compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.median_homogeneous -->, p = 0.042
<!-- metrics.json: compartment_asymmetry.pooled_heterogeneous_vs_homogeneous.p_value -->. What the
chambers contain matters more than the fact that they are connected.

**The directionality null.** One deposit recorded axonal crossings directly inside the tunnels, on
10 cultures <!-- metrics.json: axonal_directionality.n_cultures --> (Brewer 2021). That let us ask
the question a chamber readout exists to answer: if you had only the chamber recordings, could you
tell which way the axons were firing? Using all four chamber-level measures together — coupling
asymmetry, peak correlation, lag and rate imbalance — a classifier reaches a balanced accuracy of
0.3854 <!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.balanced_accuracy --> against a
chance of 0.5 <!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.chance -->.
That is worse than guessing. And it is not just a small-sample failure: with this many cultures, the
analysis could have detected a balanced accuracy of 0.6438
<!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.detectable_balanced_accuracy --> if one
had been there. We could only recover direction by instrumenting the microchannels themselves.

![Measured axonal flow against chamber-level readouts, REAL recordings. The chamber measures do not order the cultures by the direction their axons actually fired.](../outputs/figures/axonal_flow_vs_chamber.png){width=100%}

![Inter-compartment coupling lag by device type, REAL data, one value per device: five CxCx, five HpHp, six CxHp. The two same-tissue types are indistinguishable; the heterogeneous type sits apart.](../outputs/figures/compartment_lag_by_device_type.png){width=56%}

## 16. Running the model backwards, and what that cannot do

The more ambitious use of a twin is to run it backwards: given a recording, infer which knob was
turned. We did this with simulation-based inference, using a neural posterior estimator trained on
3,000 simulations <!-- metrics.json: twin_hd14_inference.n_simulations --> at the calibrated cell
density. Here the twin works as a generator of labelled data: every simulation is a training pair
whose true parameters are known exactly, which no bench experiment can supply.

![Running the SIMULATED twin backwards: draws from the prior are simulated and summarized as features, a neural posterior estimator learns to invert them and is then asked about a real recording. The bottom row is the honesty check applied before any posterior is quoted.](../outputs/figures/inference_network_simulated.png){width=100%}

The estimator is a masked autoregressive flow, built with the `sbi` library: a stack of 5
<!-- metrics.json: twin_hd14_inference.density_estimator.n_autoregressive_transforms --> autoregressive
transforms with 50 <!-- metrics.json: twin_hd14_inference.density_estimator.hidden_units_per_layer -->
hidden units each, 34,550 <!-- metrics.json: twin_hd14_inference.density_estimator.n_parameters -->
trained weights in all. It learns the joint density of the five parameters given the 19 summary
features, so it returns a full posterior rather than a point estimate, and it never has to run the
simulator again once trained. That is what makes the inference amortized: on 20 held-out
simulations, a posterior of 1,000
<!-- metrics.json: twin_hd14_inference.amortized_inference.posterior_draws --> draws takes a median
of 0.3061 s <!-- metrics.json: twin_hd14_inference.amortized_inference.median_seconds_per_posterior -->
on our workstation, against 6.69 s
<!-- metrics.json: twin_hd14_inference.amortized_inference.forward_simulation_seconds --> for one
10-second run of the simulator, which a method that simulates afresh for every recording would need
many times over. That speed holds for recordings the twin can produce; for those it cannot,
sampling stalls, which is the refusal described below.

The same generator could in principle pre-train models for assays where real data are scarce, such
as human iPSC cultures, but only once the twin is calibrated for them. Today it is calibrated on rat
devices and sits inside the real band on two of five features, so we have not done this.

Before we trust any posterior, we run two honesty checks.

| model parameter | biophysical role | posterior contraction | KS calibration (p) | uniformity status |
|---|---|---|---|---|
| `threshold_scale` | action-potential threshold | 0.661 <!-- metrics.json: twin_hd14_inference.contraction.threshold_scale.contraction --> | 0.6105 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.threshold_scale.ks_p --> | well calibrated |
| `ampa_scale` | excitatory synaptic weight | 0.599 <!-- metrics.json: twin_hd14_inference.contraction.ampa_scale.contraction --> | 0.3128 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.ampa_scale.ks_p --> | well calibrated |
| `log10_forward_ratio` | axonal pathway asymmetry | 0.438 <!-- metrics.json: twin_hd14_inference.contraction.log10_forward_ratio.contraction --> | 0.1517 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.log10_forward_ratio.ks_p --> | well calibrated |
| `gabaa_scale` | inhibitory synaptic weight | 0.3 <!-- metrics.json: twin_hd14_inference.contraction.gabaa_scale.contraction --> | 0.0174 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.gabaa_scale.ks_p --> | **over-confident** |
| `inter_in_degree` | cross-chamber connectivity | 0.269 <!-- metrics.json: twin_hd14_inference.contraction.inter_in_degree.contraction --> | 0.0121 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.inter_in_degree.ks_p --> | **over-confident** |

Contraction measures how much narrower the posterior is than the prior; 0 means the model learned
nothing. Rank uniformity asks whether the stated intervals are honest. Three knobs pass. Inhibitory
transmission does not — and it is the knob our pharmacology sections depend on, so **we do not quote
its intervals as calibrated.** Before the density correction (§13), no parameter passed and a fifth of
our simulations were silent; now 0
<!-- metrics.json: twin_hd14_inference.fraction_silent_simulations --> are. Simulating at the
calibrated density brought three knobs into calibration. Appendix G shows the full rank histograms.

**The false-disinhibition artifact.** When we applied the model to real untreated chips, it returned
an inferred inhibition scale of 0.5219
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median --> on an interval of 0.2732 to 0.8503
<!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median_ci95.1 --> — apparent
disinhibition in wells that had received no drug. That is not biology. It is the model's way of
saying "real recordings do not look like my simulations," in the only vocabulary it has. When we
reformulated inference around paired deltas, untreated chips moved to 0.9108
<!-- metrics.json: twin_hd17_delta_inference.real_data.gabaa_scale_at_baseline.median --> on 0.8583 to 0.966
<!-- metrics.json: twin_hd17_delta_inference.real_data.gabaa_scale_at_baseline.median_ci95.1 -->, and
contraction rose to 0.5375 <!-- metrics.json: twin_hd17_delta_inference.held_out_checks.contraction -->.
Most of the offset disappears, but not all of it: the interval still stops short of neutral.

**Out-of-distribution rejection.** We asked for 200 draws
<!-- metrics.json: demo_precompute.posterior_draws.requested --> per recording across 113
<!-- metrics.json: demo_precompute.posterior_draws.n_recordings --> bicuculline trials, and the
posterior could not deliver them for 9
<!-- metrics.json: demo_precompute.posterior_draws.n_short_of_requested --> of those recordings; the
worst yielded only 1 <!-- metrics.json: demo_precompute.posterior_draws.fewest_draws --> draw, and 0
<!-- metrics.json: demo_precompute.posterior_draws.n_unsampled --> failed outright. This only appeared
once the twin moved to its calibrated density: a posterior fitted to a sharper forward model is less
forgiving of observations that model cannot produce. In effect the posterior carries a built-in
out-of-distribution flag: a recording it cannot sample from is one the twin cannot produce, and in
high-content screening that recording should go to a person rather than get a forced answer. We
have not measured how well this flag separates genuine anomalies from ordinary recordings, so we
report it as a behavior, not as a validated detector.

## 17. What it takes to see directionality at all

One thing a twin is genuinely good at is telling you how much experiment you need. Across 120
simulated networks <!-- metrics.json: twin_hd16_directionality.n_networks --> with known wiring, the
cheapest configuration that reaches our target distinguishes asymmetric from symmetric wiring with an
AUC of 0.8928
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_all_features.auc -->,
using 24 electrodes per compartment
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_all_features.electrodes_per_compartment -->
and a 10-second recording
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_all_features.duration_s -->.

**The hidden catch.** Most of that signal is firing rate, not direction. A directional pathway
delivers more excitation to its target chamber, so the target simply fires more. If we lower the
receiving chamber's background drive, measured coupling asymmetry moves from 0.109
<!-- metrics.json: twin_hd11_validation.directionality_sweep.0.asymmetry_mean --> to 0.681
<!-- metrics.json: twin_hd11_validation.directionality_sweep.2.asymmetry_mean --> with the wiring
unchanged. Across the sweep, drive moves the measure further than wiring does: a span of 0.671
<!-- metrics.json: twin_hd11_validation.confound_summary.raw.drive_span --> against 0.109
<!-- metrics.json: twin_hd11_validation.confound_summary.raw.wiring_span -->. Regress firing rate
out of the features, inside each training fold, and the cheapest configuration doubles to 48
<!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.electrodes_per_compartment -->
electrodes per compartment. Twenty-four electrodes and ten seconds will tell these cultures apart,
but claiming that this reflects directional communication requires explicit rate-matching controls.

---

# ACT V: The negative results and scientific boundaries

## 18. Collection of negative results

Negative results are easy to miss when they are scattered through a long report. Collected in one
place, they are the part of our work we would most like to see challenged — and the part most likely
to save someone else an experiment. Each one tells you what not to rely on, so the last column says
what we would do instead.

| # | finding | evidence | what to do instead |
|---|---|---|---|
| 1 | no monotonic dose trend in the summary features | 0 <!-- metrics.json: feature_dose_response.verdict.n_features_moving_absolute --> of 7 <!-- metrics.json: feature_dose_response.absolute.n_features_tested --> testable features move with dose | do not read concentration from summary tables; keep the spike times so other features can be tried |
| 2 | chamber readouts miss direction | chamber classifier 0.3854 <!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.balanced_accuracy --> against 0.5 chance | put electrodes inside the microchannels |
| 3 | no cross-laboratory transfer | 0.3482 <!-- metrics.json: cross_lab_transfer.balanced_accuracy --> against 0.3333 <!-- metrics.json: cross_lab_transfer.chance --> chance | validate inside one laboratory; a cross-laboratory test needs two laboratories sharing at least two classes |
| 4 | domain adaptation backfires | CORAL drops transfer to 0.442 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy -->, standardization to 0.408 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy --> | report the unadapted transfer first; adapt only once it beats that on held-out cultures |
| 5 | synchrony was an artifact | depression and delay dispersion fail to fix synchrony | match cell density to culture and cap each electrode's radius at half the pitch (§13) |
| 6 | posterior over-confidence | `gabaa_scale` fails rank uniformity at p = 0.0174 <!-- metrics.json: twin_hd14_inference.rank_uniformity_ks_p.gabaa_scale.ks_p --> | check rank uniformity before quoting an interval; quote only parameters that pass |
| 7 | baseline read as drug effect | absolute features infer 0.5219 <!-- metrics.json: twin_hd15_real_inference.gabaa_scale_at_baseline.median --> inhibition scale on untreated chips | infer from paired deltas against each well's own baseline (§16) |
| 8 | support rejection | the calibrated twin cannot sample 9 <!-- metrics.json: demo_precompute.posterior_draws.n_short_of_requested --> of 113 <!-- metrics.json: demo_precompute.posterior_draws.n_recordings --> real recordings | treat a refusal as an out-of-distribution flag and send the recording to a person |

**On the first, which matters most.** §6 showed that the mechanism classifier had no dose
relationship, and §16 showed that the inverse model could not recover dose either. It would be
natural to blame the models, so we asked the question directly: across 113 bicuculline recordings
<!-- metrics.json: feature_dose_response.n_recordings --> on 16 devices
<!-- metrics.json: feature_dose_response.n_devices -->, does *any* feature move monotonically with
dose? Seven of the nine had enough points to test; the two burst-shape features, burst duration and
spikes per burst, did not. Using raw values, 0
<!-- metrics.json: feature_dose_response.verdict.n_features_moving_absolute --> of the 7
<!-- metrics.json: feature_dose_response.absolute.n_features_tested --> do. After subtracting each
device's own baseline, 0
<!-- metrics.json: feature_dose_response.verdict.n_features_moving_baseline_subtracted --> do — in
both cases after a Holm correction for testing seven features at once. The models were not failing to
extract a signal. The signal simply was not there.

## 19. What you cannot conclude from this work

- **No clinical or safety claims.** NeuroMPS is an exploratory analytical framework. We make no
  efficacy, diagnostic or clinical claims, and the design could not support them.
- **It is not a screening tool.** A within-platform mechanism accuracy of 0.537
  <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_mean -->,
  with an interval from 0.4216 to 0.6845
  <!-- metrics.json: mechanism_baselines.task1_within_platform.models.logistic_regression.balanced_accuracy_ci95.1 -->,
  and near-chance performance across laboratories do not make a decision instrument.
- **Interspecies support is narrow.** Rat-to-human transfer rests on 37 rat training pairs and 32
  human testing pairs, which supports the direction of the effect rather than its size.
- **The simulator is not fully reconciled.** The twin sits inside the real band on two of five
  features, varies from seed to seed, and gets GABA-A potentiation mostly wrong.
- **There is no monotonic dose tracking.** The feature deltas contain no monotonic concentration
  signal across the bicuculline series.
- **The species balance is severe.** Our asset holds 21,530 rat
  <!-- metrics.json: data_asset.counts.by_species.rat --> against 357 human
  <!-- metrics.json: data_asset.counts.by_species.human --> records — a consequence of open-access
  licensing, not a design choice.
- **The samples are small.** Our primary classification rests on 73 paired recordings from 50
  independent cultures.

---

# ACT VI: Application value, translation and open verification

## 20. Application value and translation for neural organ-on-a-chip

The main bottleneck in making neural organ-on-a-chip systems useful is not microfluidic fabrication;
it is making functional electrophysiology interpretable and reproducible. Academic centers and
commercial vendors alike produce multi-channel MEA recordings that remain hard to compare across
sites. Our results translate into practical guidance in four areas.

**Where this sits.** The US FDA Modernization Act 2.0 (2022) replaced the statutory reference to
animal tests with "nonclinical tests", a category that names cell-based assays, organ chips and
microphysiological systems, and computer modeling. That makes comparable, checkable readouts from
neural chips more useful, and it is the setting our tools are built for. The provenance chain —
license, source URL and a cryptographic checksum recorded for every file, and a build that halts on any mismatch —
can serve as an open template for documenting where such data came from. It is a record of
traceability, not a qualification process: NeuroMPS has not been qualified or reviewed by any
regulator, and nothing in this report is evidence for a submission.

**1. Design guidance for chip hardware.** Developers often assume they can validate directional
connectivity by measuring cross-correlation latencies or coupling asymmetries between open somatic
chambers. §15 shows on real devices that this does not work: chamber readouts reached 0.3854
<!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.balanced_accuracy -->
balanced accuracy where chance is 0.5, on a sample that could have detected 0.6438
<!-- metrics.json: axonal_directionality.chamber_readouts_predict_direction.detectable_balanced_accuracy -->.
To show that axons really conduct one way, you need microelectrodes inside the microchannels
themselves — and knowing that saves time otherwise spent on measures that cannot answer the question.

**2. A standardization framework for contract research.** Screening protocols should move away from
proprietary vendor summary tables toward standardized processing from raw spikes. Our feature
ablation (§8) shows that two rate metrics match the predictive power of the full nine-feature
vector, while avoiding the cross-platform variance that plagues burst-structure and synchrony
metrics (§3, §11). A smaller, more reliable feature vector is the more reproducible instrument.

**3. Cheaper assay design in silico.** Plating human iPSCs is expensive and takes weeks. The twin
gives you a testbed for pre-registering an assay: the power sweep in §17 sets a minimum electrode
count (24 electrodes per compartment, 10-second recordings) for distinguishing wiring regimes, and the
listening-radius finding in §13 gives you a way to diagnose electrode catchment overlap in physical
recordings.

**4. Paired-delta normalization for analysis pipelines.** Resting firing rates in primary cultures
span an order of magnitude. Expressing screening features as paired log-ratios against each well's
own baseline stabilizes the analysis, raises posterior contraction from 0.3
<!-- metrics.json: twin_hd14_inference.contraction.gabaa_scale.contraction --> to 0.5375
<!-- metrics.json: twin_hd17_delta_inference.held_out_checks.contraction -->, and removes most of
the false-disinhibition artifact described in §16.

## 21. How to reproduce and check this work

Every computed number in this report carries an automated pointer back to the
`outputs/metrics.json` key that produced it, and a test fails if any number drifts from that file.
To reproduce everything from a clean machine, run these steps in a terminal:

| step | command | what it does |
|---|---|---|
| 1. get the code | `git clone https://github.com/Vodolay/AI4S`, then `cd AI4S` | downloads the repository |
| 2. create the environment | install uv as the README shows, then `uv venv --python 3.11 .venv` | an isolated environment with the pinned Python |
| 3. install | `uv pip install --python .venv/bin/python -e ".[dev]"` | NeuroMPS and its pinned dependencies |
| 4. quick run | `.venv/bin/python run_all.py --quick` | every stage end to end on the reduced data tier |
| 5. full run | `.venv/bin/python run_all.py` | the full data tier, 3,000 <!-- metrics.json: twin_hd14_inference.n_simulations --> forward simulations, every figure |
| 6. check the numbers | `.venv/bin/python -m neuromps.report.verify` | re-reads every citation in this report and fails on any disagreement |
| 7. open the demo | `.venv/bin/streamlit run app/app.py` | interactive mechanism prediction, twin sweeps and record browser |

Use the environment's own `python` and `streamlit`, as shown. We ran these steps on a fresh clone,
and every cited number reproduced exactly except ten that depend on the machine: the timings, and
the counts and intervals from posterior sampling that stops at a time limit (§16). Provenance is
guarded throughout: `data/raw/MANIFEST.json` records every source's URL, license and checksum.

## 22. AI assistance disclosure

We built this project in a paired workflow with Anthropic's Claude Code. Claude wrote most of the
code and tests under step-by-step human direction: we set the plan and the phase gates, chose the
scope, approved each step, and directed the science. We also used it to draft and revise parts of
this report, which we then edited ourselves. Three practices kept that collaboration honest. Numbers
are emitted into `metrics.json` by scripts and re-checked by `neuromps.report.verify`, never typed
by hand. We report results as we observed them, including the ones that failed. And
`report/LAB_NOTEBOOK.md` records what we tried and abandoned. No AI model is part of the delivered
pipeline itself: every statistical model in it is trained from scratch on the data described here.

---

# Appendices: consolidated technical reference

## Appendix A & B — Record architecture, data levels and ingestion ETL

Every record declares a data level: `spike_times`, `per_well_features` or `fitted_curve`. The level
decides what may be computed from the record, and the build enforces that instead of trusting it — a
feature that needs spikes cannot be derived from a record holding only a fitted curve. Records
validate against `schema/neuromps_schema.json` at three conformance levels, so a partially specified
source can still enter the asset as long as it declares what it is missing.

We record quality-control flags but never apply them silently: 537
<!-- metrics.json: data_asset.counts.by_usable.False --> of 21,887
<!-- metrics.json: data_asset.n_records --> records are marked unusable, with the diagnostic reason
attached, and the flag travels with the record. Each analysis decides whether to honor it and says
which it did; if we dropped flagged records inside the pipeline, the reader could never see the drop.

Each record also carries the method string that produced its features — the exact burst rule, the
synchrony method and the active-electrode threshold — so two records computed under different
settings cannot be compared by accident.

Ingestion lives in `src/neuromps/fetch/`. It uses HTTP range requests against remote ZIP archives to
pull only the files we need, instead of downloading multi-gigabyte archives. Two server behaviors
needed special handling and now have regression tests: one EPA host returns HTTP 500 for suffix
ranges, and the same host, when offered gzip encoding, replies with a plausible `Content-Range` while
actually streaming the whole 7.5 GB file from byte zero.

## Appendix C — Data sources and licenses

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
only and is not part of the 21,887-record <!-- metrics.json: data_asset.n_records --> asset; we use
it solely for the cross-source rate-only arm. We record `tukker2020` as a skipped source, with its
reason, rather than silently leaving it out.

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

Each of the following is a test that fails, not a convention we merely follow.

- **Provenance.** The asset build fails if any record's provenance hash disagrees with the manifest.
- **One feature extractor.** The twin computes features through the same `features/mea.py` as the
  real recordings and — after a defect that survived several phases — through the same coupling
  measure on the same per-electrode input.
- **Interval discipline.** A contract test fails if any headline metric loses its confidence
  interval, and the resample count sits next to every estimate.
- **Cache meaning.** The simulation cache stores its summary names, electrode count, a hash of the
  code that defines every summary, and the calibration it was generated under; it refuses to load
  across a change in any of them.
- **Report numbers.** `neuromps.report.verify` re-reads every citation in this report and fails on
  any disagreement.
- **No clinical claims.** A test scans the reports for efficacy, diagnostic and treatment language
  and fails unless the line is a disclaimer ruling it out.

## Appendix F & G — Coupling asymmetry proof, and posterior calibration diagnostics

**Appendix F.** The measure the twin exists to use reports relative firing rate as much as pathway
direction. The cleanest demonstration holds the wiring **fixed** and changes only the drive: lowering
the receiving chamber's background drive moves asymmetry from 0.109
<!-- metrics.json: twin_hd11_validation.directionality_sweep.0.asymmetry_mean --> to 0.681
<!-- metrics.json: twin_hd11_validation.directionality_sweep.2.asymmetry_mean --> with the pathway
unchanged. Across the whole sweep, drive moves the measure further than wiring does: a span of 0.671
<!-- metrics.json: twin_hd11_validation.confound_summary.raw.drive_span --> against 0.109
<!-- metrics.json: twin_hd11_validation.confound_summary.raw.wiring_span -->. Until something rules
it out, an open-chamber asymmetry is a statement about which chamber is busier.

**Appendix G.** We trained on 3,000 simulations <!-- metrics.json: twin_hd14_inference.n_simulations --> at the
calibrated density, of which 0 <!-- metrics.json: twin_hd14_inference.fraction_silent_simulations --> produced no
spikes. The two parameters that fail rank uniformity are the connectivity parameter and the
inhibitory knob — the latter being the one §16 depends on, which is why we do not quote its intervals
as calibrated.

![SIMULATED posterior calibration: the rank of the true parameter among posterior samples. Flat would be calibrated; these are not.](../outputs/twin/hd14_posterior_ranks_simulated.png){width=100%}

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
<!-- metrics.json: microchannel_directionality.comparison.kruskal_p -->, and every Holm-adjusted
comparison against the control geometry is 1.0
<!-- metrics.json: microchannel_directionality.comparison.vs_reference_holm.rams -->. The null is not
an artifact of our geometry assumptions either: it survives every detector variant we tried, with a
worst-case Kruskal p of 0.13074
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_one_missing.kruskal_p -->.
Tolerating single-electrode dropouts triples what the detector has to work with, from 336,794
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_strict.total_events --> analyzed
events to 1,101,233
<!-- metrics.json: microchannel_directionality.geometry_robustness.variants.column_one_missing.total_events -->,
and does not change the outcome. We also measured and report a counting confound: chips with fewer
propagation events look more one-sided (Spearman ρ = −0.494
<!-- metrics.json: microchannel_directionality.event_count_confound.spearman_rho -->, p = 0.01032
<!-- metrics.json: microchannel_directionality.event_count_confound.p_value -->), which is what a
counting artifact looks like. Condition is also confounded with culture batch by the deposit's own
design
<!-- metrics.json: microchannel_directionality.batch_confound -->.

![Microchannel propagation asymmetry by device geometry, REAL recordings.](../outputs/figures/microchannel_asymmetry_real.png){width=52%}

**Appendix K.** `--quick` completes end to end in 544.7 s
<!-- metrics.json: harness_quick.total_s --> on a workstation CPU with 20 cores
<!-- metrics.json: harness_quick.cpu_count -->, and in under twenty minutes on a dual-vCPU CI runner;
the two differ because `worker_processes` correctly asks for the cores it is given. The run itself
writes that figure, and only a run that executed every stage can write it, so a partial re-run
cannot quietly shrink it. It is one measurement rather than a bound: repeat runs on the same machine
vary with whatever else the machine is doing, and the slowest we have seen crossed ten minutes. The
longest stage is the twin, at 344.3 s
<!-- metrics.json: harness_quick.by_stage_s.twin -->. Full reproduction runs through
`python run_all.py` after a full data-tier fetch. The environment is pinned in `pyproject.toml` and
recorded from what actually imported: Python `3.11.16` <!-- metrics.json: harness_quick.python -->,
NumPy `2.3.5` <!-- metrics.json: harness_quick.numpy -->, Brian2 `2.9.0`
<!-- metrics.json: harness_quick.brian2 -->, PyTorch `2.14.0` <!-- metrics.json: harness_quick.torch -->
and sbi `0.27.0` <!-- metrics.json: harness_quick.sbi -->. We hold NumPy back because its next minor
release removes `ndarray.ptp`, which Brian2 imports.

## Appendix L, M & N — Sensory validation, dose tracking and species transfer

**Appendix L, the rate-only arm.** Human sensory neurons across 291 pairs
<!-- metrics.json: rate_only_arm.n_pairs --> in `fofie2025` shift by −0.1743
<!-- metrics.json: rate_only_arm.by_mechanism.control.median_log2_change --> log2 units for vehicle
controls, against −1.9529
<!-- metrics.json: rate_only_arm.by_mechanism.na_channel_blocker.median_log2_change --> for
sodium-channel blockers, which yields 0.5012
<!-- metrics.json: rate_only_arm.across_sources.balanced_accuracy --> balanced accuracy against a chance of
0.3333 <!-- metrics.json: rate_only_arm.across_sources.chance -->.

**Appendix M, Task 2 dose tracking.** The detection rate rises from 0.179
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.0.detection_rate --> at 0.1 µM
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.0.dose_uM --> to 0.426
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.5.detection_rate --> at 30 µM
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.5.dose_uM -->.
Quoted on its own, that would look like a dose response, so here is the rest of the series: the rate
peaks at 0.48 <!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.6.detection_rate -->
at 100 µM <!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.6.dose_uM -->
and falls to 0.091
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.8.detection_rate --> at the
top of it, 500 µM
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.by_dose.8.dose_uM -->.
The overall rank correlation is not significant (ρ = 0.014
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman -->, p = 0.751
<!-- metrics.json: mechanism_baselines.task2_external_detection.screenneuropharm_bicuculline.dose_spearman_p -->).

**Appendix N, species transfer, all settings.**

| transfer setting | balanced accuracy | 95% bootstrap CI |
|---|---|---|
| human within-species | 0.646 <!-- metrics.json: species_transfer.transfer.none.human_within.balanced_accuracy --> | 0.483 to 0.78 <!-- metrics.json: species_transfer.transfer.none.human_within.balanced_accuracy_ci95.1 --> |
| rat within-species | 0.543 <!-- metrics.json: species_transfer.transfer.none.rat_within.balanced_accuracy --> | 0.458 to 0.7 <!-- metrics.json: species_transfer.transfer.none.rat_within.balanced_accuracy_ci95.1 --> |
| rat → human, zero-shot | 0.55 <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy --> | 0.4 to 0.6 <!-- metrics.json: species_transfer.transfer.none.rat_to_human.balanced_accuracy_ci95.1 --> |
| rat → human, CORAL | 0.442 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy --> | 0.302 to 0.591 <!-- metrics.json: species_transfer.transfer.coral.rat_to_human.balanced_accuracy_ci95.1 --> |
| rat → human, standardized | 0.408 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy --> | 0.267 to 0.568 <!-- metrics.json: species_transfer.transfer.standardize.rat_to_human.balanced_accuracy_ci95.1 --> |

Every interval resamples cultures, not rows. For the within-species rows it is taken over the
out-of-fold predictions pooled across the repeated grouped splits, and a culture drawn into a
resample brings all of its predictions with it; for the transfer rows it is taken over the human
test cultures.

## Appendix O — Third-party software, models and AI tools

**Software.** We use the libraries below unmodified, as dependencies; none of their source is
redistributed here. Exact versions are pinned in `pyproject.toml`, and `THIRD_PARTY.md` lists each
package's license as read from its installed metadata.

| license | packages | what they do here |
|---|---|---|
| BSD (3-clause and BSD-style) | numpy, scipy, pandas, scikit-learn, h5py, altair, xlrd | numerics, data handling, classifiers, charts, file readers |
| MIT | jsonschema, openpyxl, python-docx, pytest, ruff | schema validation, spreadsheet input, report build, tests, linting |
| Apache 2.0 | sbi, pyarrow, requests, streamlit | posterior estimation, storage, downloads, the demo app |
| Apache and BSD (combined notice) | torch | neural network backend for posterior estimation |
| Python Software Foundation | matplotlib | figures |
| CeCILL 2.1 | brian2 | spiking-network simulation of the twin |

Brian2's license is the most restrictive in the list; we use it unmodified, as a library.

**Models.** We use no pretrained or external models. Every statistical model in this report —
logistic regression, gradient boosting and the neural posterior estimator — is trained from scratch
on the data described here.

**Data.** Every dataset we used, with its license and status, is listed in Appendix C.

**AI tools.** We developed the project with Anthropic's Claude Code (§22), a commercial tool used
under Anthropic's terms of service. It is not part of the delivered pipeline, and nothing it wrote is
accepted without passing the same tests and citation checks as the rest of the code.
