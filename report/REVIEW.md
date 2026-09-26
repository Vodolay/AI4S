> **Superseded — kept as a record, not as current criticism.**
>
> This is an adversarial review of the report as it stood on 2026-09-21, when it carried 115
> citations. The report now carries 224 and has been through two further review cycles. Every
> finding below was either fixed or answered in the commits that followed; the ones that moved
> numbers are recorded in `report/LAB_NOTEBOOK.md`. Read it for how the work was checked, not for
> what is wrong with the current report.

# Adversarial review — NeuroMPS technical report

> **Historical.** This review was written against an earlier version of the technical report,
> which has since been rewritten. `report/TECHNICAL_REPORT.md` now holds the replacement, so the
> filenames, byte counts and section numbers below describe a document that no longer exists in
> that form.

**Reviewer stance:** skeptical judge. I read `report/TECHNICAL_REPORT.md`, checked its rendering in
`report/TECHNICAL_REPORT.pdf`, and cross-checked every quantitative claim against
`outputs/metrics.json`, `outputs/tables/toxcast_mea_endpoints.csv`, `outputs/twin/*.json`,
`outputs/logs/`, and the figure files themselves. Context from `report/CONTEXT_FROM_CHAT.md` and
`report/LAB_NOTEBOOK.md`.

**Report version reviewed:** `TECHNICAL_REPORT.md`, 43,429 bytes, mtime 2026-09-21; PDF built
2026-09-21, title page commit `8ee0364 (working tree dirty)`.

**Verification method:** all 115 `<!-- metrics.json: ... -->` citations were resolved
programmatically against `metrics.json`. **All 115 resolve to a live key and all match the prose
value.** That is unusually good and I want to say so before the criticism starts: the citation
apparatus works. Every problem below is a problem of *which* number was chosen, what the surrounding
sentence claims about it, or what was left uncited — not of fabricated figures.

**Weighted score: 6.8 / 10.**

---

## 1. Scores by criterion

### Problem Importance and Impact — 7.0 / 10 (weight 30%)

Standardization and provenance genuinely are the bottleneck in organ-on-chip data, the entry picks a
domain where public data actually exists, and anchoring on real openly-licensed recordings rather
than synthetic data is a real differentiator against the competing entries described in
`CONTEXT_FROM_CHAT.md`. The license-discipline worked example (§4.1, Tukker 2020 excluded at a
stated scientific cost) is the kind of thing that distinguishes a serious data asset from a scraped
pile.

The ceiling is the asset itself: 897 spike-time records of 21,272
<!-- data_asset.counts.by_data_level.spike_times / .n_records -->, 173 human records
<!-- data_asset.counts.by_species.human -->, 73 modelling pairs, no working demo. A judge asking
"who could use this next week, and for what?" gets an honest answer of "nobody yet, but the schema
and the ETL are reusable", which is worth a solid score and not a top one.

### Technical Approach and Innovation — 7.5 / 10 (weight 30%)

The engineering discipline is the strongest part of the submission and is unusual at this scale:
build-time enforcement of conformance level 2, one shared feature extractor called by both the real
ETL and the twin, cluster bootstrap over culture groups rather than rows, a simulation cache keyed
on a hash of the source of every summary function, and a `report.verify` step that re-reads the
document against the metrics. The chance-corrected network-burst threshold (§3.4) is a real
methodological improvement over the obvious implementation, and the §7.2 discovery that coupling
asymmetry tracks relative firing rate rather than pathway direction is a genuine contribution that
invalidates a measure the project itself wanted to use.

Two deductions. The plan specified nested CV and the implementation is single-level grouped CV with
fixed hyperparameters — the report admits this in §8, which is honest but does not undo it. More
seriously, the rate-matching procedure in §7.6 fails the validity criterion that `metrics.json`
itself defines for it (finding **F1** below), and the report neither reports that nor adjusts the
claim it supports.

### Results and Validation — 5.5 / 10 (weight 20%)

This is the weakest criterion and the one that would cost the most with a careful judge. The
negative results are handled well — §7.5's retraction of a spurious dose trend, the unseeded-training
disclosure, the domain-adaptation failure — and that honesty is worth real credit.

But the report's **first** headline claim ("rate and burst-fraction measures are portable across
platforms") is contradicted by the only platform-contrast analysis in `outputs/`, which the report
never reports and whose figure it captions incorrectly (findings **F2**, **F3**). The §6.6
compartment result is the pooled test that `LAB_NOTEBOOK.md` explicitly abandoned as a pooling
artefact, presented without the sound per-device-type analysis that sits beside it in
`metrics.json` (**F5**). And the fourth headline claim rests on the arm that fails its own validity
check (**F1**). Three of four headline claims have a problem of this kind.

### Reproducibility — 7.0 / 10 (weight 10%)

Pinned environment with a stated reason for the numpy hold, a one-command entry point, logged
timings, a contract test that fails when a headline metric loses its interval, cache invalidation on
meaning-change, and a report-verification script. This is well above what most entries will show.

Deductions: `--quick` does **not** regenerate the §7.6 headline — in the logged quick run every
`cheapest_configuration` entry is `null` and the validity check returns AUC 0.125 (**F4**) — while
§9 advertises `--quick` as "every figure and metric". CI has never been observed to pass (disclosed,
§8). The PDF title page reads `8ee0364 (working tree dirty)`, which tells a judge the artifact was
built from uncommitted state.

### Presentation — 6.5 / 10 (weight 10%)

Structure is clear, the summary leads with negatives, the PDF compiles cleanly with a table of
contents, and the disclaimer is placed where it should be. Prose is well above competition average.

Against that: two figure titles are truncated mid-word in the PDF, one caption is factually wrong,
one caption claims error bars the figure does not draw, one caption contradicts the figure's own
group structure, Figure 12 is illegible at print size, and Task 3 is declared in §5.2 and then never
given a results subsection (**F2**, **F6**–**F9**).

---

## 2. Every quantitative claim, checked

### 2.1 Citation audit result

| Check | Result |
|---|---|
| `<!-- metrics.json: ... -->` citations in the report | 115 |
| Keys that resolve in `metrics.json` | 115 / 115 |
| Prose values matching the resolved key (to stated precision) | 115 / 115 |
| Citations anchored to the wrong key | 0 |
| Fabricated numbers | 0 |

I attempted to break this three ways — list-indexed keys (`conformal_sweep.0.*`,
`twin_hd13_perturbation.knobs.0.*`, `calibration.ece_ci95.1`), knob label-to-index ordering, and
CI-array element order — and found no error. Knob indices 0–3 do map to `gabaa_block`,
`gabaa_potentiate`, `ampa_block`, `sodium_block` as labelled, and the agree/scored counts (4/5, 2/4,
4/4, 3/3) match both `metrics.json` and the rendered heat-map.

### 2.2 Findings

---

**F1 — CRITICAL. The §7.6 headline comes from the arm that fails the project's own validity check,
and the check reported in its place is from the other arm.**

`metrics.json` defines the criterion explicitly:

> `twin_hd16_directionality.rate_only_is_the_validity_check`: *"in the rate-matched arm the rate_only
> AUC should fall towards chance; if it does not, the matching failed and that arm says nothing about
> direction"*

The rate-matched arm's `rate_only` AUC does **not** fall towards chance:

| quantity | value | source |
|---|---|---|
| `rate_matched.rate_only` AUC range | 0.8587 – 0.9859 | `summary_by_feature_set.rate_matched.rate_only.min/.max` |
| `rate_matched.rate_only.all_better_than_chance` | `true` | same |

By the stated criterion, **the matching failed and that arm says nothing about direction.** The
headline AUC 0.9784 in §1 bullet 4 and §7.6 is
`cheapest_configuration.rate_matched_residualized.auc` — from that arm.

The report does present a validity check, but it is a different quantity from a different arm:
0.3869–0.5969 from `summary_by_feature_set.**free_rate**.rate_only_rate_residualized`. Substituting
the free-rate arm's check for the rate-matched arm's is not equivalent, and the panel those numbers
come from is not among the four plotted in Figure 12.

Supporting evidence that the matching did not work: `rate_imbalance_by_arm` shows the rate-matched
arm did not remove the imbalance but **reversed its sign and left it comparable in size** —
free-rate {3.0: −0.2756, 10.0: −0.5180} against rate-matched {3.0: **+0.3116**, 10.0: **+0.2377**} —
and `matching_quality.rate_matched.worst_relative_rate_gap` = 0.26 is *worse* than
`free_rate`'s 0.2248. Appendix B already concedes "Two attempts to fix this by matching the drive
both failed, each trading one rate mismatch for another", but the headline still quotes the failed
arm.

The defensible number is the free-rate residualised configuration, where the quoted validity check
actually applies: AUC 0.9359, 95% CI [0.8930, 0.9739]
(`cheapest_configuration.free_rate_residualized`) — but at **24** electrodes per compartment, not 6,
and that cheapest-configuration entry is never cited in the report. The report has quoted the
higher AUC at the cheaper electrode count from the invalid arm.

---

**F2 — CRITICAL. Headline claim #1 is contradicted by the report's own platform-contrast metrics,
which are never reported.**

§1 bullet 1: *"Rate and burst-fraction measures are portable across platforms; synchrony and
network-burst measures are not."*

`mechanism_baselines.task3_platform_contrast` is the only platform-contrast analysis in `outputs/`.
It says the opposite about two of the named features:

| feature | Kapucu (gabazine) | ScreenNeuroPharm (bicuculline) | sign |
|---|---:|---:|---|
| `mfr_hz` | +0.348 | −0.684 | **disagrees** |
| `synchrony_index` | +1.743 | +0.148 | **agrees** |
| `network_burst_rate_per_min` | +1.536 | −0.920 | disagrees |
| `isi_cv` | +0.573 | −0.127 | disagrees |
| `active_electrodes` | 0.000 | 0.000 | (listed as disagreeing; a zero has no sign) |

`features_disagreeing_in_sign` = `[active_electrodes, isi_cv, mfr_hz, network_burst_rate_per_min]`;
`features_agreeing_in_sign` includes `synchrony_index`.

So on the single cross-platform test available, **`mfr_hz` — the report's flagship portable feature —
is the one that flips sign, and `synchrony_index` agrees.** The `mechanism_baselines.task3_*` key is
cited nowhere in the report.

Compounding this, the three sections offered as support ("Four independent analyses converge on
this (§6.3, §6.4, §7.3)") are a *species* transfer study, a *species* shift analysis, and a
*simulation* calibration. None of them is a platform comparison. The claim is about platforms and
the evidence cited is not. (The parenthetical also lists three sections for "four analyses".)

Two further cracks in the same thesis, both visible in Figure 9 and confirmed in
`twin_hd13_perturbation`: the `gabaa_block` knob **differs** from the real direction on `mfr_hz`,
and `gabaa_potentiate` **differs** on `burst_rate_per_min` and `pct_spikes_in_bursts` — i.e. the
twin's drug knobs fail precisely on the features the report calls reliable. §7.3's "Four analyses,
one conclusion" overstates a convergence that three separate outputs contradict.

---

**F3 — MAJOR. Figure 4's caption states the opposite of the data it plots.**

Report L256 / PDF Figure 4: *"Features whose sign disagrees across platforms are the synchrony and
network-burst group."*

In the plotted figure, the two `synchrony_index` bars are **both positive** (agree) and the two
`mfr_hz` bars have **opposite signs** (disagree). The caption is false against both `metrics.json`
and the rendered image. A judge who reads the figure will see the contradiction directly.

---

**F4 — MAJOR. `--quick` does not reproduce the §7.6 headline, contrary to §9.**

§9 presents `python run_all.py --quick` as "every figure and metric, reduced settings", and §9.2
claims report numbers are enforced by `neuromps.report.verify`.

In the logged quick run, `outputs/logs/hd16_run_all_quick.log` (lines 1004–1022):

```
"cheapest_configuration": {
  "rate_matched_residualized": null,
  "free_rate_residualized": null,
  "free_rate_all_features": null
},
"validity_check_rate_only_residualized": [
  {"arm": "free_rate",    "duration_s": 10.0, "auc": 0.125, "better_than_chance": false},
  {"arm": "rate_matched", "duration_s": 10.0, "auc": 0.125, "better_than_chance": false}
]
```

All three cheapest configurations are `null` under `--quick`, and `twin_hd16_directionality.quick`
is `false` in the shipped `metrics.json` — the stored value came from a full run. So the fourth
headline claim cannot be regenerated by the advertised one-command path, and if `--quick` overwrites
`metrics.json`, `report.verify` would fail against the report's own §7.6 citations. Either way the
reproducibility claim is stronger than what the command delivers.

Also: §9 states `--quick` completes in **463 s**; the log's own final line reads `TOTAL 465.99 s`.
The 463 s figure, the 1,956 s full-twin figure and the entire §9.1 stage breakdown (~50/~60/~70/~280 s)
carry no `metrics.json` citation, in a report whose opening paragraph states that every number does.

---

**F5 — MAJOR. §6.6 reports the analysis the lab notebook abandoned, and omits the sound one.**

§6.6 reports the pooled heterogeneous-vs-homogeneous test: medians 0.138 vs 0.094, Mann-Whitney
p = 0.042, rank-biserial 0.633, 6 against 10 devices — all correctly cited to
`compartment_asymmetry.pooled_heterogeneous_vs_homogeneous`.

`LAB_NOTEBOOK.md` (2026-09-20, `e6f8e60`) records that this exact test was *"computed and then
abandoned as the headline: it is an artefact of pooling, because HpHp overlaps CxHp. The
per-device-type analysis replaced it."*

The replacement analysis sits in the same file under `compartment_asymmetry.by_device_type` and is
both stronger and methodologically clean, including an internal control the pooled test cannot
offer:

| metric | CxCx | HpHp | CxHp | Kruskal p | Holm (control pair CxCx vs HpHp) |
|---|---:|---:|---:|---:|---:|
| median \|lag\| (ms) | 5.0 | 5.0 | 11.25 | 0.0082 | **0.4785** (correctly null) |
| median peak correlation | 0.962 | 0.908 | 0.822 | 0.0053 | 0.0159 |
| median \|asymmetry\| | 0.047 | 0.096 | 0.138 | 0.0336 | 0.1905 |

None of `by_device_type` is cited anywhere in the report. Note also that the per-type Holm result
for the statistic §6.6 actually reports — `median_abs_asymmetry.pairwise_holm.CxCx_vs_CxHp` = **0.0519** —
does not clear 0.05, while the pooled version the report chose gives p = 0.042. The report has
selected the abandoned analysis that crosses the significance line over the retained one that does
not.

---

**F6 — MODERATE. Figure 6's caption describes a different statistic and a different grouping from
the figure.**

The prose above it is about pooled asymmetry, 6 against 10 devices. The figure plots **lag** in
three groups of n=5, 5, 6, with the CxCx and HpHp groups plainly separate. The caption —
*"Inter-compartment coupling lag by device type, REAL data, one value per device. Heterogeneous
(cortical-hippocampal) devices sit apart from homogeneous ones, on six devices against ten"* — mixes
the two: it names the lag statistic, then imposes the pooled 6-vs-10 framing on a three-group plot.
A reader cannot reconcile caption and axis.

---

**F7 — MODERATE. Figure 5's caption claims intervals that are not drawn.**

*"Per-feature species shift, REAL data: rat against human baseline distributions, **with
cluster-bootstrap intervals**."* The figure is a plain grouped bar chart with no error bars. The
intervals exist in `metrics.json` (`species_shift.by_feature.*.difference_ci95`) but are not
plotted. Since several of those intervals cross zero, drawing them would materially change how the
figure reads — which is the point of F8.

---

**F8 — MODERATE. The species-shift claims are hedged for one feature and not for two others that
need the same hedge.**

§6.4 correctly flags that `synchrony_index` has interval [−5.7435, 0.1868] which "reaches zero, so
the ordering is better supported than the magnitude". Good. But of the three features named as "most
shifted", two others need the same treatment and do not get it:

| feature | difference | 95% CI | crosses zero? |
|---|---:|---|---|
| `synchrony_index` | −4.179 | [−5.7435, 0.1868] | yes — **hedged in report** |
| `isi_cv` | −1.233 | [−1.6738, 0.1500] | **yes — not hedged** |
| `network_burst_rate_per_min` | −4.907 | [−9.7879, −0.3389] | no |

Separately, `burst_rate_per_min` is presented as one of the three "most invariant" features while
carrying a difference of **+1.040** log2 units — a two-fold shift — with CI [−0.6659, 2.4688]. The
ranking is by standardised difference, which is defensible, but calling a 2× median shift
"invariant" without saying so invites a challenge.

---

**F9 — MODERATE. Task 3 is declared in Methods and never reported.**

§5.2 lists four tasks. §6 has subsections for Task 1 (§6.1), Task 2 (§6.2), transfer (§6.3), and
then goes to feature shift, ablations and compartments. **Task 3 — platform contrast never receives
a results subsection.** Its figure is dropped into the middle of §6.2, where the surrounding prose
is about external detection, with no numbers, no n, and no metrics citation. A judge following the
Methods-to-Results mapping will find a declared task with no result. This is also the mechanism by
which F2 stays invisible.

---

**F10 — MINOR. Claims stated more strongly than the interval or the figure supports.**

1. **§6.1**: *"Logistic regression beats chance with an interval that excludes it, **and beats
   gradient boosting**"* — then, in the same sentence, *"the gap between them is within noise."* The
   intervals overlap heavily: LR [0.4216, 0.6845] against GB [0.2762, 0.5178]. "Beats" and "within
   noise" cannot both stand; the second clause is the correct one.
2. **§6.1 calibration**: *"predicted probabilities should be read as ordering, not as calibrated
   risk."* The reliability diagram shows accuracy **falling** from 0.667 (0.7–0.8 bin) to 0.0
   (0.8–0.9, n=1) to 0.25 (0.9–1.0, n=4). Ordering also breaks at the top of the range; the
   monotonicity the sentence claims is not there.
3. **§7.6** *"rate features after residualising … **are never better than chance**"* rests on
   `all_better_than_chance: false`, which means *not all* cells beat chance, not *no* cell does.
   With a maximum of 0.5969 the stronger reading is not licensed by the key. No `any_better_than_chance`
   key exists.
4. **§1 bullet 2 / §7.2** *"swings the measured asymmetry across most of its range"* — the sweep
   moves 0.109 → 0.681 at fixed wiring
   (`twin_hd11_validation.directionality_sweep`), which is a large and real effect, but the
   measure's range is [−1, 1]; 0.11–0.68 is not "most of" it.

---

**F11 — MINOR. Numbers without a source, in a report that promises every number has one.**

The opening paragraph states: *"Every number below carries an HTML comment naming the
`outputs/metrics.json` key it came from."* These do not:

| Location | Uncited number | Status |
|---|---|---|
| §7.2 | asymmetry "from about 0.11 to 0.68" | derivable from `twin_hd11_validation.directionality_sweep`, no key cited — and it is headline claim #2 |
| §7.2 | "raising the forward ratio from 1 to 16 barely moves it" | derivable, uncited |
| §9, §9.1 | 463 s, 1,956 s, ~50/~60/~70/~280 s | no key; log says 465.99 s |
| §5.1 | "200 in `--quick`" | no key (the 1,000 full-run count is cited) |
| §6.3 | "37 training" | only `n_test` is cited; `n_train` exists |
| §7.5 | "every other parameter is below 0.01" | true (0.0092, 0.0004, 0.0011, 0.0030) but uncited aggregate |
| §7.5 | run-to-run p-values 0.46 / 0.0092 / 0.1956, contraction drift ~0.02 | no key; disclosed but unverifiable from `outputs/` |
| §7.5, App. B | prior-version ρ = −0.197, p = 0.036 | historical, no key |
| §8 | real recordings 480 s; chambers carry 48, 36 or 60 electrodes | no key |
| §8 | remote `master` "11 commits behind" | no key, not checkable from `outputs/` |
| §3.4 | "on the order of fifty spurious network bursts per minute" | notebook-sourced, no key |
| §8 | "33 numbered limitations" | **verified correct** by counting `LAB_NOTEBOOK.md` §Known limitations |

None of these is wrong where I could check it. The issue is that the report sets itself a stricter
standard than it meets, and a judge who spot-checks the §7.2 headline will find no citation behind
it.

---

**F12 — MINOR. `active_electrodes` is counted as a sign disagreement with a median delta of exactly
0.000 on both platforms** (`task3_platform_contrast.by_source.*.median_delta.active_electrodes`).
A zero has no sign. The "5 agree / 4 disagree" split is therefore at most three substantive
disagreements plus one degenerate case. This never reaches the report because of F9, but it will
surface the moment Task 3 is written up.

---

## 3. Efficacy / safety / clinical language

**Result: clean. No violation found.**

A full-text scan for efficacy, safety, toxicity, therapeutic, treatment, cure, patient, clinical,
diagnostic and indication language returned five hits, every one of which is a disclaimer or a
negation:

| Line | Text | Verdict |
|---|---|---|
| 6–7 | "Nothing in this report is an efficacy, safety, diagnostic or clinical claim" | disclaimer |
| 67 | "No clinical, diagnostic, efficacy or safety claim is made or implied. Compound labels are mechanism classes drawn from the source publications, not indications." | disclaimer, and the second sentence pre-empts the obvious misreading |
| 260 | "is **not** a screening tool" | negation |
| 588 | description of the test that scans for such language | meta |
| 650 | "extrapolating on summary dimensions" — matched on "extrapolat", not a clinical term | false positive |

Two things I checked specifically and found handled correctly: mechanism-class labels are
explicitly described as mechanism classes rather than indications (§2), and §7.4's drug-knob
agreement is explicitly limited to *direction*, never magnitude, with the cross-platform reference
named as a limitation in the same paragraph. The §9.2 claim that a test enforces this is consistent
with what I observed in the text, though I did not run the test suite.

The disclaimer placement is good: blockquote at the top, restated in scope, and on the PDF title
page footer.

---

## 4. Real-vs-simulated labelling

**Result: substantively correct throughout — every simulated figure is labelled SIMULATED in both
caption and embedded title. Two defects, one of them in the project's own stated rule.**

| # | Figure | Caption label | Embedded title | Verdict |
|---|---|---|---|---|
| 1 | `mechanism_confusion.png` | REAL data | "(real data, Kapucu 2022)" | OK |
| 2 | `task1_reliability_real.png` | REAL data | "on real data" | OK |
| 3 | `conformal_set_size.png` | REAL data | "(real data…)" | OK but **title truncated** (F13) |
| 4 | `platform_contrast_gabaa_antagonist.png` | REAL data | "(real data)" | label OK, **content wrong** (F3) |
| 5 | `species_shift.png` | REAL data | "(real data, Kapucu 2022)" | label OK, **claims intervals not drawn** (F7) |
| 6 | `compartment_lag_by_device_type.png` | REAL data | "(real data)" | label OK, **caption mismatches figure** (F6) |
| 7 | `hd11_directionality_simulated.png` | SIMULATED | SIMULATED | label OK, **title truncated + stale claim** (F13, F14) |
| 8 | `hd12_calibration_simulated.png` | "SIMULATED against REAL" | SIMULATED … vs real | OK — exemplary |
| 9 | `hd13_drug_knob_validation_simulated.png` | "SIMULATED drug knobs against REAL pharmacology" | same | OK — exemplary |
| 10 | `hd14_posterior_ranks_simulated.png` | SIMULATED | SIMULATED | OK |
| 11 | `hd15_real_dose_response.png` | "REAL bicuculline recordings, from a posterior trained on SIMULATIONS" | same | caption **exemplary**, **filename breaks the stated rule** (F15) |
| 12 | `hd16_directionality_simulated.png` | SIMULATED | SIMULATED | label OK, **drops the figure's own caveat** (F16) |

**F13 — Two embedded figure titles are truncated mid-word in the PDF.** Figure 3: *"…so the features
do not pin a mechanis"*. Figure 7: *"coupling asymmetry depends on drive balanc"*. Both are matplotlib
titles wider than the axes; both are visible in the compiled PDF and look like a build defect.

**F14 — Figure 7's title makes a claim the current data no longer supports.** The title reads *"same
wiring can read as either direction when the target is more or less driven"* — i.e. a sign flip. In
the current `directionality_sweep` every `asymmetry_mean` is non-negative (0.005 – 0.830); no
condition reads as the reversed direction. An earlier run did contain negative values (−0.166,
−0.067). The title is stale relative to the regenerated metrics. The §7.2 prose is correctly worded
and does **not** claim a sign flip, so the defect is confined to the image.

**F15 — `outputs/twin/hd15_real_dose_response.png` violates the naming convention Appendix A
states.** Appendix A: *"Figures are written to `outputs/figures/` (real data) and `outputs/twin/`
(simulated, **with `simulated` in the filename**, per the project's own rule that simulated output is
never presented as measurement)."* This file lives in `outputs/twin/` and has no `simulated` in its
name. It is genuinely a hybrid — real recordings, simulation-trained posterior — and the caption
handles that well, but the appendix asserts a rule the tree does not satisfy. Either the filename or
the appendix needs to change; as written, a judge checking the stated rule finds it broken on the
first file they test.

**F16 — Figure 12's caption drops the caveat the figure itself carries.** The image's own title
says *"bottom row: drive adjusted to match rates, **imperfectly** (see matching_quality)"*. The
report's caption says only *"bottom row matches rates"*. The figure is more honest than the report,
which is the wrong way round — and this is the same arm as F1.

**F17 — Figure 12 is illegible at print size.** Eight panels across a full-width landscape image
scaled to the text block: axis labels, panel titles and the four-line embedded title are all below
readable size in the PDF. The one figure carrying the most load in the twin story is the one a judge
cannot read.

---

## 5. The five highest-leverage edits, in priority order

### Edit 1 — Rebuild the §7.6 headline on the arm that passes its own validity check

*Fixes F1. Biggest single risk in the submission: a judge who opens `metrics.json` finds the
criterion, applies it, and concludes the flagship positive result is unsupported.*

Replace §1 bullet 4 with:

> 4. **Direction is recoverable from simulated features once rate is residualised out** — AUC 0.9359
>    <!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.auc -->,
>    95% CI [0.8930, 0.9739]
>    <!-- metrics.json: twin_hd16_directionality.cheapest_configuration.free_rate_residualized.auc_ci95 -->,
>    from a 10 s recording on 24 electrodes per compartment. The rate-*matched* arm reaches 0.9784
>    but fails its own validity check and is reported as inconclusive (§7.6), not as the result.

And add this paragraph to §7.6, immediately after the residualisation paragraph:

> **The rate-matched arm does not pass its own check.** `metrics.json` states the criterion:
> in the rate-matched arm, rate-only features should fall towards chance, and if they do not, the
> matching failed. They do not — rate-only AUC in that arm runs from 0.8587
> <!-- metrics.json: twin_hd16_directionality.summary_by_feature_set.rate_matched.rate_only.min -->
> to 0.9859
> <!-- metrics.json: twin_hd16_directionality.summary_by_feature_set.rate_matched.rate_only.max -->.
> The drive adjustment did not remove the rate imbalance, it reversed its sign: −0.2756 and −0.5180
> in the free-rate arm against +0.3116 and +0.2377 after matching
> <!-- metrics.json: twin_hd16_directionality.rate_imbalance_by_arm -->, with a worst relative gap of
> 0.26 against 0.2248 before
> <!-- metrics.json: twin_hd16_directionality.matching_quality -->. We therefore report the
> rate-matched arm as inconclusive and take the free-rate residualised configuration as the result.

*Cost: one paragraph and a number change. Benefit: converts the most attackable claim in the report
into a demonstration that the project's own guardrail caught a failure — which is exactly the
quality the rest of the report is trading on.*

---

### Edit 2 — Write the missing Task 3 subsection and correct the claim it contradicts

*Fixes F2, F3, F9, F12. Currently the report asserts a cross-platform claim while the cross-platform
evidence is unreported and points the other way.*

Add **§6.4 Task 3 — platform contrast** (renumbering what follows), move the platform-contrast figure
there from §6.2, and write:

> The same mechanism at ≥30 µM
> <!-- metrics.json: mechanism_baselines.task3_platform_contrast.min_dose_uM --> on two platforms —
> gabazine on 16-electrode Axion wells, n = 10
> <!-- metrics.json: mechanism_baselines.task3_platform_contrast.by_source.kapucu2022.n -->, against
> bicuculline on 60-electrode MCS arrays, n = 167
> <!-- metrics.json: mechanism_baselines.task3_platform_contrast.by_source.screenneuropharm_bic.n -->
> — agrees in sign on five features and disagrees on four
> <!-- metrics.json: mechanism_baselines.task3_platform_contrast.features_agreeing_in_sign -->
> <!-- metrics.json: mechanism_baselines.task3_platform_contrast.features_disagreeing_in_sign -->.
>
> **The disagreement does not fall where the species analysis predicts.** `mfr_hz` reverses (+0.348
> against −0.684) while `synchrony_index` agrees (+1.743 against +0.148). So the portability result
> of §6.5 is a *species* result, not a platform result: rate features survive a change of species and
> do not survive this change of platform. The contrast also confounds compound (gabazine against
> bicuculline), electrode count, recording duration and culture age, so it bounds portability rather
> than explaining it. `active_electrodes` has a median delta of exactly 0.000 on both platforms and
> is counted as a disagreement by a sign test that does not exclude zeros; the substantive count is
> three.

Then rewrite §1 bullet 1 to say what the evidence supports:

> 1. **Rate and burst-fraction measures survive a change of species; synchrony and network-burst
>    measures do not (§6.5, §6.6, §7.3). Across *platforms*, none of them is safe — mean firing rate
>    itself reverses sign (§6.4).** The portable-feature result is about species, and the platform
>    contrast is the sharper limitation.

Also fix the §7.3 sentence "Four analyses, one conclusion" — with Task 3 reported, the honest version
is "three analyses agree on species; the platform contrast disagrees, and that disagreement is the
finding". And correct Figure 4's caption to: *"Features whose sign disagrees across platforms are
mean firing rate, ISI CV and network burst rate; synchrony agrees."*

---

### Edit 3 — Replace §6.6 with the per-device-type analysis the notebook retained

*Fixes F5, F6. The report currently presents an analysis its own lab notebook rejected, and a judge
who reads both will treat that as cherry-picking.*

Replace the §6.6 body with:

> Across 16 two-chamber devices
> <!-- metrics.json: compartment_asymmetry.n_devices -->, the discriminator is coupling **lag**, not
> asymmetry. Median |lag| is 11.25 ms for cortical–hippocampal devices
> <!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.CxHp.median -->
> against 5.0 ms for both same-tissue types
> <!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.CxCx.median -->
> <!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.by_type.HpHp.median -->
> (Kruskal p = 0.0082
> <!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.kruskal_p -->; Holm
> 0.0413 and 0.0257 against the two homogeneous types
> <!-- metrics.json: compartment_asymmetry.by_device_type.median_abs_lag_ms.pairwise_holm -->).
> **The internal control behaves:** the two same-tissue types are indistinguishable from each other
> (Holm 0.4785), which a pooled heterogeneous-vs-homogeneous split cannot show. That pooled test is
> retained in `metrics.json` and is not reported here: it is an artefact of pooling, since HpHp
> overlaps CxHp. On |asymmetry| alone the per-type comparison does not clear correction (Holm 0.0519),
> so lag is the claim and asymmetry is not.
>
> With five to six devices per group this supports an ordering of device types, not an effect size,
> and §7.2 gives a specific reason to distrust the asymmetry measure itself.

Then fix Figure 6's caption to match: *"Inter-compartment coupling lag by device type, REAL data, one
value per device: five CxCx, five HpHp, six CxHp. The two same-tissue types are indistinguishable;
the heterogeneous type sits apart."*

*This edit makes the section stronger, not weaker — the retained analysis has a working negative
control, which is more persuasive than p = 0.042 on a pooled comparison.*

---

### Edit 4 — Make §9's reproducibility claim match what `--quick` actually produces

*Fixes F4 and the §9.1 citation gap. Reproducibility is 10% of the score and is currently the
easiest criterion to lose points on for a claim that is checkable in one command.*

In §9, replace the bullet *"`--quick` completes in 463 s on CPU against a 600 s budget"* and the
surrounding claim with:

> - `--quick` completes in **466 s** on CPU against a 600 s budget (`outputs/logs/hd16_run_all_quick.log`,
>   final line); the full twin stage alone takes 1,956 s (`outputs/logs/hd16_twin_full.log`).
> - **`--quick` reproduces every figure and every metric except the HD16 directionality sweep.** At
>   reduced settings that sweep has too few simulations per cell to fit a classifier, so
>   `cheapest_configuration` is written as `null` and the validity check is not meaningful; the §7.6
>   numbers require `python run_all.py --stage twin` in full. This is a deliberate budget choice, and
>   `metrics.json` records which mode produced the stored values
>   (`twin_hd16_directionality.quick`).

Add the stage table's timings as metrics keys if they are not already emitted, so §9.1 stops being
the one uncited table in the document. And rebuild the PDF from a clean tree so the title page does
not read `working tree dirty` — that line is the first thing a judge sees and it silently contradicts
the whole reproducibility section.

---

### Edit 5 — Fix the four figure defects and soften four over-claims

*Fixes F3 (caption), F7, F8, F10, F13, F14, F16, F17. Small individually; together they are most of
the Presentation score and several of the credibility nicks in Results.*

**Figures — regenerate:**

1. Wrap or shorten the titles of `conformal_set_size.png` and `hd11_directionality_simulated.png` so
   they stop truncating mid-word in the PDF.
2. Change `hd11_directionality_simulated.png`'s title from *"same wiring can read as either
   direction…"* to *"drive moves the measure further than wiring does"* — the sign flip is no longer
   in the data.
3. Draw the cluster-bootstrap intervals on `species_shift.png` (they are already in
   `species_shift.by_feature.*.difference_ci95`), or delete the phrase from the caption.
4. Split `hd16_directionality_simulated.png` into two stacked figures, or drop the two panels not
   discussed, so it is legible at text-block width. Add the `rate_only_rate_residualized` panel —
   it is the panel the validity-check numbers come from and it is currently not shown.
5. Either rename `hd15_real_dose_response.png` to carry `simulated`, or amend Appendix A to state the
   rule as it actually is: *"figures in `outputs/twin/` carry `simulated` in the filename unless they
   plot real recordings under a simulation-trained model, in which case the caption states both."*

**Prose — four sentences:**

6. §6.1: delete *"and beats gradient boosting"*. Keep *"with 73 pairs the simpler model is the better
   one, and the gap between them is within noise."*
7. §6.1 calibration: after *"read as ordering, not as calibrated risk"*, add *"— and not even as
   ordering above 0.8 confidence, where accuracy falls to 0.25 on four predictions."*
8. §6.5/§6.6 species shift: extend the existing hedge to `isi_cv`, whose interval [−1.6738, 0.1500]
   also reaches zero, and note that `burst_rate_per_min` is "invariant" by standardised difference
   while carrying a 2× median shift.
9. §7.6: change *"are never better than chance"* to *"do not clear chance in every cell (maximum
   0.5969)"*, which is what `all_better_than_chance: false` licenses.

---

## 6. What I would tell the authors in one paragraph

The apparatus is better than the argument. 115 of 115 citations resolve and match, the negative
results are real and well-told, and the provenance and cache-invalidation machinery is stronger than
most submissions will show. But three of the four headline claims have a sourcing problem that a
determined judge will find in `metrics.json` itself: the directionality headline comes from the arm
that fails the validity criterion the file defines; the portability headline is contradicted by the
one platform analysis in the outputs, which is never reported; and the compartment section reports
the test the lab notebook explicitly abandoned over the one it retained. None of these requires new
computation — the better numbers are already in `outputs/`. Edits 1–3 are re-selections, not
re-analyses, and each one makes the report more impressive rather than less, because in every case
the honest version demonstrates a guardrail working. Do those three, then Edit 4 and the figure
pass, and this moves from roughly 6.8 to low-8s.
