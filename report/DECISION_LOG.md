# Decision log: the embodied-fly path and its branch points

Every crossroads on the way from the chip-to-animal study to the embodied fly, with the options
that were open, the one taken, why, and where to branch from to explore another. Results are not
restated here: each stage's numbers live in `report/ROADMAP_EMBODIED_FLY.md` §7, cited to
`outputs/metrics.json`, and each stage's registration in `preregistration/`. Commits are on
master unless marked otherwise. Decisions taken by the user are marked **(user)**; the rest were
taken by the agent and are open to revision.

## How to explore another branch

Check out the commit named in the decision, create a branch, register the alternative in
`preregistration/` before running it (fresh seeds if the original result has been seen), and run
it with its own command or `python -m neuromps.fly.pipeline --only <step>`.

## Decisions

### D1. Which fly connectome release the model runs on
- **Options:** FlyWire 630 (the published model's release), FlyWire 783 (the current one).
- **Taken:** 783 for the studies, with a registered 630-vs-783 replication of the published
  sugar result. **Why:** the current release, with the published release kept as a check.
- **Revisit at:** `fly_chip_to_animal` registration; `neuromps.fly.connectome.load_630`.

### D2. Which body simulator for the embodied fly
- **Options:** FlyGym / NeuroMechFly v2 (needs Python 3.12), the FlyMimic musculoskeletal leg
  loaded in plain MuJoCo, flybody.
- **Taken:** FlyMimic in plain MuJoCo (pinned `mujoco` as the optional `embodied` extra); flybody
  later for the proboscis. **Why:** muscle-driven, runs in the project's Python 3.11, no new
  framework. **Revisit:** a FlyGym environment would give six muscle-driven legs only if its
  musculoskeletal model covers them.

### D3. After stage 1 failed: what to change first
- **Options:** rescale the male-CNS synaptic weight further; drive the male nerve cord from
  descending neurons with the brain removed; switch connectome.
- **Taken:** the nerve cord from descending neurons (stage 1c), then the FlyWire brain joined to
  it through type-matched descending neurons (stages 1e, 1f). **Why:** the male brain avalanches
  under the published parameters (`fly_spread`); each connectome is used where its model is
  validated.

### D4. Stage 1e's precondition failed on an absolute rate line
- **Taken:** stage 1f, a relative precondition and depth of reach, on fresh seeds; 1e stays
  uninterpretable. **Why:** the absolute line had no basis.

### D5. Stage 2a failed: how to look for a walking rhythm **(user: sweep)**
- **Options offered:** fix adaptation parameters from literature; sweep them and report the whole
  sweep. **Taken (user):** the sweep (stage 2c), with a registered selection rule and confirmation.
- **Revisit:** published adaptation values, if any appear for fly leg or premotor neurons.

### D6. The glutamate sign rule **(user: go ahead)**
- **Taken by the agent:** two bounding rules (glutamate excitatory onto motor neurons only; and
  everywhere), each with the adaptation grid (stage 2d). **Unexplored:** under the corrected
  rhythm measure the selection rule would pick (glutamate everywhere, 16 mV, 100 ms); it was not
  confirmed on fresh seeds.

### D7. A different neuron model **(user: look at how others make flies walk)**
- **Found:** the walking flies online use trained controllers (Eon Systems, flybody); only
  Pugliese et al. (2025) get a rhythm from nerve-cord wiring, with a firing-rate model.
- **Taken:** reimplement their model from the paper and configuration (their code is unlicensed).
- **Size stand-in (stage 2e):** total synapse count, because the male CNS has no volumes. This
  starved DNg100; stage 2f drove every command neuron to the same rate; stage 2g used their
  front-leg network and recruitment rule; stage 2h a geometric input search.
- **Unexplored:** neuron volumes (a future male-CNS release, or MANC volumes through the
  male-CNS-to-MANC matching); their own network files under a licence.

### D8. A flaw in the registered rhythm measure
- **Taken:** a registered audit with a peak-requiring measure (prominence 0.1, calibrated on
  synthetic signals only), re-scoring stages 1d-2e on their original seeds; registered outcomes
  kept as recorded, the documents carry both. Stage 1d's D2 withdrawn.
- **Revisit:** prominence 0.05 or 0.2 give the same calibration results; a different peak
  criterion could be registered.

### D9. Where to close the loop first (stage 3)
- **Options:** the front leg (muscle-driven, thinly traced proprioceptors), a middle or hind leg
  (well traced, but only position-servoed in flybody), the proboscis.
- **Taken:** the front leg through the claw neurons; the unknown flexion/extension tuning
  bracketed by two assignments and a sham; then a perturbation test (stage 3b).
- **Unexplored:** a hind-leg loop through flybody's servos and its 22 claw neurons; hook and club
  neurons; campaniform sensilla.

### D10. Drugs in the loop (stage 4)
- **Taken:** each knob acts on every network of a trial; movement disruption uses
  |log2((x + 0.01 mm)/(x_vehicle + 0.01 mm))| with the parent's 0.5 line, 0.05 mm as sensitivity.
- **Disclosed issue:** at full GABA-A block the reach depth is negative, so that condition's
  index averages the two proboscis terms. **Revisit:** a registered rule for negative depths.

### D11. Checking against real flies (stage 5)
- **Taken:** a search protocol registered before any search; every candidate classified.
- **Unexplored:** neuron-specific simulations of the identified-neuron studies (DSOG1, 13A/13B)
  instead of class-wide knobs; a satiety state for the feeding model.

### D12. Release (stage 6)
- **Taken:** a pipeline that reruns every later fly study as its own process, wired into
  `run_all.py` as a full-run-only `embodied` stage. **Why:** one command reproduces every
  number; separate processes keep the connectomes from accumulating in memory.

### D13. Working overnight without the user **(user: continue, commit, push, update docs)**
- **Taken:** "push" read as merging finished work into master through a pull request, because
  GitHub keeps master only; every crossroads tonight is added here.

### D14. The precision a chip needs, at the movement level (overnight)
- **Options:** leave stage 4a's timing result as it stands; compute, as the parent study did at
  the neural level, the chip's change at the dose where movement first fails.
- **Taken:** the latter, as an exploratory analysis (both inputs had been seen), added to the
  pipeline as `stage4a_precision`. **Why:** the paper's assay-design message rests on the
  precision a chip needs at behavioural onset; this extends it to movement at no simulation cost.

### D15. Confirming the cell the corrected measure selects (overnight)
- **Options:** leave stage 2d's alternative selection unconfirmed (as the audit registered);
  confirm it on fresh seeds.
- **Taken:** confirm it (registration `embodied_stage2d_confirm_corrected.json`, commit 8495b57),
  with the corrected measure throughout. **Why:** it is the one loose end the audit named, and it
  is cheap. **Not taken tonight:** a hind-leg closed loop through flybody's servos (D9), a
  volume-based size rule (D7), neuron-specific drug simulations (D11), each larger.

### D16. Bringing the embodied fly into the technical report and a paper draft (overnight)
- **Taken:** Act VIII of `report/TECHNICAL_REPORT.md` (§27-§31, every number cited, the
  uncited-number budget unchanged) and `report/PAPER_DRAFT.md` (abstract to methods, every number
  cited). **Why:** the user asked for a paper and for the report to keep being retouched; the
  draft is a starting point for the team, not a submission.

### D17. Testing the model against identified-neuron experiments (overnight)
- **Options:** a hind-leg closed loop (D9), a volume size rule (D7), simulating the identified
  neurons that stage 5's published studies manipulated (D11).
- **Taken:** the identified neurons (stage 5b, registration commit c20fa08): the left front leg's
  GABAergic 13A and 13B hemilineage neurons, silenced (outputs removed) or activated (150 Hz), in
  the grooming relay of stage 1f. **Why:** it turns stage 5's indirect comparison into a direct
  one with data already in hand, and it is affordable overnight. **Not possible:** Pool et al.'s
  DSOG1 neurons are not annotated in FlyWire. **Detail:** 73 of the 76 annotated 13B neurons of
  that neuromere are in the nerve-cord network used.
- **Revisit:** subsets of 13A/13B (the study manipulated sub-populations), pulsed activation
  (the study's 70 ms pulses induced grooming-like rhythms), the femur-tibia angle as a readout
  ("legs locked in flexion").

### D18. Reading the 13A/13B manipulations at the joint (overnight)
- **Options:** pulsed 13A activation (the study's grooming-inducing protocol; a leaky
  integrate-and-fire network has no rebound, so it would very likely fail by construction); a
  hind-leg closed loop (larger); the joint readout the study actually used.
- **Taken:** the joint readout (stage 5c, registration commit 04a2387, fresh seeds): femur-tibia
  angle under the same arms as stage 5b. **Why:** stage 5b's mismatch could be a readout mismatch
  (reach versus joint posture); this tests that directly and cheaply.
- **Process note:** a test edit for this step did not apply and a piped test run hid it; fixed in
  commit fd51761. Test runs before commits are no longer piped.

### D19. Closing the loop on a well-traced leg (overnight)
- **Options:** stop stage 3 at the front-leg limit; close the loop on the right hind leg (25 claw
  neurons) through flybody's servoed tibia; build a muscle-driven hind leg (no such model at
  hand).
- **Taken:** the servoed hind leg (stage 3c, registration commit 22b082e): the walking command
  holds the right hind tibia extended (its extensor pool is tonically active, stage 2a), a 20 ms
  torque pushes it, and the return is timed under the resistance and assistance assignments and a
  sham, as in stage 3b. **Why:** it tests whether the front leg's failure was the thin sensor
  tracing, with the one body that has the leg. **Cost:** feedback changes a commanded angle, not
  muscle force; a servo's own stiffness sets much of the return.
- **Revisit:** a muscle-driven hind leg (if FlyMimic or another model adds one); the walking
  command's other legs.

### D20. Testing stage 3c's exploratory difference on its own (overnight)
- **Options:** report stage 3c's resistance-against-sham difference as exploratory and stop; test
  it alone on fresh seeds.
- **Taken:** test it (stage 3d, registration commit 8ea194a): resistance against sham, paired by
  seed, 32 fresh seeds, a one-sided sign test and a 5 ms margin. **Why:** the difference was the
  one sign that the push-following part of the feedback acts on a well-traced leg, and it was
  seen in a noisy, unstable regime, so only a registered test on new seeds can say whether it is
  real.
- **Outcome:** 0 of 2 (stage 3 stays a documented negative). **Revisit:** a muscle-driven hind
  leg.

### D21. After stage 3: replicate the headline before anything new (overnight)
- **Options:** new science that needs a model not at hand (a muscle-driven hind leg, neuron
  volumes for the size rule); 13A/13B subsets or pulsed activation (a leaky integrate-and-fire
  network has no rebound); paper and slide writing only; a registered replication of stage 4a,
  the paper's headline, which ran on one set of 8 seeds.
- **Taken:** the replication (stage 4b, registration commit b23f6a3): stage 4a's movements on 16
  fresh seeds, against the chip detections of the parent design's own fresh-seed replication,
  with each movement left out in turn. **Why:** it is the result a reviewer will press first, it
  is cheap (about 15 minutes), and the leave-one-out check names the assumption most likely to
  make it wrong (that one movement carries the disruption index).
- **Outcome:** 3 of 4 (registration commit b23f6a3). The chip is late in every family on the
  fresh seeds; without the bitter-gated proboscis GABA-A potentiation ties the chip.
- **Revisit:** a different impairment line for movement (the 0.05 mm sensitivity is reported); a
  finer dose grid for GABA-A potentiation and excitability block, where the margin is one step.

### D22. Resolving stage 4b's one-step margins (overnight)
- **Options:** leave the two thin margins reported; a finer dose grid inside them, as the
  motor-level tie-break did; a different movement impairment line (a new rule chosen after the
  results, so not taken).
- **Taken:** the finer grid (stage 4c, registration commit f387845): GABA-A potentiation at
  gaba_scale 1.6-1.9 and excitability block at threshold_scale 1.24-1.36 and 1.44-1.56, with
  stage 4b's movement seeds and both of the parent's chip seed sets, the coarse levels rerun as a
  reproduction check. **Why:** stage 4b showed that in two families the headline rests on a single
  step of the dose grid, and a finer grid is the direct way to ask what happens inside it.
- **Outcome:** 3 of 4 (registration commit f387845). Excitability block resolves chip later
  against both chip runs; GABA-A potentiation stays chip later with the full index, and tied
  without the bitter-gated proboscis.
- **Revisit:** a fourth movement (for example a walking readout, once one exists) would tell
  whether GABA-A potentiation's lateness is specific to feeding.

### D23. Stopping new science for the night (overnight)
- **Options:** further stages (each remaining branch needs a model not at hand or is expected
  to fail by construction); documents for the team.
- **Taken:** a slide outline for the deck (`report/SLIDES_OUTLINE.md`, every number cited), the
  README's document table brought up to date, and a morning summary in `PROGRESS.md`. **Why:**
  the headline has now been tested as far as the current bodies allow (stages 4a-4c), and the
  next scientific choices (D7 neuron volumes, a muscle-driven hind leg, a walking readout for
  drugs) are the team's to make.

### D24. The volume size rule **(user)**
- **Options (D7's unexplored branch):** neuron volumes from a future release, or MANC volumes
  through the male-CNS-to-MANC matching; found instead: the male CNS v1.0 neuPrint neuron table
  (same bucket, same CC-BY release) has a voxel count per body.
- **Taken:** fetch that table whole (4.65 GB, source `malecns_sizes`), checked against the
  bucket's CRC32C, because the bucket publishes no MD5 for it **(user: full file, not a
  column-only range read)**; rerun stage 2h with the published rule (stage 2i, registration
  commit d03ee27) on fresh draws. A numpy CRC32C was written rather than adding a dependency.
- **Outcome:** 6 of 7 (registration commit d03ee27): the rhythm needs the circuit and
  promotor and remotor keep a reliable, non-antiphase offset; flexor-extensor alternation fails.
- **Revisit:** normalizing by the median over the whole nerve cord rather than over the
  network (stated choice: over the network, as for the synapse-count stand-in).

### D25. Checking that stage 2i comes from the neurons' own volumes
- **Options:** report stage 2i as it stands; name the assumption most likely to make it wrong
  and test it (the project rule for results that look good).
- **Taken:** stage 2j (registration commit 1d1d644), fresh draws: the volume rule replicated,
  volumes shuffled across neurons (a fresh permutation per draw), and normalization by the
  nerve cord's median instead of the network's. **Why:** if shuffled volumes do as well, the
  result is about the spread of sizes, not the identified neurons; the normalization was a
  stated choice the published description does not fix.
- **Outcome:** 2 of 4 (registration commit 1d1d644). Shuffled volumes give no rhythm; the
  cord-median normalization changes nothing; the rhythm and the circuit's necessity replicate,
  but the promotor-remotor offset does not reach the registered reliability, so it is not claimed.
- **Revisit:** a registered test of the offset alone with more draws, if it matters for the paper.

### D26. Walking as a drug readout **(user)**
- **Options (from D22's revisit):** a fourth movement for the drug design; the only walking
  readout at hand is stage 2i's fictive front-leg rhythm (rate model, volume sizes, no body).
- **Taken:** stage 4d (registration commit 73ffde6), fresh draws: the drug knobs applied to the
  walking network (weights scaled by presynaptic class, as in the spiking model; the excitability
  knob scales the activation threshold), each draw's DNg100 input found under vehicle and held
  fixed, onsets on stage 4c's combined dose series. **Why:** it asks whether GABA-A
  potentiation's lateness is specific to feeding and whether the headline holds for a rhythm.
- **Stated choice:** the threshold knob maps to the rate model's theta; fixed inputs, so a drug
  changes the response, not the stimulus.
- **Outcome:** 2 of 4 (registration commit 73ffde6). GABA-A and GluCl block break walking first;
  acetylcholine block is chip-first; GABA-A potentiation and the threshold knob leave walking
  intact; with walking as a fourth movement the chip is still late in every family.
- **Revisit:** a readout that respects the bimodal draws (the fraction of rhythmic draws)
  registered as its own test; a threshold mapping that also shifts the drive.

### D27. Retesting the acetylcholine exception with a readout that respects the draws **(user)**
- **Options:** keep stage 4d's median readout; a fraction-of-rhythmic-draws readout; a
  per-draw paired analysis.
- **Taken:** the fraction readout on fresh draws (stage 4e, registration commit 180476f). **Why:**
  stage 4d's draws are bimodal, the median sits between the modes, and the exception entered
  the abstract; the fraction is the plainest readout of a bimodal outcome and uses the rhythm
  line registered since stage 2g.
- **Outcome:** 2 of 4 (registration commit 180476f). GABA-A and GluCl block break walking first;
  the acetylcholine exception does not replicate: mild block raises the share of rhythmic draws,
  and the two-sided index counts that as a change before the chip. The exception is withdrawn
  from the abstract, discussion, README and slides.
- **Revisit:** a one-sided walking index (only loss of rhythm counts) would be a new rule chosen
  after seeing this result; register it only with a reason that does not depend on the outcome.

### D28. The paper and new research move to a private repository **(user)**
- **Options:** keep everything public; make the repository private (would break the
  competition's links and the hosted demo, and erase stars and watchers); rewrite public history
  (destructive, and GitHub keeps every pull request's commits regardless); keep new work private
  from here on.
- **Taken:** new work private from here on. The public repository `Vodolay/AI4S` keeps the
  hackathon deliverables and everything already published (up to pull request 26). A private
  repository `Vodolay/AI4S-paper` mirrors public master and adds a `paper-draft` branch holding
  the paper draft, its outline, the slide outline (D23) and any new research toward the paper;
  those three documents were removed from public master. **Why:** the user wants the preprint
  and the research leading to it kept private until submission; what was already public stays
  so, and its history timestamps the work.
- **Revisit:** when the paper is submitted, `paper-draft` merges into public master.

### D29. Recovering the ToxCast components that have no level-0 rows
(Numbered D55 in the project's private research log, where it was first taken.)
- **Found:** the invitrodb v4.3 dump's level-0 table lacks 8 of the 17 acute MEA components,
  among them firing rate, spike number and burst number; level 3 still has their rows.
- **Options:** use only the 9 components with level 0 (no firing rate); ask the EPA for the
  missing rows; recover the wells from the dump's id allocation.
- **Taken:** recovery (`neuromps.etl.toxcast_recover`). Level-0 ids come in contiguous runs of
  equal length per component and batch, so a component's run start is predicted from the
  reference component's; rval is recovered from level 3 exactly (rval = bval + resp x
  (pval - bval) / 100). Wells are assigned by position in the reference run; for all 7 missing
  electrophysiology components concentrations agree on every row and each date's stored baseline
  is reproduced from the assigned wells on every date. Four present components have their runs in
  another order (both checks fail on them), so present components keep their own level-0 wells.
- **Assumption recorded:** positions within a date are not checked by the baseline, so the
  physical-plate assignment of a recovered well rests on the run order alone.
- **Revisit at:** `data/interim/toxcast_mea/recovery_checks.json`.

### D30. Testing the twin's dose response against real concentration-response wells
- **Found:** §33 showed the twin's knobs get the direction of the EPA acute screen's three
  positive controls right, but each is run at one saturating concentration, so §34 recorded the
  open question: nothing there tests the twin's dose response. The screen also runs about 500
  compounds over seven concentrations each, with a viability readout on the same plates.
- **Options:** leave the dose response untested (the deck's closing slide already lists it as a
  next step); match knob levels to concentrations by potency (needs an IC50 per compound and per
  platform, which §14/HD8 showed does not transfer); match by rank and score only what a rank
  match can carry.
- **Taken:** the rank match, registered in `preregistration/twin_toxcast_dose_response.json`
  before any treated-well value was computed, with the compound list and one class label per
  compound fixed in the registration from published primary-target annotation. Scores are a
  Spearman rank correlation of the twin's matched curve against the real one per compound and
  readout, which readout moves first, and two nulls the twin must beat: the compound's own real
  firing-rate curve (the §33 follow-firing-rate rule, extended to curves) and the twin's curve
  under a knob the compound is not labelled with. **Why:** the knob scales and the concentration
  axis share no unit, and every registered score is invariant to a monotone map between them, so
  a rank match supplies exactly what the scores can use.
- **Registered before looking:** firing rate counts as already seen (a private analysis looked at
  it) and is excluded from the shape tests; bins are retained only up to the first that the
  screen's own viability readout flags, so that a collapse at the top concentration cannot be
  read as pharmacology; a compound needs four retained bins to be scored.
- **Assumption recorded:** the rank match assumes the screen's concentration range covers a
  comparable span of the mechanism to the knob grid. Checked by rescoring with the grid's two
  strongest levels dropped (`sensitivity_truncated_grid`).
- **Revisit at:** `outputs/metrics.json` key `twin_toxcast_dose_response`, report §35.

### D31. Recalibrating the twin's inverse model: more simulations, then an ensemble
- **Found:** the inverse model (one masked autoregressive flow, 2,700 training simulations,
  `twin_hd14_inference`) fails the rank-uniformity check for `gabaa_scale` and
  `inter_in_degree`, so the slide withholds both.
- **Options:** keep withholding them; correct the check for five parameters (Bonferroni would
  pass both, which lowers the bar rather than fixes the model); retrain the same flow on more
  simulations; average several flows; change the flow family.
- **Taken:** a registered test of the two remedies that do not touch the bar: about 15,000 fresh
  simulations (5x; truncated at 05:00 EDT on 2026-10-28 by a prefix rule fixed in advance) for
  the same flow, and an equal-weight ensemble of five flows on the same pool, both judged on
  1,000 fresh held-out simulations by the original per-parameter test at 0.05, plus coverage,
  contraction against the old model on the same simulations, and a check near the real
  recordings. The flow family is not changed, so budget and averaging are the only differences.
  **Why:** a slide that withholds two of five parameters should say whether that is a limit of
  the budget, of the estimator, or of what the summaries carry.
- **Registered:** `preregistration/twin_inference_recalibration.json` (seven predictions and the
  assumption check), committed before any new simulation.
- **Outcome:** 2 of 7 (registration commit 0bc29aa, amended before any simulation in 176b36e).
  51 of 60 training chunks finished before the cut. On fresh simulations the round-1 model fails
  all five parameters, not two; the ensemble keeps its sharpness and passes four of five on the
  300-run test, but fails `inter_in_degree`, stays over-confident by coverage for four
  parameters, and takes about 2 s per posterior. Budget alone fixes neither failing parameter.
  Under the registered rule the slide quotes `threshold_scale` only. Report §16, change-log
  entry 45.
- **Revisit at:** `neuromps.twin.recalibration`; `twin_hd14b_recalibration` in `outputs/metrics.json`.
  Next branches, each to be registered with fresh seeds: a different flow family (neural spline
  flow) on the same pool, which would separate estimator from summaries; or summaries that carry
  connectivity more directly (the failure that neither remedy moves).

### D32. How noisy an untreated human iPSC chip is, against a rat chip from the same lab **(user)**
- **Found:** the practical-value result (§32) gives the wells an assay needs for rat cortical
  chips only. The natural human analogue, untreated wells recorded before and after a treatment
  step, exists in Kapucu 2022, but only 4 human and 6 rat such wells pass QC.
- **Options:** drop the human question; search for another licensed human iPSC set with
  re-recorded untreated wells; measure the noise inside one untreated recording, by comparing
  its two halves, on the lab's matched 48-well baselines (human DIV29, rat DIV22).
- **Taken:** the split-half comparison, registered in `preregistration/hpsc_split_half.json`
  before any half-recording feature was computed: three predictions (human firing rate noisier;
  noisier across features; replicates on the lab's 12-well developmental plates), and the wells
  per arm a 20% or 50% fall would need, reported as a lower bound. **Why:** it is the only
  comparison with enough wells in both species, and the same lab, protocol and extractor make
  the species the main difference.
- **Assumption recorded:** a species difference could be a firing-rate difference, because fewer
  spikes give noisier rates; checked on wells whose firing rates overlap between species.
- **Limit stated in advance:** split-half noise leaves out handling, compound addition and
  day-to-day variation, so the wells it implies are a lower bound, and are not compared
  numerically with the ToxCast vehicle-well numbers.
- **Revisit at:** `outputs/metrics.json` key `hpsc_split_half`.
