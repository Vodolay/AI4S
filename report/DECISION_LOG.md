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
- **Taken:** the replication (stage 4b, registration commit below): stage 4a's movements on 16
  fresh seeds, against the chip detections of the parent design's own fresh-seed replication,
  with each movement left out in turn. **Why:** it is the result a reviewer will press first, it
  is cheap (about 15 minutes), and the leave-one-out check names the assumption most likely to
  make it wrong (that one movement carries the disruption index).
- **Revisit:** a different impairment line for movement (the 0.05 mm sensitivity is reported).
