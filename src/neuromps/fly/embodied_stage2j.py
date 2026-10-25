"""Embodied fly, stage 2j: is stage 2i the neurons' own volumes?

``python -m neuromps.fly.embodied_stage2j`` runs the registered design
(`preregistration/embodied_stage2j_volume_specificity.json`): stage 2i on fresh draws under
three size arms (the volume rule; the same sizes shuffled across neurons, a fresh permutation
per draw; volume over the nerve cord's median), scored by stage 2g's code.
Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import design, scoring
from neuromps.fly import embodied_stage2g as s2g
from neuromps.fly import embodied_stage2h as s2h
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage2j_volume_specificity.json"
SEEDS = list(range(1420, 1452))
ARMS = ("volume", "volume_shuffled", "volume_cord")


def shuffled(sizes: np.ndarray, seeds: list[int]) -> np.ndarray:
    """(neurons x draws): the same size values permuted across neurons, per draw's seed."""
    return np.stack([np.random.default_rng(s).permutation(sizes) for s in seeds], axis=1)


def _init() -> None:
    s2g._init(with_volumes=True)
    sizes = s2g._STATE["sizes"]
    sizes["volume_shuffled"] = shuffled(sizes["volume"], SEEDS)


def run_job(job) -> dict:
    """One (arm, drive): stage 2h's search and stage 2g's measures on this stage's draws."""
    return s2g.run_job(job, seeds=SEEDS, search=s2h.bisect_inputs)


def _outcome(scores: dict, prefix: str) -> str:
    return next(v["outcome"] for k, v in scores.items() if k.startswith(prefix))


def predictions(arms: dict[str, dict], spec: dict) -> dict:
    """J0-J3 from each arm's P0-P6 outcomes."""
    walking = ("P1", "P2", "P5")
    found = {
        "J0_the_volume_result_replicates": (
            "volume",
            all(_outcome(arms["volume"], p) == "pass" for p in walking),
        ),
        "J1_circuit_necessity_needs_the_neurons_own_volumes": (
            "volume_shuffled",
            _outcome(arms["volume_shuffled"], "P5") == "fail",
        ),
        "J2_the_offset_needs_the_neurons_own_volumes": (
            "volume_shuffled",
            _outcome(arms["volume_shuffled"], "P2") == "fail",
        ),
        "J3_the_normalization_does_not_matter": (
            "volume_cord",
            all(_outcome(arms["volume_cord"], p) == "pass" for p in walking),
        ),
    }
    out = {}
    for name, (arm, ok) in found.items():
        outcome = "pass" if ok else "fail"
        if arm != "volume" and _outcome(arms[arm], "P0") != "pass":
            outcome = "uninterpretable"
        value = {p: _outcome(arms[arm], p) for p in ("P0", *walking)}
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-2j design."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    parent = json.loads(s2g.REGISTRATION.read_text())
    jobs = [(arm, drive) for arm in ARMS for drive in s2g.DRIVES]
    with ProcessPoolExecutor(len(jobs), initializer=_init) as pool:
        got = list(pool.map(run_job, jobs))
    table = pd.DataFrame([row for g in got for row in g["rows"]])
    arms = {arm: s2g.score(table[table["size_rule"] == arm], parent) for arm in ARMS}
    result = {
        "label": "SIMULATED left front-leg network of the male CNS: volume, shuffled, cord median",
        "predictions": predictions(arms, spec),
        "arms": {
            arm: {
                "scores": {k: {"value": v["value"], "outcome": v["outcome"]} for k, v in s.items()},
                "reported": s2h.reported(table[table["size_rule"] == arm]),
            }
            for arm, s in arms.items()
        },
        "runs_used": {f"{a}/{d}": g["runs"] for (a, d), g in zip(jobs, got, strict=True)},
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage2j_SIMULATED.csv", index=False)
    (out / "embodied_stage2j_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage2j", scoring.clean(result))
    for name, entry in result["predictions"].items():
        print(f"{name:52s} {entry['outcome']:>15}  {entry['value']}")
    for arm, s in arms.items():
        print(arm, {k[:2]: (v["outcome"], v["value"]) for k, v in s.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
