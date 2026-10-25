"""Embodied fly, stage 5c: the 13A/13B leg-neuron manipulations read at the femur-tibia joint.

``python -m neuromps.fly.embodied_stage5c`` runs the registered design
(`preregistration/embodied_stage5c_identified_neurons_joints.json`): stage 5b's relay and
arms on fresh seeds, read as the median femur-tibia angle (larger is more flexed) rather
than the reach, against the published postures. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import body, design, embodied_stage1, relay, scoring
from neuromps.fly import embodied_stage5b as s5b
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage5c_identified_neurons_joints.json"
SEEDS = list(range(1220, 1228))
TIBIA = "joint_LFTibia_pitch"
MARGIN_RAD = 0.1


def _leg(cord: dict) -> dict:
    """The leg under one cord trial: median angle of every left front-leg joint."""
    names = body.muscle_names()
    moved = body.run(embodied_stage1.activations(cord, s5b.R_FULL, names), names)
    medians = np.median(moved.joint_angles, axis=0)
    return {
        "arm": cord["drive"],
        "seed": cord["seed"],
        **{n: float(v) for n, v in zip(moved.joint_names, medians, strict=True)},
    }


def score(table: pd.DataFrame, spec: dict) -> dict:
    """J1-J4: flexion under silencing, extension under activation, by MARGIN_RAD."""
    med = table.groupby("arm")[TIBIA].median()
    base = float(med["intact"])
    expect = {"silence_13A": 1, "silence_13B": 1, "activate_13A": -1, "activate_13B": -1}
    out = {}
    for (arm, sign), name in zip(expect.items(), spec["predictions"], strict=True):
        shift = float(med[arm]) - base
        ok = shift >= MARGIN_RAD if sign > 0 else shift <= -MARGIN_RAD
        out[name] = {
            "test": spec["predictions"][name]["test"],
            "value": {
                "angle_rad": round(float(med[arm]), 4),
                "intact_rad": round(base, 4),
                "shift_rad": round(shift, 4),
            },
            "outcome": "pass" if ok else "fail",
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-5c design."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    fw, male = relay._tables()
    pairs = relay.match(fw, male)
    with ProcessPoolExecutor(memory_bounded_workers(), initializer=relay._brain_init) as pool:
        brains = list(pool.map(relay._brain, [("jon_ce", s) for s in SEEDS]))
    with ProcessPoolExecutor(
        memory_bounded_workers(per_worker_gb=8.0), initializer=s5b._cord_init, initargs=(pairs,)
    ) as pool:
        cords = list(pool.map(s5b._cord, [(b, a) for a in s5b.ARMS for b in brains]))
    with ProcessPoolExecutor(12) as pool:
        table = pd.DataFrame(list(pool.map(_leg, cords)))
    joints = [c for c in table if c.startswith("joint_")]
    result = {
        "label": "SIMULATED 13A/13B manipulations read at the front-leg joints",
        "predictions": score(table, spec),
        "joint_medians_rad": {
            a: {j: round(float(g[j].median()), 4) for j in joints} for a, g in table.groupby("arm")
        },
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage5c_SIMULATED.csv", index=False)
    (out / "embodied_stage5c_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage5c", scoring.clean(result))
    for name, entry in result["predictions"].items():
        print(f"{name:40s} {entry['outcome']:>5}  {entry['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
