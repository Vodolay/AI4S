"""Embodied fly, stage 1b: stage 1 with the male CNS weight rescaled to FlyWire's input.

``python -m neuromps.fly.embodied_stage1b`` runs the registered design
(`preregistration/embodied_stage1b_rescaled_weight.json`). The male CNS gives
each neuron about 1.7 times FlyWire's synaptic input (fly.excitability), so the
weight set on FlyWire is scaled down by that ratio. Feeding is checked first
(sugar must still drive MN9 and bitter gate it); then stage 1's leg test runs
unchanged on fresh seeds. Stage 1 stays failed as registered. Everything here
is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from neuromps.fly import design, motor
from neuromps.fly.embodied_stage1 import Config
from neuromps.fly.embodied_stage1 import main as run_leg
from neuromps.fly.study import memory_bounded_workers
from neuromps.paths import REPO_ROOT

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage1b_rescaled_weight.json"
FEEDING_SEEDS = range(720, 728)


def feeding_check(w_syn: float, spec: dict) -> dict:
    """B0: MN9 under sugar and under sugar with bitter, at the rescaled weight."""
    from neuromps.fly.replication import _context

    context = _context(design.registration())
    work = [({"w_syn_mv": w_syn}, p, s) for p in ("sugar", "sugar_bitter") for s in FEEDING_SEEDS]
    with ProcessPoolExecutor(
        memory_bounded_workers(per_worker_gb=motor.WORKERS_GB),
        initializer=motor._init,
        initargs=(context,),
    ) as pool:
        got = list(pool.map(motor._trial, work))
    sugar = float(
        np.mean([g["mn9"] for (_, p, _), g in zip(work, got, strict=True) if p == "sugar"])
    )
    bitter = float(
        np.mean([g["mn9"] for (_, p, _), g in zip(work, got, strict=True) if p == "sugar_bitter"])
    )
    ok = sugar >= 20 and bitter <= 0.5 * sugar
    return {
        "B0_feeding_still_works": {
            "test": spec["predictions"]["B0_feeding_still_works"]["test"],
            "value": {"mn9_sugar_hz": round(sugar, 3), "mn9_sugar_bitter_hz": round(bitter, 3)},
            "outcome": "pass" if ok else "fail",
        }
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: feeding check, then the leg test at the rescaled weight."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    w_syn = float(spec["design"]["w_syn_mv"])
    b0 = feeding_check(w_syn, spec)
    print("B0", b0["B0_feeding_still_works"]["outcome"], b0["B0_feeding_still_works"]["value"])
    config = Config(
        registration=REGISTRATION,
        seeds=tuple(range(710, 718)),
        w_syn_mv=w_syn,
        predictions=(
            "B1_antennal_drive_brings_the_claw_to_the_head",
            "B2_it_is_a_movement_toward_the_head",
            "B3_the_activation_scale_does_not_decide_it",
        ),
        interpretable=b0["B0_feeding_still_works"]["outcome"] == "pass",
        metrics_key="embodied_stage1b",
        stem="embodied_stage1b",
        extra=b0,
    )
    return run_leg(config=config)


if __name__ == "__main__":
    raise SystemExit(main())
