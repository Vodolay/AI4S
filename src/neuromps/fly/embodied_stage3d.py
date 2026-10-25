"""Embodied fly, stage 3d: the hind-leg contingency test, paired by seed.

``python -m neuromps.fly.embodied_stage3d`` runs the registered design
(`preregistration/embodied_stage3d_hind_leg_contingency.json`): stage 3c's closed loop on
32 fresh seeds; for each seed, the recovery time with resistance feedback against the
sham (the same seed's unperturbed feedback). Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from neuromps.fly import design, scoring
from neuromps.fly import embodied_stage3c as s3c
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage3d_hind_leg_contingency.json"
SEEDS = list(range(1320, 1352))
NEEDED, MARGIN_MS = 22, 5.0


def score(table: pd.DataFrame, spec: dict) -> dict:
    """Q1 (paired sign count) and Q2 (median paired difference)."""
    rec = table.pivot(index="seed", columns="arm", values="recovery_ms")
    diff = rec["sham"] - rec["resistance"]
    favour = int((diff > 0).sum())
    median = float(diff.median())
    found = {
        "Q1_feedback_that_follows_the_push_speeds_the_return": (
            {"seeds_favouring_resistance": favour, "seeds": int(len(diff))},
            favour >= NEEDED,
        ),
        "Q2_by_a_meaningful_margin": (round(median, 2), median >= MARGIN_MS),
    }
    return {
        n: {"test": spec["predictions"][n]["test"], "value": v, "outcome": "pass" if ok else "fail"}
        for n, (v, ok) in found.items()
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-3d design."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    arms = ("open", "resistance", "resistance_unperturbed")
    with ProcessPoolExecutor(
        memory_bounded_workers(per_worker_gb=8.0), initializer=s3c._init
    ) as pool:
        trials = list(pool.map(s3c._trial, [(a, s, None) for a in arms for s in SEEDS]))
        tonic = {t["seed"]: t["rates"] for t in trials if t["arm"] == "resistance_unperturbed"}
        trials += list(pool.map(s3c._trial, [("sham", s, tonic[s]) for s in SEEDS]))
    table = pd.DataFrame(
        [{k: v for k, v in t.items() if k not in ("rates", "tibia")} for t in trials]
    )
    reported = {
        a: {c: round(float(g[c].median()), 4) for c in ("pre_rad", "excursion_rad", "recovery_ms")}
        for a, g in table.groupby("arm")
    }
    pairs = table[table["arm"].isin(["resistance", "sham"])]
    result = {
        "label": "SIMULATED hind-leg contingency test (paired by seed)",
        "predictions": score(pairs, spec),
        "reported": reported,
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage3d_SIMULATED.csv", index=False)
    (out / "embodied_stage3d_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage3d", scoring.clean(result))
    for name, entry in result["predictions"].items():
        print(f"{name:52s} {entry['outcome']:>5}  {entry['value']}")
    print("reported", json.dumps(reported))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
