"""Embodied fly, stage 2i: stage 2h's front-leg network with Pugliese et al.'s size rule.

``python -m neuromps.fly.embodied_stage2i`` runs the registered design
(`preregistration/embodied_stage2i_volume_size_rule.json`): everything as stage 2h (the
network, the fine input search, stage 2g's measures and scoring), with neuron size taken
from the male CNS neuron volumes, median-normalized over the network. Fresh draws.
Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from neuromps.fly import design, scoring
from neuromps.fly import embodied_stage2g as s2g
from neuromps.fly import embodied_stage2h as s2h
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage2i_volume_size_rule.json"
SEEDS = list(range(1380, 1412))
RULE = "volume"


def _init() -> None:
    s2g._init(with_volumes=True)


def run_job(drive: str) -> dict:
    """One drive under the volume rule: stage 2h's search and stage 2g's measures."""
    return s2g.run_job((RULE, drive), seeds=SEEDS, search=s2h.bisect_inputs)


def score(table: pd.DataFrame, spec: dict) -> dict:
    """V0-V6: stage 2g's P0-P6 computed by its code, with this registration's names."""
    got = s2g.score(table, json.loads(s2g.REGISTRATION.read_text()))
    return {
        v: got[p] | {"test": spec["predictions"][v]["test"]}
        for p, v in zip(got, spec["predictions"], strict=True)
    }


def size_summary() -> dict:
    """How the volume sizes relate to the stage-2e stand-in, and where the drive and circuit sit."""
    sizes = s2g._STATE["sizes"]
    types = s2g._STATE["types"]
    rho = stats.spearmanr(sizes[RULE], sizes["synapse_count"]).statistic
    out = {
        "network_neurons": int(types.size),
        "volumes_missing": int(s2g._STATE["volumes_missing"]),
        "spearman_volume_vs_synapse_count": round(float(rho), 4),
    }
    for name, idx in (
        ("DNg100", s2g._STATE["drive"]["forward_walking"]),
        ("circuit", s2g._STATE["cpg"]),
    ):
        out[name] = {
            rule: round(float(np.median(sizes[rule][idx])), 4) for rule in (RULE, "synapse_count")
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-2i design."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    drives = list(s2g.DRIVES)
    with ProcessPoolExecutor(len(drives), initializer=_init) as pool:
        got = list(pool.map(run_job, drives))
    _init()
    table = pd.DataFrame([row for g in got for row in g["rows"]])
    examples = {RULE: got[drives.index("forward_walking")].get("example")}
    result = {
        "label": "SIMULATED left front-leg network of the male CNS, volume size rule",
        "runs_used": {d: g["runs"] for d, g in zip(drives, got, strict=True)},
        "sizes": size_summary(),
        "predictions": score(table, spec),
        "reported": s2h.reported(table),
        "circuit_check": s2h._check_rule(RULE, table, seeds=SEEDS),
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage2i_SIMULATED.csv", index=False)
    (out / "embodied_stage2i_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage2i", scoring.clean(result))
    s2g.plot(
        examples, table, figures_dir() / "embodied_stage2i_front_leg_SIMULATED.png", rules=(RULE,)
    )
    print("runs", result["runs_used"])
    print("sizes", json.dumps(result["sizes"]))
    for name, entry in result["predictions"].items():
        print(f"   {name:48s} {entry['outcome']:>15}  {entry['value']}")
    print("reported", json.dumps(scoring.clean(result["reported"])))
    print("circuit check", json.dumps(scoring.clean(result["circuit_check"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
