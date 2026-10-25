"""Embodied fly, stage 5b: the identified GABAergic 13A/13B leg neurons, manipulated.

``python -m neuromps.fly.embodied_stage5b`` runs the registered design
(`preregistration/embodied_stage5b_identified_neurons.json`): stage 1f's relay (FlyWire
brain under antennal drive, descending spikes replayed into the male nerve cord, the
FlyMimic leg) with the left front leg's 13A or 13B hemilineage neurons silenced (their
outputs removed) or activated (150 Hz drive), against the published direction for
grooming. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from neuromps.fly import (
    body,
    design,
    embodied_stage1,
    embodied_stage1c,
    kinematics,
    model,
    relay,
    scoring,
)
from neuromps.fly import embodied_stage5 as s5
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage5b_identified_neurons.json"
SEEDS = list(range(1210, 1218))
ARMS = ("intact", "silence_13A", "silence_13B", "activate_13A", "activate_13B")
NEAR_HEAD_MM = body.REST_DISTANCE_MM - 0.2
R_FULL = 20.0
_STATE: dict = {}


def population(table: pd.DataFrame, lineage: str) -> np.ndarray:
    """Model indices of one hemilineage's neurons in the left front-leg neuromere."""
    hit = (
        (table["trumanHl"].fillna("").astype(str) == lineage)
        & (table["somaNeuromere"].fillna("").astype(str) == "T1")
        & (table["somaSide"].fillna("").astype(str) == "L")
    )
    return np.flatnonzero(hit.to_numpy())


def silenced(con, neurons: np.ndarray):
    """The network with every outgoing connection of `neurons` set to zero synapses."""
    return replace(con, synapses=np.where(np.isin(con.pre, neurons), 0, con.synapses))


def _cord_init(pairs: dict[int, int]) -> None:
    relay._cord_init(pairs)
    con = embodied_stage1c._STATE["con"]
    table = embodied_stage1c._annotations(con.root_ids)
    pops = {k: population(table, k) for k in ("13A", "13B")}
    _STATE.update(
        pops=pops,
        nets={"intact": con, **{f"silence_{k}": silenced(con, v) for k, v in pops.items()}},
    )


def _cord(job) -> dict:
    """One nerve-cord trial of one arm, driven by one brain trial's descending spikes."""
    brain, arm = job
    state = embodied_stage1c._STATE
    con = _STATE["nets"].get(arm, _STATE["nets"]["intact"])
    excite = _STATE["pops"][arm.split("_")[1]] if arm.startswith("activate") else []
    replay = {
        t: brain["dn_spikes"][s]
        for t, s in relay._STATE["pairs"].items()
        if s in brain["dn_spikes"]
    }
    params = model.FlyParams(duration_s=embodied_stage1.DURATION_S, seed=brain["seed"])
    spikes = model.simulate(con, params, excite=excite, replay=replay)
    counts = np.bincount(spikes.neurons, minlength=con.n_neurons) / spikes.duration_s
    neurons = sorted({n for v in state["assignment"].neurons_by_muscle.values() for n in v})
    return {
        "drive": arm,
        "seed": brain["seed"],
        "spikes": {n: spikes.times[spikes.neurons == n] for n in neurons},
        "assignment": state["assignment"],
        "front_hz": float(counts[state["pools"]["front"]].mean()),
    }


def _leg(cord: dict) -> dict:
    moved = embodied_stage1._move((cord, R_FULL))
    return {
        "arm": cord["drive"],
        "seed": cord["seed"],
        "depth_mm": body.REST_DISTANCE_MM - moved["min_distance_mm"],
        "time_near_head": kinematics.time_near_head(moved["distance_mm"], NEAR_HEAD_MM),
        "front_hz": cord["front_hz"],
    }


def direction(x: float, base: float) -> str:
    """Stage 5's rule on the medians."""
    if abs(np.log2((x + s5.EPS_MM) / (base + s5.EPS_MM))) >= s5.LINE:
        return "increase" if x > base else "decrease"
    return "none"


def score(table: pd.DataFrame, spec: dict) -> dict:
    """I0-I4; I1-I4 uninterpretable if the intact reach is missing."""
    med = table.groupby("arm")["depth_mm"].median()
    i0 = med["intact"] >= 0.3
    found = {"I0_the_reach_is_there": (round(float(med["intact"]), 4), i0)}
    names = dict(zip(ARMS[1:], list(spec["predictions"])[1:], strict=True))
    for arm, name in names.items():
        d = direction(float(med[arm]), float(med["intact"]))
        found[name] = ({"depth_mm": round(float(med[arm]), 4), "direction": d}, d == "decrease")
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        if name != "I0_the_reach_is_there" and not i0:
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-5b design."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    fw, male = relay._tables()
    pairs = relay.match(fw, male)
    with ProcessPoolExecutor(memory_bounded_workers(), initializer=relay._brain_init) as pool:
        brains = list(pool.map(relay._brain, [("jon_ce", s) for s in SEEDS]))
    with ProcessPoolExecutor(
        memory_bounded_workers(per_worker_gb=8.0), initializer=_cord_init, initargs=(pairs,)
    ) as pool:
        cords = list(pool.map(_cord, [(b, a) for a in ARMS for b in brains]))
    with ProcessPoolExecutor(12) as pool:
        table = pd.DataFrame(list(pool.map(_leg, cords)))
    _cord_init(pairs)
    reported = {
        a: {c: round(float(g[c].median()), 4) for c in ("depth_mm", "time_near_head", "front_hz")}
        for a, g in table.groupby("arm")
    }
    result = {
        "label": "SIMULATED grooming reach with the 13A/13B leg neurons manipulated",
        "predictions": score(table, spec),
        "reported": reported,
        "population_sizes": {k: int(len(v)) for k, v in _STATE["pops"].items()},
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage5b_SIMULATED.csv", index=False)
    (out / "embodied_stage5b_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage5b", scoring.clean(result))
    for name, entry in result["predictions"].items():
        print(f"{name:40s} {entry['outcome']:>15}  {entry['value']}")
    print("reported", json.dumps(reported))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
