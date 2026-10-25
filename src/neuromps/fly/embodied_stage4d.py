"""Embodied fly, stage 4d: the walking rhythm as a drug readout.

``python -m neuromps.fly.embodied_stage4d`` runs the registered design
(`preregistration/embodied_stage4d_walking_readout.json`): the drug knobs applied to stage
2i's walking network (rate model, volume sizes), each draw's DNg100 input found under
vehicle and held fixed; walking onsets, and onsets of the four-movement index, against the
chip replication on stage 4c's combined dose series. Everything here is **simulated**.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import design, rate_model, scoring
from neuromps.fly import embodied_stage2g as s2g
from neuromps.fly import embodied_stage2h as s2h
from neuromps.fly import embodied_stage4a as s4a
from neuromps.fly import embodied_stage4c as s4c
from neuromps.fly.model import KNOB_OF_CLASS
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage4d_walking_readout.json"
SEEDS = list(range(1460, 1492))
RULE = "volume"
FAMILIES = ("gaba_block", "gaba_potentiate", "glucl_block", "ach_block", "excitability_block")
PREDICTIONS = (
    "W0_the_walking_rhythm_is_there",
    "W1_the_chip_is_late_for_walking",
    "W2_gaba_potentiation_lateness_is_not_specific_to_feeding",
    "W3_with_walking_the_chip_is_late_in_every_family",
)


def conditions() -> pd.DataFrame:
    """The parent's 22 conditions and stage 4c's fine levels, with their knobs."""
    coarse = s4a.conditions()
    fine = s4c.levels(json.loads(s4c.REGISTRATION.read_text()))
    fine = fine[~fine["condition"].isin(coarse["condition"])]
    return pd.concat([coarse, fine], ignore_index=True)


def knob_parts(knob: dict) -> tuple[dict[str, float], float]:
    """A condition's knob as class scales for the weights and a threshold factor."""
    classes = {cls: float(knob[k]) for cls, k in KNOB_OF_CLASS.items() if k in knob}
    return classes, float(knob.get("threshold_scale", 1.0))


def _init() -> None:
    s2g._init(with_volumes=True)


def _condition(job) -> dict:
    """One condition at the vehicle inputs: rhythm score, period and recruitment per draw."""
    name, knob, inputs = job
    classes, threshold = knob_parts(knob)
    w = rate_model.weights(s2g._STATE["con"], class_scale=classes)
    p = rate_model.draw(s2g._STATE["sizes"][RULE], SEEDS)
    p = dataclasses.replace(p, theta=p.theta * threshold)
    idx = s2g._STATE["drive"]["forward_walking"]
    run = rate_model.simulate(
        w,
        p,
        idx,
        np.broadcast_to(inputs, (len(idx), len(SEEDS))),
        s2g._STATE["mn"],
        duration_s=s2g.DURATION_S,
        dt_s=s2g.DT_S,
    )
    measures = [s2g.draw_measures(run, k) for k in range(len(SEEDS))]
    return {
        "condition": name,
        "score": [m["score"] for m in measures],
        "period_ms": [m["period_ms"] for m in measures],
        "recruited": s2g._recruited(run, "forward_walking").tolist(),
    }


def walking_disruption(walk: pd.Series) -> pd.Series:
    """|log2((x + 0.01) / (x_vehicle + 0.01))| of the per-condition median rhythm score."""
    return np.log2((walk + s4a.EPS_MM) / (walk["vehicle"] + s4a.EPS_MM)).abs()


def score(
    conds: pd.DataFrame,
    walk: pd.Series,
    feasible: int,
    chip: pd.DataFrame,
    moves: pd.DataFrame,
    spec: dict,
) -> tuple[dict, dict]:
    """W0-W3 and the onsets they rest on."""
    table = conds.assign(
        walking_disruption=walking_disruption(walk).loc[conds["condition"]].to_numpy()
    )
    walking = s4a.onset(table, "walking_disruption", s4a.IMPAIR)
    four = moves.assign(walking=walk.loc[moves.index])
    four_index = s4a.disruption(four, s4a.EPS_MM).loc[conds["condition"]].to_numpy()
    four_onset = s4a.onset(conds.assign(md=four_index), "md", s4a.IMPAIR)
    chips = s4a.onset(chip, "abs_delta_mfr", s4a.CHIP)
    w1 = {f: s4c.resolution(d, chips.get(f)) for f, d in walking.items() if d is not None}
    w3 = {f: s4c.resolution(four_onset.get(f), chips.get(f)) for f in FAMILIES}
    w0 = feasible >= 16 and walk["vehicle"] >= 0.3
    found = {
        PREDICTIONS[0]: (
            {"feasible": feasible, "median_score": round(float(walk["vehicle"]), 4)},
            w0,
        ),
        PREDICTIONS[1]: (w1, bool(w1) and all(r == "chip later" for r in w1.values())),
        PREDICTIONS[2]: (
            {
                "walking_onset": walking.get("gaba_potentiate"),
                "chip_onset": chips.get("gaba_potentiate"),
            },
            s4c.resolution(walking.get("gaba_potentiate"), chips.get("gaba_potentiate"))
            == "chip later",
        ),
        PREDICTIONS[3]: (w3, all(r == "chip later" for r in w3.values())),
    }
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        unscorable = (name != PREDICTIONS[0] and not w0) or (name == PREDICTIONS[1] and not w1)
        if unscorable:
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    onsets = {"walking": walking, "four_movement": four_onset, "chip_replication": chips}
    disruption = dict(
        zip(conds["condition"], np.round(table["walking_disruption"], 4), strict=True)
    )
    return out, {"onsets": onsets, "walking_disruption": disruption}


def _chip_series() -> pd.DataFrame:
    coarse = pd.read_csv(outputs_dir() / "fly" / "confirm_conditions_SIMULATED.csv")
    levels = json.loads((outputs_dir() / "fly" / "embodied_stage4c_SIMULATED.json").read_text())
    fine = conditions().set_index("condition")
    rows = [
        {
            "condition": c,
            "family": fine.loc[c, "family"],
            "dose": fine.loc[c, "dose"],
            "abs_delta_mfr": v["chip_replication"],
        }
        for c, v in levels["levels"].items()
        if "chip_replication" in v and c not in set(coarse["condition"])
    ]
    keep = ["condition", "family", "dose", "abs_delta_mfr"]
    return pd.concat([coarse[keep], pd.DataFrame(rows)], ignore_index=True)


def _movement_medians() -> pd.DataFrame:
    fly = outputs_dir() / "fly"
    coarse = s4a.movement_medians(pd.read_csv(fly / "embodied_stage4b_SIMULATED.csv"))
    fine = s4a.movement_medians(pd.read_csv(fly / "embodied_stage4c_SIMULATED.csv"))
    return s4c.combine(coarse, fine)


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-4d design."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    conds = conditions()
    _init()
    _, inputs, ok, runs, _ = s2h.bisect_inputs(RULE, "forward_walking", SEEDS)
    log.info("vehicle inputs: %d of %d feasible after %d runs", int(ok.sum()), len(SEEDS), runs)
    jobs = [(c, k, inputs) for c, k in zip(conds["condition"], conds["knob"], strict=True)]
    with ProcessPoolExecutor(16, initializer=_init) as pool:
        got = list(pool.map(_condition, jobs))
    rows = [
        {
            "condition": g["condition"],
            "seed": s,
            "feasible": bool(ok[k]),
            "input": float(inputs[k]),
            "score": g["score"][k],
            "period_ms": g["period_ms"][k],
            "recruited": g["recruited"][k],
        }
        for g in got
        for k, s in enumerate(SEEDS)
    ]
    trials = pd.DataFrame(rows)
    use = trials[trials["feasible"]]
    walk = use.groupby("condition")["score"].median()
    period = use.groupby("condition")["period_ms"].median()
    predictions, details = score(
        conds, walk, int(ok.sum()), _chip_series(), _movement_medians(), spec
    )
    result = {
        "label": "SIMULATED walking rhythm (rate model, volume sizes) under drug knobs",
        "predictions": predictions,
        **details,
        "medians": {
            c: {
                "score": round(float(walk[c]), 4),
                "period_ms": None if pd.isna(period[c]) else float(period[c]),
            }
            for c in conds["condition"]
        },
        "vehicle_inputs": {
            "feasible": int(ok.sum()),
            "runs": int(runs),
            "median": float(np.median(inputs[ok])),
        },
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    trials.to_csv(out / "embodied_stage4d_SIMULATED.csv", index=False)
    (out / "embodied_stage4d_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage4d", scoring.clean(result))
    for name, entry in predictions.items():
        print(f"{name:58s} {entry['outcome']:>15}  {entry['value']}")
    print("onsets", json.dumps(scoring.clean(details["onsets"])))
    print("medians", json.dumps(scoring.clean(result["medians"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
