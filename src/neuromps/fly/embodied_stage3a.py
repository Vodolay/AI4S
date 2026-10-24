"""Embodied fly, stage 3a: the grooming leg with tibia-position feedback through claw neurons.

``python -m neuromps.fly.embodied_stage3a`` runs the registered design
(`preregistration/embodied_stage3a_claw_feedback.json`): the stage-1c male nerve cord
and the FlyMimic left front leg run together (`fly.closed_loop`); the left front leg's
femoral chordotonal claw neurons fire with the femur-tibia angle under a resistance
assignment, the opposite assignment, or a non-contingent sham; open loop for
comparison. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import body, closed_loop, design, embodied_stage1c, kinematics, model, scoring
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage3a_claw_feedback.json"
SEEDS = list(range(1020, 1028))
DRIVES = ("grooming", "none")
CLAW = {"SNpp51": "extension", "SNpp50": "flexion"}  # the resistance assignment
OPPOSITE = {"extension": "flexion", "flexion": "extension"}
NEAR_HEAD_MM = body.REST_DISTANCE_MM - 0.2
DURATION_S = 1.0
_STATE: dict = {}


def _init() -> None:
    embodied_stage1c._init()
    con = embodied_stage1c._STATE["con"]
    table = embodied_stage1c._annotations(con.root_ids)
    front = (table["entryNerve"].astype(str) == "ProLN") & (table["rootSide"].astype(str) == "L")
    sensor = front & (table["superclass"] == "vnc_sensory") & table["type"].isin(list(CLAW))
    idx = np.flatnonzero(sensor.to_numpy())
    types = table["type"].astype(str).to_numpy()
    leg = (table["superclass"] == "vnc_motor") & (table["subclass"] == "fl")
    leg = leg & (table["somaSide"] == "L")
    _STATE.update(
        sensors=idx,
        resistance=[CLAW[types[i]] for i in idx],
        leg_mn=np.flatnonzero(leg.to_numpy()),
        flexor=np.flatnonzero((leg & (table["type"] == "Ti flexor MN")).to_numpy()),
        extensor=np.flatnonzero((leg & (table["type"] == "Ti extensor MN")).to_numpy()),
    )


def tunings(arm: str) -> list[str] | None:
    """Per claw neuron, its tuning in this arm (None: no position tuning)."""
    if arm == "resistance":
        return list(_STATE["resistance"])
    if arm == "assistance":
        return [OPPOSITE[t] for t in _STATE["resistance"]]
    return None


def _trial(job) -> dict:
    """One closed-loop (or open-loop) trial: readouts and the claw rate time course."""
    arm, drive, seed, sham = job
    con = embodied_stage1c._STATE["con"]
    params = model.FlyParams(duration_s=DURATION_S, stim_rate_hz=150.0, seed=seed)
    loop = closed_loop.Loop(
        neurons_by_muscle=embodied_stage1c._STATE["assignment"].neurons_by_muscle,
        sensors=_STATE["sensors"],
        tunings=None if arm in ("open", "sham") else tunings(arm),
        sham=sham,
    )
    spikes = model.simulate(
        con,
        params,
        excite=embodied_stage1c._STATE["drive"][drive],
        extend=closed_loop.extender(loop, params),
    )
    rec = loop.record
    distance, tibia = np.asarray(rec["distance"]), np.asarray(rec["tibia"])
    rates = np.asarray(rec["sensor_rates"])

    def pool_hz(idx):
        return float(np.isin(spikes.neurons, idx).sum() / max(len(idx), 1) / DURATION_S)

    return {
        "arm": arm,
        "drive": drive,
        "seed": seed,
        "median_tibia_rad": float(np.median(tibia)),
        "time_near_head": kinematics.time_near_head(distance, NEAR_HEAD_MM),
        "min_distance_mm": float(distance.min()),
        "leg_motor_spikes": int(np.isin(spikes.neurons, _STATE["leg_mn"]).sum()),
        "tibia_flexor_hz": pool_hz(_STATE["flexor"]),
        "tibia_extensor_hz": pool_hz(_STATE["extensor"]),
        "claw_mean_hz": float(rates.mean()) if rates.size else 0.0,
        "rates": rates,
        "tibia": tibia,
    }


def _median(table: pd.DataFrame, arm: str, drive: str, column: str) -> float:
    rows = table[(table["arm"] == arm) & (table["drive"] == drive)]
    return float(rows[column].median())


def score(table: pd.DataFrame, spec: dict) -> dict:
    """L0-L4 with values and outcomes; L2 uninterpretable when L1 fails."""
    quiet = table[
        (table["drive"] == "none") & table["arm"].isin(["open", "resistance", "assistance"])
    ]
    silent = int(quiet["leg_motor_spikes"].sum())
    tib = {
        a: round(_median(table, a, "grooming", "median_tibia_rad"), 4)
        for a in ("open", "resistance", "assistance", "sham")
    }
    shift = {a: round(tib[a] - tib["open"], 4) for a in ("resistance", "assistance", "sham")}
    l1 = shift["resistance"] >= 0.1
    near = _median(table, "resistance", "grooming", "time_near_head")
    near_none = _median(table, "resistance", "none", "time_near_head")
    found = {
        "L0_silent_without_drive": (silent, silent == 0),
        "L1_feedback_resists_the_extension": (tib | {"shift": shift["resistance"]}, l1),
        "L2_the_change_needs_the_loop": (shift, shift["resistance"] > abs(shift["sham"])),
        "L3_the_other_assignment_does_not_resist": (shift["assistance"], shift["assistance"] < 0.1),
        "L4_the_grooming_reach_survives_the_loop": (
            {"grooming": round(near, 4), "none": round(near_none, 4)},
            near >= 0.5 and near >= near_none + 0.25,
        ),
    }
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        if name == "L2_the_change_needs_the_loop" and not l1:
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    return out


def reported(table: pd.DataFrame) -> dict:
    """Per arm and drive: medians of every readout."""
    cols = [
        "median_tibia_rad",
        "time_near_head",
        "min_distance_mm",
        "tibia_flexor_hz",
        "tibia_extensor_hz",
        "claw_mean_hz",
        "leg_motor_spikes",
    ]
    out = {}
    for (arm, drive), g in table.groupby(["arm", "drive"], sort=False):
        out.setdefault(arm, {})[drive] = {c: round(float(g[c].median()), 4) for c in cols}
    return out


def plot(trials: list[dict], table: pd.DataFrame, path) -> None:
    """Tibia angle over time per arm (one trial), and every trial's median angle per arm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "open": "#999999",
        "resistance": "#0072B2",
        "assistance": "#D55E00",
        "sham": "#009E73",
    }
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    for arm, colour in colours.items():
        t = next(x for x in trials if x["arm"] == arm and x["drive"] == "grooming")
        left.plot(
            np.arange(t["tibia"].size) * body.CONTROL_DT_S,
            t["tibia"],
            color=colour,
            lw=1.0,
            label=arm,
        )
    left.axhline(closed_loop.EXTENDED, color="black", lw=0.5, ls=":")
    left.set_xlabel("time (s)", fontsize=7)
    left.set_ylabel("femur-tibia angle (rad; lower = extended)", fontsize=7)
    left.set_title(f"grooming drive, seed {SEEDS[0]}; dotted: extension limit", fontsize=7)
    left.legend(fontsize=6, frameon=False)
    left.tick_params(labelsize=6)
    g = table[table["drive"] == "grooming"]
    for k, (arm, colour) in enumerate(colours.items()):
        v = g.loc[g["arm"] == arm, "median_tibia_rad"].to_numpy()
        right.scatter(k + np.linspace(-0.2, 0.2, v.size), v, s=10, color=colour)
    right.set_xticks(range(len(colours)), list(colours), fontsize=6.5)
    right.set_ylabel("median femur-tibia angle (rad)", fontsize=7)
    right.set_title("every trial, grooming drive", fontsize=7)
    right.tick_params(labelsize=6)
    fig.suptitle(
        "SIMULATED closed loop: male nerve cord, FlyMimic leg, claw-neuron feedback", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-3a design."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    jobs = [
        (a, d, s, None) for a in ("open", "resistance", "assistance") for d in DRIVES for s in SEEDS
    ]
    with ProcessPoolExecutor(memory_bounded_workers(per_worker_gb=8.0), initializer=_init) as pool:
        trials = list(pool.map(_trial, jobs))
        res = {
            t["seed"]: t["rates"]
            for t in trials
            if t["arm"] == "resistance" and t["drive"] == "grooming"
        }
        sham_jobs = [
            ("sham", "grooming", s, res[SEEDS[(k + 1) % len(SEEDS)]]) for k, s in enumerate(SEEDS)
        ]
        trials += list(pool.map(_trial, sham_jobs))
    table = pd.DataFrame(
        [{k: v for k, v in t.items() if k not in ("rates", "tibia")} for t in trials]
    )
    result = {
        "label": "SIMULATED closed loop through the front-leg claw neurons",
        "predictions": score(table, spec),
        "reported": reported(table),
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage3a_SIMULATED.csv", index=False)
    (out / "embodied_stage3a_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage3a", scoring.clean(result))
    plot(trials, table, figures_dir() / "embodied_stage3a_claw_feedback_SIMULATED.png")
    for name, entry in result["predictions"].items():
        print(f"{name:46s} {entry['outcome']:>15}  {entry['value']}")
    print("reported", json.dumps(scoring.clean(result["reported"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
