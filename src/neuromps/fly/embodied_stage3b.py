"""Embodied fly, stage 3b: a perturbation test of the claw-neuron resistance reflex.

``python -m neuromps.fly.embodied_stage3b`` runs the registered design
(`preregistration/embodied_stage3b_perturbation_reflex.json`): stage 3a's closed loop
(nerve cord, leg, claw-neuron feedback) with a 20 ms flexing torque on the tibia during
the grooming hold; the return is timed under open loop, the resistance and assistance
assignments, and a sham that keeps the tonic feedback but not the part that follows the
perturbation. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import body, design, scoring
from neuromps.fly import embodied_stage3a as s3a
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage3b_perturbation_reflex.json"
SEEDS = list(range(1030, 1046))
PERTURB = (0.500, 0.520, 10.0)
PRE_S, WINDOW_END_S, RECOVERY_RAD = (0.40, 0.50), 0.60, 0.05
R_FULL, SENSITIVITY_R_FULL = 20.0, 80.0
ARMS = ("open", "resistance", "assistance")


def readouts(tibia: np.ndarray) -> dict:
    """Pre-pulse angle, excursion, and recovery time (ms after the pulse ends)."""
    ms = lambda t: int(round(t / body.CONTROL_DT_S))  # noqa: E731
    pre = float(np.median(tibia[ms(PRE_S[0]) : ms(PRE_S[1])]))
    excursion = float(tibia[ms(PERTURB[0]) : ms(WINDOW_END_S)].max() - pre)
    after = tibia[ms(PERTURB[1]) :]
    back = np.flatnonzero(after <= pre + RECOVERY_RAD)
    return {
        "pre_rad": pre,
        "excursion_rad": excursion,
        "recovery_ms": float(back[0] if back.size else after.size),
    }


def _job(arm: str, drive: str, seed: int, sham=None, r_full: float = R_FULL, perturbed=True):
    """Stage 3a's trial with this stage's arm label; 'resistance_unperturbed' runs unperturbed."""
    base = "resistance" if arm == "resistance_unperturbed" else arm
    return (base, drive, seed, sham, r_full, PERTURB if perturbed else None, arm)


def _trial(job) -> dict:
    trial = s3a._trial(job[:6])
    trial["arm"] = job[6]
    trial["r_full"] = job[4]
    return trial | readouts(trial["tibia"])


def _run(pool, r_full: float, drives: tuple[str, ...], arms: tuple[str, ...]) -> list[dict]:
    jobs = [_job(a, d, s, r_full=r_full) for a in arms for d in drives for s in SEEDS]
    jobs += [
        _job("resistance_unperturbed", "grooming", s, r_full=r_full, perturbed=False) for s in SEEDS
    ]
    trials = list(pool.map(_trial, jobs))
    tonic = {t["seed"]: t["rates"] for t in trials if t["arm"] == "resistance_unperturbed"}
    shams = [_job("sham", "grooming", s, sham=tonic[s], r_full=r_full) for s in SEEDS]
    return trials + list(pool.map(_trial, shams))


def _median(table: pd.DataFrame, arm: str, column: str, drive: str = "grooming") -> float:
    rows = table[(table["arm"] == arm) & (table["drive"] == drive)]
    return float(rows[column].median())


def score(table: pd.DataFrame, spec: dict) -> dict:
    """M0-M4; M1-M3 uninterpretable if the push fails, M2 also if M1 fails."""
    exc = _median(table, "open", "excursion_rad")
    rec = {
        a: _median(table, a, "recovery_ms") for a in ("open", "resistance", "assistance", "sham")
    }
    m0 = exc >= 0.2
    m1 = rec["resistance"] <= 0.8 * rec["open"]
    near = _median(table, "resistance", "time_near_head")
    near_none = _median(table, "resistance", "time_near_head", "none")
    found = {
        "M0_the_push_moves_the_leg": (round(exc, 4), m0),
        "M1_feedback_speeds_the_return": ({k: rec[k] for k in ("open", "resistance")}, m1),
        "M2_the_speed_needs_the_loop": (
            {k: rec[k] for k in ("resistance", "sham")},
            rec["resistance"] < rec["sham"],
        ),
        "M3_the_opposite_assignment_does_not_speed_it": (
            {k: rec[k] for k in ("open", "assistance")},
            rec["assistance"] >= rec["open"],
        ),
        "M4_the_reach_survives": (
            {"grooming": round(near, 4), "none": round(near_none, 4)},
            near >= 0.5 and near >= near_none + 0.25,
        ),
    }
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        no_push = name.startswith(("M1", "M2", "M3")) and not m0
        nothing_to_explain = name.startswith("M2") and not m1
        if no_push or nothing_to_explain:
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    return out


def reported(table: pd.DataFrame) -> dict:
    """Per r_full, arm and drive: medians of the readouts."""
    cols = [
        "pre_rad",
        "excursion_rad",
        "recovery_ms",
        "time_near_head",
        "tibia_flexor_hz",
        "tibia_extensor_hz",
        "claw_mean_hz",
    ]
    out: dict = {}
    for (r_full, arm, drive), g in table.groupby(["r_full", "arm", "drive"], sort=False):
        out.setdefault(f"r_full_{int(r_full)}", {}).setdefault(arm, {})[drive] = {
            c: round(float(g[c].median()), 4) for c in cols
        }
    return out


def plot(trials: list[dict], table: pd.DataFrame, path) -> None:
    """The tibia around the push (one seed per arm), and every trial's recovery time."""
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
    window = slice(450, 650)
    for arm, colour in colours.items():
        t = next(
            x
            for x in trials
            if x["arm"] == arm and x["drive"] == "grooming" and x["r_full"] == R_FULL
        )
        left.plot(
            np.arange(1000)[window] * body.CONTROL_DT_S,
            t["tibia"][window],
            color=colour,
            lw=1.0,
            label=arm,
        )
    left.axvspan(PERTURB[0], PERTURB[1], color="black", alpha=0.08)
    left.set_xlabel("time (s); shaded: imposed flexion", fontsize=7)
    left.set_ylabel("femur-tibia angle (rad)", fontsize=7)
    left.set_title(f"grooming drive, seed {SEEDS[0]}", fontsize=7)
    left.legend(fontsize=6, frameon=False)
    left.tick_params(labelsize=6)
    g = table[(table["drive"] == "grooming") & (table["r_full"] == R_FULL)]
    for k, (arm, colour) in enumerate(colours.items()):
        v = g.loc[g["arm"] == arm, "recovery_ms"].to_numpy()
        right.scatter(k + np.linspace(-0.2, 0.2, v.size), v, s=8, color=colour)
    right.set_xticks(range(len(colours)), list(colours), fontsize=6.5)
    right.set_ylabel("recovery time (ms after the push)", fontsize=7)
    right.set_title("every trial, r_full 20 Hz", fontsize=7)
    right.tick_params(labelsize=6)
    fig.suptitle("SIMULATED closed loop: a push on the tibia during the grooming hold", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-3b design."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    with ProcessPoolExecutor(
        memory_bounded_workers(per_worker_gb=8.0), initializer=s3a._init
    ) as pool:
        trials = _run(pool, R_FULL, ("grooming", "none"), ARMS)
        trials += _run(pool, SENSITIVITY_R_FULL, ("grooming",), ("open", "resistance"))
    table = pd.DataFrame(
        [{k: v for k, v in t.items() if k not in ("rates", "tibia")} for t in trials]
    )
    main_table = table[table["r_full"] == R_FULL]
    result = {
        "label": "SIMULATED perturbation test of the claw-neuron reflex",
        "predictions": score(main_table, spec),
        "reported": reported(table),
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage3b_SIMULATED.csv", index=False)
    (out / "embodied_stage3b_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage3b", scoring.clean(result))
    plot(trials, table, figures_dir() / "embodied_stage3b_perturbation_SIMULATED.png")
    for name, entry in result["predictions"].items():
        print(f"{name:46s} {entry['outcome']:>15}  {entry['value']}")
    print("reported", json.dumps(scoring.clean(result["reported"])))
    return 0


def chordotonal_counts() -> dict:
    """Exploratory (not registered): chordotonal-organ and claw neurons per leg nerve and side
    in the male CNS, to show how thinly the front leg's proprioceptors are traced."""
    from neuromps.fly import malecns

    ann = malecns._annotations(malecns.RAW)
    sensory = ann[ann["superclass"].astype(str) == "vnc_sensory"]
    nerves = {"ProLN": "front", "MesoLN": "middle", "MetaLN": "hind"}
    out: dict = {}
    for nerve, leg in nerves.items():
        for side in ("L", "R"):
            rows = sensory[
                (sensory["entryNerve"].astype(str) == nerve)
                & (sensory["rootSide"].astype(str) == side)
            ]
            cho = rows[rows["subclass"].astype(str) == "chordotonal organ"]
            claw = cho[cho["type"].astype(str).isin(list(s3a.CLAW))]
            out[f"{leg}_{side}"] = {"chordotonal": int(len(cho)), "claw": int(len(claw))}
    return out


def check() -> int:
    """Write the exploratory chordotonal counts to metrics.json."""
    result = {
        "label": "EXPLORATORY, not registered: proprioceptor tracing per leg (male CNS)",
        "chordotonal_by_leg": chordotonal_counts(),
    }
    (outputs_dir() / "fly" / "embodied_stage3b_check.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    update_metrics("embodied_stage3b_check", result)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(check() if "--check" in sys.argv else main())
