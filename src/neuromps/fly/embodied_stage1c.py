"""Embodied fly, stage 1c: the male nerve cord driven from descending neurons, brain removed.

``python -m neuromps.fly.embodied_stage1c`` runs the registered design
(`preregistration/embodied_stage1c_nerve_cord_from_descending_neurons.json`):
the MaleCNS network cut down to the nerve cord and its descending and ascending
neurons; the antennal-grooming descending neurons (aDN1, aDN2), the giant fibre,
MDN or BDN2 driven at 150 Hz; the left front-leg motor neurons mapped onto the
FlyMimic leg as in stage 1; and the claw's approach to the antenna scored, with
the motor pools of the three left legs alongside. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import body, connectome, design, embodied_stage1, model, muscle_map, scoring
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = (
    REPO_ROOT / "preregistration" / "embodied_stage1c_nerve_cord_from_descending_neurons.json"
)
NERVE_CORD = (
    "vnc_intrinsic",
    "vnc_motor",
    "vnc_sensory",
    "vnc_efferent",
    "ascending_neuron",
    "sensory_ascending",
    "descending_neuron",
)
DRIVES = {
    "grooming": ("DNg62", "DNge078"),
    "giant_fibre": ("DNp01",),
    "backward_walking": ("MDN",),
    "forward_walking": ("DNg100",),
    "none": (),
}
SEEDS = range(730, 738)
W_SYN = 0.275
W_SENSITIVITY = 0.1596
LEGS = {"front": "fl", "middle": "ml", "hind": "hl"}
_STATE: dict = {}


def _annotations(root_ids: np.ndarray) -> pd.DataFrame:
    from neuromps.fly import malecns

    full = pd.read_feather(malecns.RAW / malecns.source.FILES[1])
    return full[full["status"] == "Traced"].set_index("bodyId").reindex(root_ids).reset_index()


def _init() -> None:
    from neuromps.fly import malecns

    con, ann = malecns.cached()
    keep = ann["superclass"].isin(NERVE_CORD).to_numpy()
    sub = connectome.subnetwork(con, keep)
    table = _annotations(sub.root_ids)
    motor = table["superclass"].isin(muscle_map.MOTOR) & (table["somaSide"] == "L")
    _STATE.update(
        con=sub,
        assignment=muscle_map.assign(table),
        drive={k: np.flatnonzero(table["type"].isin(v).to_numpy()) for k, v in DRIVES.items()},
        pools={
            leg: np.flatnonzero((motor & (table["subclass"] == code)).to_numpy())
            for leg, code in LEGS.items()
        },
        all_motor=np.flatnonzero(table["superclass"].isin(muscle_map.MOTOR).to_numpy()),
    )


def _trial(job) -> dict:
    drive, seed, w_syn = job
    con = _STATE["con"]
    params = model.FlyParams(
        duration_s=embodied_stage1.DURATION_S, stim_rate_hz=150.0, seed=seed, w_syn_mv=w_syn
    )
    spikes = model.simulate(con, params, excite=_STATE["drive"][drive])
    counts = np.bincount(spikes.neurons, minlength=con.n_neurons) / spikes.duration_s
    neurons = sorted({n for v in _STATE["assignment"].neurons_by_muscle.values() for n in v})
    return {
        "drive": drive,
        "seed": seed,
        "w_syn": w_syn,
        "spikes": {n: spikes.times[spikes.neurons == n] for n in neurons},
        "assignment": _STATE["assignment"],
        "pool_rates": {leg: float(counts[idx].mean()) for leg, idx in _STATE["pools"].items()},
        "motor_spikes": int(counts[_STATE["all_motor"]].sum() * spikes.duration_s),
        "drive_neurons": int(_STATE["drive"][drive].size),
    }


def score(table: pd.DataFrame, pools: pd.DataFrame, motor_spikes_none: int, spec: dict) -> dict:
    """C0-C4 with values and outcomes."""

    def c1(scale):
        med = table[table["r_full"] == scale].groupby("drive")["min_distance_mm"].median()
        others = [d for d in DRIVES if d != "grooming"]
        ok = all(med["grooming"] <= med[d] - 0.2 for d in others)
        return {k: round(float(v), 4) for k, v in med.items()}, bool(ok)

    primary, ok1 = c1(20.0)
    rates = pools.groupby("drive")[list(LEGS)].mean()
    g = rates.loc["grooming"]
    controls = ("giant_fibre", "backward_walking", "forward_walking")
    found = {
        "C0_the_nerve_cord_is_silent_without_drive": (motor_spikes_none, motor_spikes_none == 0),
        "C1_grooming_neurons_bring_the_claw_to_the_head": (primary, ok1),
        "C2_the_front_leg_is_the_one_that_moves": (
            {leg: round(float(g[leg]), 3) for leg in LEGS},
            bool(g["front"] > g["middle"] and g["front"] > g["hind"]),
        ),
        "C3_grooming_neurons_recruit_the_front_leg_most": (
            {d: round(float(rates.loc[d, "front"]), 3) for d in ("grooming", *controls)},
            bool(all(g["front"] > rates.loc[d, "front"] for d in controls)),
        ),
    }
    others = {f"r_full_{s:g}": c1(s) for s in (10.0, 40.0)}
    found["C4_the_activation_scale_does_not_decide_it"] = (
        {k: v[0] for k, v in others.items()},
        all(v[1] for v in others.values()),
    )
    return {
        n: {"test": spec["predictions"][n]["test"], "value": v, "outcome": "pass" if ok else "fail"}
        for n, (v, ok) in found.items()
    }


def plot(moves: list[dict], pools: pd.DataFrame, path) -> None:
    """Claw-antenna distance per drive, and the three left legs' motor pools per drive."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "grooming": "#D55E00",
        "giant_fibre": "#009E73",
        "backward_walking": "#CC79A7",
        "forward_walking": "#0072B2",
        "none": "#999999",
    }
    labels = {
        "grooming": "aDN1 + aDN2 (antennal grooming)",
        "giant_fibre": "giant fibre (escape)",
        "backward_walking": "MDN (backward walking)",
        "forward_walking": "BDN2 (forward walking)",
        "none": "no drive",
    }
    primary = [m for m in moves if m["r_full"] == 20.0]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    t = np.arange(len(primary[0]["distance_mm"])) * body.CONTROL_DT_S
    for drive in DRIVES:
        traces = np.stack([m["distance_mm"] for m in primary if m["drive"] == drive])
        left.plot(t, np.median(traces, axis=0), color=colours[drive], lw=1.2, label=labels[drive])
    left.axhline(body.REST_DISTANCE_MM, color="black", ls=":", lw=0.8, label="passive rest")
    left.set_xlabel("time (s)", fontsize=7)
    left.set_ylabel("claw to antenna (mm), median of 8", fontsize=7)
    left.legend(fontsize=5, frameon=False)
    left.tick_params(labelsize=6)
    means = pools.groupby("drive")[list(LEGS)].mean().reindex(list(DRIVES))
    width = 0.16
    for k, drive in enumerate(DRIVES):
        right.bar(
            np.arange(len(LEGS)) + (k - 2) * width, means.loc[drive], width, color=colours[drive]
        )
    right.set_xticks(range(len(LEGS)), [f"left {leg} leg" for leg in LEGS], fontsize=6.5)
    right.set_ylabel("motor pool rate (Hz)", fontsize=7)
    right.tick_params(labelsize=6)
    fig.suptitle(
        "SIMULATED male nerve cord driven from descending neurons, brain removed", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(work: list[tuple[str, int, float]]) -> tuple[list[dict], list[dict]]:
    """Network trials for (drive, seed, w_syn) jobs, then the leg at every activation scale."""
    with ProcessPoolExecutor(memory_bounded_workers(per_worker_gb=8.0), initializer=_init) as pool:
        trials = list(pool.map(_trial, work))
    jobs_ = [(t, r) for t in trials for r in embodied_stage1.R_FULL]
    with ProcessPoolExecutor(12) as pool:
        moves = list(pool.map(embodied_stage1._move, jobs_))
    for m, (t, _) in zip(moves, jobs_, strict=True):
        m["w_syn"] = t["w_syn"]
    return trials, moves


def main(argv: list[str] | None = None) -> int:
    """CLI: run the registered stage-1c design and score it."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    work = [(d, s, W_SYN) for d in DRIVES for s in SEEDS]
    work += [(d, s, W_SENSITIVITY) for d in ("grooming", "none") for s in SEEDS]
    trials, moves = run(work)
    main_trials = [t for t in trials if t["w_syn"] == W_SYN]
    table = pd.DataFrame(
        [{k: v for k, v in m.items() if k not in ("distance_mm", "mean_activation")} for m in moves]
    )
    pools = pd.DataFrame(
        [
            {
                "drive": t["drive"],
                "seed": t["seed"],
                "w_syn": t["w_syn"],
                "motor_spikes": t["motor_spikes"],
                **t["pool_rates"],
            }
            for t in trials
        ]
    )
    main_table, main_pools = table[table["w_syn"] == W_SYN], pools[pools["w_syn"] == W_SYN]
    none_spikes = int(main_pools[main_pools["drive"] == "none"]["motor_spikes"].sum())
    sens = table[(table["w_syn"] == W_SENSITIVITY) & (table["r_full"] == 20.0)]
    result = {
        "label": "SIMULATED male nerve cord driven from descending neurons, brain removed",
        "predictions": score(main_table, main_pools, none_spikes, spec),
        "drive_neurons": {
            d: next(t["drive_neurons"] for t in main_trials if t["drive"] == d) for d in DRIVES
        },
        "coverage": main_trials[0]["assignment"].coverage,
        "sensitivity_rescaled_weight": {
            "median_min_distance_mm": sens.groupby("drive")["min_distance_mm"]
            .median()
            .round(4)
            .to_dict()
        },
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage1c_SIMULATED.csv", index=False)
    pools.to_csv(out / "embodied_stage1c_pools_SIMULATED.csv", index=False)
    (out / "embodied_stage1c_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage1c", scoring.clean(result))
    plot(
        [m for m in moves if m["w_syn"] == W_SYN],
        main_pools,
        figures_dir() / "embodied_stage1c_leg_SIMULATED.png",
    )
    for name, entry in result["predictions"].items():
        print(f"{name:50s} {entry['outcome']:>5}  {entry['value']}")
    print("drive neurons", result["drive_neurons"], "coverage", result["coverage"])
    print("sensitivity", result["sensitivity_rescaled_weight"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
