"""Embodied fly, stage 1: does antennal drive move the front leg toward the head? (open loop)

``python -m neuromps.fly.embodied_stage1`` runs the registered design
(`preregistration/embodied_stage1_open_loop_leg.json`): the male CNS model under
antennal (JO-CE), sugar or no drive; the left front-leg motor neurons' spikes,
filtered into rates and mapped onto the FlyMimic leg's muscles; the leg moved
in MuJoCo; and the claw's distance to the antenna scored. Open loop: the leg's
movement does not feed back into the network (that is stage 3). Everything here
is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import body, design, jobs, model, muscle_map, scoring
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage1_open_loop_leg.json"
DRIVES = ("jon_ce", "sugar", "none")
SEEDS = range(700, 708)
R_FULL = (20.0, 10.0, 40.0)
DURATION_S = 1.0
TAU_S = 0.02
WORKERS_GB = 12.0
_STATE: dict = {}


def _init(context: jobs.Context) -> None:
    from neuromps.fly import malecns

    jobs.init(context, "malecns")
    con = jobs._STATE["con"]
    full = pd.read_feather(malecns.RAW / malecns.source.FILES[1])
    full = full[full["status"] == "Traced"].set_index("bodyId").reindex(con.root_ids)
    _STATE["assignment"] = muscle_map.assign(full.reset_index(drop=True))


def _trial(job) -> dict:
    """One 1 s trial: spike times of every mapped left front-leg motor neuron."""
    drive, seed = job
    ctx, con = jobs._STATE["context"], jobs._STATE["con"]
    sets = {"jon_ce": "jon_ce", "sugar": "sugar_grn_right", "none": None}[drive]
    params = jobs._base(duration_s=DURATION_S, stim_rate_hz=150.0, seed=seed)
    excite = con.index_of(ctx.sets[sets]) if sets else []
    spikes = model.simulate(con, params, excite=excite)
    neurons = sorted({n for v in _STATE["assignment"].neurons_by_muscle.values() for n in v})
    return {
        "drive": drive,
        "seed": seed,
        "spikes": {n: spikes.times[spikes.neurons == n] for n in neurons},
        "assignment": _STATE["assignment"],
    }


def activations(trial: dict, r_full: float, names: list[str]) -> np.ndarray:
    """Time x muscle activation of one trial at one activation scale."""
    neurons = sorted(trial["spikes"])
    column = {n: k for k, n in enumerate(neurons)}
    rates = np.stack(
        [
            muscle_map.filtered_rate(trial["spikes"][n], DURATION_S, body.CONTROL_DT_S, TAU_S)
            for n in neurons
        ],
        axis=1,
    )
    by_muscle = trial["assignment"].neurons_by_muscle
    return np.stack(
        [
            muscle_map.activation(rates, [column[n] for n in by_muscle.get(name, [])], r_full)
            for name in names
        ],
        axis=1,
    )


def score(table: pd.DataFrame, spec: dict) -> dict:
    """S1a-S1c with values and outcomes."""

    def s1a(scale):
        med = table[table["r_full"] == scale].groupby("drive")["min_distance_mm"].median()
        ok = med["jon_ce"] <= med["sugar"] - 0.2 and med["jon_ce"] <= med["none"] - 0.2
        return {k: round(float(v), 4) for k, v in med.items()}, bool(ok)

    primary, ok_a = s1a(20.0)
    jon = table[(table["r_full"] == 20.0) & (table["drive"] == "jon_ce")]
    below = int((jon["min_distance_mm"] < body.REST_DISTANCE_MM).sum())
    others = {f"r_full_{s:g}": s1a(s) for s in (10.0, 40.0)}
    found = {
        "S1a_antennal_drive_brings_the_claw_to_the_head": (primary, ok_a),
        "S1b_it_is_a_movement_toward_the_head": (
            {"trials_below_rest": below, "trials": int(len(jon))},
            below >= 6,
        ),
        "S1c_the_activation_scale_does_not_decide_it": (
            {k: v[0] for k, v in others.items()},
            all(v[1] for v in others.values()),
        ),
    }
    return {
        n: {"test": spec["predictions"][n]["test"], "value": v, "outcome": "pass" if ok else "fail"}
        for n, (v, ok) in found.items()
    }


def _move(args) -> dict:
    trial, r_full = args
    names = body.muscle_names()
    act = activations(trial, r_full, names)
    moved = body.run(act, names)
    return {
        "drive": trial["drive"],
        "seed": trial["seed"],
        "r_full": r_full,
        "min_distance_mm": float(moved.distance_mm.min()),
        "mean_distance_mm": float(moved.distance_mm.mean()),
        "distance_mm": moved.distance_mm,
        "mean_activation": act.mean(axis=0),
    }


def plot(moves: list[dict], names: list[str], path) -> None:
    """Claw-antenna distance over time per drive, and mean muscle activation per drive."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"jon_ce": "#D55E00", "sugar": "#0072B2", "none": "#999999"}
    labels = {"jon_ce": "antennal (JO-CE) drive", "sugar": "sugar drive", "none": "no drive"}
    primary = [m for m in moves if m["r_full"] == R_FULL[0]]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    t = np.arange(len(primary[0]["distance_mm"])) * body.CONTROL_DT_S
    for drive in DRIVES:
        traces = np.stack([m["distance_mm"] for m in primary if m["drive"] == drive])
        left.plot(t, np.median(traces, axis=0), color=colours[drive], lw=1.3, label=labels[drive])
        left.fill_between(
            t, traces.min(axis=0), traces.max(axis=0), color=colours[drive], alpha=0.15, lw=0
        )
    left.axhline(body.REST_DISTANCE_MM, color="black", ls=":", lw=0.8, label="passive rest")
    left.set_xlabel("time (s)", fontsize=7)
    left.set_ylabel("claw to antenna (mm)", fontsize=7)
    left.legend(fontsize=5.5, frameon=False)
    left.tick_params(labelsize=6)
    short = [n.replace("LFC_", "").replace("LFF_", "").replace("LFTibia_", "tibia ") for n in names]
    width = 0.27
    for k, drive in enumerate(DRIVES):
        mean = np.mean([m["mean_activation"] for m in primary if m["drive"] == drive], axis=0)
        right.barh(np.arange(len(names)) + (k - 1) * width, mean, width, color=colours[drive])
    right.set_yticks(range(len(names)), short, fontsize=5)
    right.invert_yaxis()
    right.set_xlabel("mean muscle activation", fontsize=7)
    right.tick_params(labelsize=6)
    fig.suptitle("SIMULATED male CNS driving the FlyMimic front leg, open loop", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: run the registered stage-1 design and score it."""
    from neuromps.fly.replication import _context
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    context = _context(design.registration())
    work = [(d, s) for d in DRIVES for s in SEEDS]
    with ProcessPoolExecutor(
        memory_bounded_workers(per_worker_gb=WORKERS_GB), initializer=_init, initargs=(context,)
    ) as pool:
        trials = list(pool.map(_trial, work))
    with ProcessPoolExecutor(12) as pool:
        moves = list(pool.map(_move, [(t, r) for t in trials for r in R_FULL]))
    table = pd.DataFrame(
        [{k: v for k, v in m.items() if k not in ("distance_mm", "mean_activation")} for m in moves]
    )
    names = body.muscle_names()
    activation = pd.DataFrame(
        [
            {
                "drive": m["drive"],
                "seed": m["seed"],
                **dict(zip(names, m["mean_activation"], strict=True)),
            }
            for m in moves
            if m["r_full"] == R_FULL[0]
        ]
    )
    result = {
        "label": "SIMULATED male CNS driving the FlyMimic leg (open loop)",
        "predictions": score(table, spec),
        "coverage": trials[0]["assignment"].coverage,
        "mean_activation_by_drive": activation.groupby("drive")[names]
        .mean()
        .round(4)
        .to_dict(orient="index"),
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage1_SIMULATED.csv", index=False)
    (out / "embodied_stage1_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage1", scoring.clean(result))
    plot(moves, names, figures_dir() / "embodied_stage1_leg_SIMULATED.png")
    for name, entry in result["predictions"].items():
        print(f"{name:48s} {entry['outcome']:>5}  {entry['value']}")
    print("coverage", result["coverage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
