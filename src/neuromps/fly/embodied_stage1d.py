"""Embodied fly, stage 1d: does a grooming drive hold the leg at the head while walking swings it?

``python -m neuromps.fly.embodied_stage1d`` repeats stage 1c's design on fresh seeds
and scores the readouts registered in
`preregistration/embodied_stage1d_reach_versus_swing.json`: the time the claw spends
near the head, the rhythmicity of its movement, and stage 1c's motor-level
specificity. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from neuromps.fly import body, design, embodied_stage1c, kinematics, scoring
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage1d_reach_versus_swing.json"
SEEDS = range(740, 748)
NEAR_HEAD_MM = body.REST_DISTANCE_MM - 0.2
DRIVES = embodied_stage1c.DRIVES
LEGS = embodied_stage1c.LEGS
CONTROLS = ("giant_fibre", "backward_walking", "forward_walking")


def readouts(moves: list[dict]) -> pd.DataFrame:
    """Per trial and activation scale: time near the head and rhythmicity."""
    return pd.DataFrame(
        [
            {
                "drive": m["drive"],
                "seed": m["seed"],
                "r_full": m["r_full"],
                "time_near_head": kinematics.time_near_head(m["distance_mm"], NEAR_HEAD_MM),
                "rhythmicity": kinematics.rhythmicity(m["distance_mm"]),
            }
            for m in moves
        ]
    )


def score(table: pd.DataFrame, pools: pd.DataFrame, spec: dict) -> dict:
    """D1-D4 with values and outcomes."""

    def d1(scale):
        med = table[table["r_full"] == scale].groupby("drive")["time_near_head"].median()
        ok = med["grooming"] >= 0.5 and all(
            med["grooming"] >= med[d] + 0.25 for d in DRIVES if d != "grooming"
        )
        return {k: round(float(v), 4) for k, v in med.items()}, bool(ok)

    primary, ok1 = d1(20.0)
    rhythm = table[table["r_full"] == 20.0].groupby("drive")["rhythmicity"].median()
    ok2 = rhythm["forward_walking"] >= 0.3 and rhythm["forward_walking"] > rhythm["grooming"]
    others = {f"r_full_{s:g}": d1(s) for s in (10.0, 40.0)}
    rates = pools.groupby("drive")[list(LEGS)].mean()
    g = rates.loc["grooming"]
    ok4 = (
        g["front"] > g["middle"]
        and g["front"] > g["hind"]
        and all(g["front"] > rates.loc[d, "front"] for d in CONTROLS)
    )
    found = {
        "D1_grooming_holds_the_claw_at_the_head": (primary, ok1),
        "D2_forward_walking_swings_the_leg": (
            {k: round(float(v), 4) for k, v in rhythm.items()},
            bool(ok2),
        ),
        "D3_the_activation_scale_does_not_decide_it": (
            {k: v[0] for k, v in others.items()},
            all(v[1] for v in others.values()),
        ),
        "D4_the_front_leg_specificity_replicates": (
            {
                "grooming": {leg: round(float(g[leg]), 3) for leg in LEGS},
                "front_under_controls": {
                    d: round(float(rates.loc[d, "front"]), 3) for d in CONTROLS
                },
            },
            bool(ok4),
        ),
    }
    return {
        n: {"test": spec["predictions"][n]["test"], "value": v, "outcome": "pass" if ok else "fail"}
        for n, (v, ok) in found.items()
    }


def plot(table: pd.DataFrame, path) -> None:
    """Time near the head against rhythmicity, one point per trial, per drive."""
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
        "giant_fibre": "giant fibre",
        "backward_walking": "MDN (backward walking)",
        "forward_walking": "BDN2 (forward walking)",
        "none": "no drive",
    }
    rows = table[table["r_full"] == 20.0]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for drive in DRIVES:
        d = rows[rows["drive"] == drive]
        ax.scatter(
            d["time_near_head"],
            d["rhythmicity"],
            s=18,
            color=colours[drive],
            label=labels[drive],
            alpha=0.85,
            edgecolors="none",
        )
    ax.set_xlabel("fraction of the second with the claw near the head", fontsize=7)
    ax.set_ylabel("rhythmicity of the claw's movement", fontsize=7)
    ax.set_xlim(-0.03, 1.03)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.5, frameon=False, loc="upper center")
    ax.set_title("SIMULATED male nerve cord moving the FlyMimic leg: reach or swing?", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: run the registered stage-1d design and score it."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    trials, moves = embodied_stage1c.run(
        [(d, s, embodied_stage1c.W_SYN) for d in DRIVES for s in SEEDS]
    )
    table = readouts(moves)
    pools = pd.DataFrame(
        [{"drive": t["drive"], "seed": t["seed"], **t["pool_rates"]} for t in trials]
    )
    result = {
        "label": "SIMULATED male nerve cord moving the FlyMimic leg: reach or swing",
        "predictions": score(table, pools, spec),
        "near_head_threshold_mm": NEAR_HEAD_MM,
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage1d_SIMULATED.csv", index=False)
    pools.to_csv(out / "embodied_stage1d_pools_SIMULATED.csv", index=False)
    (out / "embodied_stage1d_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage1d", scoring.clean(result))
    plot(table, figures_dir() / "embodied_stage1d_reach_swing_SIMULATED.png")
    for name, entry in result["predictions"].items():
        print(f"{name:46s} {entry['outcome']:>5}  {entry['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
