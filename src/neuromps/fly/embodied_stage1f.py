"""Embodied fly, stage 1f: the brain-to-nerve-cord relay, scored by depth of reach, fresh seeds.

``python -m neuromps.fly.embodied_stage1f`` reruns stage 1e's relay
(`fly.relay.run`) on fresh seeds and scores what
`preregistration/embodied_stage1f_relay_depth_of_reach.json` registers: a relative
precondition on the grooming descending neurons, and the closest approach of the
claw to the antenna. Stage 1e stays uninterpretable as registered. Everything
here is **simulated**.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from neuromps.fly import design, embodied_stage1c, relay, scoring
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage1f_relay_depth_of_reach.json"
SEEDS = range(760, 768)
LEGS = embodied_stage1c.LEGS


def score(table: pd.DataFrame, pools: pd.DataFrame, grooming_rate: dict, spec: dict) -> dict:
    """F0-F4; F1-F4 uninterpretable if F0 fails."""

    def f1(scale):
        med = table[table["r_full"] == scale].groupby("drive")["min_distance_mm"].median()
        ok = all(med["jon_ce"] <= med[d] - 0.2 for d in ("sugar", "none"))
        return {k: round(float(v), 4) for k, v in med.items()}, bool(ok)

    primary, ok1 = f1(20.0)
    rates = pools.groupby("drive")[list(LEGS)].mean()
    j = rates.loc["jon_ce"]
    others = {f"r_full_{s:g}": f1(s) for s in (10.0, 40.0)}
    g = grooming_rate
    found = {
        "F0_the_grooming_neurons_answer_antennal_input": (
            {k: round(v, 3) for k, v in g.items()},
            g["jon_ce"] > g["sugar"] and g["jon_ce"] > g["none"],
        ),
        "F1_antennal_input_reaches_deepest": (primary, ok1),
        "F2_the_front_leg_is_the_one_that_moves": (
            {
                "jon_ce": {leg: round(float(j[leg]), 3) for leg in LEGS},
                "front_under_sugar": round(float(rates.loc["sugar", "front"]), 3),
            },
            bool(
                j["front"] > j["middle"]
                and j["front"] > j["hind"]
                and j["front"] > rates.loc["sugar", "front"]
            ),
        ),
        "F3_the_activation_scale_does_not_decide_it": (
            {k: v[0] for k, v in others.items()},
            all(v[1] for v in others.values()),
        ),
        "F4_the_reach_runs_through_the_grooming_neurons": (
            {"intact": primary["jon_ce"], "lesioned": primary["jon_ce_lesion"]},
            bool(primary["jon_ce_lesion"] >= primary["jon_ce"] + 0.2),
        ),
    }
    gate = found["F0_the_grooming_neurons_answer_antennal_input"][1]
    return {
        n: {
            "test": spec["predictions"][n]["test"],
            "value": v,
            "outcome": ("pass" if ok else "fail")
            if gate or n.startswith("F0")
            else "uninterpretable",
        }
        for n, (v, ok) in found.items()
    }


def plot(table: pd.DataFrame, path) -> None:
    """Closest approach of the claw per condition, one point per trial."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["jon_ce", "jon_ce_lesion", "sugar", "none"]
    labels = ["antennal input", "antennal input,\naDN1+aDN2 removed", "sugar input", "no input"]
    colours = ["#D55E00", "#E69F00", "#0072B2", "#999999"]
    rows = table[table["r_full"] == 20.0]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for k, cond in enumerate(order):
        values = rows[rows["drive"] == cond]["min_distance_mm"].to_numpy()
        ax.bar(k, np.median(values), color=colours[k], alpha=0.55)
        ax.scatter(
            np.full(values.size, k) + np.linspace(-0.15, 0.15, values.size),
            values,
            s=10,
            color=colours[k],
            zorder=3,
        )
    ax.set_xticks(range(len(order)), labels, fontsize=6)
    ax.set_ylabel("closest claw-to-antenna distance (mm)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("SIMULATED FlyWire brain -> male nerve cord -> leg: depth of reach", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: run the relay on fresh seeds and score the registered predictions."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    ran = relay.run(SEEDS)
    result = {
        "label": "SIMULATED FlyWire brain -> male nerve cord -> FlyMimic leg, depth of reach",
        "predictions": score(ran["table"], ran["pools"], ran["grooming_rate"], spec),
        "matching": ran["matching"],
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    ran["table"].to_csv(out / "embodied_stage1f_SIMULATED.csv", index=False)
    ran["pools"].to_csv(out / "embodied_stage1f_pools_SIMULATED.csv", index=False)
    (out / "embodied_stage1f_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage1f", scoring.clean(result))
    plot(ran["table"], figures_dir() / "embodied_stage1f_depth_SIMULATED.png")
    for name, entry in result["predictions"].items():
        print(f"{name:48s} {entry['outcome']:>15}  {entry['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
