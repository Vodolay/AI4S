"""Embodied fly, stage 2f: the rate model with every command neuron driven to the same rate.

``python -m neuromps.fly.embodied_stage2f`` runs the registered design
(`preregistration/embodied_stage2f_rate_model_driven.json`): stage 2e's firing-rate
network of the male nerve cord, two size rules (synapse count; none), each driven
neuron given the input that alone brings it to the target rate, fresh draws, and
stage 2a's tests scored with the corrected rhythm measure of the audit. Everything
here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import design, fictive, rate_model, scoring
from neuromps.fly import embodied_stage2c as s2c
from neuromps.fly import embodied_stage2e as s2e
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage2f_rate_model_driven.json"
SEEDS = list(range(900, 932))
SIZE_RULES = ("synapse_count", "none")
TARGET_HZ, SENSITIVITY_HZ = 50.0, (25.0, 100.0)
METHOD = "audited"
# condition -> (stage-2e condition it reuses, target rate)
CONDITIONS = {
    "forward_walking": ("forward_walking", TARGET_HZ),
    "forward_walking_25": ("forward_walking", SENSITIVITY_HZ[0]),
    "forward_walking_100": ("forward_walking", SENSITIVITY_HZ[1]),
    "grooming": ("grooming", TARGET_HZ),
    "dnb08": ("dnb08", TARGET_HZ),
    "none": ("none", TARGET_HZ),
    "lesion": ("lesion", TARGET_HZ),
}
_STATE: dict = {}


def _init() -> None:
    s2e._init()
    st = s2e._STATE
    driven = np.concatenate([v for v in st["drive"].values() if len(v)])
    st["record"] = np.unique(np.concatenate([st["record"], driven]))
    _STATE["sizes"] = {"synapse_count": st["size"], "none": np.ones_like(st["size"])}


def _simulate(rule: str, name: str) -> rate_model.Run:
    st = s2e._STATE
    base, target = CONDITIONS[name]
    drive, _, lesion = s2e.CONDITIONS[base]
    p = rate_model.draw(_STATE["sizes"][rule], SEEDS)
    idx = st["drive"][drive]
    amplitude = rate_model.drive_input(p, idx, target) if len(idx) else 0.0
    return rate_model.simulate(
        st["w"],
        p,
        idx,
        amplitude,
        st["record"],
        duration_s=s2e.DURATION_S,
        dt_s=s2e.DT_S,
        silenced=st["cpg"] if lesion else None,
    )


def run_condition(job) -> list[dict]:
    """All draws of one (size rule, condition): stage 2a's measures, corrected, per draw."""
    rule, name = job
    run = _simulate(rule, name)
    base = CONDITIONS[name][0]
    rows = s2e.condition_rows(run, base, SEEDS, METHOD)
    dng100 = np.searchsorted(s2e._STATE["record"], s2e._STATE["drive"]["forward_walking"])
    start = int(round(s2e.START_S / 1e-3))
    for rep, row in enumerate(rows):
        row["condition"], row["size_rule"] = name, rule
        row["dng100_hz"] = float(run.rates[start:, dng100, rep].mean())
    return rows


def _median(table: pd.DataFrame, condition: str, column: str) -> float:
    return float(table.loc[table["condition"] == condition, column].median())


def score(table: pd.DataFrame, spec: dict) -> dict:
    """S0-S7 for one size rule; S1-S6 uninterpretable when S0 fails."""
    fw = table[table["condition"] == "forward_walking"]
    on = _median(table, "forward_walking", "dng100_hz")
    s0 = 0.5 * TARGET_HZ <= on <= 2 * TARGET_HZ
    rhythm = {
        c: round(_median(table, c, "rhythmicity_L1_flexor"), 4)
        for c in ("forward_walking", "forward_walking_25", "forward_walking_100", "grooming")
    }
    rhythm["lesion"] = round(_median(table, "lesion", "rhythmicity_L1_flexor"), 4)
    s3 = fictive._close(fw["phase_L1flex_L1ext"], 0.5)
    s4 = fictive._close(fw["phase_L1_R1"], 0.5)
    s5 = (fictive._close(fw["phase_L1_L2"], 0.5), fictive._close(fw["phase_L1_L3"], 0.0))
    period = s2c._period_ms(fw)
    s1 = rhythm["forward_walking"] >= 0.3
    found = {
        "S0_the_command_is_on": (round(on, 2), s0),
        "S1_the_walking_command_makes_a_rhythm": (
            {
                k: rhythm[k]
                for k in ("forward_walking", "forward_walking_25", "forward_walking_100")
            },
            s1,
        ),
        "S2_more_than_grooming": (
            {k: rhythm[k] for k in ("forward_walking", "grooming")},
            rhythm["forward_walking"] > rhythm["grooming"],
        ),
        "S3_antagonists_alternate": (s3, s3 is not None and s3 <= 0.15),
        "S4_left_and_right_alternate": (s4, s4 is not None and s4 <= 0.15),
        "S5_a_tripod": (
            {"L1_L2_from_half": s5[0], "L1_L3_from_zero": s5[1]},
            None not in s5 and max(s5) <= 0.15,
        ),
        "S6_the_rhythm_needs_the_three_neuron_circuit": (
            {k: rhythm[k] for k in ("forward_walking", "lesion")},
            s1 and rhythm["lesion"] < 0.3 and rhythm["lesion"] <= 0.5 * rhythm["forward_walking"],
        ),
        "S7_a_fly_like_step_period": (
            period,
            period is not None and s2c.PERIOD_MS[0] <= period <= s2c.PERIOD_MS[1],
        ),
    }
    out = {}
    for n, (v, ok) in found.items():
        outcome = "pass" if ok else "fail"
        command_off = n != "S0_the_command_is_on" and not s0
        nothing_to_lesion = n == "S6_the_rhythm_needs_the_three_neuron_circuit" and not s1
        if command_off or nothing_to_lesion:
            outcome = "uninterpretable"
        out[n] = {"test": spec["predictions"][n]["test"], "value": v, "outcome": outcome}
    return out


def plot(rows: list[dict], table: pd.DataFrame, path) -> None:
    """Per size rule: one forward-walking draw's tibia pools, and every draw's corrected rhythm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "L1_flexor": "#D55E00",
        "L1_extensor": "#E69F00",
        "R1_flexor": "#0072B2",
        "L2_flexor": "#009E73",
        "L3_flexor": "#CC79A7",
    }
    order = list(CONDITIONS)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})
    for r, rule in enumerate(SIZE_RULES):
        signals = next(
            x["signals"]
            for x in rows
            if x["size_rule"] == rule and x["condition"] == "forward_walking" and "signals" in x
        )
        left, right = axes[r]
        t = s2e.START_S + np.arange(signals["L1_flexor"].size) * 1e-3
        for key, colour in colours.items():
            left.plot(t, signals[key], color=colour, lw=0.9, label=key.replace("_", " "))
        left.set_xlim(s2e.START_S, 1.2)
        left.set_ylabel("pool rate (Hz)", fontsize=7)
        left.set_title(f"size rule '{rule}': DNg100 driven to 50 Hz, draw 900", fontsize=7)
        left.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=2)
        left.tick_params(labelsize=6)
        sub = table[table["size_rule"] == rule]
        for k, name in enumerate(order):
            v = sub.loc[sub["condition"] == name, "rhythmicity_L1_flexor"].to_numpy()
            right.scatter(k + np.linspace(-0.2, 0.2, v.size), v, s=6, color="#0072B2")
        right.axhline(0.3, color="black", lw=0.6, ls="--")
        right.set_ylim(-0.03, 1.0)
        right.set_xticks(
            range(len(order)), [o.replace("_", " ") for o in order], fontsize=5, rotation=35
        )
        right.set_ylabel("corrected L1 flexor rhythm", fontsize=7)
        right.set_title("every draw; dashed: registered 0.3", fontsize=7)
        right.tick_params(labelsize=6)
    axes[1, 0].set_xlabel("time (s)", fontsize=7)
    fig.suptitle(
        "SIMULATED male nerve cord as a firing-rate network, command neurons driven alike",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-2f design."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    jobs = [(rule, name) for rule in SIZE_RULES for name in CONDITIONS]
    with ProcessPoolExecutor(len(jobs), initializer=_init) as pool:
        rows = [r for got in pool.map(run_condition, jobs) for r in got]
    table = s2c._frame(rows)
    result = {"label": "SIMULATED rate model of the male nerve cord, command neurons driven alike"}
    for rule in SIZE_RULES:
        sub = table[table["size_rule"] == rule]
        result[rule] = {"predictions": score(sub, spec), "reported": s2e.reported(sub)}
    result["registration"] = design.provenance(REGISTRATION)
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage2f_SIMULATED.csv", index=False)
    (out / "embodied_stage2f_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage2f", scoring.clean(result))
    plot(rows, table, figures_dir() / "embodied_stage2f_rate_model_driven_SIMULATED.png")
    for rule in SIZE_RULES:
        print("size rule:", rule)
        for name, entry in result[rule]["predictions"].items():
            print(f"   {name:46s} {entry['outcome']:>15}  {entry['value']}")
        print("   reported", json.dumps(scoring.clean(result[rule]["reported"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
