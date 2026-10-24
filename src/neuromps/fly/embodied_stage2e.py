"""Embodied fly, stage 2e: Pugliese et al.'s firing-rate model on the male nerve cord.

``python -m neuromps.fly.embodied_stage2e`` runs the registered design
(`preregistration/embodied_stage2e_rate_model_walking.json`): the stage-1c nerve
cord as a firing-rate network (`fly.rate_model`), 32 parameter draws, driven by
DNg100, the grooming neurons, DNb08 or nothing, with the three-neuron rhythm
generator silenced as a lesion; stage 2a's measures on the tibia pools. Nothing
is tuned. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import design, embodied_stage1c, fictive, rate_model, scoring
from neuromps.fly import embodied_stage2c as s2c
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage2e_rate_model_walking.json"
SEEDS = list(range(860, 892))
DRIVES = {
    "forward_walking": ("DNg100",),
    "grooming": ("DNg62", "DNge078"),
    "dnb08": ("DNb08",),
    "none": (),
}
CPG = ("IN17A001", "INXXX466", "IN16B036")
LEG_SUBCLASSES = ("fl", "ml", "hl")
AMPLITUDE, SENSITIVITY_AMPLITUDE = 250.0, 150.0
DT_S, CHECK_DT_S = 1e-4, 5e-5
DURATION_S, START_S = 2.0, 0.2
RHYTHMIC = 0.5
CONDITIONS = {
    "forward_walking": ("forward_walking", AMPLITUDE, False),
    "grooming": ("grooming", AMPLITUDE, False),
    "dnb08": ("dnb08", AMPLITUDE, False),
    "none": ("none", AMPLITUDE, False),
    "lesion": ("forward_walking", AMPLITUDE, True),
    "forward_walking_150": ("forward_walking", SENSITIVITY_AMPLITUDE, False),
}
_STATE: dict = {}


def _init() -> None:
    fictive._init()
    con = embodied_stage1c._STATE["con"]
    table = embodied_stage1c._annotations(con.root_ids)
    types = table["type"].astype(str)
    leg_mn = table["superclass"].eq("vnc_motor") & table["subclass"].isin(LEG_SUBCLASSES)
    pools = fictive._STATE["pools"]
    pool_idx = np.unique(np.concatenate(list(pools.values())))
    legs = np.flatnonzero(leg_mn.to_numpy())
    _STATE.update(
        con=con,
        w=rate_model.weights(con),
        size=rate_model.sizes(con),
        drive={k: np.flatnonzero(types.isin(v).to_numpy()) for k, v in DRIVES.items()},
        cpg=np.flatnonzero(types.isin(CPG).to_numpy()),
        record=np.unique(np.concatenate([pool_idx, legs])),
        leg_mn=legs,
    )


def _signals(run: rate_model.Run, rep: int) -> tuple[dict, dict]:
    record = _STATE["record"]
    start = int(round(START_S / 1e-3))
    signals, rates = {}, {}
    for (leg, kind), idx in fictive._STATE["pools"].items():
        cols = np.searchsorted(record, idx)
        trace = run.rates[:, cols, rep].mean(axis=1)
        signals[f"{leg}_{kind}"] = trace[start:]
        rates[f"{leg}_{kind}"] = float(trace.mean())
    return signals, rates


def _rhythmic_fraction(run: rate_model.Run, rep: int, method: str = "registered") -> float | None:
    """Fraction of active leg motor neurons (mean >= 1 Hz) with rhythmicity >= 0.5."""
    record = _STATE["record"]
    cols = np.searchsorted(record, _STATE["leg_mn"])
    start = int(round(START_S / 1e-3))
    traces = run.rates[start:, cols, rep]
    active = traces.mean(axis=0) >= fictive.MIN_RATE_HZ
    if not active.any():
        return None
    scores = [fictive.rhythmicity(traces[:, j], method) for j in np.flatnonzero(active)]
    return float(np.mean(np.asarray(scores) >= RHYTHMIC))


def simulate_condition(name: str, seeds: list[int], dt_s: float) -> rate_model.Run:
    """All replicates of one condition, integrated together."""
    drive, amplitude, lesion = CONDITIONS[name]
    p = rate_model.draw(_STATE["size"], seeds)
    return rate_model.simulate(
        _STATE["w"],
        p,
        _STATE["drive"][drive],
        amplitude,
        _STATE["record"],
        duration_s=DURATION_S,
        dt_s=dt_s,
        silenced=_STATE["cpg"] if lesion else None,
    )


def condition_rows(
    run: rate_model.Run, name: str, seeds: list[int], method: str = "registered"
) -> list[dict]:
    """Stage 2a's measures per replicate of one simulated condition."""
    drive = CONDITIONS[name][0]
    rows = []
    for rep, seed in enumerate(seeds):
        signals, rates = _signals(run, rep)
        trial = {"drive": drive, "seed": seed, "signals": signals, "rates": rates}
        row = fictive.measures(trial, method=method)
        row["condition"] = name
        row["period_L1_flexor_s"] = (
            s2c.period_s(signals["L1_flexor"], method=method)
            if rates["L1_flexor"] >= fictive.MIN_RATE_HZ
            else None
        )
        row["active_neurons"] = int(np.sum(run.mean_rate[:, rep] >= fictive.MIN_RATE_HZ))
        row["leg_mn_rhythmic_fraction"] = _rhythmic_fraction(run, rep, method)
        row |= {f"rate_{k}": v for k, v in rates.items()}
        rows.append(row | ({"signals": signals} if rep == 0 else {}))
    return rows


def run_condition(job) -> list[dict]:
    """All replicates of one condition: stage 2a's measures per replicate."""
    name, seeds, dt_s = job
    return condition_rows(simulate_condition(name, seeds, dt_s), name, seeds)


def _median(table: pd.DataFrame, condition: str, column: str) -> float:
    return float(table.loc[table["condition"] == condition, column].median())


def score(table: pd.DataFrame, spec: dict) -> dict:
    """R0-R7 with values and outcomes."""
    rate_cols = [c for c in table if c.startswith("rate_")]
    none = table[table["condition"] == "none"]
    fw = table[table["condition"] == "forward_walking"]
    rhythm = {
        c: round(_median(table, c, "rhythmicity_L1_flexor"), 4)
        for c in ("forward_walking", "forward_walking_150", "grooming", "lesion")
    }
    r3 = fictive._close(fw["phase_L1flex_L1ext"], 0.5)
    r4 = fictive._close(fw["phase_L1_R1"], 0.5)
    r5 = (fictive._close(fw["phase_L1_L2"], 0.5), fictive._close(fw["phase_L1_L3"], 0.0))
    period = s2c._period_ms(fw)
    silent = float(none[rate_cols].to_numpy().max())
    found = {
        "R0_silent_without_drive": (silent, silent < 0.01),
        "R1_the_walking_command_makes_a_rhythm": (
            {k: rhythm[k] for k in ("forward_walking", "forward_walking_150")},
            rhythm["forward_walking"] >= 0.3 and rhythm["forward_walking_150"] >= 0.3,
        ),
        "R2_more_than_grooming": (
            {k: rhythm[k] for k in ("forward_walking", "grooming")},
            rhythm["forward_walking"] > rhythm["grooming"],
        ),
        "R3_antagonists_alternate": (r3, r3 is not None and r3 <= 0.15),
        "R4_left_and_right_alternate": (r4, r4 is not None and r4 <= 0.15),
        "R5_a_tripod": (
            {"L1_L2_from_half": r5[0], "L1_L3_from_zero": r5[1]},
            None not in r5 and max(r5) <= 0.15,
        ),
        "R6_the_rhythm_needs_the_three_neuron_circuit": (
            {k: rhythm[k] for k in ("forward_walking", "lesion")},
            rhythm["lesion"] < 0.3 and rhythm["lesion"] <= 0.5 * rhythm["forward_walking"],
        ),
        "R7_a_fly_like_step_period": (
            period,
            period is not None and s2c.PERIOD_MS[0] <= period <= s2c.PERIOD_MS[1],
        ),
    }
    return {
        n: {"test": spec["predictions"][n]["test"], "value": v, "outcome": "pass" if ok else "fail"}
        for n, (v, ok) in found.items()
    }


def reported(table: pd.DataFrame) -> dict:
    """Per condition: medians of rhythmicity, period, activity, pool rates, rhythmic MNs."""
    out = {}
    for name, g in table.groupby("condition", sort=False):
        period = g["period_L1_flexor_s"].dropna()
        frac = g["leg_mn_rhythmic_fraction"].dropna()
        out[name] = {
            "rhythmicity_L1_flexor": round(float(g["rhythmicity_L1_flexor"].median()), 4),
            "period_ms": None if period.empty else round(float(period.median() * 1000), 1),
            "active_neurons": int(g["active_neurons"].median()),
            "leg_mn_rhythmic_fraction": None if frac.empty else round(float(frac.median()), 3),
            "flexor_hz": round(
                float(
                    g[[c for c in g if c.endswith("_flexor") and c.startswith("rate_")]]
                    .mean()
                    .mean()
                ),
                2,
            ),
            "extensor_hz": round(
                float(
                    g[[c for c in g if c.endswith("_extensor") and c.startswith("rate_")]]
                    .mean()
                    .mean()
                ),
                2,
            ),
        }
    return out


def plot(rows: list[dict], table: pd.DataFrame, path) -> None:
    """One replicate's tibia pools under DNb08, and every replicate's rhythmicity per condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals = next(r["signals"] for r in rows if r["condition"] == "dnb08" and "signals" in r)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    colours = {
        "L1_flexor": "#D55E00",
        "L1_extensor": "#E69F00",
        "R1_flexor": "#0072B2",
        "L2_flexor": "#009E73",
        "L3_flexor": "#CC79A7",
    }
    t = START_S + np.arange(signals["L1_flexor"].size) * 1e-3
    for key, colour in colours.items():
        left.plot(t, signals[key], color=colour, lw=0.9, label=key.replace("_", " "))
    left.set_xlim(START_S, 1.2)
    left.set_xlabel("time (s)", fontsize=7)
    left.set_ylabel("pool rate (Hz)", fontsize=7)
    left.set_title(
        "DNb08 drive, one draw (seed 860); under DNg100 drive the pools stay silent", fontsize=7
    )
    left.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=2)
    left.tick_params(labelsize=6)
    order = list(CONDITIONS)
    for k, name in enumerate(order):
        values = table.loc[table["condition"] == name, "rhythmicity_L1_flexor"].to_numpy()
        right.scatter(k + np.linspace(-0.2, 0.2, values.size), values, s=6, color="#0072B2")
    right.axhline(0.3, color="black", lw=0.6, ls="--")
    right.set_xticks(
        range(len(order)), [o.replace("_", " ") for o in order], fontsize=5.5, rotation=30
    )
    right.set_ylabel("L1 flexor rhythmicity", fontsize=7)
    right.set_title("every draw; dashed: registered 0.3", fontsize=7)
    right.tick_params(labelsize=6)
    fig.suptitle(
        "SIMULATED male nerve cord as a firing-rate network (Pugliese et al. model)", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def driven_neurons() -> dict:
    """Exploratory: each driven neuron's size stand-in and its drive-only rate (mean draws)."""
    out = {}
    for name, idx in _STATE["drive"].items():
        mean = {k: v[0] for k, v in rate_model.DISTRIBUTIONS.items()}
        sizes = _STATE["size"][idx]
        x = AMPLITUDE - mean["theta"] * sizes
        rate = np.maximum(mean["r_max"] * np.tanh((mean["a"] / sizes / mean["r_max"]) * x), 0)
        if len(idx):
            out[name] = {
                "size_median_normalized": [round(float(v), 2) for v in sizes],
                "drive_only_rate_hz": [round(float(v), 2) for v in rate],
            }
    q = np.quantile(_STATE["size"], [0.5, 0.99, 0.999])
    out["network_size_quantiles"] = {
        "q50": 1.0,
        "q99": round(float(q[1]), 2),
        "q999": round(float(q[2]), 2),
    }
    return out


def check_step(pool) -> dict:
    """The registered numerical check on the first draw; stops the stage on a mismatch."""
    coarse, fine = pool.map(
        run_condition,
        [("forward_walking", SEEDS[:1], DT_S), ("forward_walking", SEEDS[:1], CHECK_DT_S)],
    )
    a, b = coarse[0]["rhythmicity_L1_flexor"], fine[0]["rhythmicity_L1_flexor"]
    if abs(a - b) > 0.05:
        raise RuntimeError(f"rhythmicity {a:.3f} at 0.1 ms vs {b:.3f} at 0.05 ms")
    return {"rhythmicity_0.1ms": a, "rhythmicity_0.05ms": b}


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-2e design."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    with ProcessPoolExecutor(len(CONDITIONS), initializer=_init) as pool:
        check = check_step(pool)
        log.info("numerical check %s", check)
        rows = [
            r
            for got in pool.map(run_condition, [(c, SEEDS, DT_S) for c in CONDITIONS])
            for r in got
        ]
    table = s2c._frame(rows)
    _init()
    result = {
        "label": "SIMULATED firing-rate model (Pugliese et al.) of the male nerve cord",
        "predictions": score(table, spec),
        "numerical_check": check,
        "reported": reported(table),
        "exploratory_driven_neurons": driven_neurons(),
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage2e_SIMULATED.csv", index=False)
    (out / "embodied_stage2e_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage2e", scoring.clean(result))
    plot(rows, table, figures_dir() / "embodied_stage2e_rate_model_SIMULATED.png")
    for name, entry in result["predictions"].items():
        print(f"{name:46s} {entry['outcome']:>5}  {entry['value']}")
    print("numerical check", check)
    print("reported", json.dumps(scoring.clean(result["reported"]), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
