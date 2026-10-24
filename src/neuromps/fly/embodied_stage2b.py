"""Embodied fly, stage 2b: proboscis extension from the FlyWire brain (open loop).

``python -m neuromps.fly.embodied_stage2b`` runs the registered design
(`preregistration/embodied_stage2b_proboscis_extension.json`): sugar, bitter,
sugar with bitter, antennal or no input to the published FlyWire 783 model; the
proboscis motor pools, named by cell type after the muscle roles in McKellar et
al. 2020, turned into activations; per joint the extensor's activation minus its
antagonist's commands the flybody proboscis. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import behaviour, connectome, design, model, muscle_map, proboscis, scoring
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage2b_proboscis_extension.json"
DRIVES = {
    "sugar": ("sugar_grn_right", None),
    "bitter": ("bitter_grn", None),
    "sugar_bitter": ("sugar_grn_right", "bitter_grn"),
    "jon_ce": ("jon_ce", None),
    "none": (None, None),
}
POOLS = {
    "rostrum_protractor": ("CB0701",),
    "rostrum_retractor": ("CB0720", "CB0789", "CB0861", "CB0783"),
    "haustellum_extensor": ("CB0845", "CB0871"),
    "haustellum_flexor": ("CB0703", "CB0875", "CB0882"),
    "labellum_extensor": ("CB0858",),
    "labellum_abductor": ("CB0762",),
}
SCORED = ("rostrum_protractor", "rostrum_retractor", "haustellum_extensor", "haustellum_flexor")
JOINT_POOLS = {
    "rostrum": ("rostrum_protractor", "rostrum_retractor"),
    "haustellum": ("haustellum_extensor", "haustellum_flexor"),
}
SEEDS = range(810, 818)
R_FULL = (20.0, 10.0, 40.0)
DURATION_S = 1.0
DT_S = 0.001
TAU_S = 0.02
STIM_HZ = 150.0
STEP_CHECK_S = 5e-6
_STATE: dict = {}


def _init() -> None:
    con = connectome.cached()
    ann = connectome.read_annotations().drop_duplicates("root_id")
    present = set(con.root_ids.tolist())
    _STATE.update(con=con, sets=behaviour.neuron_sets())
    _STATE["pools"] = {
        name: con.index_of(
            [r for r in ann[ann["cell_type"].isin(types)]["root_id"] if r in present]
        )
        for name, types in POOLS.items()
    }


def _brain(job) -> dict:
    """One FlyWire trial: each pool's mean filtered rate (Hz, 1 ms steps) and mean spike rate."""
    drive, seed = job
    con, sets = _STATE["con"], _STATE["sets"]
    first, second = DRIVES[drive]
    excite = con.index_of(sets[first]) if first else []
    excite2 = con.index_of(sets[second]) if second else []
    params = model.FlyParams(
        seed=seed, stim_rate_hz=STIM_HZ, stim2_rate_hz=STIM_HZ if second else 0.0
    )
    spikes = model.simulate(con, params, excite=excite, excite2=excite2)
    traces, rates = {}, {}
    for name, idx in _STATE["pools"].items():
        trains = [
            muscle_map.filtered_rate(spikes.times[spikes.neurons == n], DURATION_S, DT_S, TAU_S)
            for n in idx
        ]
        traces[name] = np.mean(trains, axis=0)
        rates[name] = float(np.isin(spikes.neurons, idx).sum() / len(idx) / DURATION_S)
    return {"drive": drive, "seed": seed, "traces": traces, "rates": rates}


def activation(trace: np.ndarray, r_full: float) -> np.ndarray:
    """A pool's activation: its mean rate over `r_full`, clipped to [0, 1]."""
    return np.clip(np.asarray(trace, float) / r_full, 0.0, 1.0)


def commands(traces: dict[str, np.ndarray], r_full: float) -> dict[str, np.ndarray]:
    """Per joint, the servo command from extensor minus antagonist activation."""
    return {
        joint: proboscis.command(
            activation(traces[ext], r_full) - activation(traces[opp], r_full), joint
        )
        for joint, (ext, opp) in JOINT_POOLS.items()
    }


def onset_s(signal: np.ndarray) -> float | None:
    """First time |signal| reaches half its own peak; None for a flat signal."""
    mag = np.abs(np.asarray(signal, float))
    if mag.max() <= 0:
        return None
    return float(np.argmax(mag >= 0.5 * mag.max()) * DT_S)


def _body(job) -> dict:
    """One trial at one activation scale: peak extension and the time-averaged activations."""
    trial, r_full = job
    cmd = commands(trial["traces"], r_full)
    ext = proboscis.extension_mm(cmd)
    row = {"drive": trial["drive"], "seed": trial["seed"], "r_full": r_full}
    row["peak_extension_mm"] = float(ext.max())
    for name in SCORED:
        row[f"act_{name}"] = float(activation(trial["traces"][name], r_full).mean())
    for joint, signal in cmd.items():
        row[f"onset_{joint}_s"] = onset_s(signal)
    for name, rate in trial["rates"].items():
        row[f"rate_{name}"] = rate
    return row | {"trace": ext}


def _median(table: pd.DataFrame, drive: str, column: str) -> float:
    return float(table.loc[table["drive"] == drive, column].median())


def _g1(table: pd.DataFrame, e_full: float) -> tuple[dict, bool]:
    peak = {d: _median(table, d, "peak_extension_mm") for d in DRIVES}
    controls = ("bitter", "jon_ce", "none")
    ok = peak["sugar"] >= 0.5 * e_full and all(
        peak["sugar"] - peak[c] >= 0.25 * e_full for c in controls
    )
    return {d: round(v, 4) for d, v in peak.items()}, ok


def _g3(table: pd.DataFrame) -> tuple[dict, bool]:
    sugar = _median(table, "sugar", "peak_extension_mm")
    both = _median(table, "sugar_bitter", "peak_extension_mm")
    return {"sugar": round(sugar, 4), "sugar_bitter": round(both, 4)}, both <= 0.5 * sugar


def score(table: pd.DataFrame, spec: dict, e_full: float) -> dict:
    """G0-G5 with values and outcomes."""
    main = table[table["r_full"] == R_FULL[0]]
    none = main[main["drive"] == "none"]
    silent = float(none[[f"rate_{p}" for p in SCORED]].to_numpy().sum())
    act = {p: round(_median(main, "sugar", f"act_{p}"), 4) for p in SCORED}
    g1, ok1 = _g1(main, e_full)
    g3, ok3 = _g3(main)
    sens = {}
    for r in R_FULL[1:]:
        sub = table[table["r_full"] == r]
        sens[f"r_full_{int(r)}"] = {"G1": _g1(sub, e_full), "G3": _g3(sub)}
    found = {
        "G0_silent_without_drive": (silent, silent == 0),
        "G1_sugar_extends_the_proboscis": (g1 | {"e_full": round(e_full, 4)}, ok1),
        "G2_both_joints_are_driven_to_extend": (
            act,
            act["rostrum_protractor"] > act["rostrum_retractor"]
            and act["haustellum_extensor"] > act["haustellum_flexor"],
        ),
        "G3_bitter_gates_extension": (g3, ok3),
        "G4_bitter_alone_does_not_extend": (
            g1["bitter"],
            g1["bitter"] <= 0.25 * e_full,
        ),
        "G5_the_activation_scale_does_not_decide_it": (
            {k: {g: v[g][0] for g in v} for k, v in sens.items()},
            all(v["G1"][1] and v["G3"][1] for v in sens.values()),
        ),
    }
    return {
        n: {"test": spec["predictions"][n]["test"], "value": v, "outcome": "pass" if ok else "fail"}
        for n, (v, ok) in found.items()
    }


def reported(table: pd.DataFrame) -> dict:
    """Labellum pool rates and command onsets per drive (r_full 20 Hz), without prediction."""
    main = table[table["r_full"] == R_FULL[0]]
    out = {}
    for drive, g in main.groupby("drive", sort=False):
        out[drive] = {
            c.removeprefix("rate_"): round(float(g[c].median()), 2)
            for c in g
            if c.startswith("rate_")
        } | {
            c: (None if g[c].isna().all() else float(g[c].median()))
            for c in g
            if c.startswith("onset_")
        }
    return out


def plot(rows: list[dict], table: pd.DataFrame, e_full: float, path) -> None:
    """Extension over time per drive (one trial each), and every trial's peak extension."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "sugar": "#E69F00",
        "sugar_bitter": "#0072B2",
        "bitter": "#009E73",
        "jon_ce": "#CC79A7",
        "none": "#999999",
    }
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    for drive, colour in colours.items():
        row = next(r for r in rows if r["drive"] == drive and r["r_full"] == R_FULL[0])
        t = np.arange(row["trace"].size) * DT_S
        left.plot(t, row["trace"], color=colour, lw=1.0, label=drive.replace("_", " + "))
    left.axhline(e_full, color="black", lw=0.6, ls="--")
    left.set_xlabel("time (s)", fontsize=7)
    left.set_ylabel("proboscis extension (mm)", fontsize=7)
    left.set_title("one trial per input; dashed: both joints at full extension", fontsize=7)
    left.legend(fontsize=5.5, frameon=False, loc="center right")
    left.tick_params(labelsize=6)
    main = table[table["r_full"] == R_FULL[0]]
    for k, (drive, colour) in enumerate(colours.items()):
        values = main.loc[main["drive"] == drive, "peak_extension_mm"].to_numpy()
        jitter = np.linspace(-0.15, 0.15, values.size)
        right.scatter(k + jitter, values, s=10, color=colour)
    right.axhline(e_full, color="black", lw=0.6, ls="--")
    right.set_xticks(range(len(colours)), [d.replace("_", "+") for d in colours], fontsize=6)
    right.set_ylabel("peak extension (mm)", fontsize=7)
    right.set_title("every trial (r_full 20 Hz)", fontsize=7)
    right.tick_params(labelsize=6)
    fig.suptitle(
        "SIMULATED FlyWire brain -> proboscis motor neurons -> flybody proboscis", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def check_step(e_full: float) -> float:
    """The registered numerical check: E_full at half the step; stops the stage on a mismatch."""
    finer = proboscis.full_extension_mm(step_s=STEP_CHECK_S)
    if abs(finer - e_full) > 0.01:
        raise RuntimeError(f"E_full {e_full:.4f} mm at 0.01 ms vs {finer:.4f} at 0.005 ms")
    return finer


def main(argv: list[str] | None = None) -> int:
    """CLI: run the registered stage-2b design and score it."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    e_full = proboscis.full_extension_mm()
    finer = check_step(e_full)
    log.info("E_full %.4f mm (%.4f at the finer step)", e_full, finer)
    workers = memory_bounded_workers(per_worker_gb=6.0)
    with ProcessPoolExecutor(workers, initializer=_init) as pool:
        trials = list(pool.map(_brain, [(d, s) for d in DRIVES for s in SEEDS]))
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_body, [(t, r) for r in R_FULL for t in trials]))
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "trace"} for r in rows])
    result = {
        "label": "SIMULATED proboscis extension from the FlyWire brain",
        "predictions": score(table, spec, e_full),
        "numerical_check": {"e_full_mm": e_full, "e_full_finer_step_mm": finer},
        "reported": reported(table),
        "pool_sizes": {k: int(len(v)) for k, v in _pool_sizes().items()},
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage2b_SIMULATED.csv", index=False)
    (out / "embodied_stage2b_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage2b", scoring.clean(result))
    plot(rows, table, e_full, figures_dir() / "embodied_stage2b_proboscis_SIMULATED.png")
    for name, entry in result["predictions"].items():
        print(f"{name:42s} {entry['outcome']:>5}  {entry['value']}")
    print("numerical check", result["numerical_check"])
    print("reported", json.dumps(scoring.clean(result["reported"]), indent=1))
    return 0


def _pool_sizes() -> dict:
    _init()
    return _STATE["pools"]


if __name__ == "__main__":
    raise SystemExit(main())
