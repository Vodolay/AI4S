"""Embodied fly, stage 2a: fictive walking in the male nerve cord (motor neurons, no body).

``python -m neuromps.fly.fictive`` runs the registered design
(`preregistration/embodied_stage2a_fictive_walking.json`): the stage-1c nerve cord
driven by its forward-walking command neuron (BDN2), the grooming neurons, MDN or
nothing; the tibia flexor and extensor motor pools of all six legs turned into
rates; their rhythmicity and their phases relative to one another. Everything here
is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import design, embodied_stage1c, kinematics, model, muscle_map, scoring
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage2a_fictive_walking.json"
DRIVES = {
    "forward_walking": ("DNg100",),
    "grooming": ("DNg62", "DNge078"),
    "backward_walking": ("MDN",),
    "none": (),
}
LEGS = {
    "L1": ("fl", "L"),
    "R1": ("fl", "R"),
    "L2": ("ml", "L"),
    "R2": ("ml", "R"),
    "L3": ("hl", "L"),
    "R3": ("hl", "R"),
}
MUSCLES = {"flexor": "Ti flexor MN", "extensor": "Ti extensor MN"}
SEEDS = range(800, 808)
DURATION_S = 2.0
DT_S = 0.001
TAU_S = 0.02
START_S = 0.2
MIN_RATE_HZ = 1.0
LAGS_S = (0.05, 0.5)
_STATE: dict = {}


def circular_distance(p: float, q: float) -> float:
    """Distance between two phases on the unit circle of cycles."""
    d = abs(p - q) % 1.0
    return float(min(d, 1.0 - d))


def phase(a: np.ndarray, b: np.ndarray, dt_s: float = DT_S) -> float | None:
    """Phase of b relative to a, in cycles; None if a has no rhythm to measure against."""
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    if np.dot(a, a) <= 1e-12 or np.dot(b, b) <= 1e-12:
        return None
    lo, hi = (int(round(v / dt_s)) for v in LAGS_S)
    auto = [np.dot(a[:-k], a[k:]) for k in range(lo, min(hi, a.size - 1) + 1)]
    period = lo + int(np.argmax(auto))
    norm = np.sqrt(np.dot(a, a) * np.dot(b, b))
    cross = [np.dot(a[: a.size - k], b[k:]) / norm for k in range(period)]
    return float(int(np.argmax(cross)) / period)


def _init() -> None:
    embodied_stage1c._init()
    sub = embodied_stage1c._STATE["con"]
    table = embodied_stage1c._annotations(sub.root_ids)
    motor = table["superclass"].isin(muscle_map.MOTOR)
    _STATE["pools"] = {
        (leg, kind): np.flatnonzero(
            (
                motor
                & (table["subclass"] == code)
                & (table["somaSide"] == side)
                & (table["type"] == mtype)
            ).to_numpy()
        )
        for leg, (code, side) in LEGS.items()
        for kind, mtype in MUSCLES.items()
    }
    _STATE["drive"] = {
        k: np.flatnonzero(table["type"].isin(v).to_numpy()) for k, v in DRIVES.items()
    }


def _trial(job) -> dict:
    """One trial: the twelve pool rate signals (Hz, 1 ms), from START_S on, and mean rates."""
    drive, seed = job
    con = embodied_stage1c._STATE["con"]
    spikes = model.simulate(
        con,
        model.FlyParams(duration_s=DURATION_S, stim_rate_hz=150.0, seed=seed),
        excite=_STATE["drive"][drive],
    )
    start = int(round(START_S / DT_S))
    signals, rates = {}, {}
    for key, idx in _STATE["pools"].items():
        trains = [
            muscle_map.filtered_rate(spikes.times[spikes.neurons == n], DURATION_S, DT_S, TAU_S)
            for n in idx
        ]
        signal = np.mean(trains, axis=0) if trains else np.zeros(int(DURATION_S / DT_S))
        signals[f"{key[0]}_{key[1]}"] = signal[start:]
        rates[f"{key[0]}_{key[1]}"] = float(
            np.isin(spikes.neurons, idx).sum() / max(idx.size, 1) / DURATION_S
        )
    return {"drive": drive, "seed": seed, "signals": signals, "rates": rates}


def measures(trial: dict) -> dict:
    """Rhythmicity of L1 flexor and the registered phases (None where a pool is too quiet)."""
    s, r = trial["signals"], trial["rates"]

    def ph(a, b):
        if r[a] < MIN_RATE_HZ or r[b] < MIN_RATE_HZ:
            return None
        return phase(s[a], s[b])

    return {
        "drive": trial["drive"],
        "seed": trial["seed"],
        "rhythmicity_L1_flexor": kinematics.rhythmicity(s["L1_flexor"], start_s=0.0),
        "phase_L1flex_L1ext": ph("L1_flexor", "L1_extensor"),
        "phase_L1_R1": ph("L1_flexor", "R1_flexor"),
        "phase_L1_L2": ph("L1_flexor", "L2_flexor"),
        "phase_L1_L3": ph("L1_flexor", "L3_flexor"),
        **{f"rate_{k}": v for k, v in r.items()},
    }


def _close(values: pd.Series, target: float) -> float | None:
    if values.isna().any():
        return None
    return float(np.median([circular_distance(v, target) for v in values]))


def score(table: pd.DataFrame, spec: dict) -> dict:
    """W0-W4 with values and outcomes; an undefined phase counts as a failure."""
    none = table[table["drive"] == "none"]
    silent = float(none[[c for c in table if c.startswith("rate_")]].to_numpy().sum())
    fw = table[table["drive"] == "forward_walking"]
    groom = table[table["drive"] == "grooming"]
    r_fw, r_gr = (
        float(fw["rhythmicity_L1_flexor"].median()),
        float(groom["rhythmicity_L1_flexor"].median()),
    )
    w2 = _close(fw["phase_L1flex_L1ext"], 0.5)
    w3 = _close(fw["phase_L1_R1"], 0.5)
    w4a, w4b = _close(fw["phase_L1_L2"], 0.5), _close(fw["phase_L1_L3"], 0.0)
    found = {
        "W0_silent_without_drive": (silent, silent == 0),
        "W1_the_walking_command_makes_a_rhythm": (
            {"forward_walking": round(r_fw, 4), "grooming": round(r_gr, 4)},
            r_fw >= 0.3 and r_fw > r_gr,
        ),
        "W2_antagonists_alternate": (w2, w2 is not None and w2 <= 0.15),
        "W3_left_and_right_alternate": (w3, w3 is not None and w3 <= 0.15),
        "W4_a_tripod": (
            {"L1_L2_from_half": w4a, "L1_L3_from_zero": w4b},
            w4a is not None and w4b is not None and w4a <= 0.15 and w4b <= 0.15,
        ),
    }
    return {
        n: {"test": spec["predictions"][n]["test"], "value": v, "outcome": "pass" if ok else "fail"}
        for n, (v, ok) in found.items()
    }


def plot(trials: list[dict], table: pd.DataFrame, path) -> None:
    """One forward-walking trial's flexor rhythms on four legs, and every trial's phases."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    example = next(t for t in trials if t["drive"] == "forward_walking")
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    t = START_S + np.arange(example["signals"]["L1_flexor"].size) * DT_S
    colours = {
        "L1_flexor": "#D55E00",
        "L1_extensor": "#E69F00",
        "R1_flexor": "#0072B2",
        "L2_flexor": "#009E73",
        "L3_flexor": "#CC79A7",
    }
    for k, (key, colour) in enumerate(colours.items()):
        sig = example["signals"][key]
        span = np.ptp(sig) or 1.0
        left.plot(
            t, (sig - sig.min()) / span + 1.2 * k, color=colour, lw=0.9, label=key.replace("_", " ")
        )
    left.set_yticks([])
    left.set_xlabel("time (s)", fontsize=7)
    left.set_title(
        "forward-walking drive (BDN2): tibia motor pools, one trial (normalized)", fontsize=7
    )
    left.legend(fontsize=5.5, frameon=False, loc="upper right")
    left.tick_params(labelsize=6)
    fw = table[table["drive"] == "forward_walking"]
    pairs = [
        ("phase_L1flex_L1ext", "L1 flexor -> extensor", 0.5),
        ("phase_L1_R1", "L1 -> R1", 0.5),
        ("phase_L1_L2", "L1 -> L2", 0.5),
        ("phase_L1_L3", "L1 -> L3", 0.0),
    ]
    for k, (col, _label, target) in enumerate(pairs):
        values = fw[col].dropna().to_numpy()
        right.scatter(values, np.full(values.size, k), s=14, color="#0072B2", zorder=3)
        right.scatter([target], [k], marker="|", s=200, color="black")
    right.set_yticks(range(len(pairs)), [p[1] for p in pairs], fontsize=6.5)
    right.set_xlim(-0.02, 1.02)
    right.set_xlabel("phase (cycles); bar = registered target", fontsize=7)
    right.tick_params(labelsize=6)
    fig.suptitle(
        "SIMULATED male nerve cord, fictive walking (motor neurons only, no body)", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: run the registered stage-2a design and score it."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    with ProcessPoolExecutor(memory_bounded_workers(per_worker_gb=8.0), initializer=_init) as pool:
        trials = list(pool.map(_trial, [(d, s) for d in DRIVES for s in SEEDS]))
    table = pd.DataFrame([measures(t) for t in trials])
    mdn = table[table["drive"] == "backward_walking"]
    result = {
        "label": "SIMULATED fictive walking in the male nerve cord",
        "predictions": score(table, spec),
        "backward_walking_reported": {c: mdn[c].median() for c in mdn if c.startswith("phase")}
        | {"rhythmicity_L1_flexor": float(mdn["rhythmicity_L1_flexor"].median())},
        "pool_sizes": {f"{k[0]}_{k[1]}": int(v.size) for k, v in _pool_sizes().items()},
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage2a_SIMULATED.csv", index=False)
    (out / "embodied_stage2a_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage2a", scoring.clean(result))
    plot(trials, table, figures_dir() / "embodied_stage2a_fictive_walking_SIMULATED.png")
    for name, entry in result["predictions"].items():
        print(f"{name:40s} {entry['outcome']:>5}  {entry['value']}")
    print("backward walking (reported)", result["backward_walking_reported"])
    return 0


def _pool_sizes() -> dict:
    _init()
    return _STATE["pools"]


if __name__ == "__main__":
    raise SystemExit(main())
