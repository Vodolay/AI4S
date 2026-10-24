"""Score the registered output-tract test: descending neurons as an electrode array.

``python -m neuromps.fly.tract`` re-simulates the parent sweep's spontaneous
recordings (same seeds) and reads each one three ways: the parent's planar
array, 64 of its electrodes, and 64 single-unit electrodes on descending
neurons, the brain's output tract. Behavioural onset comes from the parent
sweep. See `preregistration/fly_output_tract_electrode.json`. Everything here
is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import checks, connectome, design, electrodes, jobs, model, scoring
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "fly_output_tract_electrode.json"
SPONTANEOUS_SEEDS = (100, 101, 102)
NOISE_SEEDS = range(1000, 1012)
N_ELECTRODES = 64
THRESHOLD = 0.156
ARRAYS = ("planar", "planar_64", "tract_64")


def subset(array: electrodes.VirtualArray, n: int, seed: int) -> electrodes.VirtualArray:
    """`n` of an array's electrodes, drawn without replacement."""
    keys = sorted(array.units)
    chosen = np.random.default_rng(seed).choice(len(keys), n, replace=False)
    picked = [keys[i] for i in sorted(chosen)]
    return electrodes.VirtualArray(
        {k: array.units[k] for k in picked}, {k: array.region[k] for k in picked}
    )


def tract_array(descending: np.ndarray, n: int, seed: int) -> electrodes.VirtualArray:
    """Single-unit electrodes on `n` descending neurons."""
    drawn = np.sort(np.random.default_rng(seed).choice(descending, n, replace=False))
    units = {f"DN_{k}": np.array([u]) for k, u in enumerate(drawn)}
    return electrodes.VirtualArray(units, dict.fromkeys(units, "descending"))


def _arrays() -> dict:
    planar = jobs._STATE["array"]
    return {
        "planar": planar,
        "planar_64": subset(planar, N_ELECTRODES, 1),
        "tract_64": tract_array(jobs._STATE["context"].descending, N_ELECTRODES, 2),
    }


def _record(args) -> dict:
    overrides, seed = args
    ctx, con = jobs._STATE["context"], jobs._STATE["con"]
    d = ctx.spec["design"]
    params = jobs._base(
        duration_s=d["spontaneous"]["duration_s"],
        background_mean_mv=d["background_mean_mv"],
        seed=seed,
        **overrides,
    )
    spikes = model.simulate(con, params)
    return {
        name: jobs._fill_silent(electrodes.mea_features(spikes, a)) for name, a in _arrays().items()
    }


def deltas(per_seed: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """Per array and condition: delta log2 firing rate of the seed means against vehicle."""
    from neuromps.twin.perturbation import simulated_delta

    out: dict[str, dict[str, float]] = {}
    for name in ARRAYS:
        means = {
            c: {"mfr_hz": float(np.mean([r[name]["mfr_hz"] for r in rs]))}
            for c, rs in per_seed.items()
        }
        out[name] = {c: simulated_delta(m, means["vehicle"])["mfr_hz"] for c, m in means.items()}
    return out


def onset(parent: pd.DataFrame, threshold: float) -> dict[str, tuple[float, str] | None]:
    """Per family: (dose, condition) of the first behaviour-impairing level, from the parent."""
    out = {}
    for family, rows in parent[parent["family"] != "vehicle"].groupby("family"):
        hit = rows[rows["disruption"] >= threshold].sort_values("dose")
        out[family] = None if hit.empty else (float(hit.iloc[0]["dose"]), hit.iloc[0]["condition"])
    return out


def flagged_in_time(parent: pd.DataFrame, delta: dict[str, float], first: dict) -> dict:
    """Per family: does |delta| reach THRESHOLD at a dose no later than behavioural onset?"""
    out = {}
    for family, start in first.items():
        if start is None:
            out[family] = None
            continue
        rows = parent[parent["family"] == family].sort_values("dose")
        hits = [
            r["dose"]
            for _, r in rows.iterrows()
            if abs(delta.get(r["condition"]) or 0.0) >= THRESHOLD
        ]
        out[family] = bool(hits and min(hits) <= start[0])
    return out


def score(per_seed: dict, noise: list[dict], parent: pd.DataFrame, spec: dict) -> dict:
    """T0-T3 with values and outcomes."""
    d = deltas(per_seed)
    impair = design.registration()["design"]["behaviour_impairment_threshold"]
    first = onset(parent, impair)
    parent_delta = dict(zip(parent["condition"], parent["delta.mfr_hz"], strict=True))
    t0 = max(abs(d["planar"][c] - parent_delta[c]) for c in parent_delta if c != "vehicle")
    at_onset = {
        name: {f: (None if s is None else round(abs(d[name][s[1]]), 4)) for f, s in first.items()}
        for name in ARRAYS
    }
    scored = [f for f, s in first.items() if s is not None]
    t1 = sum(at_onset["tract_64"][f] > at_onset["planar_64"][f] for f in scored)
    alarms = {
        name: checks.false_alarms([r[name] for r in noise], THRESHOLD)["mfr_hz"] for name in ARRAYS
    }
    in_time = {name: flagged_in_time(parent, d[name], first) for name in ARRAYS}
    count = {name: int(sum(bool(v) for v in in_time[name].values())) for name in ARRAYS}
    predictions = {
        "T0_planar_reproduces": (round(t0, 6), t0 <= 0.001),
        "T1_tract_moves_more_at_onset": ({"families": int(t1), "scored": len(scored)}, t1 >= 4),
        "T2_tract_is_not_noise": (alarms["tract_64"], alarms["tract_64"] <= 0.05),
        "T3_tract_flags_in_time_more_often": (
            {"tract_64": count["tract_64"], "planar_64": count["planar_64"]},
            count["tract_64"] > count["planar_64"],
        ),
    }
    return {
        "predictions": {
            n: {
                "test": spec["predictions"][n]["test"],
                "value": v,
                "outcome": "pass" if ok else "fail",
            }
            for n, (v, ok) in predictions.items()
        },
        "abs_delta_mfr_at_onset": at_onset,
        "false_alarm_rate": alarms,
        "flagged_in_time": in_time,
        "families_flagged_in_time": count,
        "delta_mfr": {
            n: {c: (None if v is None else round(v, 4)) for c, v in d[n].items()} for n in ARRAYS
        },
        "threshold": THRESHOLD,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: run the recordings and score the registered test."""
    from neuromps.fly.study import _context, memory_bounded_workers

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec, reg = design.registration(), json.loads(REGISTRATION.read_text())
    context, _ = _context(spec, connectome.cached())
    conds = design.conditions(spec)
    work = [(c.overrides, s) for c in conds for s in SPONTANEOUS_SEEDS]
    work += [({}, s) for s in NOISE_SEEDS]
    with ProcessPoolExecutor(
        memory_bounded_workers(), initializer=jobs.init, initargs=(context, "783")
    ) as pool:
        got = list(pool.map(_record, work))
    n = len(conds) * len(SPONTANEOUS_SEEDS)
    per_seed: dict[str, list[dict]] = {}
    for (c, _), r in zip([(c, s) for c in conds for s in SPONTANEOUS_SEEDS], got[:n], strict=True):
        per_seed.setdefault(c.name, []).append(r)
    parent = pd.read_csv(outputs_dir() / "fly" / "conditions_SIMULATED.csv")
    result = {
        "label": "SIMULATED output-tract electrode vs planar array",
        **score(per_seed, got[n:], parent, reg),
        "registration": design.provenance(REGISTRATION),
    }
    (outputs_dir() / "fly" / "tract_electrode_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("fly_output_tract", scoring.clean(result))
    for name, entry in result["predictions"].items():
        print(f"{name:38s} {entry['outcome']:>5}  {entry['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
