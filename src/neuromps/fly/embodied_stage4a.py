"""Embodied fly, stage 4a: the chip-to-animal drug design with movement as the readout.

``python -m neuromps.fly.embodied_stage4a`` runs the registered design
(`preregistration/embodied_stage4a_drugs_and_movement.json`): every condition of the
parent design (`fly_chip_to_animal`) applied to every network of a trial; proboscis
extension under sugar and under sugar with bitter (stage 2b) and the grooming reach
through the brain-to-nerve-cord relay (stage 1f); movement onset per family compared
with the recorded chip, tract and planar-64 detections and the parent's neural onset.
Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from neuromps.fly import body, design, embodied_stage1, relay, scoring
from neuromps.fly import embodied_stage2b as s2b
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage4a_drugs_and_movement.json"
PARENT = REPO_ROOT / "preregistration" / "fly_chip_to_animal.json"
SEEDS = list(range(1100, 1108))
EPS_MM, EPS_SENSITIVITY_MM, IMPAIR, CHIP = 0.01, 0.05, 0.5, 0.5
MOVEMENTS = ("proboscis_sugar", "proboscis_sugar_bitter", "reach_depth")
R_FULL = 20.0


def conditions() -> pd.DataFrame:
    """The parent's conditions (name, family, level, dose) and each one's knob."""
    table = pd.read_csv(outputs_dir() / "fly" / "conditions_SIMULATED.csv")
    knobs = {
        f: c["knob"] for f, c in json.loads(PARENT.read_text())["design"]["conditions"].items()
    }
    table = table[["condition", "family", "level", "dose"]].copy()
    table["knob"] = [
        {} if f == "vehicle" else {knobs[f]: float(lv)}
        for f, lv in zip(table["family"], table["level"], strict=True)
    ]
    return table


def _init() -> None:
    s2b._init()
    relay._brain_init()


def _brain(job) -> dict:
    """One FlyWire trial under a condition: proboscis pools (sugar drives) or DN spikes."""
    condition, drive, seed, knob = job
    if drive == "jon_ce":
        out = relay._brain(("jon_ce", seed, knob))
    else:
        out = s2b._brain((drive, seed, knob))
    return out | {"condition": condition, "drive": drive}


def _proboscis(trial: dict) -> dict:
    row = s2b._body((trial, R_FULL))
    return {
        "condition": trial["condition"],
        "drive": trial["drive"],
        "seed": trial["seed"],
        "value": row["mean_extension_mm"],
    }


def _cord(job) -> dict:
    brain, knob = job
    out = relay._cord((brain, False, set(), knob))
    return out | {"condition": brain["condition"]}


def _reach(cord: dict) -> dict:
    moved = embodied_stage1._move((cord, R_FULL))
    return {
        "condition": cord["condition"],
        "drive": "jon_ce",
        "seed": cord["seed"],
        "value": body.REST_DISTANCE_MM - moved["min_distance_mm"],
    }


def disruption(medians: pd.DataFrame, eps: float) -> pd.Series:
    """Per condition: mean over movements of |log2((x + eps) / (x_vehicle + eps))|."""
    base = medians.loc["vehicle"]
    ratio = np.log2((medians + eps) / (base + eps)).abs()
    return ratio.mean(axis=1)


def onset(table: pd.DataFrame, column: str, threshold: float) -> dict:
    """Per family: the lowest dose where `column` reaches `threshold` (None if never)."""
    out = {}
    for family, rows in table[table["family"] != "vehicle"].groupby("family"):
        hit = rows[rows[column] >= threshold].sort_values("dose")
        out[family] = None if hit.empty else float(hit.iloc[0]["dose"])
    return out


def flagged_in_time(table: pd.DataFrame, delta: dict, threshold: float, first: dict) -> dict:
    """Per family with a movement onset: does |delta| reach `threshold` by that dose?"""
    out = {}
    for family, start in first.items():
        if start is None:
            continue
        rows = table[table["family"] == family]
        hits = [
            d
            for c, d in zip(rows["condition"], rows["dose"], strict=True)
            if abs(delta.get(c) or 0.0) >= threshold
        ]
        out[family] = bool(hits and min(hits) <= start + 1e-9)
    return out


def _spearman(x, y) -> float:
    return float(stats.spearmanr(x, y).statistic)


def _family_series(conds: pd.DataFrame, medians: pd.DataFrame, family: str, column: str):
    rows = conds[conds["family"].isin(["vehicle", family])].sort_values("dose")
    return rows["dose"].to_numpy(), medians.loc[rows["condition"], column].to_numpy()


def score(conds: pd.DataFrame, medians: pd.DataFrame, spec: dict) -> tuple[dict, dict]:
    """N0-N5 and the per-family onsets they rest on."""
    parent = pd.read_csv(outputs_dir() / "fly" / "conditions_SIMULATED.csv")
    tract = json.loads((outputs_dir() / "fly" / "tract_electrode_SIMULATED.json").read_text())
    table = conds.copy()
    table["movement_disruption"] = disruption(medians, EPS_MM).loc[table["condition"]].to_numpy()
    first = {
        "movement": onset(table, "movement_disruption", IMPAIR),
        "neural_behaviour": onset(parent, "disruption", IMPAIR),
        "chip": onset(parent, "abs_delta_mfr", CHIP),
    }
    in_time = {
        arr: flagged_in_time(table, tract["delta_mfr"][arr], tract["threshold"], first["movement"])
        for arr in ("tract_64", "planar_64")
    }
    vehicle = medians.loc["vehicle"]
    n0 = vehicle["proboscis_sugar"] >= 0.1 and vehicle["reach_depth"] >= 0.3
    n1 = _spearman(*_family_series(conds, medians, "gaba_block", "proboscis_sugar_bitter"))
    n2 = {
        m: _spearman(*_family_series(conds, medians, "ach_block", m))
        for m in ("proboscis_sugar", "reach_depth")
    }
    impaired = {f: d for f, d in first["movement"].items() if d is not None}
    n3 = {
        f: {
            "chip": first["chip"][f],
            "movement": d,
            "chip_first": first["chip"][f] is not None and first["chip"][f] <= d + 1e-9,
        }
        for f, d in impaired.items()
    }
    both = {
        f: (d, first["neural_behaviour"][f])
        for f, d in impaired.items()
        if first["neural_behaviour"][f] is not None
    }
    counts = {arr: int(sum(v for v in in_time[arr].values())) for arr in in_time}
    found = {
        "N0_the_movements_are_there": (
            {k: round(float(vehicle[k]), 4) for k in ("proboscis_sugar", "reach_depth")},
            n0,
        ),
        "N1_disinhibition_releases_the_bitter_gate_in_movement": (round(n1, 4), n1 >= 0.8),
        "N2_excitation_block_stops_movement": (
            {k: round(v, 4) for k, v in n2.items()},
            all(v <= -0.8 for v in n2.values()),
        ),
        "N3_the_chip_flags_before_movement_fails": (
            n3,
            bool(impaired) and all(v["chip_first"] for v in n3.values()),
        ),
        "N4_the_output_tract_still_leads": (counts, counts["tract_64"] > counts["planar_64"]),
        "N5_movement_fails_no_earlier_than_neural_behaviour": (
            {f: {"movement": a, "neural_behaviour": b} for f, (a, b) in both.items()},
            bool(both) and all(a >= b - 1e-9 for a, b in both.values()),
        ),
    }
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        if name != "N0_the_movements_are_there" and not n0:
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    sensitivity = onset(
        conds.assign(md=disruption(medians, EPS_SENSITIVITY_MM).loc[conds["condition"]].to_numpy()),
        "md",
        IMPAIR,
    )
    details = {
        "onsets": first,
        "flagged_in_time": in_time,
        "movement_onset_eps_005mm": sensitivity,
        "movement_disruption": {
            c: round(float(v), 4)
            for c, v in zip(table["condition"], table["movement_disruption"], strict=True)
        },
    }
    return out, details


def plot(conds: pd.DataFrame, medians: pd.DataFrame, path) -> None:
    """Each movement against dose, per family (median over trials; vehicle at dose 0)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "gaba_block": "#D55E00",
        "gaba_potentiate": "#E69F00",
        "glucl_block": "#009E73",
        "ach_block": "#0072B2",
        "excitability_block": "#CC79A7",
    }
    labels = {
        "proboscis_sugar": "proboscis | sugar (mm)",
        "proboscis_sugar_bitter": "proboscis | sugar + bitter (mm)",
        "reach_depth": "grooming reach depth (mm)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    for ax, m in zip(axes, MOVEMENTS, strict=True):
        for family, colour in colours.items():
            x, y = _family_series(conds, medians, family, m)
            ax.plot(x, y, "o-", color=colour, lw=1.0, ms=3, label=family.replace("_", " "))
        ax.set_xlabel("dose (fraction of the knob's range)", fontsize=7)
        ax.set_title(labels[m], fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].legend(fontsize=5.5, frameon=False)
    fig.suptitle(
        "SIMULATED fly: movements under transmitter-class drug knobs (median of 8 trials)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-4a design."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    conds = conditions()
    knob = dict(zip(conds["condition"], conds["knob"], strict=True))
    jobs = [
        (c, d, s, knob[c])
        for c in conds["condition"]
        for d in ("sugar", "sugar_bitter", "jon_ce")
        for s in SEEDS
    ]
    with ProcessPoolExecutor(memory_bounded_workers(per_worker_gb=12.0), initializer=_init) as pool:
        brains = list(pool.map(_brain, jobs))
    log.info("brain trials done: %d", len(brains))
    with ProcessPoolExecutor(memory_bounded_workers(per_worker_gb=4.0)) as pool:
        rows = list(pool.map(_proboscis, [b for b in brains if b["drive"] != "jon_ce"]))
    fw, male = relay._tables()
    pairs = relay.match(fw, male)
    jon = [b for b in brains if b["drive"] == "jon_ce"]
    with ProcessPoolExecutor(
        memory_bounded_workers(per_worker_gb=8.0), initializer=relay._cord_init, initargs=(pairs,)
    ) as pool:
        cords = list(pool.map(_cord, [(b, knob[b["condition"]]) for b in jon]))
    with ProcessPoolExecutor(12) as pool:
        rows += list(pool.map(_reach, cords))
    trials = pd.DataFrame(rows)
    trials["movement"] = trials["drive"].map(
        {
            "sugar": "proboscis_sugar",
            "sugar_bitter": "proboscis_sugar_bitter",
            "jon_ce": "reach_depth",
        }
    )
    medians = trials.groupby(["condition", "movement"])["value"].median().unstack()[list(MOVEMENTS)]
    predictions, details = score(conds, medians, spec)
    result = {
        "label": "SIMULATED drug knobs with movement as the readout",
        "predictions": predictions,
        **details,
        "medians": {
            c: {m: round(float(medians.loc[c, m]), 4) for m in MOVEMENTS} for c in medians.index
        },
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    trials.to_csv(out / "embodied_stage4a_SIMULATED.csv", index=False)
    (out / "embodied_stage4a_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage4a", scoring.clean(result))
    plot(conds, medians, figures_dir() / "embodied_stage4a_drugs_movement_SIMULATED.png")
    for name, entry in predictions.items():
        print(f"{name:52s} {entry['outcome']:>15}  {entry['value']}")
    print("onsets", json.dumps(scoring.clean(details["onsets"])))
    print("flagged in time", json.dumps(scoring.clean(details["flagged_in_time"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
