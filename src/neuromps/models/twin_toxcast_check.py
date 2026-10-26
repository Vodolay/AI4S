"""Out-of-sample check: the twin's drug knobs against ToxCast's real positive controls.

The calibrated twin (Act III) was tuned on one laboratory's baselines and its knobs
were checked (HD13) against two other laboratories' drug pairs. The EPA acute MEA
screen is a fourth setting it has never seen: rat cortical cultures on 48-well
Axion plates, with three positive controls run on almost every exposure date -
bicuculline and picrotoxin (GABA-A block) and tetrodotoxin (sodium-channel block).

For each, the real per-feature change is the well's log2 change from its own
baseline minus the median of the same date's vehicle wells, summarised over wells
with a bootstrap over dates. The twin's prediction is its own paired change
(the same simulated chip untreated and under the knob, fresh noise) through the
same ``features/mea.py`` path, over eight chips. Agreement is scored on direction
for the six features both share.

Registered in ``preregistration/twin_toxcast_positive_controls.json`` before the
twin was run and before any positive-control statistic was computed; the twin's
predictions are committed (``--predict``) before the real side is computed.
Research decision support only: this checks a model's directions, not any
compound's effect.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neuromps.models.toxcast_noise_floors import WELLS, log2_change, provenance
from neuromps.paths import REPO_ROOT, twin_dir

log = logging.getLogger("neuromps.models")
REGISTRATION = REPO_ROOT / "preregistration" / "twin_toxcast_positive_controls.json"
PREDICTIONS_NAME = "toxcast_positive_controls_predictions_simulated.json"
KNOBS = {"gabaa_block": {"gabaa_scale": 0.2}, "sodium_block": {"threshold_scale": 1.6}}
DRUGS = {
    "Bicuculline": {"knob": "gabaa_block"},
    "Picrotoxin": {"knob": "gabaa_block"},
    "Tetrodotoxin": {"knob": "sodium_block"},
}
MAPPING = {
    "mfr_hz": 2448,
    "burst_rate_per_min": 2449,
    "burst_duration_s": 2450,
    "spikes_per_burst": 2451,
    "pct_spikes_in_bursts": 2453,
    "synchrony_index": 2461,
}
BUILD_SEEDS = tuple(range(9101, 9109))
DURATION_S = 40.0
MIN_TWIN_CHANGE = 0.1
MIN_SIGN_SHARE = 7 / 8
MIN_TWIN_CHIPS = 5
MIN_REAL_WELLS = 10
RESAMPLES = 2_000
REAL_SEED = 20261028
PREDICTION_KEYS = {
    "P1_bicuculline": "Bicuculline",
    "P2_picrotoxin": "Picrotoxin",
    "P3_tetrodotoxin": "Tetrodotoxin",
}


def _sign(value: float) -> str | None:
    if value is None or not np.isfinite(value) or value == 0:
        return None
    return "up" if value > 0 else "down"


def twin_direction(changes: np.ndarray) -> dict:
    """The registered rule: |median| >= 0.1 log2 and >= 7/8 of defined chips agree."""
    changes = np.asarray(changes, dtype=float)
    defined = changes[np.isfinite(changes)]
    out = {"n_defined": int(defined.size), "median": None, "n_same_sign": 0, "direction": None}
    if defined.size == 0:
        return out
    median = float(np.median(defined))
    sign = _sign(median)
    same = int((np.sign(defined) == np.sign(median)).sum())
    out.update(median=round(median, 4), n_same_sign=same)
    if (
        defined.size >= MIN_TWIN_CHIPS
        and abs(median) >= MIN_TWIN_CHANGE
        and same >= MIN_SIGN_SHARE * defined.size
    ):
        out["direction"] = sign
    return out


def _simulate(job: tuple) -> dict:
    """One chip's paired change under one knob: untreated and treated, fresh noise."""
    from neuromps.models.twin_prior import _features, pair_deltas

    knob, build = job
    baseline = _features({}, build, build + 100_000)
    treated = _features(KNOBS[knob], build, build + 200_000)
    deltas = pair_deltas(treated, baseline)
    return {"knob": knob, "build_seed": build, **{f: deltas.get(f"delta.{f}") for f in MAPPING}}


def simulate_changes(processes: int, seeds=BUILD_SEEDS) -> pd.DataFrame:
    """Every registered (knob, chip) pair, simulated in parallel."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(knob, s) for knob in KNOBS for s in seeds]
    with ProcessPoolExecutor(processes) as pool:
        return pd.DataFrame(list(pool.map(_simulate, jobs)))


def twin_predictions(changes: pd.DataFrame) -> dict:
    """knob -> feature -> the registered direction rule applied over chips."""
    return {
        knob: {f: twin_direction(rows[f].to_numpy(dtype=float)) for f in MAPPING}
        for knob, rows in changes.groupby("knob")
    }


def vehicle_corrected(wells: pd.DataFrame, acid: int, spid: str) -> pd.DataFrame:
    """Each positive-control well's log2 change minus its date's vehicle median."""
    rows = wells[wells["acid"] == acid].assign(x=lambda d: log2_change(d["rval"].to_numpy()))
    rows = rows[np.isfinite(rows["x"])]
    vehicle = rows[rows["wllt"] == "n"].groupby("apid")["x"].median().rename("vehicle")
    treated = rows[(rows["wllt"] == "p") & (rows["spid"] == spid)].join(vehicle, on="apid")
    treated = treated[np.isfinite(treated["vehicle"])]
    return treated.assign(effect=treated["x"] - treated["vehicle"])[["apid", "effect"]]


def real_effect(effects: pd.DataFrame, n_resamples: int, seed: int) -> dict:
    """Median effect over wells with a 95% interval from resampling dates."""
    values = effects["effect"].to_numpy(dtype=float)
    labels, codes = np.unique(effects["apid"].to_numpy(), return_inverse=True)
    members = [values[codes == g] for g in range(labels.size)]
    rng = np.random.default_rng(seed)
    boot = [
        float(
            np.median(
                np.concatenate([members[g] for g in rng.integers(0, labels.size, labels.size)])
            )
        )
        for _ in range(n_resamples)
    ]
    low, high = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    median = float(np.median(values))
    established = values.size >= MIN_REAL_WELLS and (low > 0 or high < 0)
    return {
        "wells": int(values.size),
        "dates": int(labels.size),
        "median": round(median, 4),
        "median_fraction": round(2**median - 1, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "established": bool(established),
        "direction": _sign(median),
    }


def score_cells(twin: dict, real: dict, drugs: dict = DRUGS) -> dict:
    """Per drug and feature: both directions and whether they agree (None: not scored)."""
    out = {}
    for drug, spec in drugs.items():
        out[drug] = {}
        for feature, ref in real[drug].items():
            predicted = twin[spec["knob"]][feature]["direction"]
            scored = predicted is not None and ref["established"] and ref["direction"] is not None
            out[drug][feature] = {
                "twin_direction": predicted,
                "real_direction": ref["direction"],
                "real_established": ref["established"],
                "agrees": (predicted == ref["direction"]) if scored else None,
            }
    return out


def drug_summary(cells: dict) -> dict:
    """Scored and agreeing cells for one drug."""
    scored = [c["agrees"] for c in cells.values() if c["agrees"] is not None]
    return {"scored": len(scored), "agreeing": int(sum(scored))}


def evaluate_predictions(cells: dict) -> dict:
    """The registered pass/fail tests P1-P4."""
    out = {}
    for key, drug in PREDICTION_KEYS.items():
        s = drug_summary(cells[drug])
        ok = s["scored"] >= 3 and s["agreeing"] >= (2 / 3) * s["scored"]
        out[key] = {**s, "pass": bool(ok)}
    rate = {d: c["mfr_hz"]["agrees"] for d, c in cells.items() if c["mfr_hz"]["agrees"] is not None}
    out["P4_firing_rate"] = {"scored": rate, "pass": bool(rate) and all(rate.values())}
    return out


def magnitudes(twin: dict, real: dict, cells: dict) -> dict:
    """M1 (firing-rate rank across drugs) and M2 (feature rank within drug), descriptive."""
    gaba, sodium = (abs(twin[k]["mfr_hz"]["median"] or 0) for k in ("gabaa_block", "sodium_block"))
    ttx = abs(real["Tetrodotoxin"]["mfr_hz"]["median"])
    m1 = {
        "twin_sodium_over_gabaa": sodium > gaba,
        "real_ttx_over_bicuculline": ttx > abs(real["Bicuculline"]["mfr_hz"]["median"]),
        "real_ttx_over_picrotoxin": ttx > abs(real["Picrotoxin"]["mfr_hz"]["median"]),
    }
    m2 = {}
    for drug, spec in DRUGS.items():
        scored = [f for f, c in cells[drug].items() if c["agrees"] is not None]
        if len(scored) < 3:
            m2[drug] = {"features": len(scored), "spearman": None}
            continue
        t = [abs(twin[spec["knob"]][f]["median"]) for f in scored]
        r = [abs(real[drug][f]["median"]) for f in scored]
        rho = spearmanr(t, r).statistic
        m2[drug] = {"features": len(scored), "spearman": round(float(rho), 4)}
    return {"M1_drug_rank_on_firing_rate": m1, "M2_feature_rank_within_drug": m2}


COLOURS = {"twin": "#D55E00", "real": "#0072B2"}


def plot(twin: dict, real: dict, cells: dict, path: Path) -> None:
    """Twin median change against real median change, per drug and shared feature."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = list(MAPPING)
    x = np.arange(len(features))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (drug, spec) in zip(axes, DRUGS.items(), strict=True):
        t = [twin[spec["knob"]][f]["median"] for f in features]
        r = [real[drug][f]["median"] for f in features]
        ci = np.array([real[drug][f]["ci95"] for f in features])
        ax.axhline(0, color="grey", lw=0.6)
        ax.errorbar(
            x + 0.12,
            r,
            yerr=[np.subtract(r, ci[:, 0]), ci[:, 1] - r],
            fmt="o",
            ms=4,
            color=COLOURS["real"],
            label="REAL ToxCast wells (vs same-date vehicle)",
        )
        twin_values = [np.nan if v is None else v for v in t]
        ax.plot(
            x - 0.12,
            twin_values,
            "s",
            ms=4,
            color=COLOURS["twin"],
            label=f"SIMULATED twin, {spec['knob']} knob",
        )
        for i, f in enumerate(features):
            mark = {True: "agree", False: "differ", None: "n/s"}[cells[drug][f]["agrees"]]
            ax.text(i, 1.02, mark, transform=ax.get_xaxis_transform(), ha="center", fontsize=6)
        labels = [f.replace("_", " ") for f in features]
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=6.5)
        ax.set_title(drug, fontsize=8, pad=12)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("median log2 change", fontsize=7)
    axes[0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.suptitle(
        "SIMULATED twin knobs vs REAL ToxCast positive controls (rat cortex): direction check; "
        "n/s = not scored",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_or_simulate(predict: bool) -> dict:
    """The registered twin predictions: the committed file, or a fresh simulation."""
    from neuromps.parallel import worker_processes
    from neuromps.paths import for_reading

    path = twin_dir() / PREDICTIONS_NAME
    if not predict and for_reading(path).exists():
        return json.loads(for_reading(path).read_text())
    changes = simulate_changes(worker_processes(cap=16))
    record = {
        "label": "SIMULATED: the calibrated twin's paired changes under each knob, 8 chips",
        "registration": provenance(REGISTRATION),
        "knobs": KNOBS,
        "build_seeds": list(BUILD_SEEDS),
        "duration_s": DURATION_S,
        "per_chip": json.loads(changes.to_json(orient="records")),
        "predictions": twin_predictions(changes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def real_side(wells: pd.DataFrame) -> dict:
    """drug -> feature -> the registered real effect."""
    return {
        drug: {
            f: real_effect(vehicle_corrected(wells, acid, drug), RESAMPLES, REAL_SEED + acid)
            for f, acid in MAPPING.items()
        }
        for drug in DRUGS
    }


def main(argv: list[str] | None = None) -> int:
    """--predict: simulate and write the twin's predictions only; else score them."""
    from neuromps.metrics import update_metrics
    from neuromps.paths import figures_dir

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predict", action="store_true", help="simulate the twin's predictions only")
    p.add_argument("--quick", action="store_true", help="accepted for run_all; nothing changes")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.predict and not WELLS.exists():
        log.warning("%s not built; skipping the positive-control check", WELLS)
        return 0
    twin = load_or_simulate(args.predict)
    if args.predict:
        print(json.dumps(twin["predictions"], indent=1))
        return 0
    wells = pd.read_parquet(WELLS, columns=["acid", "wllt", "spid", "apid", "rval"])
    real = real_side(wells)
    cells = score_cells(twin["predictions"], real)
    results = {
        "label": "SIMULATED twin knobs vs REAL ToxCast acute MEA positive controls (directions)",
        "registration": provenance(REGISTRATION),
        "predictions_file": str(Path("outputs/twin") / PREDICTIONS_NAME),
        "twin": twin["predictions"],
        "real": real,
        "cells": cells,
        "by_drug": {d: drug_summary(c) for d, c in cells.items()},
        "predictions": evaluate_predictions(cells),
        **magnitudes(twin["predictions"], real, cells),
    }
    figures_dir().mkdir(parents=True, exist_ok=True)
    plot(twin["predictions"], real, cells, figures_dir() / "twin_toxcast_positive_controls.png")
    update_metrics("twin_toxcast_positive_controls", results)
    print(json.dumps({k: results[k] for k in ("by_drug", "predictions")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
