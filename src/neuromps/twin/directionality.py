"""HD16: can summary features tell symmetric from asymmetric connectivity?

The twin exists to say something about pathway direction, so the question is
whether direction is recoverable from MEA summaries *at all*, and at what cost
in electrodes and recording time. This runs on simulated data, where the wiring
is known, and reports discrimination as an AUC with a bootstrap interval.

Three design choices matter for reading the result:

- **Electrode count is varied by subsampling one fixed array**, not by shrinking
  the grid. A smaller grid would also shrink the chamber and move the neurons,
  which would conflate "fewer electrodes" with "different culture".
- **Recording length is varied by truncating one simulation**, so the 10 s and
  60 s rows describe the same network rather than different draws.
- **Three feature sets are scored separately.** HD11 showed coupling asymmetry
  tracks relative firing rate, so a discriminator can succeed by reading rate
  imbalance instead of direction. If `rate_only` scores as well as
  `coupling_only`, the task is not being solved by directional information.

All outputs are simulated and labelled as such.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.features.mea import extract_features
from neuromps.metrics import update_metrics
from neuromps.twin.inference import SUMMARY_NAMES, default_layout, summary_vector
from neuromps.twin.mea import (
    electrode_coupling,
    electrode_positions,
    place_neurons,
    record_electrodes,
)
from neuromps.twin.network import NetworkParams, simulate

log = logging.getLogger("neuromps.twin")
REPO_ROOT = Path(__file__).resolve().parents[3]
TWIN_DIR = REPO_ROOT / "outputs" / "twin"
GRID_FILE = TWIN_DIR / "hd16_directionality_simulated.parquet"
QUICK_GRID_FILE = TWIN_DIR / "hd16_directionality_simulated_quick.parquet"

SYMMETRIC_RATIO = 1.0
ASYMMETRIC_RATIOS = (3.0, 10.0)
DURATIONS_S = (10.0, 30.0, 60.0)
ELECTRODE_COUNTS = (6, 12, 24, 48)
AUC_TARGET = 0.8

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "all": tuple(SUMMARY_NAMES),
    "coupling_only": ("peak_correlation", "lag_ms", "asymmetry", "coupling_measured"),
    "rate_only": ("c0_mfr_hz", "c1_mfr_hz", "log_rate_ratio"),
}


def _sub_grid(rows: int, cols: int, count: int) -> tuple[int, int]:
    """Rows and columns of an evenly spaced sub-grid holding `count` electrodes."""
    options = [
        (r, count // r)
        for r in range(1, rows + 1)
        if count % r == 0 and count // r <= cols and r <= rows
    ]
    if not options:
        return rows, cols
    target = rows / cols
    return min(options, key=lambda rc: abs(rc[0] / rc[1] - target))


def subsample_electrodes(heard: dict[str, np.ndarray], per_compartment: int) -> dict:
    """Keep a spatially spread subset of electrodes from each compartment.

    An evenly spaced sub-grid is taken, so a 6-electrode array still spans the
    whole chamber the way a sparser real device would. Taking a corner of the
    array instead would confound electrode count with the area covered.
    """
    layout = default_layout()
    kept = {}
    for compartment in (0, 1):
        names = [k for k in heard if k.startswith(f"c{compartment}_")]
        if not names:
            continue
        if per_compartment >= len(names):
            kept.update({name: heard[name] for name in names})
            continue
        n_rows, n_cols = _sub_grid(layout.rows, layout.cols, per_compartment)
        rows = np.linspace(0, layout.rows - 1, n_rows).round().astype(int)
        cols = np.linspace(0, layout.cols - 1, n_cols).round().astype(int)
        for row in dict.fromkeys(rows.tolist()):
            for col in dict.fromkeys(cols.tolist()):
                name = f"c{compartment}_r{row}_k{col}"
                if name in heard:
                    kept[name] = heard[name]
    return kept


def truncate(heard: dict[str, np.ndarray], duration_s: float) -> dict:
    """Keep only the spikes in the first `duration_s` of the recording."""
    return {name: times[times < duration_s] for name, times in heard.items()}


def summarise(heard: dict[str, np.ndarray], duration_s: float) -> np.ndarray:
    """Summary vector for one recording, through the same path as the simulator."""
    features = {
        c: extract_features({k: v for k, v in heard.items() if k.startswith(f"c{c}_")}, duration_s)
        for c in (0, 1)
    }
    return summary_vector(features, electrode_coupling(heard, duration_s))


def _grid_job(args) -> list[dict]:
    """One simulation, summarised at every duration and electrode count."""
    ratio, rep, durations, electrode_counts, n_per_compartment = args
    layout = default_layout()
    params = NetworkParams(
        n_per_compartment=n_per_compartment,
        duration_s=max(durations),
        seed=1000 + rep,
        inter_in_degree=9.0,
        forward_ratio=ratio,
    )
    result = simulate(params)
    positions, compartment = place_neurons(params, layout, seed=1000 + rep)
    heard = record_electrodes(
        result.spike_times,
        result.spike_neurons,
        positions,
        compartment,
        electrode_positions(layout),
        layout,
    )
    rows = []
    for duration_s in durations:
        windowed = truncate(heard, duration_s)
        for count in electrode_counts:
            vector = summarise(subsample_electrodes(windowed, count), duration_s)
            rows.append(
                {
                    "forward_ratio": float(ratio),
                    "rep": int(rep),
                    "duration_s": float(duration_s),
                    "electrodes_per_compartment": int(count),
                    "symmetric": bool(ratio == SYMMETRIC_RATIO),
                    **dict(zip(SUMMARY_NAMES, vector, strict=True)),
                }
            )
    return rows


def run_grid(
    n_reps: int,
    ratios: tuple[float, ...],
    durations: tuple[float, ...],
    electrode_counts: tuple[int, ...],
    n_per_compartment: int = 150,
    processes: int | None = None,
) -> pd.DataFrame:
    """Simulate the grid in parallel, one network per (ratio, rep)."""
    import multiprocessing as mp

    jobs = [
        (ratio, rep, durations, electrode_counts, n_per_compartment)
        for ratio in ratios
        for rep in range(n_reps)
    ]
    processes = processes or max(1, min(mp.cpu_count() - 2, 16))
    log.info("simulating %d networks on %d processes", len(jobs), processes)
    with mp.Pool(processes) as pool:
        results = pool.map(_grid_job, jobs, chunksize=1)
    return pd.DataFrame([row for rows in results for row in rows])


def discriminate(
    frame: pd.DataFrame, features: tuple[str, ...], n_resamples: int = 1000, seed: int = 0
) -> dict:
    """Out-of-fold AUC for symmetric vs asymmetric, with a bootstrap interval.

    Rows are independent simulations, so this bootstraps rows directly - unlike
    the real-data analyses, where resampling has to be over devices.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = (~frame["symmetric"]).to_numpy(dtype=int)
    if len(np.unique(y)) < 2 or len(y) < 10:
        return {"auc": None, "note": "too few or single-class simulations"}
    x = frame[list(features)].to_numpy(dtype=float)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    predicted = cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, predicted))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(float(roc_auc_score(y[idx], predicted[idx])))
    low, high = np.percentile(draws, [2.5, 97.5]) if draws else (None, None)
    return {
        "auc": round(auc, 4),
        "auc_ci95": [round(float(low), 4), round(float(high), 4)],
        "n_simulations": int(len(y)),
        "bootstrap_resamples": int(n_resamples),
        "better_than_chance": bool(low is not None and low > 0.5),
    }


def sweep(frame: pd.DataFrame, n_resamples: int = 1000, seed: int = 0) -> list[dict]:
    """Discrimination for every (duration, electrode count, feature set) cell."""
    rows = []
    for duration_s in sorted(frame["duration_s"].unique()):
        for count in sorted(frame["electrodes_per_compartment"].unique()):
            cell = frame[
                (frame["duration_s"] == duration_s) & (frame["electrodes_per_compartment"] == count)
            ]
            for name, features in FEATURE_SETS.items():
                rows.append(
                    {
                        "duration_s": float(duration_s),
                        "electrodes_per_compartment": int(count),
                        "feature_set": name,
                        **discriminate(cell, features, n_resamples=n_resamples, seed=seed),
                    }
                )
    return rows


def cheapest_configuration(rows: list[dict], target: float = AUC_TARGET) -> dict | None:
    """Smallest recording that reaches the AUC target with an interval above chance.

    "Smallest" is ranked by recording seconds first, then electrode count: time
    on a rig is the scarcer resource in a screening context.
    """
    qualifying = [
        r
        for r in rows
        if r["feature_set"] == "all"
        and r.get("auc") is not None
        and r["auc"] >= target
        and r.get("better_than_chance")
    ]
    if not qualifying:
        return None
    return min(qualifying, key=lambda r: (r["duration_s"], r["electrodes_per_compartment"]))


def grid_config(
    n_reps: int,
    durations: tuple[float, ...],
    counts: tuple[int, ...],
    n_per_compartment: int,
) -> dict:
    """The configuration a cached grid was produced under.

    Cached alongside the grid so a run cannot silently inherit a grid built for
    a different question - the mistake the SBI cache made in HD15, where a
    posterior trained on one electrode layout was applied under another.
    """
    from neuromps.twin.inference import summary_pipeline_fingerprint

    return {
        "n_reps": int(n_reps),
        "ratios": [SYMMETRIC_RATIO, *ASYMMETRIC_RATIOS],
        "durations_s": [float(d) for d in durations],
        "electrode_counts": [int(c) for c in counts],
        "n_per_compartment": int(n_per_compartment),
        "summary_pipeline_sha256": summary_pipeline_fingerprint(),
    }


def cached_grid(path: Path, config: dict) -> pd.DataFrame | None:
    """The cached grid if it was built under exactly this configuration."""
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
        return None
    if json.loads(meta.read_text()) != config:
        log.info("cached grid was built under a different configuration; rebuilding")
        return None
    return pd.read_parquet(path)


def plot_sweep(rows: list[dict], path: Path, title: str) -> None:
    """AUC against recording length, one panel per feature set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {6: "#0072B2", 12: "#E69F00", 24: "#009E73", 48: "#CC79A7"}
    names = list(FEATURE_SETS)
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names, strict=True):
        for count in sorted({r["electrodes_per_compartment"] for r in rows}):
            cells = [
                r
                for r in rows
                if r["feature_set"] == name
                and r["electrodes_per_compartment"] == count
                and r.get("auc") is not None
            ]
            if not cells:
                continue
            cells.sort(key=lambda r: r["duration_s"])
            x = [r["duration_s"] for r in cells]
            y = [r["auc"] for r in cells]
            low = [r["auc_ci95"][0] for r in cells]
            high = [r["auc_ci95"][1] for r in cells]
            colour = colours.get(count, "#555555")
            ax.plot(x, y, "o-", color=colour, label=f"{count} electrodes")
            ax.fill_between(x, low, high, color=colour, alpha=0.15)
        ax.axhline(0.5, color="black", ls=":", lw=1)
        ax.axhline(AUC_TARGET, color="black", ls="--", lw=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("recording length (s)")
        ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel("AUC, symmetric vs asymmetric")
    axes[-1].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run the directionality grid and report what it takes to see direction."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="few reps, one recording length")
    p.add_argument("--reps", type=int, default=None, help="simulations per wiring condition")
    p.add_argument("--rebuild", action="store_true", help="ignore the cached grid")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    TWIN_DIR.mkdir(parents=True, exist_ok=True)

    n_reps = args.reps or (4 if args.quick else 40)
    durations = (10.0,) if args.quick else DURATIONS_S
    counts = (12, 48) if args.quick else ELECTRODE_COUNTS
    resamples = 200 if args.quick else 1000

    n_per_compartment = 80 if args.quick else 150
    path = QUICK_GRID_FILE if args.quick else GRID_FILE
    config = grid_config(n_reps, durations, counts, n_per_compartment)
    frame = None if args.rebuild else cached_grid(path, config)
    if frame is not None:
        log.info("reusing cached grid: %d rows", len(frame))
    else:
        frame = run_grid(
            n_reps,
            (SYMMETRIC_RATIO, *ASYMMETRIC_RATIOS),
            durations,
            counts,
            n_per_compartment=n_per_compartment,
        )
        frame.to_parquet(path, index=False)
        path.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n")

    rows = sweep(frame, n_resamples=resamples, seed=args.seed)
    cheapest = cheapest_configuration(rows)
    silent = float((frame["c0_mfr_hz"] == 0).mean())
    results = {
        "label": "SIMULATED - directionality discrimination, ground-truth wiring known",
        "question": "can MEA summary features tell symmetric from asymmetric connectivity",
        "symmetric_ratio": SYMMETRIC_RATIO,
        "asymmetric_ratios": list(ASYMMETRIC_RATIOS),
        "n_networks": int(frame.groupby(["forward_ratio", "rep"]).ngroups),
        "fraction_silent": round(silent, 3),
        "auc_target": AUC_TARGET,
        "cheapest_configuration_all_features": cheapest,
        "sweep": rows,
        "bootstrap_resamples": resamples,
        "quick": bool(args.quick),
        "grid_config": config,
    }
    plot_sweep(
        rows,
        TWIN_DIR / "hd16_directionality_simulated.png",
        "SIMULATED: telling symmetric from asymmetric wiring, by recording length\n"
        "dashed = AUC target 0.8, dotted = chance; shaded = 95% bootstrap interval",
    )
    update_metrics("twin_hd16_directionality", results)
    headline = [
        r for r in rows if r["feature_set"] == "all" and r["electrodes_per_compartment"] == 48
    ]
    print(
        json.dumps(
            {
                "n_networks": results["n_networks"],
                "cheapest_configuration_all_features": cheapest,
                "all_features_48_electrodes": headline,
                "rate_only_vs_coupling_only": [
                    r
                    for r in rows
                    if r["feature_set"] in ("rate_only", "coupling_only")
                    and r["electrodes_per_compartment"] == 48
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
