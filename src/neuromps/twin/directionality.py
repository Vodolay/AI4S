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
    ratio, rep, durations, electrode_counts, n_per_compartment, drive, arm = args
    layout = default_layout()
    params = NetworkParams(
        n_per_compartment=n_per_compartment,
        duration_s=max(durations),
        seed=1000 + rep,
        inter_in_degree=9.0,
        forward_ratio=ratio,
        background_rate_hz=drive["background_rate_hz"],
        background_scale_target=drive["background_scale_target"],
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
                    "arm": arm,
                    "background_rate_hz": float(drive["background_rate_hz"]),
                    "background_scale_target": float(drive["background_scale_target"]),
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
    drives: dict[float, dict[str, float]] | None = None,
    arm: str = "free_rate",
) -> pd.DataFrame:
    """Simulate the grid in parallel, one network per (ratio, rep)."""
    import multiprocessing as mp

    drives = drives or {}
    default_drive = {"background_rate_hz": 10.0, "background_scale_target": 1.0}
    jobs = [
        (
            ratio,
            rep,
            durations,
            electrode_counts,
            n_per_compartment,
            drives.get(ratio, default_drive),
            arm,
        )
        for ratio in ratios
        for rep in range(n_reps)
    ]
    processes = processes or max(1, min(mp.cpu_count() - 2, 16))
    log.info("simulating %d networks on %d processes", len(jobs), processes)
    with mp.Pool(processes) as pool:
        results = pool.map(_grid_job, jobs, chunksize=1)
    return pd.DataFrame([row for rows in results for row in rows])


def _classifier(kind: str, seed: int):
    """The discriminator. `linear` is the interpretable baseline, `nonlinear` the ceiling.

    Reporting only a linear model would confuse "the features do not carry
    direction" with "a linear boundary does not separate them", so both are run
    wherever the full feature set is scored.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "nonlinear":
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def discriminate(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    n_resamples: int = 1000,
    seed: int = 0,
    kind: str = "linear",
) -> dict:
    """Out-of-fold AUC for symmetric vs asymmetric, with a bootstrap interval.

    Rows are independent simulations, so this bootstraps rows directly - unlike
    the real-data analyses, where resampling has to be over devices.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    y = (~frame["symmetric"]).to_numpy(dtype=int)
    if len(np.unique(y)) < 2 or len(y) < 10:
        return {"auc": None, "note": "too few or single-class simulations"}
    x = frame[list(features)].to_numpy(dtype=float)
    model = _classifier(kind, seed)
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
        "classifier": kind,
    }


RATE_COLUMNS = ("c0_mfr_hz", "c1_mfr_hz")
# Four feature sets fit legibly at text width; eight did not. These are the ones
# the report argues from, including the set the validity check is computed on.
PLOTTED = ("all", "rate_only", "all_rate_residualized", "rate_only_rate_residualized")


def residualize(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Remove the part of each feature that both chambers' firing rates explain.

    Drive-matching turned out to be unreliable: every setting that fixed one
    chamber's rate moved the other's, and a classifier reads whatever mismatch
    is left. This needs no intervention in the simulation. Each feature is
    regressed on `c0_mfr_hz` and `c1_mfr_hz` across the cell and replaced by its
    residual, so a discriminator that still succeeds is not using rate.

    It removes *linear* dependence only; a feature related to rate in a curved
    way keeps part of that relation.
    """
    design = np.column_stack(
        [np.ones(len(frame))] + [frame[c].to_numpy(dtype=float) for c in RATE_COLUMNS]
    )
    out = frame.copy()
    for name in features:
        y = frame[name].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        # A feature the rates explain exactly leaves floating-point dust, which a
        # standardising classifier rescales to unit variance and can separate
        # perfectly. Anything that small is zero.
        if np.std(residual) <= 1e-8 * max(np.std(y), 1e-12):
            residual = np.zeros_like(residual)
        out[name] = residual
    return out


SCORED = (
    ("all", "linear", False),
    ("all", "nonlinear", False),
    ("coupling_only", "linear", False),
    ("rate_only", "linear", False),
    ("all", "linear", True),
    ("coupling_only", "linear", True),
    ("rate_only", "linear", True),
)


def sweep(frame: pd.DataFrame, n_resamples: int = 1000, seed: int = 0) -> list[dict]:
    """Discrimination for every (duration, electrode count, feature set) cell."""
    if "arm" in frame.columns and frame["arm"].nunique() > 1:
        rows = []
        for arm in sorted(frame["arm"].unique()):
            for row in sweep(frame[frame["arm"] == arm], n_resamples=n_resamples, seed=seed):
                rows.append({**row, "arm": arm})
        return rows
    rows = []
    for duration_s in sorted(frame["duration_s"].unique()):
        for count in sorted(frame["electrodes_per_compartment"].unique()):
            cell = frame[
                (frame["duration_s"] == duration_s) & (frame["electrodes_per_compartment"] == count)
            ]
            for name, kind, residual in SCORED:
                label = name if kind == "linear" else f"{name}_{kind}"
                data = residualize(cell, FEATURE_SETS[name]) if residual else cell
                rows.append(
                    {
                        "duration_s": float(duration_s),
                        "electrodes_per_compartment": int(count),
                        "feature_set": f"{label}_rate_residualized" if residual else label,
                        "rate_residualized": bool(residual),
                        **discriminate(
                            data,
                            FEATURE_SETS[name],
                            n_resamples=n_resamples,
                            seed=seed,
                            kind=kind,
                        ),
                    }
                )
    return rows


def rate_imbalance_by_arm(frame: pd.DataFrame) -> dict:
    """Median rate imbalance per arm and wiring condition, to show matching worked."""
    out: dict[str, dict[str, float]] = {}
    for arm in sorted(frame.get("arm", pd.Series(["free_rate"])).unique()):
        subset = frame[frame["arm"] == arm] if "arm" in frame.columns else frame
        out[arm] = {
            str(ratio): round(float(np.median(part["log_rate_ratio"])), 4)
            for ratio, part in subset.groupby("forward_ratio")
        }
    return out


def summary_by_feature_set(rows: list[dict]) -> dict:
    """Range of AUC across configurations, per arm and feature set.

    The report's claims are of the form "rate alone reaches 0.86-0.98", which
    needs a stable key to cite; the per-cell rows are positional and would break
    the moment the grid changes.
    """
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row.get("auc") is None:
            continue
        arm = out.setdefault(row.get("arm", "free_rate"), {})
        cell = arm.setdefault(row["feature_set"], {"min": 1.0, "max": 0.0, "n_cells": 0})
        cell["min"] = round(min(cell["min"], row["auc"]), 4)
        cell["max"] = round(max(cell["max"], row["auc"]), 4)
        cell["n_cells"] += 1
        cell["all_better_than_chance"] = bool(
            cell.get("all_better_than_chance", True) and row["better_than_chance"]
        )
    return out


def cheapest_configuration(
    rows: list[dict],
    target: float = AUC_TARGET,
    arm: str = "rate_matched",
    feature_set: str = "all_rate_residualized",
) -> dict | None:
    """Smallest recording whose AUC *interval* clears the target.

    The lower bound has to clear the target, not the point estimate: with 120
    simulations per cell the estimate wanders by about 0.1, and a design
    recommendation drawn from the optimistic end of that wander would not
    reproduce. "Smallest" is ranked by recording seconds first, then electrode
    count, because time on a rig is the scarcer resource in screening.
    """
    qualifying = [
        r
        for r in rows
        if r["feature_set"] == feature_set
        and r.get("arm", arm) == arm
        and r.get("auc") is not None
        and r.get("auc_ci95")
        and r["auc_ci95"][0] >= target
    ]
    if not qualifying:
        return None
    return min(qualifying, key=lambda r: (r["duration_s"], r["electrodes_per_compartment"]))


def rate_matching_drives(
    ratios: tuple[float, ...],
    global_rates: tuple[float, ...] = (10.0, 11.0, 12.0),
    target_scales: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.75, 0.65),
    n_reps: int = 8,
    duration_s: float = 10.0,
    n_per_compartment: int = 150,
) -> dict[float, dict[str, float]]:
    """Background drive that makes each chamber fire at the symmetric condition's rate.

    An asymmetric pathway does two things at once: it carries influence one way,
    and it changes how much excitation each chamber receives, which moves both
    chambers' firing rates. A discriminator can then separate the conditions
    from rate alone, without using any directional information - HD11 found the
    same confound in the coupling measure, and the free-rate arm below shows it
    again.

    Matching only the *ratio* is not enough: it leaves the absolute rates shifted,
    which is just as readable. Both knobs are therefore searched together, and
    both chambers' median rates are matched to the symmetric reference.
    """
    reference = None
    chosen: dict[float, dict[str, float]] = {}
    for ratio in ratios:
        if ratio == SYMMETRIC_RATIO:
            frame = _probe(ratio, 10.0, 1.0, n_reps, duration_s, n_per_compartment)
            reference = (
                float(np.median(frame["c0_mfr_hz"])),
                float(np.median(frame["c1_mfr_hz"])),
            )
            chosen[ratio] = {"background_rate_hz": 10.0, "background_scale_target": 1.0}
            continue
        best, best_distance = None, np.inf
        for rate in global_rates:
            for scale in target_scales:
                frame = _probe(ratio, rate, scale, n_reps, duration_s, n_per_compartment)
                got = (float(np.median(frame["c0_mfr_hz"])), float(np.median(frame["c1_mfr_hz"])))
                distance = sum(
                    abs(g - r) / max(r, 1e-6) for g, r in zip(got, reference, strict=True)
                )
                log.info(
                    "  ratio %.1f drive %.1f/%.2f -> rates %.2f/%.2f (target %.2f/%.2f)",
                    ratio,
                    rate,
                    scale,
                    got[0],
                    got[1],
                    reference[0],
                    reference[1],
                )
                if distance < best_distance:
                    best, best_distance = (
                        {
                            "background_rate_hz": rate,
                            "background_scale_target": scale,
                        },
                        distance,
                    )
        chosen[ratio] = best
    return chosen


def _probe(
    ratio: float,
    background_rate_hz: float,
    target_scale: float,
    n_reps: int,
    duration_s: float,
    n_per_compartment: int,
) -> pd.DataFrame:
    """A few short simulations at one drive setting, for the matching search."""
    return run_grid(
        n_reps,
        (ratio,),
        (duration_s,),
        (default_layout().electrodes_per_compartment,),
        n_per_compartment=n_per_compartment,
        drives={
            ratio: {
                "background_rate_hz": background_rate_hz,
                "background_scale_target": target_scale,
            }
        },
    )


def matching_quality(frame: pd.DataFrame) -> dict:
    """How closely the matched arm reproduces the symmetric condition's rates.

    Reported so the arm can be dismissed when matching failed, rather than read
    as a result about wiring.
    """
    out = {}
    for arm, part in frame.groupby("arm"):
        reference = part[part["forward_ratio"] == SYMMETRIC_RATIO]
        if reference.empty:
            continue
        ref = {c: float(np.median(reference[c])) for c in ("c0_mfr_hz", "c1_mfr_hz")}
        per_ratio, worst = {}, 0.0
        for ratio, cell in part.groupby("forward_ratio"):
            if ratio == SYMMETRIC_RATIO:
                continue
            gaps = {
                column: round(abs(float(np.median(cell[column])) - value) / max(value, 1e-6), 4)
                for column, value in ref.items()
            }
            per_ratio[str(ratio)] = gaps
            worst = max(worst, *gaps.values())
        out[str(arm)] = {
            "worst_relative_rate_gap": round(worst, 4),
            "relative_rate_gap_by_ratio": per_ratio,
        }
    return out


def grid_config(
    n_reps: int,
    durations: tuple[float, ...],
    counts: tuple[int, ...],
    n_per_compartment: int,
    drive_scales: dict | None = None,
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
        "rate_matched_drives": {str(k): v for k, v in (drive_scales or {}).items()},
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
    names = [n for n in PLOTTED if any(r["feature_set"] == n for r in rows)]
    arms = sorted({r.get("arm", "free_rate") for r in rows})
    fig, axes = plt.subplots(
        len(arms),
        len(names),
        figsize=(3.6 * len(names), 3.2 * len(arms)),
        sharey=True,
        squeeze=False,
    )
    for row_index, arm in enumerate(arms):
        for ax, name in zip(axes[row_index], names, strict=True):
            _panel(ax, rows, name, arm, colours)
            ax.set_title(f"{name}\n{arm}", fontsize=10)
            ax.tick_params(labelsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("recording length (s)", fontsize=10)
    for row in axes:
        row[0].set_ylabel("AUC", fontsize=10)
    axes[0][-1].legend(fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _panel(ax, rows: list[dict], name: str, arm: str, colours: dict) -> None:
    """One AUC-vs-duration panel for one feature set and one arm."""
    for count in sorted({r["electrodes_per_compartment"] for r in rows}):
        cells = [
            r
            for r in rows
            if r["feature_set"] == name
            and r.get("arm", "free_rate") == arm
            and r["electrodes_per_compartment"] == count
            and r.get("auc") is not None
        ]
        if not cells:
            continue
        cells.sort(key=lambda r: r["duration_s"])
        colour = colours.get(count, "#555555")
        x = [r["duration_s"] for r in cells]
        ax.plot(x, [r["auc"] for r in cells], "o-", color=colour, label=f"{count} electrodes")
        ax.fill_between(
            x,
            [r["auc_ci95"][0] for r in cells],
            [r["auc_ci95"][1] for r in cells],
            color=colour,
            alpha=0.15,
        )
    ax.axhline(0.5, color="black", ls=":", lw=1)
    ax.axhline(AUC_TARGET, color="black", ls="--", lw=1)
    ax.set_ylim(0.3, 1.02)


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
    ratios = (SYMMETRIC_RATIO, *ASYMMETRIC_RATIOS)
    neutral = {"background_rate_hz": 10.0, "background_scale_target": 1.0}
    drives = (
        {r: neutral for r in ratios}
        if args.quick
        else rate_matching_drives(ratios, n_per_compartment=n_per_compartment)
    )
    log.info("rate-matching drives: %s", drives)
    config = grid_config(n_reps, durations, counts, n_per_compartment, drives)
    frame = None if args.rebuild else cached_grid(path, config)
    if frame is not None:
        log.info("reusing cached grid: %d rows", len(frame))
    else:
        free = run_grid(
            n_reps,
            ratios,
            durations,
            counts,
            n_per_compartment=n_per_compartment,
            arm="free_rate",
        )
        matched = run_grid(
            n_reps,
            ratios,
            durations,
            counts,
            n_per_compartment=n_per_compartment,
            drives=drives,
            arm="rate_matched",
        )
        frame = pd.concat([free, matched], ignore_index=True)
        frame.to_parquet(path, index=False)
        path.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n")

    rows = sweep(frame, n_resamples=resamples, seed=args.seed)
    cheapest = {
        "rate_matched_residualized": cheapest_configuration(rows, arm="rate_matched"),
        "free_rate_residualized": cheapest_configuration(rows, arm="free_rate"),
        "free_rate_all_features": cheapest_configuration(rows, arm="free_rate", feature_set="all"),
    }
    silent = float((frame["c0_mfr_hz"] == 0).mean())
    results = {
        "label": "SIMULATED - directionality discrimination, ground-truth wiring known",
        "question": "can MEA summary features tell symmetric from asymmetric connectivity",
        "symmetric_ratio": SYMMETRIC_RATIO,
        "asymmetric_ratios": list(ASYMMETRIC_RATIOS),
        "n_networks": int(frame.groupby(["forward_ratio", "rep"]).ngroups),
        "fraction_silent": round(silent, 3),
        "auc_target": AUC_TARGET,
        "cheapest_configuration": cheapest,
        "summary_by_feature_set": summary_by_feature_set(rows),
        "rate_imbalance_by_arm": rate_imbalance_by_arm(frame),
        "matching_quality": matching_quality(frame),
        "rate_only_is_the_validity_check": (
            "in the rate-matched arm the rate_only AUC should fall towards chance; if it does "
            "not, the matching failed and that arm says nothing about direction"
        ),
        "arms": {
            "free_rate": "asymmetric wiring left to raise the target chamber's firing rate",
            "rate_matched": "target drive lowered so the rate imbalance matches symmetric",
        },
        "sweep": rows,
        "bootstrap_resamples": resamples,
        "quick": bool(args.quick),
        "grid_config": config,
    }
    plot_sweep(
        rows,
        TWIN_DIR / "hd16_directionality_simulated.png",
        "SIMULATED: telling symmetric from asymmetric wiring, by recording length\n"
        "top row: drive left alone; bottom row: drive adjusted to match rates, imperfectly\n"
        "(see matching_quality); rate_residualized is the controlled comparison\n"
        "dashed = AUC target 0.8, dotted = chance; shaded = 95% bootstrap interval",
    )
    update_metrics("twin_hd16_directionality", results)
    print(
        json.dumps(
            {
                "n_networks": results["n_networks"],
                "rate_imbalance_by_arm": results["rate_imbalance_by_arm"],
                "matching_quality": results["matching_quality"],
                "cheapest_configuration": cheapest,
                "summary_by_feature_set": summary_by_feature_set(rows),
                "validity_check_rate_only_residualized": [
                    {k: r[k] for k in ("arm", "duration_s", "auc", "better_than_chance")}
                    for r in rows
                    if r["feature_set"] == "rate_only_rate_residualized"
                    and r["electrodes_per_compartment"] == 48
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
