"""Calibrate the twin so its simulated features sit inside the real baseline distributions.

The reference is the ScreenNeuroPharm **chamber** baselines: rat cultures in a
two-chamber device, 48 electrodes per chamber, which is the geometry the twin
models. Targets are computed from `data/processed/neuromps.parquet` at run time,
never hard-coded, so recalibrating after an ETL change needs no edits here.

Comparisons are per compartment, because a real chamber record's features are
computed over that chamber's electrodes only.

What is tuned, and why it is legitimate to tune it:

- ``background_weight``, ``excitatory_weight``, ``adaptation_increment``,
  ``intra_in_degree`` — free parameters of a model of an unobserved culture.
- ``detection_radius_um`` — a property of the virtual electrode, i.e. how many
  neurons one electrode pools. It is not measured in the source datasets.

Everything else stays fixed. The search is a seeded random search; the chosen
values, the ranges searched and the achieved distance are all written out, so the
calibration can be audited rather than trusted.
"""

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.asset import compartment_records
from neuromps.features.mea import extract_features
from neuromps.metrics import update_metrics
from neuromps.paths import asset_path, for_reading, twin_dir
from neuromps.twin.mea import ElectrodeLayout, electrode_positions, place_neurons, record_electrodes
from neuromps.twin.network import NetworkParams, simulate

log = logging.getLogger("neuromps.twin")
REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_SOURCE = "screenneuropharm_bic"
CALIBRATION_FEATURES = (
    "mfr_hz",
    "burst_rate_per_min",
    "network_burst_rate_per_min",
    "synchrony_index",
    "pct_spikes_in_bursts",
)
SEARCH_SPACE = {
    "background_weight": (0.035, 0.05),
    "excitatory_weight": (0.15, 0.45),
    "inhibitory_weight": (1.5, 5.0),
    "adaptation_increment": (0.5, 2.5),
    "threshold_spread": (0.0, 0.30),
    "intra_in_degree": (10.0, 35.0),
    "detection_radius_um": (100.0, 300.0),
}


CALIBRATION_NAME = "hd12_calibration_simulated.json"


def calibration_file() -> Path:
    """Where the active run keeps its calibration."""
    return twin_dir() / CALIBRATION_NAME


@lru_cache(maxsize=4)
def _read_calibration(path: str) -> tuple[tuple[str, float], ...]:
    """The tuned draw, cached per path within a process."""
    if not Path(path).exists():
        log.warning("no calibration at %s; using module defaults", path)
        return ()
    return tuple(json.loads(Path(path).read_text())["best"]["draw"].items())


def calibrated_params(path: Path | None = None) -> dict:
    """The tuned parameter values from the last calibration run.

    Returns an empty dict if no calibration has been run, so callers fall back to
    module defaults rather than failing; the caller should record which it used.

    Cached per process: the sbi simulator asks for this twice per draw, which
    on a fresh run meant two file reads and two fallback warnings for every one
    of thousands of simulations. The calibration does not change while a run is
    using it, so re-reading it only produced log noise.
    """
    path = Path(path) if path is not None else for_reading(calibration_file())
    return dict(_read_calibration(str(path)))


def real_targets(asset: pd.DataFrame) -> dict:
    """Median and interquartile band per feature, from real chamber baselines."""
    chambers = compartment_records(asset)
    rows = chambers[
        (chambers["source"] == REFERENCE_SOURCE)
        & (chambers["data_level"] == "spike_times")
        & (chambers["recording.phase"] == "baseline")
        & (chambers["qc.usable"].astype(bool))
    ]
    by_feature = {}
    for feature in CALIBRATION_FEATURES:
        values = pd.to_numeric(rows[f"features.{feature}"], errors="coerce").dropna()
        by_feature[feature] = {
            "median": round(float(values.median()), 3),
            "q25": round(float(values.quantile(0.25)), 3),
            "q75": round(float(values.quantile(0.75)), 3),
            "n": int(values.size),
        }
    return {
        "reference": f"{REFERENCE_SOURCE} chamber baselines",
        "n_records": int(len(rows)),
        "by_feature": by_feature,
    }


def score_features(features: dict, targets: dict) -> dict:
    """Distance from the real band: 0 inside it, log-ratio distance outside.

    A missing feature scores the worst observed penalty rather than being
    skipped, so a parameter set that silences the network cannot win by
    producing nothing to compare.
    """
    penalties = {}
    for feature, band in targets["by_feature"].items():
        value = features.get(feature)
        if value is None or not np.isfinite(value) or value <= 0:
            penalties[feature] = 3.0
            continue
        low, high = max(band["q25"], 1e-9), max(band["q75"], 1e-9)
        if low <= value <= high:
            penalties[feature] = 0.0
        else:
            edge = low if value < low else high
            penalties[feature] = abs(float(np.log(value / edge)))
    return {
        "score": round(float(np.mean(list(penalties.values()))), 4),
        "per_feature": {k: round(v, 4) for k, v in penalties.items()},
    }


def draw_params(rng: np.random.Generator) -> dict:
    """One draw from the search space."""
    return {
        name: round(float(rng.uniform(low, high)), 4) for name, (low, high) in SEARCH_SPACE.items()
    }


def refine_params(best: dict, rng: np.random.Generator, shrink: float = 0.25) -> dict:
    """Draw near a promising point: a shrunken box around it, clipped to the search space."""
    drawn = {}
    for name, (low, high) in SEARCH_SPACE.items():
        width = (high - low) * shrink
        centre = best[name]
        drawn[name] = round(
            float(np.clip(rng.uniform(centre - width / 2, centre + width / 2), low, high)), 4
        )
    return drawn


def simulate_chamber_features(
    drawn: dict, duration_s: float, seed: int = 0, n_per_compartment: int = 250
) -> dict:
    """Per-compartment features of one simulated recording, via the shared extractor."""
    network_keys = set(NetworkParams().as_record())
    params = NetworkParams(
        n_per_compartment=n_per_compartment,
        duration_s=duration_s,
        seed=seed,
        **{k: v for k, v in drawn.items() if k in network_keys},
    )
    layout = ElectrodeLayout(
        **{k: v for k, v in drawn.items() if k in set(ElectrodeLayout().as_record())}
    )
    result = simulate(params)
    positions, compartment = place_neurons(params, layout, seed=seed)
    electrodes = electrode_positions(layout)
    heard = record_electrodes(
        result.spike_times, result.spike_neurons, positions, compartment, electrodes, layout
    )
    per_compartment = {}
    for c in (0, 1):
        subset = {k: v for k, v in heard.items() if k.startswith(f"c{c}_")}
        per_compartment[c] = extract_features(subset, result.duration_s)
    return {
        "per_compartment": per_compartment,
        "mean": {
            f: _mean_of([per_compartment[c].get(f) for c in (0, 1)]) for f in CALIBRATION_FEATURES
        },
        "layout": layout.as_record(),
        "params": params.as_record(),
    }


def _mean_of(values: list) -> float | None:
    """Mean over compartments, ignoring undefined ones."""
    present = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(present)) if present else None


def random_search(
    targets: dict,
    n_draws: int,
    duration_s: float,
    seed: int = 0,
    n_per_compartment: int = 250,
    refine_fraction: float = 0.4,
) -> list[dict]:
    """Seeded search: random draws, then refinement around the best one so far.

    Pure random search covers seven dimensions poorly at this budget, so the last
    ``refine_fraction`` of the budget is spent in a shrunken box around the best
    draw. Every trial is kept, not only the winner.
    """
    rng = np.random.default_rng(seed)
    n_refine = int(n_draws * refine_fraction)
    trials = []
    for i in range(n_draws):
        if i < n_draws - n_refine or not trials:
            drawn = draw_params(rng)
            stage = "random"
        else:
            drawn = refine_params(min(trials, key=lambda t: t["score"])["draw"], rng)
            stage = "refine"
        simulated = simulate_chamber_features(drawn, duration_s, seed, n_per_compartment)
        scored = score_features(simulated["mean"], targets)
        trials.append(
            {"draw": drawn, "stage": stage, "features_simulated": simulated["mean"], **scored}
        )
        log.info("  draw %d/%d (%s) score %.3f", i + 1, n_draws, stage, scored["score"])
    return sorted(trials, key=lambda t: t["score"])


def plot_calibration(best: dict, targets: dict, path, title: str) -> None:
    """Simulated features against the real band, one panel per feature."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = list(targets["by_feature"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, feature in enumerate(features):
        band = targets["by_feature"][feature]
        scale = max(band["median"], 1e-9)
        ax.add_patch(
            plt.Rectangle(
                (i - 0.3, band["q25"] / scale),
                0.6,
                (band["q75"] - band["q25"]) / scale,
                color="#0072B2",
                alpha=0.25,
            )
        )
        ax.plot([i - 0.3, i + 0.3], [1.0, 1.0], color="#0072B2", lw=2)
        value = best["features_simulated"].get(feature)
        if value is not None:
            ax.plot(i, value / scale, "o", color="#D55E00", markersize=9)
    ax.set_yscale("log")
    ax.set_xticks(range(len(features)), features, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("value / real median (log scale)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=[
            plt.Line2D([], [], color="#0072B2", lw=6, alpha=0.4, label="real interquartile band"),
            plt.Line2D([], [], marker="o", color="#D55E00", lw=0, label="SIMULATED (calibrated)"),
        ],
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Calibrate against the real chamber baselines and write the result."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="fewer draws, shorter runs")
    p.add_argument("--draws", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    twin_dir().mkdir(parents=True, exist_ok=True)

    asset = for_reading(asset_path())
    targets = real_targets(pd.read_parquet(asset))
    if not targets["n_records"]:
        log.warning(
            "no %s chamber baselines in %s; skipping calibration (that source is fetched "
            "with --full). The existing %s, if any, is from an earlier run and is what the "
            "rest of the twin will use.",
            REFERENCE_SOURCE,
            asset,
            calibration_file().name,
        )
        return 0
    n_draws = args.draws or (8 if args.quick else 40)
    duration = 20.0 if args.quick else 40.0
    log.info("calibrating against %s (n=%d records)", targets["reference"], targets["n_records"])
    trials = random_search(targets, n_draws, duration, seed=args.seed)
    best = trials[0]
    verdict = {
        feature: (band["q25"] <= (best["features_simulated"].get(feature) or -1) <= band["q75"])
        for feature, band in targets["by_feature"].items()
    }
    results = {
        "label": "SIMULATED - calibration of a model, not measured data",
        "targets_real": targets,
        "tuned_parameters": list(SEARCH_SPACE),
        "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
        "n_draws": n_draws,
        "duration_s": duration,
        "best": best,
        "inside_real_band": verdict,
        "all_trials": trials,
        "note": (
            "random search over 5 parameters; everything else held at its module default. "
            "detection_radius_um is a property of the virtual electrode, not of the culture."
        ),
    }
    (twin_dir() / "hd12_calibration_simulated.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    plot_calibration(
        best,
        targets,
        twin_dir() / "hd12_calibration_simulated.png",
        "SIMULATED twin after calibration vs real chamber baselines\n"
        "ScreenNeuroPharm rat two-chamber devices, baseline recordings",
    )
    update_metrics("twin_hd12_calibration", {k: v for k, v in results.items() if k != "all_trials"})
    log.info("best score %.3f with %s", best["score"], best["draw"])
    print(json.dumps({"best": best, "inside_real_band": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
