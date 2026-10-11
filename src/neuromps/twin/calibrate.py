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
from neuromps.parallel import worker_processes
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
# Features on a bounded scale, with the support that scale spans. A log-ratio
# distance is right for a rate and wrong here: it compresses the whole top of a
# bounded range, so synchrony 0.96 reads as only 14% away from a band ending at
# 0.84 while a rate off by the same factor would dominate the score. That let
# the search buy rate accuracy with synchrony error, which is what it did. A
# logit distance on the feature's own support treats a move near the ceiling as
# the large change it is. STTC spans -1 to 1, so a negative value is a real
# answer - an anti-correlated network - not a failed run.
BOUNDED_FEATURES = {
    "synchrony_index": (-1.0, 1.0),
    "pct_spikes_in_bursts": (0.0, 100.0),
}
# Keeps the logit finite when a simulation lands on or past the support edge.
LOGIT_EPS = 1e-3
# The largest electrode radius at which neighbouring catchment discs do not
# overlap, so one neuron is heard by one electrode. Above it the same spike
# times enter several electrodes' trains and inflate pairwise STTC by
# construction - at the 295.9 um the previous search chose, each neuron was
# heard by 5.7 electrodes. Derived from the layout so the two cannot drift.
MAX_DETECTION_RADIUS_UM = ElectrodeLayout().pitch_um / 2
SEARCH_SPACE = {
    # Widened upward: with the electrode no longer able to pool 30 neurons, the
    # network has to produce the firing rate itself, and independent background
    # drive is the one lever that raises rate while lowering synchrony.
    # The floor matters as much as the ceiling now that the neuron count is
    # searched: an electrode pooling 47 neurons needs each of them to fire
    # around 0.3 Hz, so the low-drive end has to be reachable or the whole
    # high-density region is scored on drive it would never actually use.
    "background_weight": (0.005, 0.12),
    "background_rate_hz": (1.0, 40.0),
    "excitatory_weight": (0.15, 0.45),
    # Widened: the previous search settled at 4.84 against a ceiling of 5.0.
    "inhibitory_weight": (1.0, 7.0),
    "adaptation_increment": (0.5, 2.5),
    "threshold_spread": (0.0, 0.30),
    "intra_in_degree": (10.0, 35.0),
    "detection_radius_um": (40.0, MAX_DETECTION_RADIUS_UM),
    # Held at 250 since the twin was written, which is 130 neurons/mm2 in a
    # 1.92 mm2 chamber against roughly 1000-3000 in a real dissociated culture.
    # At a radius that does not double-count neurons that leaves about four
    # neurons per electrode, so the previous search bought its firing rate by
    # inflating the electrode instead. Cell density is a property of an
    # unobserved culture, so it belongs in the search like the rest.
    "n_per_compartment": (250, 1500),
}
# Parameters that count things and must stay whole numbers.
INTEGER_PARAMS = ("n_per_compartment",)
# Every draw is scored on more than one network seed, so the search cannot win
# by finding a lucky realisation; `synchrony_index` varies most between seeds.
SCORE_SEEDS = (0, 1)
# A chosen value this close to its bound, as a fraction of the range, means the
# space is binding and the reported optimum is not an interior one.
BOUNDARY_FRACTION = 0.02
# Seeds the search never saw. Choosing the best of many draws on a handful of
# seeds fits the draw to those realisations as well as to the data, so the
# winner is re-run here and both scores are reported. The gap between them is
# the optimism, and a calibration that does not state it is overstating itself.
HELD_OUT_SEEDS = (2, 3, 4, 5, 6)


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


def _logit(value: float, support: tuple[float, float]) -> float:
    """Position on a bounded scale, as a log-odds."""
    low, high = support
    p = (value - low) / (high - low)
    p = min(max(p, LOGIT_EPS), 1.0 - LOGIT_EPS)
    return float(np.log(p / (1.0 - p)))


def _penalty(feature: str, value: float, band: dict) -> float:
    """Distance from the band for one feature: 0 inside, positive outside."""
    support = BOUNDED_FEATURES.get(feature)
    if support is None:
        low, high = max(band["q25"], 1e-9), max(band["q75"], 1e-9)
        if low <= value <= high:
            return 0.0
        edge = low if value < low else high
        return abs(float(np.log(value / edge)))
    low, high = band["q25"], band["q75"]
    if low <= value <= high:
        return 0.0
    edge = low if value < low else high
    return abs(_logit(value, support) - _logit(edge, support))


def score_features(features: dict, targets: dict) -> dict:
    """Distance from the real band: 0 inside it, positive outside.

    Rates are scored on a log-ratio and bounded features on a logit distance
    over their own support; see `BOUNDED_FEATURES` for why the two differ.

    A missing feature scores the worst observed penalty rather than being
    skipped, so a parameter set that silences the network cannot win by
    producing nothing to compare. A bounded feature reaches that sentinel only
    when it is absent or not finite: unlike a rate, zero and below are values it
    can legitimately take.
    """
    penalties = {}
    for feature, band in targets["by_feature"].items():
        value = features.get(feature)
        bounded = feature in BOUNDED_FEATURES
        if value is None or not np.isfinite(value) or (value <= 0 and not bounded):
            penalties[feature] = 3.0
            continue
        penalties[feature] = _penalty(feature, float(value), band)
    return {
        "score": round(float(np.mean(list(penalties.values()))), 4),
        "per_feature": {k: round(v, 4) for k, v in penalties.items()},
    }


def _as_drawn(name: str, value: float) -> float | int:
    """Round a drawn value: whole numbers for counts, four decimals otherwise."""
    return int(round(value)) if name in INTEGER_PARAMS else round(float(value), 4)


def draw_params(rng: np.random.Generator) -> dict:
    """One draw from the search space."""
    return {
        name: _as_drawn(name, rng.uniform(low, high)) for name, (low, high) in SEARCH_SPACE.items()
    }


def refine_params(best: dict, rng: np.random.Generator, shrink: float = 0.25) -> dict:
    """Draw near a promising point: a shrunken box around it, clipped to the search space."""
    drawn = {}
    for name, (low, high) in SEARCH_SPACE.items():
        width = (high - low) * shrink
        centre = best[name]
        drawn[name] = _as_drawn(
            name, float(np.clip(rng.uniform(centre - width / 2, centre + width / 2), low, high))
        )
    return drawn


def simulate_chamber_features(
    drawn: dict, duration_s: float, seed: int = 0, n_per_compartment: int = 250
) -> dict:
    """Per-compartment features of one simulated recording, via the shared extractor."""
    network_keys = set(NetworkParams().as_record())
    # A drawn value wins over the default, so the search cannot report a size it
    # never simulated.
    params = NetworkParams(
        **{
            "n_per_compartment": n_per_compartment,
            "duration_s": duration_s,
            "seed": seed,
            **{k: v for k, v in drawn.items() if k in network_keys},
        }
    )
    layout = ElectrodeLayout(
        **{k: v for k, v in drawn.items() if k in set(ElectrodeLayout().as_record())}
    )
    # The same layout to both: the wiring rule and the electrodes must agree on
    # where the neurons are, or they describe different cultures.
    result = simulate(params, layout)
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


def boundary_hits(draw: dict, space: dict | None = None) -> list[str]:
    """Parameters whose chosen value sits on an edge of the space it was drawn from.

    A value at a bound means the search wanted to go further and could not, so
    the reported optimum is not an interior one and the space is part of the
    result. The previous calibration settled on two such values and nothing said
    so.
    """
    space = SEARCH_SPACE if space is None else space
    hits = []
    for name, (low, high) in space.items():
        value = draw.get(name)
        if value is None or high <= low:
            continue
        margin = (high - low) * BOUNDARY_FRACTION
        if value <= low + margin or value >= high - margin:
            hits.append(name)
    return hits


def summarise_held_out(runs: list[dict], selection_score: float, band: tuple) -> dict:
    """Compare the winner's score on unseen seeds with the score it was chosen on."""
    scored = [r for r in runs if r.get("score") is not None]
    values = [
        r["features"].get("synchrony_index")
        for r in scored
        if r["features"].get("synchrony_index") is not None
    ]
    if not scored:
        return {"n_seeds": 0, "score_mean": None, "selection_score": selection_score}
    scores = [r["score"] for r in scored]
    low, high = band
    return {
        "n_seeds": len(scored),
        "seeds": [r["seed"] for r in scored],
        "score_mean": round(float(np.mean(scores)), 4),
        "score_sd": round(float(np.std(scores, ddof=1)), 4) if len(scores) > 1 else 0.0,
        "selection_score": round(float(selection_score), 4),
        "optimism": round(float(np.mean(scores) - selection_score), 4),
        "synchrony_index_mean": round(float(np.mean(values)), 4) if values else None,
        "synchrony_index_sd": (round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0),
        "synchrony_index_in_band_fraction": (
            round(float(np.mean([low <= v <= high for v in values])), 4) if values else None
        ),
    }


def evaluate_held_out(
    draw: dict,
    targets: dict,
    duration_s: float,
    seeds: tuple[int, ...] = HELD_OUT_SEEDS,
    n_per_compartment: int = 250,
    processes: int | None = None,
) -> list[dict]:
    """Re-run one draw on seeds the search never scored it on."""
    import multiprocessing as mp

    jobs = [(0, draw, duration_s, seed, targets, n_per_compartment) for seed in seeds]
    with mp.Pool(processes or min(worker_processes(), len(jobs))) as pool:
        results = pool.map(_search_job, jobs, chunksize=1)
    return [{**r, "seed": seed} for r, seed in zip(results, seeds, strict=True)]


def _search_job(args) -> dict:
    """One draw at one seed. Module level so the pool can pickle it."""
    index, drawn, duration_s, seed, targets, n_per_compartment = args
    try:
        simulated = simulate_chamber_features(drawn, duration_s, seed, n_per_compartment)
    except Exception:  # noqa: BLE001 - one bad draw must not stop the search
        log.exception("calibration draw failed for %s", drawn)
        return {"index": index, "features": {}, "per_feature": {}, "score": None}
    scored = score_features(simulated["mean"], targets)
    return {"index": index, "features": simulated["mean"], **scored}


def _evaluate(
    draws: list[dict],
    stage: str,
    targets: dict,
    duration_s: float,
    seeds: tuple[int, ...],
    n_per_compartment: int,
    processes: int,
) -> list[dict]:
    """Score every draw on every seed in parallel, averaging over the seeds."""
    import multiprocessing as mp

    jobs = [
        (i, drawn, duration_s, seed, targets, n_per_compartment)
        for i, drawn in enumerate(draws)
        for seed in seeds
    ]
    with mp.Pool(processes) as pool:
        results = pool.map(_search_job, jobs, chunksize=1)
    trials = []
    for i, drawn in enumerate(draws):
        runs = [r for r in results if r["index"] == i and r["score"] is not None]
        if not runs:
            trials.append(
                {
                    "draw": drawn,
                    "stage": stage,
                    "features_simulated": {},
                    "score": 3.0,
                    "per_feature": {},
                    "n_seeds": 0,
                }
            )
            continue
        trials.append(
            {
                "draw": drawn,
                "stage": stage,
                "n_seeds": len(runs),
                "features_simulated": _mean_over_runs(runs, "features"),
                "per_feature": _mean_over_runs(runs, "per_feature"),
                "score": round(float(np.mean([r["score"] for r in runs])), 4),
            }
        )
    return trials


def _mean_over_runs(runs: list[dict], key: str) -> dict:
    """Mean of one result dictionary across seeds, ignoring undefined entries."""
    names = {n for r in runs for n in r[key]}
    out = {}
    for name in names:
        values = [
            r[key][name] for r in runs if r[key].get(name) is not None and np.isfinite(r[key][name])
        ]
        out[name] = round(float(np.mean(values)), 4) if values else None
    return out


def random_search(
    targets: dict,
    n_draws: int,
    duration_s: float,
    seed: int = 0,
    n_per_compartment: int = 250,
    refine_fraction: float = 0.4,
    seeds: tuple[int, ...] = SCORE_SEEDS,
    processes: int | None = None,
) -> list[dict]:
    """Seeded search: random draws, then refinement around the best one so far.

    Pure random search covers eight dimensions poorly at this budget, so the
    last ``refine_fraction`` is spent in shrunken boxes around the best draw.
    Refinement runs in waves rather than one draw at a time: a wave re-centres
    on the best result so far, which keeps the search sequential where that
    matters and parallel where it does not. Every trial is kept, not only the
    winner.
    """
    rng = np.random.default_rng(seed)
    processes = processes or worker_processes()
    n_refine = int(n_draws * refine_fraction)
    wave_size = max(1, min(processes, n_draws))
    log.info(
        "%d draws x %d seeds on %d processes (%d refined)", n_draws, len(seeds), processes, n_refine
    )
    trials = _evaluate(
        [draw_params(rng) for _ in range(n_draws - n_refine)],
        "random",
        targets,
        duration_s,
        seeds,
        n_per_compartment,
        processes,
    )
    remaining = n_refine
    while remaining > 0:
        wave = min(wave_size, remaining)
        centre = min(trials, key=lambda t: t["score"])["draw"]
        trials += _evaluate(
            [refine_params(centre, rng) for _ in range(wave)],
            "refine",
            targets,
            duration_s,
            seeds,
            n_per_compartment,
            processes,
        )
        remaining -= wave
        log.info("  best so far %.4f (%d trials)", min(t["score"] for t in trials), len(trials))
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
    n_draws = args.draws or (6 if args.quick else 160)
    duration = 12.0 if args.quick else 40.0
    seeds = SCORE_SEEDS[:1] if args.quick else SCORE_SEEDS
    log.info("calibrating against %s (n=%d records)", targets["reference"], targets["n_records"])
    trials = random_search(targets, n_draws, duration, seed=args.seed, seeds=seeds)
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
        "n_seeds_per_draw": len(seeds),
        "duration_s": duration,
        "best": best,
        "inside_real_band": verdict,
        "boundary_hits": boundary_hits(best["draw"]),
        "held_out": summarise_held_out(
            evaluate_held_out(best["draw"], targets, duration),
            selection_score=best["score"],
            band=(
                targets["by_feature"]["synchrony_index"]["q25"],
                targets["by_feature"]["synchrony_index"]["q75"],
            ),
        ),
        "all_trials": trials,
        "note": (
            f"random search over {len(SEARCH_SPACE)} parameters, each draw scored on "
            f"{len(seeds)} network seed(s); everything else held at its module default. "
            "detection_radius_um is a property of the virtual electrode, not of the "
            f"culture, and is capped at {MAX_DETECTION_RADIUS_UM:g} um - half the "
            "electrode pitch - so no neuron is heard by two electrodes."
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
    hits = results["boundary_hits"]
    if hits:
        log.warning(
            "the chosen draw sits on a search-space bound for %s; the optimum is not "
            "interior and the space is part of the result",
            ", ".join(hits),
        )
    held = results["held_out"]
    if held.get("optimism") is not None:
        log.info(
            "held-out score %.4f on %d unseen seeds vs %.4f at selection (optimism %.4f)",
            held["score_mean"],
            held["n_seeds"],
            held["selection_score"],
            held["optimism"],
        )
    log.info("best score %.3f with %s", best["score"], best["draw"])
    print(json.dumps({"best": best, "inside_real_band": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
