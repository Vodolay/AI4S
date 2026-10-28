"""Train and judge the registered estimators of the recalibration study (D31).

The registration is `preregistration/twin_inference_recalibration.json`; the
seeds, the truncation rule and the calibration statistics are in
`neuromps.twin.recalibration`. This module puts them together: it assembles the
training pools, trains (or reloads) M1, M2 and M3, evaluates them beside HD14's
saved posterior (M0) on the same fresh held-out simulations, scores each
registered prediction and writes the result under ``twin_hd14b_recalibration``.
Every number it writes is about SIMULATED data.
"""

import logging
from pathlib import Path

import numpy as np

from neuromps.paths import interim_dir, simulation_cache, twin_dir
from neuromps.twin import recalibration as rc
from neuromps.twin.inference import (
    PARAMETER_NAMES,
    POSTERIOR_NAME,
    _rank_uniformity,
    amortized_timing,
    check_cache_matches_code,
    estimator_architecture,
    holdout_size,
    load_simulations,
    within_prior,
)

log = logging.getLogger("neuromps.twin.recalibration")
FAILED_BEFORE = ("gabaa_scale", "inter_in_degree")
FIGURE_NAME = "hd14b_recalibration_simulated.png"
REGISTRATION_COMMITS = {"registered": "0bc29aa", "amended": "176b36e"}
# The quick tier trains to convergence too: on its few dozen simulations that takes
# seconds, while a flow stopped after a few epochs accepts ~0.1% of its proposals and
# rejection sampling from it outlasts the whole quick budget.
QUICK = {"members": 2, "max_epochs": None, "n_samples": 50, "n_eval": 10}


def models_dir() -> Path:
    """Trained posteriors of this study (git-ignored: they are rebuilt from the cache)."""
    return interim_dir() / "recalibration" / "models"


def hd14_pool() -> tuple[np.ndarray, np.ndarray]:
    """HD14's cached simulations, refused if the summary code has moved on."""
    cached = load_simulations(simulation_cache())
    check_cache_matches_code(cached)
    inside = within_prior(cached["theta"])
    return cached["theta"][inside], cached["x"][inside]


def training_pools(hd14: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """HD14's own training set, and the registered pool: all of HD14's plus every new chunk.

    ``hd14`` is HD14's pool when the caller has already loaded and checked it.
    """
    theta_old, x_old = hd14 if hd14 is not None else hd14_pool()
    new = rc.load_set(rc.TRAIN)
    n_train_old = len(theta_old) - holdout_size(len(theta_old))
    return {
        "hd14_train": (theta_old[:n_train_old], x_old[:n_train_old]),
        "pool": (np.vstack([theta_old, new["theta"]]), np.vstack([x_old, new["x"]])),
        "budget": {
            "hd14_simulations": int(len(theta_old)),
            "hd14_train": int(n_train_old),
            "new_train_target": int(rc.TRAIN.n),
            "new_train_chunks_completed": int(new["chunks"]),
            "new_train_chunks_registered": int(rc.n_chunks(rc.TRAIN)),
            "new_train_simulations": int(len(new["theta"])),
            "new_train_failed": int(new["failed"]),
            "pool_simulations": int(len(theta_old) + len(new["theta"])),
            "truncated_by_deadline": bool(new["chunks"] < rc.n_chunks(rc.TRAIN)),
        },
    }


def pool_fingerprint(theta: np.ndarray, x: np.ndarray) -> str:
    """Short hash of a training pool, so a saved model is reused only for the same data."""
    import hashlib

    payload = np.ascontiguousarray(theta, dtype=float).tobytes()
    payload += np.ascontiguousarray(x, dtype=float).tobytes()
    return hashlib.sha256(payload).hexdigest()[:12]


def load_or_train(name: str, seeds, theta, x, max_epochs=None) -> list:
    """Members of one estimator, reloaded if this exact pool and seed were trained before."""
    import torch

    members = []
    tag = f"{name}_n{len(theta)}_{pool_fingerprint(theta, x)}"
    for seed in seeds:
        path = models_dir() / f"{tag}_seed{seed}_simulated.pt"
        if path.exists() and max_epochs is None:
            members.append(torch.load(path, weights_only=False))
            continue
        members.extend(rc.train_members(theta, x, (seed,), max_epochs=max_epochs))
        if max_epochs is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(members[-1], path)
    return members


def summarize(ranks: np.ndarray, widths: np.ndarray, index, n_samples: int) -> dict:
    """KS, Holm-adjusted p (reported only), coverage and contraction on one subset."""
    ks = _rank_uniformity(ranks[index], n_samples)
    return {
        "n": int(len(ranks[index])),
        "ks": ks,
        "holm_adjusted_p": rc.holm_adjust({k: v["ks_p"] for k, v in ks.items()}),
        "all_pass_alpha": bool(all(v["ks_p"] >= rc.ALPHA for v in ks.values())),
        "coverage": rc.coverage_from_ranks(ranks[index], n_samples),
        "contraction": rc.contraction_from_widths(widths[index].mean(axis=0)),
    }


def sample_ranks(posterior, held: dict, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Per held-out simulation: SBC ranks and posterior widths, from the registered seed."""
    import torch

    torch.manual_seed(rc.EVALUATION_SEED)
    return rc.ranks_and_widths(posterior, held["theta"], held["x"], n_samples)


def summarize_subsets(ranks: np.ndarray, widths: np.ndarray, near_index, n_samples: int) -> dict:
    """H300, H1000 and the near-real subset, summarised from the same ranks."""
    primary = np.arange(min(rc.PRIMARY_N, len(ranks)))
    out = {
        "H300": summarize(ranks, widths, primary, n_samples),
        "H1000": summarize(ranks, widths, np.arange(len(ranks)), n_samples),
    }
    if near_index is not None and len(near_index):
        out["near_real"] = summarize(ranks, widths, near_index, n_samples)
    return out


def evaluate(posterior, held: dict, near_index, n_samples: int) -> tuple[dict, np.ndarray]:
    """One estimator on H300, H1000 and the near-real subset, all from one sampling pass."""
    ranks, widths = sample_ranks(posterior, held, n_samples)
    return summarize_subsets(ranks, widths, near_index, n_samples), ranks


def real_recordings() -> np.ndarray:
    """Summary vectors of the real two-chamber recordings, exactly as HD15 builds them."""
    import pandas as pd

    from neuromps.paths import asset_path, for_reading
    from neuromps.twin.inference import SUMMARY_NAMES
    from neuromps.twin.real_inference import real_summaries

    path = for_reading(asset_path())
    if not path.exists():
        return np.empty((0, len(SUMMARY_NAMES)))
    summaries = real_summaries(pd.read_parquet(path))
    return summaries[list(SUMMARY_NAMES)].to_numpy(dtype=float)


def _zscore(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``values`` in units of ``reference``'s per-dimension mean and standard deviation."""
    sd = reference.std(axis=0)
    return (values - reference.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def near_real_selection(held_x: np.ndarray, pool_x: np.ndarray) -> tuple[np.ndarray, dict]:
    """The registered near-real subset, and how near the real recordings actually are.

    The distance from a held-out simulation to its nearest neighbour among the
    others is recorded beside them as the scale a "near" distance is read on.
    """
    from scipy.spatial.distance import cdist

    real_x = real_recordings()
    if not len(real_x):
        return np.array([], dtype=int), {"n_real": 0}
    k = min(rc.NEAR_REAL_N, len(held_x))
    index, real_to_sim = rc.nearest_to_real(held_x, real_x, scale_from=pool_x, k=k)
    held_z, real_z = _zscore(held_x, pool_x), _zscore(real_x, pool_x)
    gaps = cdist(held_z, held_z)
    np.fill_diagonal(gaps, np.inf)
    to_real = cdist(held_z, real_z).min(axis=1)
    return index, {
        "n_real": int(len(real_x)),
        "n_selected": int(len(index)),
        "real_to_nearest_simulation": _quantiles(real_to_sim),
        "selected_simulation_to_nearest_real": _quantiles(to_real[index]),
        "simulation_to_nearest_other_simulation": _quantiles(gaps.min(axis=1)),
        "note": "Euclidean distance on the 19 summaries z-scored with the training pool",
    }


def _quantiles(values: np.ndarray) -> dict:
    """Median, 90th percentile and maximum, rounded for the record."""
    return {
        "median": round(float(np.median(values)), 3),
        "q90": round(float(np.quantile(values, 0.9)), 3),
        "max": round(float(np.max(values)), 3),
    }


# --- registered predictions -------------------------------------------------------


def _ps(block: dict) -> dict:
    """KS p value per parameter on one subset."""
    return {name: row["ks_p"] for name, row in block["ks"].items()}


def _over_confident_cells(block: dict) -> list[str]:
    """Parameter-level cells whose coverage falls below the registered floor."""
    return [
        f"{name} {level}"
        for name, cells in block["coverage"].items()
        for level, cell in cells.items()
        if cell["over_confident"]
    ]


def _contraction_floor(results: dict) -> dict:
    """P3's floor: M0's contraction on the same H300, less the registered margin."""
    m0 = results["M0"]["H300"]["contraction"]
    return {n: round(m0[n]["contraction"] - rc.CONTRACTION_MARGIN, 3) for n in PARAMETER_NAMES}


def score_predictions(results: dict, timing: dict) -> dict:
    """Each registered prediction with the value it was judged on and pass or fail."""
    m2 = results["M2"]
    before = {m: {n: _ps(results[m]["H300"])[n] for n in FAILED_BEFORE} for m in ("M0", "M1")}
    floor = _contraction_floor(results)
    shrink = {n: m2["H300"]["contraction"][n]["contraction"] for n in PARAMETER_NAMES}
    seconds = timing["M2"]["median_seconds_per_posterior"]
    rows = {
        "P0_hd14_failure_replicates": (min(before["M0"].values()) < rc.ALPHA, before["M0"]),
        "P1_ensemble_all_calibrated": (m2["H300"]["all_pass_alpha"], _ps(m2["H300"])),
        "P2_ensemble_coverage": (
            not _over_confident_cells(m2["H1000"]),
            _over_confident_cells(m2["H1000"]),
        ),
        "P3_no_contraction_lost": (all(shrink[n] >= floor[n] for n in shrink), shrink),
        "P4_still_amortised": (seconds < rc.TIMING_LIMIT_S, seconds),
        "P5_budget_alone_suffices": (min(before["M1"].values()) >= rc.ALPHA, before["M1"]),
        "P6_ensemble_calibrated_at_higher_power": (
            m2["H1000"]["all_pass_alpha"],
            _ps(m2["H1000"]),
        ),
    }
    out = {
        name: {"passed": bool(ok), "outcome": "pass" if ok else "fail", "value": value}
        for name, (ok, value) in rows.items()
    }
    out["P3_no_contraction_lost"]["floor"] = floor
    return {
        "by_prediction": out,
        "n_passed": sum(row["passed"] for row in out.values()),
        "n_registered": len(out),
    }


def assumption_check(results: dict) -> dict:
    """The registered check near the real recordings, for the adopted estimator."""
    block = results["M2"].get("near_real")
    if block is None:
        return {"evaluated": False, "reason": "no real two-chamber recordings in the asset"}
    cells = _over_confident_cells(block)
    return {
        "evaluated": True,
        "assumption": "calibration averaged over the prior holds where the real recordings lie",
        "passed": bool(block["all_pass_alpha"] and not cells),
        "ks_p": _ps(block),
        "over_confident_cells": cells,
    }


def adopted_parameters(results: dict) -> list[str]:
    """Parameters the slide may quote under the registered adoption rule."""
    m2 = results["M2"]
    flagged = {cell.split()[0] for cell in _over_confident_cells(m2["H1000"])}
    passing = _ps(m2["H300"])
    return [n for n in PARAMETER_NAMES if passing[n] >= rc.ALPHA and n not in flagged]


# --- figure -----------------------------------------------------------------------

COLOURS = {"M0": "#E69F00", "M1": "#56B4E9", "M2": "#0072B2", "M3": "#009E73"}
LABELS = {
    "M0": "HD14: one flow, 2,700 sims",
    "M1": "one flow, larger pool",
    "M2": "five-flow ensemble, larger pool",
    "M3": "five-flow ensemble, 2,700 sims",
}


def _rank_row(axes, ranks: np.ndarray, n_samples: int, colour: str, label: str) -> None:
    """One row of rank histograms with the band a uniform rank would fill 99% of the time."""
    from scipy.stats import binom

    bins = 10
    low, high = binom.ppf([0.005, 0.995], len(ranks), 1 / bins)
    for i, (ax, name) in enumerate(zip(axes, PARAMETER_NAMES, strict=True)):
        ax.axhspan(low, high, color="#BBBBBB", alpha=0.5, lw=0)
        ax.hist(ranks[:, i], bins=bins, range=(0, n_samples), color=colour)
        ax.set_title(name, fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel(f"{label}\ncount", fontsize=6.5)


def _coverage_row(axes, results: dict) -> None:
    """Empirical against nominal coverage on H1000, per parameter and estimator."""
    levels = rc.COVERAGE_LEVELS
    for ax, name in zip(axes, PARAMETER_NAMES, strict=True):
        ax.plot([0.4, 1.0], [0.4, 1.0], color="black", lw=0.8, ls="--")
        for model, colour in COLOURS.items():
            if model not in results:
                continue
            cells = results[model]["H1000"]["coverage"][name]
            values = [cells[rc.level_key(level)]["coverage"] for level in levels]
            ax.plot(levels, values, marker="o", ms=2.5, lw=1, color=colour, label=LABELS[model])
        ax.set_xlabel("nominal", fontsize=6.5)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("empirical coverage", fontsize=6.5)


def _legend_below(fig, axes) -> None:
    """The coverage row's legend, once for the figure, under the bottom row.

    Inside a panel it covered threshold_scale's lines; below the axes it covers
    nothing. Placed after the layout is fixed, then kept by ``bbox_inches="tight"``.
    """
    handles, labels = axes[0].get_legend_handles_labels()
    to_figure = fig.transFigure.inverted()
    bottom = min(ax.get_tightbbox().transformed(to_figure).y0 for ax in axes)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, bottom - 0.01),
        ncol=len(labels),
        fontsize=6,
        frameon=False,
    )


def calibration_figure(ranks: dict, results: dict, n_samples: int, n_pool: int):
    """HD14 against the ensemble on the same fresh held-out simulations (the figure object)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(PARAMETER_NAMES), figsize=(7.6, 5.4))
    _rank_row(axes[0], ranks["M0"], n_samples, COLOURS["M0"], "HD14 (one flow)")
    _rank_row(axes[1], ranks["M2"], n_samples, COLOURS["M2"], "ensemble")
    _coverage_row(axes[2], results)
    n_held = len(ranks["M0"])
    fig.suptitle(
        f"SIMULATED: calibration of the twin's inverse model on {n_held:,} fresh held-out "
        f"simulations\nrows 1-2: rank of the true value among {n_samples} posterior samples "
        f"(grey: 99% band for a calibrated model); row 3: credible-interval coverage\n"
        f"HD14 = one flow on 2,700 simulations; ensemble = five flows on {n_pool:,}",
        fontsize=7.5,
    )
    fig.tight_layout()
    _legend_below(fig, axes[2])
    return fig


def plot_calibration(ranks: dict, results: dict, n_samples: int, n_pool: int, path) -> None:
    """Draw `calibration_figure` to ``path``."""
    import matplotlib.pyplot as plt

    fig = calibration_figure(ranks, results, n_samples, n_pool)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- the run ----------------------------------------------------------------------


def _estimators(pools: dict, quick: bool) -> dict:
    """M0 as saved; M1, M2 and M3 trained (or reloaded) with their registered seeds."""
    import torch

    from neuromps.paths import for_reading

    epochs = QUICK["max_epochs"] if quick else None
    m2_seeds = rc.M2_SEEDS[: QUICK["members"]] if quick else rc.M2_SEEDS
    pool = pools["pool"]
    models = {
        "M0": torch.load(for_reading(twin_dir() / POSTERIOR_NAME), weights_only=False),
        "M1": load_or_train("M1", (rc.M1_SEED,), *pool, max_epochs=epochs)[0],
        "M2": rc.ensemble_posterior(load_or_train("M2", m2_seeds, *pool, max_epochs=epochs)),
    }
    if not quick:
        members = load_or_train("M3", rc.M3_SEEDS, *pools["hd14_train"])
        models["M3"] = rc.ensemble_posterior(members)
    return models


def _quick_inputs() -> tuple[dict, dict] | None:
    """Token pools for the CI path: the quick tier's own small cache, split in two."""
    if not simulation_cache().exists():
        return None
    theta, x = hd14_pool()
    n_eval = QUICK["n_eval"]
    train = (theta[:-n_eval], x[:-n_eval])
    pools = {"pool": train, "hd14_train": train, "budget": {"quick_token_run": True}}
    return pools, {"theta": theta[-n_eval:], "x": x[-n_eval:], "chunks": 0, "failed": 0}


def _timings(models: dict, held_x: np.ndarray, quick: bool) -> dict:
    """Amortised cost of a posterior for M0, M1 and M2, timed in the same process."""
    kwargs = {"n_observations": 2, "n_draws": 100} if quick else {"n_observations": 20}
    return {m: amortized_timing(models[m], held_x, **kwargs) for m in ("M0", "M1", "M2")}


def _payload(pools, held, near_info, results, timing, quick: bool) -> dict:
    """What goes under ``twin_hd14b_recalibration``."""
    payload = {
        "label": "SIMULATED - recalibration of the twin's inverse model (D31)",
        "registration": rc.REGISTRATION,
        "registration_commits": REGISTRATION_COMMITS,
        "quick_token_run": bool(quick),
        "budget": {
            **pools["budget"],
            "held_out": int(len(held["theta"])),
            "held_out_failed": int(held["failed"]),
        },
        "posterior_samples": QUICK["n_samples"] if quick else rc.N_POSTERIOR_SAMPLES,
        "alpha": rc.ALPHA,
        "near_real": near_info,
        "results": results,
        "amortized_inference": timing,
        "predictions": score_predictions(results, timing),
        "assumption_check": assumption_check(results),
        "adopted_parameters": adopted_parameters(results),
    }
    return payload


def run(quick: bool = False) -> dict:
    """Train, evaluate and record the registered comparison; returns the headline."""
    from neuromps.metrics import update_metrics

    if quick:
        inputs = _quick_inputs()
        if inputs is None:
            log.warning("no quick simulation cache; run neuromps.twin.inference --quick first")
            return {}
        pools, held = inputs
    else:
        pools, held = training_pools(), rc.load_set(rc.HELD_OUT)
        if not len(held["theta"]):
            log.warning("no held-out simulations; run with --simulate first")
            return {}
    n_samples = QUICK["n_samples"] if quick else rc.N_POSTERIOR_SAMPLES
    near_index, near_info = near_real_selection(held["x"], pools["pool"][1])
    models = _estimators(pools, quick)
    results, ranks = {}, {}
    for name, posterior in models.items():
        log.info("evaluating %s on %d held-out simulations", name, len(held["theta"]))
        results[name], ranks[name] = evaluate(posterior, held, near_index, n_samples)
    timing = _timings(models, held["x"], quick)
    payload = _payload(pools, held, near_info, results, timing, quick)
    payload["density_estimator"] = estimator_architecture(models["M1"])
    twin_dir().mkdir(parents=True, exist_ok=True)
    n_pool = len(pools["pool"][0])
    plot_calibration(ranks, results, n_samples, n_pool, twin_dir() / FIGURE_NAME)
    if not quick:
        payload["post_hoc"] = post_hoc_checks(models["M0"])
    update_metrics(rc.METRICS_KEY, payload)
    keys = ("budget", "predictions", "assumption_check", "adopted_parameters")
    return {k: payload[k] for k in keys}


# --- post hoc (decided after seeing the results) ----------------------------------

RESIMULATED_ROWS = (0, 1, 2, 2700, 2701, 2999)


def same_rows(cached: np.ndarray, fresh: np.ndarray) -> dict:
    """How many re-simulated summary vectors equal their cached ones exactly."""
    equal = [bool(np.array_equal(a, b)) for a, b in zip(cached, fresh, strict=True)]
    worst = float(np.max(np.abs(np.asarray(cached) - np.asarray(fresh)))) if len(equal) else 0.0
    return {"n_checked": len(equal), "n_identical": int(sum(equal)), "max_abs_difference": worst}


def _resimulate_hd14(rows: tuple[int, ...]) -> dict:
    """Re-run HD14's own draws with today's simulator and compare with its cache."""
    import multiprocessing as mp

    from neuromps.twin.inference import SIMULATED_S, _simulate_job

    cached = load_simulations(simulation_cache())
    jobs = [(cached["theta"][i], None, SIMULATED_S, i) for i in rows]
    with mp.Pool(min(len(rows), rc.MAX_PROCESSES)) as pool:
        fresh = pool.map(_simulate_job, jobs)
    return {"rows": list(rows), **same_rows(cached["x"][list(rows)], np.vstack(fresh))}


def post_hoc_checks(m0) -> dict:
    """Two checks added after M0 failed all five parameters on fresh simulations.

    HD14 passed three parameters on its own held-out set and this study's M0
    fails all five on fresh ones. Two explanations other than chance are ruled
    in or out here: a simulator that no longer reproduces HD14's cache (a
    distribution shift), and an evaluation that differs from HD14's own.
    """
    import torch

    theta, x = hd14_pool()
    n_hold = holdout_size(len(theta))
    torch.manual_seed(rc.EVALUATION_SEED)
    samples = rc.N_POSTERIOR_SAMPLES
    ranks, widths = rc.ranks_and_widths(m0, theta[-n_hold:], x[-n_hold:], samples)
    block = summarize(ranks, widths, np.arange(len(ranks)), samples)
    return {
        "label": "POST HOC: added after seeing the registered results; not registered",
        "simulator_reproduces_hd14_cache": _resimulate_hd14(RESIMULATED_ROWS),
        "M0_on_hd14_own_holdout": {
            "n": block["n"],
            "ks": block["ks"],
            "contraction": block["contraction"],
            "coverage": block["coverage"],
        },
    }


def rescore_section(section: dict) -> dict:
    """The recorded section with its predictions scored again from its recorded results."""
    timing = section["amortized_inference"]
    return {**section, "predictions": score_predictions(section["results"], timing)}


def rescore() -> dict:
    """Score the registered predictions again from metrics.json, without re-evaluating."""
    import json

    from neuromps.metrics import update_metrics
    from neuromps.paths import metrics_path

    section = rescore_section(json.loads(metrics_path().read_text())[rc.METRICS_KEY])
    update_metrics(rc.METRICS_KEY, section)
    return section["predictions"]


def add_post_hoc() -> dict:
    """Run only the post hoc checks and add them to the recorded section."""
    import json

    import torch

    from neuromps.metrics import update_metrics
    from neuromps.paths import metrics_path

    section = json.loads(metrics_path().read_text())[rc.METRICS_KEY]
    m0 = torch.load(twin_dir() / POSTERIOR_NAME, weights_only=False)
    section["post_hoc"] = post_hoc_checks(m0)
    update_metrics(rc.METRICS_KEY, section)
    return section["post_hoc"]
