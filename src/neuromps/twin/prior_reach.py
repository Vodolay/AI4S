"""Can a wider prior reach the real recordings? The twin's prior against misspecification (D38).

D31 (`twin/recalibration.py`) found the 113 real two-chamber recordings far from
every twin simulation. This module holds what the registered test
(`preregistration/twin_prior_reach.json`) fixes in advance: the widened prior
box, the seeds, the cut-off, the one reference scale distances are read on, and
the rules each prediction is scored by. `twin/prior_reach_run.py` puts them
together. The simulator, the summaries and the cache fingerprints are HD14's,
imported unchanged; only the box the parameters are drawn from differs.
Everything simulated here is labelled SIMULATED.
"""

import logging
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from neuromps.paths import interim_dir
from neuromps.twin import recalibration as rc
from neuromps.twin.inference import (
    PARAMETER_NAMES,
    PRIOR_BOUNDS,
    SUMMARY_NAMES,
    _simulate_job,
    check_cache_matches_code,
    draw_parameters,
    load_simulations,
    save_simulations,
)

log = logging.getLogger("neuromps.twin.prior_reach")

METRICS_KEY = "twin_prior_reach"
REGISTRATION = "preregistration/twin_prior_reach.json"
# Registered: no chunk starts at or after this moment (a running one finishes).
DEADLINE = datetime(2026, 10, 29, 6, 0, tzinfo=ZoneInfo("America/New_York"))
MAX_PROCESSES = 12
MAX_NEW = 3000
SIMULATED_S = rc.SIMULATED_S
# Registered from the pre-registration diagnostic: the three parameters whose raw
# HD14 posterior for real recordings runs past an edge of the box; the other two
# do not, and keep HD14's bounds.
NEW_BOUNDS = {
    **PRIOR_BOUNDS,
    "log10_forward_ratio": (-1.3, 1.3),
    "ampa_scale": (0.4, 1.8),
    "threshold_scale": (0.85, 1.35),
}
HELD_OUT = rc.SimulationSet("heldout", draw_base=38500, network_base=38_500_000, n=600)
TRAIN = rc.SimulationSet("train", draw_base=38000, network_base=38_000_000, n=2400)
TRAIN_SEED = 3801
EVALUATION_SEED = 3802
DIAGNOSTIC_SEED = 3803
N_POSTERIOR_SAMPLES = 200
N_EDGE_DRAWS = 1000
NEAR_REAL_N = 200
HD14_HELD = 300
ALPHA = 0.05
R1_RATIO = 0.5
R2_QUANTILE = 0.9
R2_SHARE = 0.5
MIN_NEW = 600
MIN_TRAIN = 1000


def cache_root() -> Path:
    """Where this study's chunks live (git-ignored; a symlink to the main checkout's cache)."""
    return interim_dir() / "prior_reach"


# --- the box and the draws ---------------------------------------------------------


def within_box(theta: np.ndarray, bounds: dict) -> np.ndarray:
    """Mask of draws inside ``bounds`` (inclusive), in `PARAMETER_NAMES` order."""
    theta = np.atleast_2d(theta)
    inside = np.ones(len(theta), dtype=bool)
    for i, name in enumerate(PARAMETER_NAMES):
        low, high = bounds[name]
        inside &= (theta[:, i] >= low) & (theta[:, i] <= high)
    return inside


def chunk_parameters(
    sim_set: rc.SimulationSet, k: int, chunk_size: int = rc.CHUNK_SIZE
) -> tuple[np.ndarray, list[int]]:
    """Draws from the new box and network seeds of chunk ``k`` (D31's chunk seed rule)."""
    draw_seed, seeds = rc.chunk_seeds(sim_set, k, chunk_size)
    return draw_parameters(len(seeds), draw_seed, NEW_BOUNDS), seeds


def simulate_chunk(
    sim_set: rc.SimulationSet, k: int, processes: int, root: Path | None = None
) -> Path:
    """Simulate one chunk in parallel; cache it whole, with its box and HD14's fingerprints."""
    import multiprocessing as mp

    theta, seeds = chunk_parameters(sim_set, k)
    jobs = [(theta[i], None, SIMULATED_S, seed) for i, seed in enumerate(seeds)]
    if processes > 1:
        with mp.Pool(processes) as pool:
            x = np.vstack(pool.map(_simulate_job, jobs, chunksize=1))
    else:
        x = np.vstack([_simulate_job(job) for job in jobs])
    draw_seed, _ = rc.chunk_seeds(sim_set, k)
    metadata = {
        "study": "D38 prior reach",
        "set": sim_set.name,
        "chunk": k,
        "draw_seed": draw_seed,
        "network_seeds": [seeds[0], seeds[-1]],
        "n_requested": len(seeds),
        "duration_s": SIMULATED_S,
        "prior_bounds": {name: list(NEW_BOUNDS[name]) for name in PARAMETER_NAMES},
        "label": "SIMULATED",
    }
    path = rc.chunk_path(sim_set, k, root or cache_root())
    partial = path.with_name(path.stem + ".partial.npz")
    save_simulations(partial, theta, x, metadata)
    partial.replace(path)
    return path


def simulate_set(
    sim_set: rc.SimulationSet,
    processes: int = MAX_PROCESSES,
    deadline: datetime | None = DEADLINE,
    root: Path | None = None,
    max_chunks: int | None = None,
) -> list[int]:
    """Simulate every missing chunk in order, starting none at or after ``deadline``."""
    root = root or cache_root()
    for k in range(min(rc.n_chunks(sim_set), max_chunks or rc.n_chunks(sim_set))):
        if rc.chunk_path(sim_set, k, root).exists():
            continue
        if not rc.may_start_chunk(datetime.now(tz=ZoneInfo("America/New_York")), deadline):
            log.info("%s: cut-off reached before chunk %d; stopping", sim_set.name, k)
            break
        log.info("%s: simulating chunk %d of %d", sim_set.name, k + 1, rc.n_chunks(sim_set))
        simulate_chunk(sim_set, k, processes, root)
    return rc.completed_prefix(sim_set, root)


def load_set(sim_set: rc.SimulationSet, root: Path | None = None) -> dict:
    """The completed prefix of a set; refused if a chunk came from another box or code."""
    root = root or cache_root()
    expected = {name: list(NEW_BOUNDS[name]) for name in PARAMETER_NAMES}
    thetas, xs = [], []
    chunks = rc.completed_prefix(sim_set, root)
    for k in chunks:
        cached = load_simulations(rc.chunk_path(sim_set, k, root))
        check_cache_matches_code(cached)
        if cached["metadata"].get("prior_bounds") != expected:
            raise SystemExit(f"chunk {k} of {sim_set.name} was drawn from another prior box")
        thetas.append(cached["theta"])
        xs.append(cached["x"])
    if not chunks:
        empty = (np.empty((0, len(PARAMETER_NAMES))), np.empty((0, len(SUMMARY_NAMES))))
        return {"theta": empty[0], "x": empty[1], "chunks": 0, "failed": 0}
    theta, x = np.vstack(thetas), np.vstack(xs)
    finite = np.isfinite(x).all(axis=1)
    return {
        "theta": theta[finite],
        "x": x[finite],
        "chunks": len(chunks),
        "failed": int((~finite).sum()),
    }


# --- distances on one fixed scale --------------------------------------------------


def reference_scale(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-summary mean and SD of the reference pool (HD14's), as D31 standardised.

    A constant summary keeps unit scale, exactly as `recalibration_run._zscore`.
    """
    sd = reference.std(axis=0)
    return reference.mean(axis=0), np.where(sd > 0, sd, 1.0)


def zscore(values: np.ndarray, scale: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """``values`` in units of a fixed reference scale."""
    mean, sd = scale
    return (np.asarray(values, dtype=float) - mean) / sd


def nearest_distance(query_z: np.ndarray, pool_z: np.ndarray) -> np.ndarray:
    """Euclidean distance from each query row to its nearest pool row."""
    from scipy.spatial.distance import cdist

    return cdist(query_z, pool_z).min(axis=1)


def equal_size(x: np.ndarray, n: int) -> np.ndarray:
    """The first ``n`` rows: a pool cut to the size it is compared at."""
    return x[: min(n, len(x))]


def distance_shares(query_z: np.ndarray, pool_z: np.ndarray, names=SUMMARY_NAMES) -> dict:
    """Per summary, its share of the squared distance from each query to its nearest pool row."""
    from scipy.spatial.distance import cdist

    nearest = cdist(query_z, pool_z).argmin(axis=1)
    squared = (query_z - pool_z[nearest]) ** 2
    total = squared.sum(axis=0)
    per_row = squared / np.where(
        squared.sum(axis=1, keepdims=True) > 0, squared.sum(1, keepdims=True), 1
    )
    return {
        "share_of_total": {n: round(float(total[j] / total.sum()), 4) for j, n in enumerate(names)},
        "median_share_per_query": {
            n: round(float(np.median(per_row[:, j])), 4) for j, n in enumerate(names)
        },
        "nearest_index": nearest,
    }


def _position(values: np.ndarray, pool: np.ndarray, z_values: np.ndarray) -> dict:
    """Where one summary's real values sit in one pool's distribution."""
    quantile = (pool[None, :] < values[:, None]).mean(axis=1)
    low, high = np.quantile(pool, [0.01, 0.99])
    return {
        "real_median": round(float(np.median(values)), 4),
        "real_median_z": round(float(np.median(z_values)), 3),
        "pool_q01": round(float(low), 4),
        "pool_q99": round(float(high), 4),
        "median_real_quantile_in_pool": round(float(np.median(quantile)), 3),
        "share_below_q01": round(float(np.mean(values < low)), 3),
        "share_above_q99": round(float(np.mean(values > high)), 3),
        "share_outside_pool_range": round(
            float(np.mean((values < pool.min()) | (values > pool.max()))), 3
        ),
    }


def feature_positions(real_x: np.ndarray, pool_x: np.ndarray, scale, names=SUMMARY_NAMES) -> dict:
    """Per summary: where the real recordings fall in a simulated pool's distribution."""
    real_z = zscore(real_x, scale)
    return {n: _position(real_x[:, j], pool_x[:, j], real_z[:, j]) for j, n in enumerate(names)}


def sensitivity(theta: np.ndarray, x: np.ndarray) -> dict:
    """Spearman correlation of each summary with each parameter across a pool."""
    from scipy.stats import spearmanr

    out = {}
    for j, name in enumerate(SUMMARY_NAMES):
        rho = {
            p: round(float(spearmanr(theta[:, i], x[:, j])[0]), 3)
            for i, p in enumerate(PARAMETER_NAMES)
        }
        strongest = max(rho, key=lambda p: abs(rho[p]) if math.isfinite(rho[p]) else -1)
        out[name] = {"rho": rho, "strongest": strongest, "max_abs_rho": abs(rho[strongest])}
    return out


# --- the box edges, seen from the real recordings ----------------------------------


def edge_mass(posterior, real_x: np.ndarray, bounds: dict, n_draws: int = N_EDGE_DRAWS) -> dict:
    """Share of a posterior's raw draws beyond each bound of ``bounds``, per real recording.

    Raw means not rejected at the prior: a flow shown a recording it was not
    trained near puts its mass where it extrapolates to, and the side it leaves
    the box on says which bound the recording pushes against.
    """
    import torch

    below = np.zeros((len(real_x), len(PARAMETER_NAMES)))
    above = np.zeros_like(below)
    accepted = np.zeros(len(real_x))
    for r, observation in enumerate(real_x):
        draws = posterior.sample(
            (n_draws,),
            x=torch.as_tensor(observation, dtype=torch.float32),
            show_progress_bars=False,
            reject_outside_prior=False,
        )
        draws = draws.detach().numpy() if hasattr(draws, "detach") else np.asarray(draws)
        for i, name in enumerate(PARAMETER_NAMES):
            below[r, i] = np.mean(draws[:, i] < bounds[name][0])
            above[r, i] = np.mean(draws[:, i] > bounds[name][1])
        accepted[r] = within_box(draws, bounds).mean()
    return _edge_summary(below, above, accepted, bounds, n_draws)


def _edge_summary(below, above, accepted, bounds: dict, n_draws: int) -> dict:
    """Per parameter, how much raw mass leaves the box on each side."""
    by_parameter = {
        name: {
            "bounds": list(bounds[name]),
            "median_share_below": round(float(np.median(below[:, i])), 4),
            "median_share_above": round(float(np.median(above[:, i])), 4),
            "recordings_mostly_below": round(float(np.mean(below[:, i] > 0.5)), 3),
            "recordings_mostly_above": round(float(np.mean(above[:, i] > 0.5)), 3),
        }
        for i, name in enumerate(PARAMETER_NAMES)
    }
    return {
        "n_recordings": int(len(accepted)),
        "n_draws_per_recording": int(n_draws),
        "median_acceptance": round(float(np.median(accepted)), 4),
        "by_parameter": by_parameter,
    }


# --- the registered rules -----------------------------------------------------------


def _outcome(ok: bool, interpretable: bool) -> str:
    """Pass, fail, or uninterpretable when a registered precondition is not met."""
    if not interpretable:
        return "uninterpretable"
    return "pass" if ok else "fail"


def score_r1(new_median: float, old_median: float, n_new: int, reproduced: bool) -> dict:
    """R1 reach: the new pool's median real-to-nearest distance is at most half the old one's."""
    ratio = new_median / old_median if old_median > 0 else math.inf
    interpretable = n_new >= MIN_NEW and reproduced
    return {
        "outcome": _outcome(ratio <= R1_RATIO, interpretable),
        "new_pool_median": round(float(new_median), 3),
        "hd14_pool_median": round(float(old_median), 3),
        "ratio": round(float(ratio), 3),
        "threshold_ratio": R1_RATIO,
    }


def old_region_share(hd14_z: np.ndarray, new_z: np.ndarray, n_held: int = HD14_HELD) -> dict:
    """R2's statistic: how many HD14 held-out simulations the new pool still has a neighbour for.

    HD14's last ``n_held`` simulations are its held-out set. Near means within
    the 90th percentile of their distances to HD14's own first ``m`` simulations,
    where ``m`` is the size of the new pool compared (at most HD14's training size).
    """
    held, reference = hd14_z[-n_held:], hd14_z[:-n_held]
    m = min(len(reference), len(new_z))
    threshold = float(np.quantile(nearest_distance(held, reference[:m]), R2_QUANTILE))
    to_new = nearest_distance(held, new_z[:m])
    return {
        "share": round(float(np.mean(to_new <= threshold)), 4),
        "threshold_distance": round(threshold, 4),
        "n_hd14_held_out": int(n_held),
        "n_reference": int(m),
        "median_distance_to_new_pool": round(float(np.median(to_new)), 4),
    }


def score_r2(share: float, n_new: int, reproduced: bool) -> dict:
    """R2: at least half of HD14's held-out simulations keep a near neighbour in the new pool."""
    interpretable = n_new >= MIN_NEW and reproduced
    return {
        "outcome": _outcome(share >= R2_SHARE, interpretable),
        "share": share,
        "threshold_share": R2_SHARE,
    }


def score_r3(ks_p: float, n_held: int, n_train: int) -> dict:
    """R3: KS rank test of threshold_scale near the real recordings, p >= 0.05."""
    interpretable = n_held >= HELD_OUT.n and n_train >= MIN_TRAIN
    return {"outcome": _outcome(ks_p >= ALPHA, interpretable), "ks_p": ks_p, "alpha": ALPHA}
