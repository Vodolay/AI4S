"""Redraw the recalibration figure from the recorded run, checking every value (D31).

The figure of `neuromps.twin.recalibration_run` is drawn from ranks that exist only
in memory during the run. This module rebuilds them without re-running anything the
record does not fix: it simulates only the registered held-out set and exactly the
training chunks the record says completed (``budget.new_train_chunks_completed``),
retrains the estimators with their registered seeds, draws the ranks with the
registered evaluation seed, and recomputes every value under
``twin_hd14b_recalibration.results`` (plus the budget and the near-real distances).
The figure is redrawn only if every recomputed value equals the recorded one.

The ranks are persisted (``data/interim/recalibration/ranks_simulated.npz``), so a
later redraw takes seconds. Nothing here writes to ``outputs/metrics.json``: the
timing (`amortized_timing`) is not recomputed and every recorded value stays as
recorded. All of it is SIMULATED data.
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuromps.paths import interim_dir, metrics_path, twin_dir
from neuromps.twin import recalibration as rc
from neuromps.twin import recalibration_run as rr

log = logging.getLogger("neuromps.twin.recalibration")

RANKS_NAME = "ranks_simulated.npz"
CHECKED_KEYS = ("results", "near_real", "budget")


@dataclass
class RankCache:
    """Per-estimator ranks and posterior widths on the held-out set, and how they were made."""

    ranks: dict[str, np.ndarray]
    widths: dict[str, np.ndarray]
    near_index: np.ndarray
    meta: dict


def ranks_path() -> Path:
    """Where the persisted ranks live (git-ignored, beside the simulation chunks)."""
    return interim_dir() / "recalibration" / RANKS_NAME


# --- persisting the ranks ---------------------------------------------------------


def save_ranks(path: Path, cache: RankCache) -> Path:
    """Write the ranks whole or not at all (written under another name, then renamed)."""
    arrays = {f"ranks__{m}": r for m, r in cache.ranks.items()}
    arrays |= {f"widths__{m}": w for m, w in cache.widths.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez(
        partial,
        near_index=np.asarray(cache.near_index, dtype=int),
        meta=np.array(json.dumps(cache.meta, sort_keys=True)),
        **arrays,
    )
    partial.replace(path)
    return path


def load_ranks(path: Path) -> RankCache:
    """Read back what `save_ranks` wrote."""
    with np.load(path, allow_pickle=False) as data:
        ranks = {k.split("__", 1)[1]: data[k] for k in data.files if k.startswith("ranks__")}
        widths = {k.split("__", 1)[1]: data[k] for k in data.files if k.startswith("widths__")}
        return RankCache(ranks, widths, data["near_index"], json.loads(str(data["meta"])))


# --- recomputing and comparing ----------------------------------------------------


def results_from_ranks(cache: RankCache) -> dict:
    """Every estimator's H300, H1000 and near-real blocks, as the run computes them."""
    n_samples = int(cache.meta["n_samples"])
    return {
        m: rr.summarize_subsets(cache.ranks[m], cache.widths[m], cache.near_index, n_samples)
        for m in cache.ranks
    }


def differences(recorded, recomputed, path: str = "") -> list[tuple[str, object, object]]:
    """Every leaf where two records differ, as (dotted path, recorded, recomputed).

    Leaves are compared with ``==``: the record holds values rounded for printing, and
    the recomputation rounds them the same way, so equal means equal as printed.
    """
    if isinstance(recorded, dict) and isinstance(recomputed, dict):
        out = []
        for key in sorted(set(recorded) | set(recomputed), key=str):
            where = f"{path}.{key}" if path else str(key)
            if key not in recorded or key not in recomputed:
                out.append((where, recorded.get(key), recomputed.get(key)))
                continue
            out.extend(differences(recorded[key], recomputed[key], where))
        return out
    return [] if recorded == recomputed else [(path, recorded, recomputed)]


def count_values(record) -> int:
    """Number of leaves (lists count as one value) a comparison looks at."""
    if isinstance(record, dict):
        return sum(count_values(v) for v in record.values())
    return 1


def largest_difference(found: list[tuple[str, object, object]]) -> float:
    """Largest absolute difference among numeric leaves (booleans and missing ignored)."""
    gaps = [
        abs(float(a) - float(b))
        for _, a, b in found
        if _is_number(a) and _is_number(b) and math.isfinite(float(a) - float(b))
    ]
    return max(gaps, default=0.0)


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def check_against_record(section: dict, cache: RankCache) -> dict:
    """Recompute results from ``cache`` and compare them, the budget and the near-real
    distances with the recorded section. Timing is not recomputed and not compared."""
    recomputed = {
        "results": results_from_ranks(cache),
        "near_real": cache.meta["near_real"],
        "budget": cache.meta["budget"],
    }
    recomputed = json.loads(json.dumps(recomputed, default=str))
    recorded = {key: section.get(key) for key in CHECKED_KEYS}
    found = differences(recorded, recomputed)
    return {
        "equal": not found,
        "n_values_checked": count_values(recorded),
        "n_differ": len(found),
        "largest_numeric_difference": largest_difference(found),
        "differences": found,
    }


# --- regenerating the ranks -------------------------------------------------------


def recorded_train_chunks(section: dict) -> int:
    """The number of training chunks the recorded run used; refuses if there is none."""
    chunks = (section.get("budget") or {}).get("new_train_chunks_completed")
    if not isinstance(chunks, int) or chunks < 1:
        raise ValueError("no recorded training-chunk count; nothing to regenerate from")
    return chunks


def accept_hd14_cache(cached: dict, resimulate) -> dict:
    """Check HD14's cache, tolerating only a later summary-code fingerprint that reproduces it.

    HD14's cache and the recorded run share one summary-code fingerprint; later commits
    added definition options, off by default, to the feature extractor, which changes
    the fingerprint without changing a value. Such a cache is accepted only if
    ``resimulate(cached)`` (re-running cached rows under today's code) reproduces every
    checked row exactly. Every other guard (summary names, layout, calibration) stays
    strict, and the end-to-end equality check with the record still gates the figure.
    """
    from neuromps.twin.inference import check_cache_matches_code, summary_pipeline_fingerprint

    current = summary_pipeline_fingerprint()
    old = cached["metadata"].get("summary_pipeline_sha256")
    as_current = {**cached, "metadata": {**cached["metadata"], "summary_pipeline_sha256": current}}
    check_cache_matches_code(as_current)
    if old == current:
        return {"summary_code": "current"}
    rows = resimulate(cached)
    if rows["n_identical"] != rows["n_checked"] or not rows["n_checked"]:
        raise SystemExit(
            f"HD14's cache (summary code {old}) does not reproduce under the current code "
            f"{current}: {rows['n_identical']} of {rows['n_checked']} rows identical"
        )
    return {
        "summary_code": "older, reproduced",
        "cached_fingerprint": old,
        "current_fingerprint": current,
        "resimulated": rows,
    }


def hd14_pool_checked() -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    """HD14's pool as `recalibration_run.hd14_pool` returns it, under `accept_hd14_cache`."""
    from neuromps.paths import simulation_cache
    from neuromps.twin.inference import load_simulations, within_prior

    cached = load_simulations(simulation_cache())
    accepted = accept_hd14_cache(cached, lambda _c: rr._resimulate_hd14(rr.RESIMULATED_ROWS))
    log.info("HD14 cache accepted: %s", accepted)
    inside = within_prior(cached["theta"])
    return (cached["theta"][inside], cached["x"][inside]), accepted


def regenerate(section: dict, processes: int) -> RankCache:
    """Simulate what the record used and this machine lacks, retrain, and redraw the ranks.

    Only the registered held-out set and the first ``new_train_chunks_completed``
    training chunks are simulated; no chunk beyond the record is ever started.
    """
    rc.simulate_set(rc.HELD_OUT, processes)
    rc.simulate_set(rc.TRAIN, processes, deadline=None, max_chunks=recorded_train_chunks(section))
    hd14, accepted = hd14_pool_checked()
    pools, held = rr.training_pools(hd14), rc.load_set(rc.HELD_OUT)
    budget = {
        **pools["budget"],
        "held_out": int(len(held["theta"])),
        "held_out_failed": int(held["failed"]),
    }
    near_index, near_info = rr.near_real_selection(held["x"], pools["pool"][1])
    models = rr._estimators(pools, quick=False)
    ranks, widths = {}, {}
    for name, posterior in models.items():
        log.info("drawing ranks for %s on %d held-out simulations", name, len(held["theta"]))
        ranks[name], widths[name] = rr.sample_ranks(posterior, held, rc.N_POSTERIOR_SAMPLES)
    meta = {
        "label": "SIMULATED - per-simulation SBC ranks of the D31 estimators",
        "n_samples": rc.N_POSTERIOR_SAMPLES,
        "evaluation_seed": rc.EVALUATION_SEED,
        "held_out_fingerprint": rr.pool_fingerprint(held["theta"], held["x"]),
        "pool_fingerprint": rr.pool_fingerprint(*pools["pool"]),
        "near_real": near_info,
        "budget": budget,
        "hd14_cache": accepted,
    }
    return RankCache(ranks, widths, near_index, meta)


# --- the redraw -------------------------------------------------------------------


def _print_check(check: dict) -> None:
    """The comparison, every differing value listed."""
    print(
        f"checked {check['n_values_checked']} recorded values under "
        f"{rc.METRICS_KEY}.{{{', '.join(CHECKED_KEYS)}}}: "
        f"{check['n_differ']} differ; largest numeric difference "
        f"{check['largest_numeric_difference']:g}"
    )
    for where, recorded, recomputed in check["differences"]:
        print(f"  DIFFERS {where}: recorded {recorded!r}, recomputed {recomputed!r}")


def replot(
    metrics: Path | None = None,
    ranks: Path | None = None,
    figure: Path | None = None,
    processes: int = rc.MAX_PROCESSES,
) -> int:
    """Redraw the figure if every recomputed value equals the record; 0 if drawn, 1 if not.

    Uses the persisted ranks when present, else regenerates (and persists) them.
    Reads ``metrics``; never writes it.
    """
    section = json.loads(Path(metrics or metrics_path()).read_text())[rc.METRICS_KEY]
    ranks = Path(ranks or ranks_path())
    if ranks.exists():
        log.info("using persisted ranks %s", ranks)
        cache = load_ranks(ranks)
    else:
        cache = regenerate(section, processes)
        save_ranks(ranks, cache)
        log.info("persisted ranks to %s", ranks)
    check = check_against_record(section, cache)
    _print_check(check)
    if not check["equal"]:
        print("figure NOT redrawn: the recomputed values do not reproduce the record")
        return 1
    figure = Path(figure or twin_dir() / rr.FIGURE_NAME)
    results = results_from_ranks(cache)
    n_pool = int(cache.meta["budget"]["pool_simulations"])
    rr.plot_calibration(cache.ranks, results, int(cache.meta["n_samples"]), n_pool, figure)
    print(f"every value reproduced; figure redrawn: {figure}")
    return 0
