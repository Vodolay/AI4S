"""Run, score and record the registered prior-reach study of the twin (D38).

The registration is `preregistration/twin_prior_reach.json`; the box, seeds,
cut-off, scale and scoring rules are in `neuromps.twin.prior_reach`. This module
loads HD14's pool (the reference), the new pool and the real recordings, scores
R1-R3 and the assumption checks, and writes ``twin_prior_reach`` to metrics, the
results file and the figure. Every simulation here is SIMULATED; the real
recordings are the committed asset's two-chamber summaries, as HD15 builds them.
"""

import json
import logging
from pathlib import Path

import numpy as np

from neuromps.paths import REPO_ROOT, simulation_cache, twin_dir
from neuromps.twin import prior_reach as pr
from neuromps.twin import recalibration as rc
from neuromps.twin.inference import (
    PARAMETER_NAMES,
    POSTERIOR_NAME,
    PRIOR_BOUNDS,
    _rank_uniformity,
    check_cache_matches_code,
    load_simulations,
    summary_pipeline_fingerprint,
    train_posterior,
    within_prior,
)
from neuromps.twin.recalibration_run import (
    RESIMULATED_ROWS,
    _quantiles,
    _resimulate_hd14,
    pool_fingerprint,
    real_recordings,
)

log = logging.getLogger("neuromps.twin.prior_reach")
FIGURE_NAME = "prior_reach_simulated.png"
RESULTS_NAME = "prior_reach_SIMULATED.json"
LABEL = "SIMULATED twin pools against REAL two-chamber recordings - prior reach (D38)"


# --- inputs -------------------------------------------------------------------------


def hd14_reference() -> dict:
    """HD14's 3,000 cached simulations, checked against everything but the summary-code hash.

    The summary-code hash moved after HD14 ran (two extractor options, off by
    default, were added to `features/mea.py`). Whether the cached summaries are
    still what today's code produces is the registered reproduction check,
    not an assumption: `reproduction_check` re-simulates six of HD14's rows.
    """
    cached = load_simulations(simulation_cache())
    current = summary_pipeline_fingerprint()
    meta = cached["metadata"]
    check_cache_matches_code({**cached, "metadata": {**meta, "summary_pipeline_sha256": current}})
    inside = within_prior(cached["theta"])
    return {
        "theta": cached["theta"][inside],
        "x": cached["x"][inside],
        "summary_code": {
            "cached": meta.get("summary_pipeline_sha256"),
            "current": current,
            "matches": meta.get("summary_pipeline_sha256") == current,
        },
    }


def reproduction_check() -> dict:
    """Registered: today's simulator reproduces HD14's cached summaries exactly (D31's rows)."""
    got = _resimulate_hd14(RESIMULATED_ROWS)
    return {**got, "passed": bool(got["n_identical"] == got["n_checked"])}


def new_pool() -> dict:
    """The new box's completed simulations, held-out set first, then the training set."""
    held, train = pr.load_set(pr.HELD_OUT), pr.load_set(pr.TRAIN)
    return {
        "held": held,
        "train": train,
        "theta": np.vstack([held["theta"], train["theta"]]),
        "x": np.vstack([held["x"], train["x"]]),
    }


# --- the pre-registration diagnostic ----------------------------------------------


def _self_gap(z: np.ndarray) -> np.ndarray:
    """Distance from each simulation to its nearest other simulation in the same pool."""
    from scipy.spatial.distance import cdist

    gaps = cdist(z, z)
    np.fill_diagonal(gaps, np.inf)
    return gaps.min(axis=1)


def _nearest_parameters(theta: np.ndarray, index: np.ndarray) -> dict:
    """Quantiles of the parameters of the simulations nearest the real recordings."""
    return {
        name: [round(float(q), 3) for q in np.quantile(theta[index, i], [0.1, 0.5, 0.9])]
        for i, name in enumerate(PARAMETER_NAMES)
    }


def diagnostic(reference: dict, real_x: np.ndarray, posterior) -> dict:
    """Where the real recordings sit against HD14's pool, and which box edges they push.

    Seen before registration (declared there). Distances are on HD14's pool scale.
    """
    import torch

    scale = pr.reference_scale(reference["x"])
    pool_z, real_z = pr.zscore(reference["x"], scale), pr.zscore(real_x, scale)
    shares = pr.distance_shares(real_z, pool_z)
    ranked = sorted(shares["share_of_total"].items(), key=lambda kv: -kv[1])
    torch.manual_seed(pr.DIAGNOSTIC_SEED)
    return {
        "label": "SIMULATED HD14 pool against REAL recordings; seen before registration",
        "scale": "z-scored with HD14's 3,000 simulations (the registered reference scale)",
        "n_real": int(len(real_x)),
        "n_hd14": int(len(pool_z)),
        "real_to_nearest_hd14_simulation": _quantiles(pr.nearest_distance(real_z, pool_z)),
        "hd14_simulation_to_nearest_other": _quantiles(_self_gap(pool_z)),
        "distance_share_of_total": shares["share_of_total"],
        "largest_shares": [{"summary": k, "share": v} for k, v in ranked[:5]],
        "feature_positions": pr.feature_positions(real_x, reference["x"], scale),
        "sensitivity": pr.sensitivity(reference["theta"], reference["x"]),
        "nearest_simulation_parameters_q10_q50_q90": _nearest_parameters(
            reference["theta"], shares["nearest_index"]
        ),
        "hd14_posterior_edges": pr.edge_mass(posterior, real_x, PRIOR_BOUNDS),
    }


def hd14_posterior():
    """HD14's saved posterior (D14/HD14), as HD15 applies it to the real recordings."""
    import torch

    from neuromps.paths import for_reading

    return torch.load(for_reading(twin_dir() / POSTERIOR_NAME), weights_only=False)


# --- R1 and R2: reach, and the old region ------------------------------------------


def _median_reach(real_z: np.ndarray, pool_z: np.ndarray) -> float:
    return float(np.median(pr.nearest_distance(real_z, pool_z)))


def reach(reference: dict, new: dict, real_x: np.ndarray, reproduced: bool) -> dict:
    """R1 on HD14's scale at equal pool sizes, with what is left of the gap and where."""
    scale = pr.reference_scale(reference["x"])
    n_new = len(new["x"])
    old_z = pr.zscore(pr.equal_size(reference["x"], n_new), scale)
    new_z, real_z = pr.zscore(new["x"], scale), pr.zscore(real_x, scale)
    to_new = pr.nearest_distance(real_z, new_z)
    shares = pr.distance_shares(real_z, new_z)
    gap = float(np.median(_self_gap(pr.zscore(reference["x"], scale))))
    return {
        "R1": pr.score_r1(
            float(np.median(to_new)), _median_reach(real_z, old_z), n_new, reproduced
        ),
        "n_compared_each": int(min(n_new, len(reference["x"]))),
        "real_to_nearest_new_simulation": _quantiles(to_new),
        "real_to_nearest_hd14_simulation": _quantiles(pr.nearest_distance(real_z, old_z)),
        "hd14_neighbour_distance_median": round(gap, 3),
        "share_real_within_hd14_neighbour_distance": {
            "new_pool": round(float(np.mean(to_new <= gap)), 3),
            "hd14_pool": round(float(np.mean(pr.nearest_distance(real_z, old_z) <= gap)), 3),
        },
        "remaining_distance_share_of_total": shares["share_of_total"],
        "nearest_new_simulation_parameters_q10_q50_q90": _nearest_parameters(
            new["theta"], shares["nearest_index"]
        ),
        "feature_positions_in_new_pool": pr.feature_positions(real_x, new["x"], scale),
    }


def scale_check(reference: dict, new: dict, real_x: np.ndarray) -> dict:
    """Assumption check A2: the R1 ratio on two other scales (new pool's own, and pooled)."""
    n_new = len(new["x"])
    old = pr.equal_size(reference["x"], n_new)
    out = {}
    for name, ref in (("new_pool_scale", new["x"]), ("pooled_scale", np.vstack([old, new["x"]]))):
        scale = pr.reference_scale(ref)
        real_z = pr.zscore(real_x, scale)
        ratio = _median_reach(real_z, pr.zscore(new["x"], scale)) / _median_reach(
            real_z, pr.zscore(old, scale)
        )
        out[name] = round(float(ratio), 3)
    out["all_at_most_threshold"] = bool(all(v <= pr.R1_RATIO for v in out.values()))
    return out


def old_region(reference: dict, new: dict, reproduced: bool) -> dict:
    """R2 on HD14's scale."""
    scale = pr.reference_scale(reference["x"])
    stats = pr.old_region_share(pr.zscore(reference["x"], scale), pr.zscore(new["x"], scale))
    return {"R2": pr.score_r2(stats["share"], len(new["x"]), reproduced), **stats}


# --- R3: calibration near the real recordings --------------------------------------


def models_dir() -> Path:
    """Trained flows of this study (git-ignored; rebuilt from the cache)."""
    return pr.cache_root() / "models"


def load_or_train(theta: np.ndarray, x: np.ndarray):
    """The registered flow on the new training set, reloaded if this exact pool was trained."""
    import torch

    tag = f"flow_n{len(theta)}_{pool_fingerprint(theta, x)}_seed{pr.TRAIN_SEED}_simulated.pt"
    path = models_dir() / tag
    if path.exists():
        return torch.load(path, weights_only=False)
    torch.manual_seed(pr.TRAIN_SEED)
    posterior = train_posterior(theta, x, bounds=pr.NEW_BOUNDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(posterior, path)
    return posterior


def _contraction(widths: np.ndarray) -> dict:
    """1 - mean posterior SD / prior SD, against the new box."""
    out = {}
    for i, name in enumerate(PARAMETER_NAMES):
        low, high = pr.NEW_BOUNDS[name]
        prior_sd = (high - low) / np.sqrt(12)
        out[name] = round(float(1 - widths[:, i].mean() / prior_sd), 3)
    return out


def _summarise(ranks: np.ndarray, widths: np.ndarray, index: np.ndarray) -> dict:
    """KS, coverage and contraction on one subset of the held-out simulations."""
    return {
        "n": int(len(index)),
        "ks": _rank_uniformity(ranks[index], pr.N_POSTERIOR_SAMPLES),
        "coverage": rc.coverage_from_ranks(ranks[index], pr.N_POSTERIOR_SAMPLES),
        "contraction": _contraction(widths[index]),
    }


def calibration(new: dict, real_x: np.ndarray, reference: dict) -> dict:
    """R3 with the registered flow, plus its view of the real recordings (reported only)."""
    import torch

    held, train = new["held"], new["train"]
    flow = load_or_train(train["theta"], train["x"])
    k = min(pr.NEAR_REAL_N, len(held["x"]))
    near, _ = rc.nearest_to_real(held["x"], real_x, scale_from=reference["x"], k=k)
    torch.manual_seed(pr.EVALUATION_SEED)
    ranks, widths = rc.ranks_and_widths(flow, held["theta"], held["x"], pr.N_POSTERIOR_SAMPLES)
    near_block = _summarise(ranks, widths, near)
    ks_p = near_block["ks"]["threshold_scale"]["ks_p"]
    torch.manual_seed(pr.DIAGNOSTIC_SEED)
    return {
        "R3": pr.score_r3(ks_p, len(held["x"]), len(train["x"])),
        "n_train": int(len(train["x"])),
        "near_real": near_block,
        "all_held_out": _summarise(ranks, widths, np.arange(len(ranks))),
        "new_flow_edges_on_real": pr.edge_mass(flow, real_x, pr.NEW_BOUNDS),
        "density_estimator": _architecture(flow),
    }


def _architecture(flow) -> dict:
    from neuromps.twin.inference import estimator_architecture

    return estimator_architecture(flow)


# --- the record ---------------------------------------------------------------------


def budget(new: dict) -> dict:
    """How many of the registered simulations finished before the cut."""
    held, train = new["held"], new["train"]
    return {
        "registered_max": pr.MAX_NEW,
        "held_out_chunks": f"{held['chunks']} of {rc.n_chunks(pr.HELD_OUT)}",
        "train_chunks": f"{train['chunks']} of {rc.n_chunks(pr.TRAIN)}",
        "held_out_chunks_completed": int(held["chunks"]),
        "train_chunks_completed": int(train["chunks"]),
        "new_simulations_finished": int(len(new["x"]) + held["failed"] + train["failed"]),
        "new_simulations_usable": int(len(new["x"])),
        "held_out_usable": int(len(held["x"])),
        "train_usable": int(len(train["x"])),
        "failed": int(held["failed"] + train["failed"]),
        "truncated_by_cut_off": bool(train["chunks"] < rc.n_chunks(pr.TRAIN)),
    }


def registration_record() -> dict:
    """Hash and first commit of the registration (the provenance helper other studies use)."""
    from neuromps.models.toxcast_noise_floors import provenance

    return provenance(REPO_ROOT / pr.REGISTRATION)


def reading(r1: dict, r2: dict, r3: dict) -> str:
    """The registered reading rule applied to the outcomes."""
    if r1["outcome"] == "uninterpretable":
        return "uninterpretable: the registered preconditions for R1 were not met"
    if r1["outcome"] == "pass":
        head = "prior range: the widened box reaches the real recordings (R1 pass)"
    else:
        head = (
            "misspecification within the widened box: widening the three edge-loaded parameters "
            "does not halve the distance (R1 fail); the summaries carrying what remains are "
            "where the twin does not produce the real recordings"
        )
    return f"{head}; old region kept: {r2['outcome']}; calibration near real (R3): {r3['outcome']}"


def run() -> dict:
    """Score the registered predictions on what is cached and write every output."""
    from neuromps.metrics import update_metrics
    from neuromps.twin.prior_reach_figure import plot

    reference, real_x, new = hd14_reference(), real_recordings(), new_pool()
    if not len(new["held"]["x"]):
        log.warning("no new simulations; run with --simulate first")
        return {}
    reproduction = reproduction_check()
    ok = reproduction["passed"]
    r1, r2 = reach(reference, new, real_x, ok), old_region(reference, new, ok)
    r3 = calibration(new, real_x, reference)
    payload = {
        "label": LABEL,
        "registration": registration_record(),
        "budget": budget(new),
        "reference_scale": "per-summary mean and SD of HD14's 3,000 simulations, for every pool",
        "predictions": {
            "R1_reach": r1["R1"],
            "R2_old_region_kept": r2["R2"],
            "R3_calibrated_near_real": r3["R3"],
        },
        "reach": r1,
        "old_region": r2,
        "calibration": r3,
        "assumption_checks": {
            "A1_hd14_cache_reproduced": {**reproduction, "summary_code": reference["summary_code"]},
            "A2_reach_not_an_artifact_of_the_scale": scale_check(reference, new, real_x),
        },
        "diagnostic": diagnostic(reference, real_x, hd14_posterior()),
    }
    payload["reading"] = reading(r1["R1"], r2["R2"], r3["R3"])
    twin_dir().mkdir(parents=True, exist_ok=True)
    plot(reference, new, real_x, payload, twin_dir() / FIGURE_NAME)
    (twin_dir() / RESULTS_NAME).write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    update_metrics(pr.METRICS_KEY, json.loads(json.dumps(payload, default=_plain)))
    return {k: payload[k] for k in ("budget", "predictions", "assumption_checks", "reading")}


def _plain(value):
    """JSON fallback for numpy scalars and arrays."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def rescore_section(section: dict) -> dict:
    """Score R1-R3 again from the recorded values (the quick tier: no simulation, no training)."""
    ok = section["assumption_checks"]["A1_hd14_cache_reproduced"]["passed"]
    n_new = section["budget"]["new_simulations_usable"]
    r1 = section["reach"]["R1"]
    r3 = section["calibration"]
    predictions = {
        "R1_reach": pr.score_r1(r1["new_pool_median"], r1["hd14_pool_median"], n_new, ok),
        "R2_old_region_kept": pr.score_r2(section["old_region"]["share"], n_new, ok),
        "R3_calibrated_near_real": pr.score_r3(
            r3["near_real"]["ks"]["threshold_scale"]["ks_p"],
            section["budget"]["held_out_usable"],
            r3["n_train"],
        ),
    }
    return {**section, "predictions": predictions}


def recorded_section() -> dict:
    """The recorded ``twin_prior_reach`` section: this run's, else the full run's committed one."""
    from neuromps.paths import full_run_equivalent, metrics_path

    for path in (metrics_path(), full_run_equivalent(metrics_path())):
        if path.exists():
            section = json.loads(path.read_text()).get(pr.METRICS_KEY)
            if section:
                return section
    return {}


def rescore() -> dict:
    """The quick tier: re-score the committed record; milliseconds, nothing simulated."""
    from neuromps.metrics import update_metrics

    section = recorded_section()
    if "predictions" not in section:
        log.warning("no recorded %s run to re-score; the full tier produces it", pr.METRICS_KEY)
        return {}
    section = rescore_section(section)
    update_metrics(pr.METRICS_KEY, section)
    return section["predictions"]


def recorded_chunks() -> tuple[int, int] | None:
    """Held-out and training chunks the recorded run used, if there is a record."""
    from neuromps.paths import metrics_path

    if not metrics_path().exists():
        return None
    record = (json.loads(metrics_path().read_text()).get(pr.METRICS_KEY) or {}).get("budget")
    if not record:
        return None
    return record["held_out_chunks_completed"], record["train_chunks_completed"]


def simulate(processes: int, deadline_applies: bool = True) -> dict:
    """Simulate the held-out set first, then the training set, under the registered cut-off.

    With a record, a rerun simulates exactly the recorded chunks (a clean clone's
    full run, hours of simulation); without one, the registered cut-off applies.
    """
    recorded = recorded_chunks()
    deadline = pr.DEADLINE if deadline_applies and recorded is None else None
    held_max, train_max = recorded if recorded else (None, None)
    held = pr.simulate_set(pr.HELD_OUT, processes, deadline, max_chunks=held_max)
    train = pr.simulate_set(pr.TRAIN, processes, deadline, max_chunks=train_max)
    return {"held_out_chunks": len(held), "train_chunks": len(train)}


def main(argv: list[str] | None = None) -> int:
    """Diagnose, simulate, or score the registered prior-reach study."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="re-score the recorded run only")
    p.add_argument("--diagnostic", action="store_true", help="the pre-registration diagnostic")
    p.add_argument("--simulate", action="store_true", help="simulate the registered sets, stop")
    p.add_argument("--processes", type=int, default=pr.MAX_PROCESSES)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.quick:
        print(json.dumps(rescore(), indent=2, default=_plain))
        return 0
    if args.diagnostic:
        out = diagnostic(hd14_reference(), real_recordings(), hd14_posterior())
        print(json.dumps(out, indent=2, default=_plain))
        return 0
    processes = min(args.processes, pr.MAX_PROCESSES)
    print(json.dumps(simulate(processes), indent=2))
    if args.simulate:
        return 0
    print(json.dumps(run(), indent=2, default=_plain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
