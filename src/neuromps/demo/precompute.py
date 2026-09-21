"""Precompute everything the demo app shows, so no interaction runs a simulator.

A Brian2 run takes seconds and an `sbi` posterior sample can take longer; doing
either inside a Streamlit callback would make the app feel broken. This stage
runs them once, ahead of time, and writes the results to `outputs/demo/`:

- a grid of twin simulations over forward/backward ratio, GABA-A scale and AMPA
  scale, each with its simulated MEA features (from the same `features/mea.py`
  as every other part of the project), a short spike-raster snapshot per
  compartment, and posterior samples conditioned on that run;
- posterior samples for each real two-chamber recording, the HD15 result;
- a mechanism classifier and conformal threshold, fitted once, so Tab 2 can
  answer a slider move with a `predict_proba` call.

The trained posterior estimator itself is not copied here; the path to it is
recorded so the app can load the one the twin stage wrote.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.features.mea import extract_features
from neuromps.metrics import update_metrics
from neuromps.paths import asset_path, demo_dir, for_reading
from neuromps.twin.inference import (
    PARAMETER_NAMES,
    PRIOR_BOUNDS,
    SUMMARY_NAMES,
    default_layout,
    params_from_vector,
    summary_vector,
)
from neuromps.twin.mea import (
    electrode_coupling,
    electrode_positions,
    place_neurons,
    record_electrodes,
)
from neuromps.twin.network import simulate
from neuromps.twin.real_inference import posterior_file

log = logging.getLogger("neuromps.demo")
REPO_ROOT = Path(__file__).resolve().parents[3]


def runs_dir() -> Path:
    """Per-grid-point simulation files of the active run."""
    return demo_dir() / "runs"


RASTER_SECONDS = 2.0
FIXED_IN_DEGREE = 9.0
FIXED_THRESHOLD = 1.0


def grid_axes(quick: bool = False) -> dict[str, np.ndarray]:
    """Slider values, spanning the prior the posterior was trained under."""
    n_ratio, n_gabaa, n_ampa = (3, 3, 2) if quick else (5, 5, 3)
    low, high = PRIOR_BOUNDS["log10_forward_ratio"]
    return {
        "forward_backward_ratio": np.round(10 ** np.linspace(low, high, n_ratio), 2),
        "gabaa_scale": np.round(np.linspace(*PRIOR_BOUNDS["gabaa_scale"], n_gabaa), 3),
        "ampa_scale": np.round(np.linspace(*PRIOR_BOUNDS["ampa_scale"], n_ampa), 3),
    }


def grid_points(axes: dict[str, np.ndarray]) -> list[dict]:
    """Every combination of the axes, in a stable order."""
    return [
        {"forward_backward_ratio": float(r), "gabaa_scale": float(g), "ampa_scale": float(a)}
        for r in axes["forward_backward_ratio"]
        for g in axes["gabaa_scale"]
        for a in axes["ampa_scale"]
    ]


def theta_of(point: dict) -> np.ndarray:
    """The parameter vector the posterior was trained on, for one grid point."""
    return np.array(
        [
            np.log10(point["forward_backward_ratio"]),
            FIXED_IN_DEGREE,
            point["gabaa_scale"],
            point["ampa_scale"],
            FIXED_THRESHOLD,
        ]
    )


def simulate_point(point: dict, duration_s: float, seed: int, n_per_compartment: int) -> dict:
    """One simulation, returning what the app needs: features, summary and a raster."""
    theta = theta_of(point)
    params = params_from_vector(theta, n_per_compartment, duration_s, seed)
    layout = default_layout()
    result = simulate(params)
    positions, compartment = place_neurons(params, layout, seed=seed)
    heard = record_electrodes(
        result.spike_times,
        result.spike_neurons,
        positions,
        compartment,
        electrode_positions(layout),
        layout,
    )
    features = {
        c: extract_features(
            {k: v for k, v in heard.items() if k.startswith(f"c{c}_")}, result.duration_s
        )
        for c in (0, 1)
    }
    coupling = electrode_coupling(heard, result.duration_s)
    return {
        "theta": theta,
        "summary": summary_vector(features, coupling),
        "features": features,
        "raster": _raster_snapshot(result, n_per_compartment),
        "coupling": coupling,
    }


def _raster_snapshot(result, n_per_compartment: int) -> dict[str, np.ndarray]:
    """The first seconds of spikes per compartment, for the app's raster plot.

    Neuron indices are renumbered within their compartment so both panels plot
    against the same y range.
    """
    compartment = result.compartment_of_neuron[result.spike_neurons]
    early = result.spike_times < RASTER_SECONDS
    snapshot = {}
    for c in (0, 1):
        keep = early & (compartment == c)
        neurons = result.spike_neurons[keep]
        offset = 0 if c == 0 else n_per_compartment
        snapshot[f"c{c}_times"] = result.spike_times[keep].astype(np.float32)
        snapshot[f"c{c}_neurons"] = (neurons - offset).astype(np.int16)
    return snapshot


def simulate_for_app(args: tuple) -> dict:
    """One simulation, in a form the app can call across a process boundary.

    Brian2 installs a SIGINT handler when it is imported, and `signal.signal`
    only works in a main thread - so importing it inside Streamlit's script
    runner raises. Running it in a child process gives it a main thread of its
    own, and keeps a simulator out of the web server either way.
    """
    point, duration_s, seed, n_per_compartment = args
    out = simulate_point(point, duration_s, seed, n_per_compartment)
    return {
        "features": out["features"],
        "raster": {k: v for k, v in out["raster"].items()},
        "summary": out["summary"],
        "coupling": out["coupling"],
    }


def _simulate_job(args) -> dict:
    """Worker entry point: one grid point."""
    point, duration_s, seed, n_per_compartment = args
    try:
        return {"point": point, **simulate_point(point, duration_s, seed, n_per_compartment)}
    except Exception:  # noqa: BLE001 - one bad draw must not lose the whole grid
        log.exception("grid point failed: %s", point)
        return {"point": point, "summary": None}


def run_grid(
    points: list[dict], duration_s: float, n_per_compartment: int, seed: int
) -> list[dict]:
    """Simulate every grid point in parallel."""
    import multiprocessing as mp

    jobs = [(point, duration_s, seed + i, n_per_compartment) for i, point in enumerate(points)]
    processes = max(1, min(mp.cpu_count() - 2, 16))
    log.info("simulating %d grid points on %d processes", len(jobs), processes)
    with mp.Pool(processes) as pool:
        return pool.map(_simulate_job, jobs, chunksize=1)


def sample_posterior(posterior, x: np.ndarray, n_samples: int, max_seconds: float = 20.0):
    """Posterior samples for one observation, time-capped like the HD15 analysis."""
    import torch

    samples = posterior.sample(
        (n_samples,),
        x=torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float32),
        show_progress_bars=False,
        max_sampling_time=max_seconds,
        return_partial_on_timeout=True,
    )
    return samples.detach().numpy() if hasattr(samples, "detach") else np.asarray(samples)


def save_run(path: Path, entry: dict, samples: np.ndarray) -> None:
    """One grid point: parameters, summary, features, raster and posterior samples."""
    np.savez_compressed(
        path,
        theta=entry["theta"],
        summary=entry["summary"],
        posterior_samples=samples,
        features=np.array(json.dumps(entry["features"])),
        coupling=np.array(json.dumps(entry["coupling"])),
        **entry["raster"],
    )


def _observed_range(values: np.ndarray) -> list[float]:
    """The 1st to 99th percentile of a feature, widened if it would collapse."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [-1.0, 1.0]
    low, high = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    if high - low < 0.2:
        middle = 0.5 * (low + high)
        low, high = middle - 0.5, middle + 0.5
    return [low, high]


def fit_mechanism_model(pairs: pd.DataFrame, alpha: float = 0.1, seed: int = 0) -> dict:
    """A classifier and conformal threshold fitted once, for the app to apply live.

    The app must answer a slider move immediately, so it cannot refit. The
    threshold comes from the same grouped calibration split the reported
    conformal numbers use, so a set shown in the app means what the report says.
    """
    from neuromps.models.conformal import _fit_and_calibrate
    from neuromps.models.dataset import delta_columns

    x = pairs[delta_columns()].to_numpy()
    y = pairs["label"].to_numpy()
    threshold, model = _fit_and_calibrate(x, y, pairs["group"].to_numpy(), alpha, seed)
    # nanpercentile, not percentile: three delta columns are missing for the
    # sources that publish no burst statistics, and a NaN bound reaches the app
    # as a slider with equal ends, which Streamlit rejects.
    ranges = {column: _observed_range(x[:, i]) for i, column in enumerate(delta_columns())}
    return {
        "model": model,
        "threshold": float(threshold),
        "alpha": alpha,
        "labels": [str(c) for c in model.classes_],
        "feature_ranges": ranges,
        "n_pairs": int(len(pairs)),
    }


def real_posteriors(posterior, quick: bool, n_samples: int) -> dict:
    """Posterior samples for the real two-chamber recordings, the HD15 result.

    Only two-chamber recordings can be shown here: the twin's summary vector has
    a slot for each compartment, so the single-chamber sources in the asset
    (which carry the gabazine, NBQX and TTX arms) cannot be fed to it at all.
    In this asset that leaves the bicuculline series.
    """
    from neuromps.twin.real_inference import real_summaries

    asset = for_reading(asset_path())
    summaries = real_summaries(pd.read_parquet(asset))
    if quick and len(summaries) > 12:
        summaries = summaries.iloc[:: len(summaries) // 12].head(12).reset_index(drop=True)
    x = summaries[list(SUMMARY_NAMES)].to_numpy(dtype=float)
    if not len(x):  # no two-chamber source in the asset: it is fetched with --full
        log.warning("no two-chamber recordings in %s; the demo ships a simulated grid only", asset)
        return {
            "samples": np.empty((0, n_samples, len(PARAMETER_NAMES))),
            "records": summaries.drop(columns=list(SUMMARY_NAMES)),
        }
    samples = np.stack([sample_posterior(posterior, row, n_samples) for row in x])
    return {"samples": samples, "records": summaries.drop(columns=list(SUMMARY_NAMES))}


def main(argv: list[str] | None = None) -> int:
    """Precompute the demo's grid, posteriors and classifier into outputs/demo/."""
    import pickle

    import torch

    from neuromps.models.dataset import load_pairs

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="3x3x2 grid, fewer posterior samples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runs_dir().mkdir(parents=True, exist_ok=True)
    posterior_path = posterior_file()  # resolved once: it may log a fallback
    if not posterior_path.exists():
        log.warning("no trained posterior at %s; run the twin stage first", posterior_path)
        return 0

    timings: dict[str, float] = {}
    axes = grid_axes(args.quick)
    points = grid_points(axes)
    duration_s = 5.0 if args.quick else 10.0
    n_per_compartment = 80 if args.quick else 150
    n_grid_samples = 60 if args.quick else 200
    n_real_samples = 60 if args.quick else 200

    started = time.time()
    entries = run_grid(points, duration_s, n_per_compartment, args.seed)
    timings["simulate_grid_s"] = round(time.time() - started, 1)

    posterior = torch.load(posterior_path, weights_only=False)
    torch.manual_seed(args.seed)
    started = time.time()
    rows = []
    for index, entry in enumerate(entries):
        if entry.get("summary") is None:
            continue
        samples = sample_posterior(posterior, entry["summary"], n_grid_samples)
        path = runs_dir() / f"run_{index:03d}.npz"
        save_run(path, entry, samples)
        rows.append(
            {
                **entry["point"],
                "run_file": path.name,
                "silent": bool(entry["features"][0]["mfr_hz"] in (0.0, None)),
                **{f"c0_{k}": v for k, v in entry["features"][0].items()},
                **{f"c1_{k}": v for k, v in entry["features"][1].items()},
            }
        )
    timings["grid_posteriors_s"] = round(time.time() - started, 1)
    pd.DataFrame(rows).to_parquet(demo_dir() / "grid.parquet", index=False)

    started = time.time()
    real = real_posteriors(posterior, args.quick, n_real_samples)
    np.savez_compressed(demo_dir() / "real_posteriors.npz", samples=real["samples"])
    real["records"].to_parquet(demo_dir() / "real_records.parquet", index=False)
    timings["real_posteriors_s"] = round(time.time() - started, 1)

    started = time.time()
    fitted = fit_mechanism_model(load_pairs())
    # pickle, not joblib: joblib arrives only as a scikit-learn dependency and is
    # not declared in pyproject.toml, and this object is small.
    (demo_dir() / "mechanism_model.pkl").write_bytes(pickle.dumps(fitted))
    timings["fit_classifier_s"] = round(time.time() - started, 1)

    results = {
        "label": "precomputed demo inputs; simulated grid plus real-recording posteriors",
        "grid": {k: [float(v) for v in values] for k, values in axes.items()},
        "n_grid_points": len(points),
        "n_grid_usable": len(rows),
        "n_real_recordings": int(real["samples"].shape[0]),
        "posterior_samples_per_point": n_grid_samples,
        "simulated_duration_s": duration_s,
        "n_per_compartment": n_per_compartment,
        "parameter_names": list(PARAMETER_NAMES),
        "fixed_parameters": {
            "inter_in_degree": FIXED_IN_DEGREE,
            "threshold_scale": FIXED_THRESHOLD,
        },
        "posterior_file": str(posterior_path.relative_to(REPO_ROOT)),
        "conformal_alpha": fitted["alpha"],
        "timings_s": timings,
        "quick": bool(args.quick),
    }
    (demo_dir() / "meta.json").write_text(json.dumps(results, indent=2) + "\n")
    update_metrics("demo_precompute", results)
    print(
        json.dumps(
            {
                k: results[k]
                for k in ("n_grid_points", "n_grid_usable", "n_real_recordings", "timings_s")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
