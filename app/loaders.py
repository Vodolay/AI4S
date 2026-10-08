"""Cached readers for everything the demo app shows.

Nothing here computes a result: the numbers come from `metrics.json` and the
precomputed `demo/` directory, written by `run_all.py --stage demo`. The one
exception is `live_simulation`, which the user asks for explicitly by pressing
a button, and which is cached on its parameter tuple so a repeat press is
instant.

The app shows the full run. On a machine where only `--quick` has been run -
a fresh clone following the README - the full run's files are absent, so each
path falls back to the quick run's copy and `run_root_note()` says which is on
screen. Resolution happens at import: a run finishes before the app starts.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from neuromps import paths

REPO_ROOT = Path(__file__).resolve().parents[1]


def _shown(full: Path, quick: Path) -> Path:
    """The full run's file, or the quick run's when only that one exists."""
    return full if full.exists() or not quick.exists() else quick


DEMO_DIR = _shown(paths.OUTPUTS_ROOT / "demo", paths.OUTPUTS_ROOT / paths.QUICK / "demo")
METRICS = _shown(
    paths.OUTPUTS_ROOT / "metrics.json", paths.OUTPUTS_ROOT / paths.QUICK / "metrics.json"
)
ASSET = _shown(
    paths.PROCESSED_ROOT / "neuromps.parquet",
    paths.PROCESSED_ROOT / paths.QUICK / "neuromps.parquet",
)
PROVENANCE = ASSET.parent / "provenance.json"
SCHEMA = REPO_ROOT / "schema" / "neuromps_schema.json"
FIGURES = _shown(paths.OUTPUTS_ROOT / "figures", paths.OUTPUTS_ROOT / paths.QUICK / "figures")
TWIN_FIGURES = _shown(paths.OUTPUTS_ROOT / "twin", paths.OUTPUTS_ROOT / paths.QUICK / "twin")
SHOWN_PATHS = (DEMO_DIR, METRICS, ASSET, FIGURES, TWIN_FIGURES)


def run_root_note() -> str:
    """One line for the app when any of this came from a quick run, not the full one.

    A machine can hold a mixture - the full run's figures beside a quick run's
    asset - so this names what fell back rather than claiming the whole page did.
    """
    quick = sorted({p.name for p in SHOWN_PATHS if paths.QUICK in p.parts})
    if not quick:
        return ""
    return (
        f"Showing a `--quick` run for: {', '.join(quick)}. That run uses the small-tier "
        "sources with fewer draws. Run `python run_all.py --full` for the numbers the "
        "report cites."
    )


def diagram(name: str) -> Path | None:
    """A generated model diagram, or None when the figures have not been drawn.

    The three diagrams are written by `neuromps.report.diagrams` during the twin
    stage. A clone that has not run the pipeline has the metrics but not the
    figures, and the app should say so rather than break.
    """
    candidate = FIGURES / name
    return candidate if candidate.exists() else None


MISSING_DEMO = (
    "The demo inputs are missing. Run `python run_all.py --quick` (or "
    "`python run_all.py --stage demo`) to precompute them."
)


def _by_compartment(features: dict) -> dict[int, dict]:
    """Compartment-keyed features, whatever a JSON round-trip did to the keys.

    JSON has no integer keys, so a features dict written to disk comes back
    keyed "0" and "1". Normalising here means no caller has to know that.
    """
    return {int(key): value for key, value in features.items()}


def _stamp(path: Path) -> float:
    """Modification time, so a cache entry expires when a stage is rerun."""
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_metrics(_stamp_value: float = 0.0) -> dict:
    """Every reported number, exactly as the report cites it."""
    return json.loads(METRICS.read_text()) if METRICS.exists() else {}


@st.cache_data(show_spinner=False)
def load_asset(_stamp_value: float = 0.0) -> pd.DataFrame:
    """The harmonized record table."""
    return pd.read_parquet(ASSET) if ASSET.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_schema(_stamp_value: float = 0.0) -> dict:
    """The record schema the asset validates against."""
    return json.loads(SCHEMA.read_text()) if SCHEMA.exists() else {}


@st.cache_data(show_spinner=False)
def load_provenance(_stamp_value: float = 0.0) -> dict:
    """The provenance record written beside the asset."""
    return json.loads(PROVENANCE.read_text()) if PROVENANCE.exists() else {}


@st.cache_data(show_spinner=False)
def load_demo_meta(_stamp_value: float = 0.0) -> dict:
    """Grid axes, timings and sizes recorded by the demo stage."""
    path = DEMO_DIR / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_grid(_stamp_value: float = 0.0) -> pd.DataFrame:
    """One row per precomputed twin simulation."""
    path = DEMO_DIR / "grid.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_run(run_file: str) -> dict:
    """One grid point: raster, features, summary and posterior samples."""
    with np.load(DEMO_DIR / "runs" / run_file, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files if k not in ("features", "coupling")}
        out["features"] = _by_compartment(json.loads(str(data["features"])))
        out["coupling"] = json.loads(str(data["coupling"]))
    return out


@st.cache_data(show_spinner=False)
def load_real_posteriors(_stamp_value: float = 0.0) -> tuple[np.ndarray, pd.DataFrame]:
    """Posterior samples per real two-chamber recording, with their metadata."""
    samples = DEMO_DIR / "real_posteriors.npz"
    records = DEMO_DIR / "real_records.parquet"
    if not samples.exists() or not records.exists():
        return np.empty((0, 0, 0)), pd.DataFrame()
    with np.load(samples, allow_pickle=False) as data:
        return data["samples"], pd.read_parquet(records)


@st.cache_data(show_spinner=False)
def load_pairs(_stamp_value: float = 0.0) -> pd.DataFrame:
    """Treated/baseline pairs, the rows the mechanism classifier is defined on."""
    from neuromps.models.dataset import load_pairs as build

    return build(ASSET) if ASSET.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_mechanism_model() -> dict:
    """The fitted classifier and conformal threshold, loaded once per session."""
    path = DEMO_DIR / "mechanism_model.pkl"
    return pickle.loads(path.read_bytes()) if path.exists() else {}


@st.cache_resource(show_spinner=False)
def load_posterior_estimator():
    """The trained `sbi` posterior, for the optional live run."""
    import torch

    from neuromps.twin.real_inference import POSTERIOR_NAME

    posterior = TWIN_FIGURES / POSTERIOR_NAME  # TWIN_FIGURES is the twin output dir
    if not posterior.exists():
        return None
    return torch.load(posterior, weights_only=False)


@st.cache_data(show_spinner=False)
def real_baseline_features(_stamp_value: float = 0.0) -> dict[str, list[float]]:
    """Untreated human-derived recordings, for comparing a simulated run against.

    This is the distribution the twin was calibrated towards, so it is what makes
    a simulated feature vector readable as close to or far from real data.
    """
    records = load_asset(_stamp(ASSET))
    if records.empty:
        return {}
    human = records[
        (records["cell_source.species"] == "human")
        & (records["recording.phase"] == "baseline")
        & records["qc.usable"]
    ]
    if human.empty:
        human = records[(records["recording.phase"] == "baseline") & records["qc.usable"]]
    return {
        column.removeprefix("features."): human[column].dropna().tolist()
        for column in human.columns
        if column.startswith("features.")
    }


@st.cache_data(show_spinner=True)
def live_simulation(ratio: float, gabaa: float, ampa: float, duration_s: float = 6.0) -> dict:
    """Run Brian2 once at these values, in its own interpreter. Cached on the tuple.

    The subprocess is not an optimisation. Brian2 registers a signal handler on
    import, which raises inside Streamlit's script-runner thread, and a spawned
    multiprocessing child would re-import the app script. A plain subprocess
    avoids both, and keeps the simulator out of the server process.
    """
    import subprocess
    import sys
    import tempfile
    import time

    started = time.time()
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "live.npz"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "neuromps.demo.simulate_one",
                "--ratio",
                str(ratio),
                "--gabaa",
                str(gabaa),
                "--ampa",
                str(ampa),
                "--duration",
                str(duration_s),
                "--out",
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=REPO_ROOT,
            check=False,
        )
        if completed.returncode != 0 or not target.exists():
            return {"error": (completed.stderr or "simulation failed").strip()[-600:]}
        with np.load(target, allow_pickle=False) as data:
            raster = {k: data[k] for k in data.files if k.endswith(("_times", "_neurons"))}
            result = {
                "summary": data["summary"],
                "features": _by_compartment(json.loads(str(data["features"]))),
                "coupling": json.loads(str(data["coupling"])),
                "raster": raster,
            }
    result["elapsed_s"] = round(time.time() - started, 2)
    result["duration_s"] = duration_s
    return result


def stamps() -> dict[str, float]:
    """Cache keys for the files the app reads."""
    return {
        "metrics": _stamp(METRICS),
        "asset": _stamp(ASSET),
        "demo": _stamp(DEMO_DIR / "meta.json"),
    }
