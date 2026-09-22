"""Turn the asset into a modelling table: exposure recordings paired with their baselines.

A pair is one recording unit (well, array or compartment) measured before and
during exposure, described by the log2 ratio of each feature. Ratios rather than
absolute values, because absolute firing rates differ by platform and culture far
more than drug effects do, and the label is a drug mechanism.

Each pair carries a ``group`` (source + species + unit) so cross-validation can
keep a culture on one side of the split: two conditions of the same well are not
independent samples.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.paths import asset_path, for_reading

log = logging.getLogger("neuromps.models")
REPO_ROOT = Path(__file__).resolve().parents[3]
DELTA_FEATURES = (
    "mfr_hz",
    "burst_rate_per_min",
    "burst_duration_s",
    "spikes_per_burst",
    "pct_spikes_in_bursts",
    "network_burst_rate_per_min",
    "synchrony_index",
    "isi_cv",
    "active_electrodes",
)
FLOOR = 0.05  # counts a silent recording as this fraction of its baseline, not as -inf
CONTROL = "control"


def delta_columns() -> list[str]:
    """Column names of the delta feature block."""
    return [f"delta.{f}" for f in DELTA_FEATURES]


def log_ratio(exposure: float | None, baseline: float | None) -> float:
    """log2(exposure / baseline).

    A silent exposure is floored at ``FLOOR`` of its baseline instead of running
    to -inf, so "activity abolished" stays a finite, comparable number. A zero or
    missing baseline gives NaN: the ratio is genuinely undefined, and inventing a
    value here would put a fabricated effect size into the model.
    """
    if exposure is None or baseline is None:
        return float("nan")
    exposure, baseline = float(exposure), float(baseline)
    if np.isnan(exposure) or np.isnan(baseline) or baseline <= 0:
        return float("nan")
    return float(np.log2(max(exposure, FLOOR * baseline) / baseline))


def _pair_key(row: pd.Series) -> tuple:
    """Identity of the culture unit, independent of the recording phase."""
    plate = str(row.get("plate_id") or "")
    for phase_word in ("Baseline", "Pharma", "TTX"):
        plate = plate.replace(phase_word, "")
    compartment = row.get("device.compartment_id")
    return (
        row["source"],
        row["cell_source.species"],
        row["unit_id"],
        plate,
        compartment if isinstance(compartment, str) else "",
    )


def build_pairs(records: pd.DataFrame, usable_only: bool = True) -> pd.DataFrame:
    """Pair each exposure recording with the baseline of the same unit.

    Args:
        records: the flattened asset (spike-time records only are used).
        usable_only: drop pairs where either recording failed QC.

    Returns:
        One row per pair: identity, label, dose and ``delta.<feature>`` columns.
    """
    df = records[records["data_level"] == "spike_times"].copy()
    if usable_only:
        df = df[df["qc.usable"].astype(bool)]
    if df.empty:  # `apply` on an empty frame returns a frame, not a column
        return pd.DataFrame(columns=PAIR_COLUMNS + delta_columns())
    df["_key"] = df.apply(_pair_key, axis=1)
    baselines = {row["_key"]: row for _, row in df[df["recording.phase"] == "baseline"].iterrows()}
    rows, unlabelled = [], 0
    for _, exposure in df[df["recording.phase"] == "exposure"].iterrows():
        if is_unlabelled_exposure(exposure):
            unlabelled += 1
            continue
        baseline = baselines.get(exposure["_key"])
        if baseline is None:
            continue
        rows.append(_pair_row(exposure, baseline))
    if unlabelled:
        log.info("dropped %d exposures with a compound applied but no mechanism", unlabelled)
    if not rows:
        return pd.DataFrame(columns=PAIR_COLUMNS + delta_columns())
    return pd.DataFrame(rows)


# Every column `_pair_row` produces. An empty table must carry them too: a
# source that failed to download leaves no pairs, and a consumer indexing
# `pairs["source"]` then raises a KeyError about a missing column instead of
# saying that there were no pairs. That is what a blocked fetch looked like in
# CI, and it took a traceback to work out.
PAIR_COLUMNS = [
    "pair_id",
    "group",
    "source",
    "species",
    "system",
    "unit_id",
    "compound",
    "dose_value",
    "dose_unit",
    "label",
]


def is_unlabelled_exposure(row: pd.Series) -> bool:
    """Whether a compound was applied to this recording without being identified.

    Such a recording must never become a pair. :func:`_pair_row` falls back to
    the control label when ``mechanism_class`` is missing, so an unlabelled
    exposure would enter the classifier as a control - a treated recording
    teaching the model what untreated looks like. Trujillo's drug sessions are
    the case in hand: the deposit records that a compound was applied and not
    which well received which, so the mechanism is genuinely unknown rather
    than absent.
    """
    role = row.get("compound.role")
    label = row.get("compound.mechanism_class")
    return role == "test" and not (isinstance(label, str) and label)


def _pair_row(exposure: pd.Series, baseline: pd.Series) -> dict:
    """One modelling row from an exposure/baseline pair."""
    label = exposure.get("compound.mechanism_class")
    label = CONTROL if not isinstance(label, str) or not label else label
    deltas = {
        f"delta.{f}": log_ratio(exposure.get(f"features.{f}"), baseline.get(f"features.{f}"))
        for f in DELTA_FEATURES
    }
    return {
        "pair_id": f"{exposure['record_id']}|{baseline['record_id']}",
        "group": f"{exposure['source']}:{exposure['cell_source.species']}:{exposure['unit_id']}",
        "source": exposure["source"],
        "species": exposure["cell_source.species"],
        "system": exposure["cell_source.system"],
        "unit_id": exposure["unit_id"],
        "compound": exposure.get("compound.name"),
        "dose_value": exposure.get("compound.dose_value"),
        "dose_unit": exposure.get("compound.dose_unit"),
        "label": label,
        **deltas,
    }


def load_pairs(asset: Path | None = None, usable_only: bool = True) -> pd.DataFrame:
    """Build the modelling table straight from the built asset.

    Reads the active run's asset by default; see :mod:`neuromps.paths`.
    """
    asset = Path(asset) if asset is not None else for_reading(asset_path())
    return build_pairs(pd.read_parquet(asset), usable_only=usable_only)
