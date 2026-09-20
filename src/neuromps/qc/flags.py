"""Cross-record QC flags: replicate divergence, feature outliers, missing baseline pairs.

These need context that a single record does not have (its replicate wells, its
own baseline recording), so they are applied as a pass over the whole record set
after the per-source ETL has run.

Flags are informational: they never silently change ``qc.usable``. A diverging
replicate may still be a real measurement, and the consumer decides.
"""

import logging
from collections.abc import Iterable, Sequence

import numpy as np

log = logging.getLogger("neuromps.qc")

MAD_TO_SIGMA = 1.4826
MEAN_AD_TO_SIGMA = 1.2533  # fallback scale when the MAD is exactly zero
DEFAULT_Z = 3.5
MIN_GROUP = 3
FLAG_FEATURES = ("mfr_hz", "burst_rate_per_min", "pct_spikes_in_bursts", "synchrony_index")


def robust_z(values: Sequence[float]) -> list[float]:
    """Median/MAD z-scores, robust to the outliers we are looking for.

    When more than half the values are identical the MAD is exactly zero, which
    is common for replicate wells and would hide the single divergent one. In
    that case the mean absolute deviation is used as the scale instead. If that
    is zero too there is genuinely no spread, and the scores are all zero.
    """
    arr = np.asarray(list(values), dtype=float)
    median = float(np.nanmedian(arr))
    scale = MAD_TO_SIGMA * float(np.nanmedian(np.abs(arr - median)))
    if scale == 0:
        scale = MEAN_AD_TO_SIGMA * float(np.nanmean(np.abs(arr - median)))
    if scale == 0 or not np.isfinite(scale):
        return [0.0] * arr.size
    return list((arr - median) / scale)


def replicate_group(record: dict) -> tuple:
    """Key identifying wells that should be replicates of each other."""
    compound = record.get("compound", {})
    return (
        record["source"],
        record.get("plate_id"),
        record.get("recording", {}).get("phase"),
        compound.get("name"),
        compound.get("dose_value"),
        record["cell_source"]["species"],
    )


def _pair_key(record: dict) -> tuple:
    """Key pairing a well's recordings across phases (plate name minus the phase word)."""
    plate = str(record.get("plate_id") or "")
    for phase_word in ("Baseline", "Pharma", "TTX"):
        plate = plate.replace(phase_word, "")
    return (record["source"], plate, record["unit_id"])


def _add(record: dict, flag: str, reason: str | None = None) -> None:
    """Add a QC flag once, keeping the list sorted and unique, and note why."""
    flags = set(record.setdefault("qc", {}).setdefault("flags", []))
    flags.add(flag)
    record["qc"]["flags"] = sorted(flags)
    if reason:
        existing = record["qc"].get("note") or ""
        parts = [p for p in existing.split("; ") if p and p != reason]
        record["qc"]["note"] = "; ".join([*parts, reason])


def add_cross_record_flags(
    records: list[dict], z_threshold: float = DEFAULT_Z, min_group: int = MIN_GROUP
) -> list[dict]:
    """Flag diverging replicates, outlier feature values and unpaired exposures.

    Args:
        records: schema records; modified in place and returned.
        z_threshold: robust z above which a well counts as diverging.
        min_group: replicate groups smaller than this are never flagged.
    """
    _flag_replicates(records, z_threshold, min_group)
    _flag_missing_baselines(records)
    return records


def _flag_replicates(records: list[dict], z_threshold: float, min_group: int) -> None:
    """Compare each well against its replicate group, feature by feature."""
    groups: dict[tuple, list[dict]] = {}
    for rec in records:
        if rec.get("features"):
            groups.setdefault(replicate_group(rec), []).append(rec)
    for group in groups.values():
        if len(group) < min_group:
            continue
        for feature in FLAG_FEATURES:
            values = [r["features"].get(feature) for r in group]
            if sum(v is not None for v in values) < min_group:
                continue
            filled = [np.nan if v is None else float(v) for v in values]
            for rec, z in zip(group, robust_z(filled), strict=True):
                if np.isfinite(z) and abs(z) > z_threshold:
                    reason = f"{feature} z={z:+.1f} vs replicates"
                    _add(rec, "replicate_divergence", reason)
                    _add(rec, "outlier_feature_value")


def _flag_missing_baselines(records: Iterable[dict]) -> None:
    """Flag exposure recordings whose well has no baseline recording."""
    records = list(records)
    have_baseline = {
        _pair_key(r) for r in records if r.get("recording", {}).get("phase") == "baseline"
    }
    for rec in records:
        is_exposure = rec.get("recording", {}).get("phase") == "exposure"
        if is_exposure and _pair_key(rec) not in have_baseline:
            _add(rec, "missing_baseline_pair")
