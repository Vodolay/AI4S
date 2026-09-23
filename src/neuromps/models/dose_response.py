"""Is the bicuculline signal in the features at all, or only missing from the twin?

§7.5 reports that the twin's posterior recovers no bicuculline dose trend from
113 real two-chamber recordings, and §7.5's paired estimator does not rescue it.
Both are statements about an *inverse model*. Neither says whether the dose
signal is present in the summary features the estimator was given.

This asks that directly: across the same recordings, does each of the nine
features move monotonically with dose? The answer localises the failure.

- If features move with dose and the posterior does not, the inverse model is
  what is broken, and the twin is the thing to fix.
- If the features do not move either, there is nothing for any estimator to
  recover from these summaries, and the twin is being blamed for its input.

Measured twice, because the two ask different things. **Absolute** summaries are
what §7.5's first estimator saw. **Baseline-subtracted** summaries are what its
paired successor saw: each exposure minus the untreated recording of its own
device, which is how pharmacology is quantified. Intervals cluster on the
device, and p-values are Holm-corrected across the nine features because asking
nine questions and reporting the smallest is how a null becomes a finding.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from neuromps.features.mea import FEATURE_NAMES

log = logging.getLogger("neuromps.models")

# The summary table names each feature per chamber; a device's response is the
# mean of its two chambers, which is what the twin's summary vector also pools.
CHAMBERS = ("c0", "c1")
RESAMPLES = 1000


def chamber_mean(summaries: pd.DataFrame, feature: str) -> pd.Series:
    """One value per recording: the mean over the chambers that carry it."""
    columns = [f"{c}_{feature}" for c in CHAMBERS if f"{c}_{feature}" in summaries.columns]
    if not columns:
        return pd.Series(np.nan, index=summaries.index)
    return summaries[columns].replace(0.0, np.nan).mean(axis=1, skipna=True)


def baseline_subtracted(summaries: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Each exposure's feature minus the untreated recording of the same device.

    A device with no untreated recording contributes nothing, for the same
    reason the paired estimator drops it: there is no baseline to subtract.
    """
    values = chamber_mean(summaries, feature)
    frame = pd.DataFrame(
        {
            "device_id": summaries["device_id"].to_numpy(),
            "dose": summaries["dose_value"].fillna(0.0).to_numpy(dtype=float),
            "value": values.to_numpy(dtype=float),
        }
    )
    rows = []
    for _device, group in frame.groupby("device_id"):
        untreated = group[group["dose"] <= 0]
        if untreated.empty or not np.isfinite(untreated["value"].iloc[0]):
            continue
        reference = float(untreated["value"].iloc[0])
        treated = group[group["dose"] > 0].copy()
        treated["value"] = treated["value"] - reference
        rows.append(treated)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=frame.columns)


def absolute_values(summaries: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Every recording's feature against its dose, baseline included at zero."""
    return pd.DataFrame(
        {
            "device_id": summaries["device_id"].to_numpy(),
            "dose": summaries["dose_value"].fillna(0.0).to_numpy(dtype=float),
            "value": chamber_mean(summaries, feature).to_numpy(dtype=float),
        }
    )


def _trend(frame: pd.DataFrame, seed: int) -> dict:
    """Rank correlation of a feature with dose, clustered on the device."""
    from scipy.stats import spearmanr

    from neuromps.models.bootstrap import group_bootstrap_metric

    frame = frame.astype({"value": float, "dose": float})
    frame = frame[np.isfinite(frame["value"].to_numpy()) & np.isfinite(frame["dose"].to_numpy())]
    dose = frame["dose"].to_numpy(dtype=float)
    value = frame["value"].to_numpy(dtype=float)
    groups = frame["device_id"].to_numpy()
    if len(frame) < 6 or len(set(dose)) < 3 or len(set(value)) < 2:
        return {"n": int(len(frame)), "note": "too few points or doses to fit a trend"}

    def metric(idx: np.ndarray) -> float:
        if len(set(dose[idx])) < 2 or len(set(value[idx])) < 2:
            return float("nan")
        return float(spearmanr(dose[idx], value[idx]).statistic)

    observed = spearmanr(dose, value)
    boot = group_bootstrap_metric(metric, groups, RESAMPLES, seed)
    interval = boot.get("ci95")
    return {
        "n": int(len(frame)),
        "n_devices": int(len(set(groups))),
        "spearman_rho": round(float(observed.statistic), 4),
        "naive_p_value": round(float(observed.pvalue), 6),
        "spearman_rho_ci95": interval,
        "excludes_zero": None
        if not interval or interval[0] is None
        else bool(interval[0] > 0 or interval[1] < 0),
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment over the features tested together."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(ordered)
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (n - rank) * p))
        adjusted[name] = round(running, 6)
    return adjusted


def trends(summaries: pd.DataFrame, seed: int = 0) -> dict:
    """Per-feature dose trends, absolute and baseline-subtracted."""
    out = {}
    for label, builder in (
        ("absolute", absolute_values),
        ("baseline_subtracted", baseline_subtracted),
    ):
        by_feature, raw_p = {}, {}
        for feature in FEATURE_NAMES:
            entry = _trend(builder(summaries, feature), seed)
            by_feature[feature] = entry
            if "naive_p_value" in entry:
                raw_p[feature] = entry["naive_p_value"]
        for feature, adjusted in holm(raw_p).items():
            by_feature[feature]["holm_p"] = adjusted
            by_feature[feature]["survives_correction"] = bool(
                adjusted < 0.05 and by_feature[feature].get("excludes_zero")
            )
        moving = sorted(f for f, e in by_feature.items() if e.get("survives_correction"))
        out[label] = {
            "by_feature": by_feature,
            "features_moving_with_dose": moving,
            "n_features_moving": len(moving),
            "n_features_tested": len(raw_p),
        }
    return out


def verdict(result: dict) -> dict:
    """Where the §7.5 failure sits, stated from the two comparisons."""
    absolute = result["absolute"]["n_features_moving"]
    paired = result["baseline_subtracted"]["n_features_moving"]
    if absolute or paired:
        where = (
            "the features carry a dose signal that the twin's posterior does not recover, "
            "so the inverse model is what fails"
        )
    else:
        where = (
            "no feature moves monotonically with dose after correction, so there is no dose "
            "signal in these summaries for any estimator to recover"
        )
    return {
        "n_features_moving_absolute": absolute,
        "n_features_moving_baseline_subtracted": paired,
        "reading": where,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: does the dose signal exist in the features the twin was given?"""
    import argparse
    import json

    from neuromps.metrics import update_metrics
    from neuromps.paths import asset_path, for_reading
    from neuromps.twin.real_inference import real_summaries

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="fewer resamples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asset = for_reading(asset_path())
    if not asset.exists():
        log.warning("no asset at %s; skipping", asset)
        update_metrics("feature_dose_response", {"status": "no asset"})
        return 0
    summaries = real_summaries(pd.read_parquet(asset))
    if summaries.empty:
        log.warning("no two-chamber recordings in this asset; skipping")
        update_metrics("feature_dose_response", {"status": "no two-chamber recordings"})
        return 0
    result = {
        "label": "REAL bicuculline recordings, features against dose",
        "question": (
            "does the dose signal exist in the summary features, or only fail to be recovered "
            "by the twin's posterior"
        ),
        "n_recordings": int(len(summaries)),
        "n_devices": int(summaries["device_id"].nunique()),
        "bootstrap_resamples": RESAMPLES,
        **trends(summaries, seed=args.seed),
    }
    result["verdict"] = verdict(result)
    update_metrics("feature_dose_response", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("absolute", "baseline_subtracted")},
            indent=2,
        )
    )
    for label in ("absolute", "baseline_subtracted"):
        print(f"\n{label}:")
        for feature, entry in result[label]["by_feature"].items():
            print(f"  {feature:30s} {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
