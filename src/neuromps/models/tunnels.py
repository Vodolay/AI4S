"""Does coupling asymmetry track firing rate on a device with instrumented tunnels?

§7.2 found, in the twin, that the coupling-asymmetry measure moves further when
one chamber's drive changes than when the wiring changes — it reads relative
activity, not pathway direction. That is the report's sharpest methodological
finding, and until now it had been shown in simulation and in devices whose
connecting channels carry no electrodes.

Brewer's four-chamber cultures are the test that was missing. The tunnels
between chambers sit over electrodes, so the chambers are genuinely separate
populations connected through a measured path, and each culture yields six
chamber pairs. If asymmetry is tracking rate rather than direction, it should
line up with the two chambers' rate difference here as well, on real tissue and
without a simulator in sight.

The test is a rank correlation between a pair's asymmetry and the log ratio of
its two chambers' firing rates, resampled over **cultures** rather than pairs:
the six pairs of one culture share four populations and are anything but
independent.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neuromps.metrics import update_metrics
from neuromps.models.bootstrap import group_bootstrap_metric
from neuromps.paths import asset_path, figures_dir, for_reading

log = logging.getLogger("neuromps.models")
SOURCE = "brewer2021"
RESAMPLES = 2000
MIN_PAIRS = 8


def chamber_pairs(records: pd.DataFrame) -> pd.DataFrame:
    """One row per unordered chamber pair: its asymmetry and its rate ratio.

    Each pair is taken in one fixed orientation (compartments sorted), so a
    pair contributes once. Taking both orientations would double the rows with
    sign-flipped copies of the same measurement and halve every interval.
    """
    rows = []
    for device_id, device in records.groupby("device.device_id"):
        by_compartment = {
            str(r["device.compartment_id"]): r for _, r in device.iterrows()
        }
        for first, second in itertools.combinations(sorted(by_compartment), 2):
            link = _link(by_compartment[first], second)
            if link is None or link.get("asymmetry") is None:
                continue
            rate_first = _rate(by_compartment[first])
            rate_second = _rate(by_compartment[second])
            if not (rate_first > 0 and rate_second > 0):
                continue
            rows.append(
                {
                    "device_id": str(device_id),
                    "pair": f"{first}->{second}",
                    "asymmetry": float(link["asymmetry"]),
                    "peak_correlation": _value(link.get("peak_correlation")),
                    "lag_ms": _value(link.get("lag_ms")),
                    "log2_rate_ratio": float(np.log2(rate_first / rate_second)),
                }
            )
    return pd.DataFrame(rows)


def _link(row: pd.Series, to_compartment: str) -> dict | None:
    """The coupling entry from this compartment to the named one."""
    raw = row.get("coupling")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    # The asset stores this column as JSON text, as `models/asymmetry.py` also
    # has to allow for.
    links = json.loads(raw) if isinstance(raw, str) else raw
    for link in links or []:
        if link.get("to_compartment") == to_compartment:
            return link
    return None


def _rate(row: pd.Series) -> float:
    """Mean firing rate, or 0 when it is missing."""
    value = row.get("features.mfr_hz")
    return float(value) if value is not None and np.isfinite(float(value)) else 0.0


def _value(value) -> float | None:
    """Null-safe float for a metrics block."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return float(value)


def rate_tracking(pairs: pd.DataFrame, seed: int = 0) -> dict:
    """Rank correlation between asymmetry and the pair's rate ratio."""
    asymmetry = pairs["asymmetry"].to_numpy(dtype=float)
    ratio = pairs["log2_rate_ratio"].to_numpy(dtype=float)
    devices = pairs["device_id"].to_numpy()

    def metric(idx: np.ndarray) -> float:
        a, r = asymmetry[idx], ratio[idx]
        if len(set(a)) < 3 or len(set(r)) < 3:
            return float("nan")
        return float(spearmanr(a, r).statistic)

    interval = group_bootstrap_metric(metric, devices, n_resamples=RESAMPLES, seed=seed)
    observed = spearmanr(asymmetry, ratio)
    return {
        "spearman_rho": round(float(observed.statistic), 4),
        "spearman_p": round(float(observed.pvalue), 6),
        "spearman_rho_ci95": interval.get("ci95"),
        "n_pairs": int(len(pairs)),
        "n_devices": int(pairs["device_id"].nunique()),
        "excludes_zero": _excludes_zero(interval),
    }


def _excludes_zero(interval: dict) -> bool:
    """Whether a bootstrap interval lies wholly on one side of zero."""
    low, high = interval.get("ci95", [float("nan"), float("nan")])
    return bool(np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0))


def plot_pairs(pairs: pd.DataFrame, result: dict, path: Path) -> Path:
    """Asymmetry against rate ratio, one point per chamber pair."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    devices = sorted(pairs["device_id"].unique())
    colours = plt.get_cmap("cividis")(np.linspace(0.1, 0.9, len(devices)))
    for colour, device in zip(colours, devices, strict=False):
        rows = pairs[pairs["device_id"] == device]
        ax.scatter(
            rows["log2_rate_ratio"], rows["asymmetry"], s=34, color=colour, edgecolor="black",
            linewidth=0.4, label=device,
        )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("log2 firing-rate ratio of the two chambers")
    ax.set_ylabel("measured coupling asymmetry")
    ax.set_title(
        "REAL four-chamber cultures: coupling asymmetry against rate imbalance\n"
        f"Spearman rho = {result['spearman_rho']}, 95% CI "
        f"{result['spearman_rho_ci95']}, {result['n_pairs']} pairs "
        f"in {result['n_devices']} cultures",
        fontsize=9,
    )
    ax.legend(fontsize=6, ncol=2, title="culture", title_fontsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(records: pd.DataFrame, seed: int = 0) -> dict | None:
    """The whole analysis, or None when the source is not in the asset."""
    subset = records[records["source"] == SOURCE]
    if subset.empty:
        log.info("brewer2021 not in the asset; skipping the tunnel analysis")
        return None
    pairs = chamber_pairs(subset)
    if len(pairs) < MIN_PAIRS:
        log.warning("only %d chamber pairs with measurable coupling; skipping", len(pairs))
        return {"n_pairs": int(len(pairs)), "note": "too few pairs with coupling above the floor"}
    result = rate_tracking(pairs, seed)
    result["label"] = (
        "coupling asymmetry against chamber rate imbalance, four-chamber cultures "
        "with instrumented tunnels"
    )
    result["median_abs_asymmetry"] = round(float(pairs["asymmetry"].abs().median()), 4)
    # How much rate imbalance there was to detect an effect in. A null over a
    # narrow range says less than the same null over a wide one, and §7.2's
    # simulated effect was produced by driving one chamber far harder than the
    # other.
    imbalance = pairs["log2_rate_ratio"].abs()
    result["abs_log2_rate_ratio"] = {
        "median": round(float(imbalance.median()), 3),
        "p90": round(float(imbalance.quantile(0.9)), 3),
        "max": round(float(imbalance.max()), 3),
    }
    result["pairs_dropped_below_noise_floor"] = int(
        len(subset) * 3 // 2 - len(pairs)
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI: test the rate-tracking explanation on real four-chamber devices."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="accepted for pipeline symmetry")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asset = for_reading(asset_path())
    if not asset.exists():
        log.warning("no asset at %s; skipping", asset)
        return 0
    records = pd.read_parquet(asset)
    result = run(records, seed=args.seed)
    if result is None:
        return 0
    if "spearman_rho" in result:
        pairs = chamber_pairs(records[records["source"] == SOURCE])
        figure = plot_pairs(pairs, result, figures_dir() / "tunnel_asymmetry_vs_rate.png")
        result["figure"] = figure.name
    update_metrics("tunnel_rate_tracking", result)
    log.info(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
