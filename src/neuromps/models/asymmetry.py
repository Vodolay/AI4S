"""Do heterogeneous two-population devices couple more asymmetrically than homogeneous ones?

The digital twin's claim is about directed connectivity between compartments, so
the asset needs a measured counterpart. The ScreenNeuroPharm archive supplies
one: devices whose two chambers hold the same population (CxCx, HpHp) and
devices whose chambers hold different ones (CxHp).

The test is deliberately conservative:

- **One value per device.** Conditions of one chip are not independent, so each
  device contributes the median over its conditions.
- **Magnitude, not signed direction.** Which electrode-colour group holds which
  population is not recorded, so the sign is arbitrary per device; only
  |asymmetry| is comparable.
- **Chamber-to-chamber links only.** The interface strip lies between the
  chambers and is not a chamber.

With about six devices per group this can support "these groups differ", not an
effect size anyone should quote as a constant.
"""

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

log = logging.getLogger("neuromps.models")
CHAMBERS = ("red", "green")
HETEROGENEOUS = ("CxHp",)
DEVICE_FAMILY = "screenneuropharm_bic"
MIN_PER_GROUP = 3


@dataclass(frozen=True)
class ChamberPair:
    """One measured link between two chambers of a device."""

    device_id: str
    device_type: str
    asymmetry: float
    lag_ms: float | None
    peak_correlation: float | None


def _links(records: pd.DataFrame) -> list[ChamberPair]:
    """Chamber-to-chamber coupling links from compartment records."""
    out = []
    for _, row in records.iterrows():
        compartment = row.get("device.compartment_id")
        device_id = row.get("device.device_id")
        if compartment not in CHAMBERS or not isinstance(device_id, str):
            continue
        raw = row.get("coupling")
        for link in json.loads(raw) if isinstance(raw, str) else (raw or []):
            if link.get("to_compartment") not in CHAMBERS:
                continue
            if link.get("asymmetry") is None:
                continue
            out.append(
                ChamberPair(
                    device_id=device_id,
                    device_type=device_id.split("-")[1],
                    asymmetry=float(link["asymmetry"]),
                    lag_ms=link.get("lag_ms"),
                    peak_correlation=link.get("peak_correlation"),
                )
            )
    return out


def device_asymmetry(records: pd.DataFrame) -> pd.DataFrame:
    """One row per device: median |asymmetry| and |lag| over its conditions."""
    links = _links(records)
    if not links:
        return pd.DataFrame(columns=["device_id", "device_type", "median_abs_asymmetry"])
    frame = pd.DataFrame([link.__dict__ for link in links])
    frame["abs_asymmetry"] = frame["asymmetry"].abs()
    frame["abs_lag_ms"] = frame["lag_ms"].abs()
    grouped = frame.groupby(["device_id", "device_type"], as_index=False).agg(
        median_abs_asymmetry=("abs_asymmetry", "median"),
        median_abs_lag_ms=("abs_lag_ms", "median"),
        median_peak_correlation=("peak_correlation", "median"),
        n_conditions=("asymmetry", "size"),
    )
    grouped["heterogeneous"] = grouped["device_type"].isin(HETEROGENEOUS)
    return grouped


def by_device_type(devices: pd.DataFrame, column: str = "median_abs_asymmetry") -> dict:
    """Per-type medians and a three-way test.

    Pooling the two homogeneous types can hide the fact that they differ from
    each other, which would make a 'heterogeneous vs homogeneous' result an
    artefact of how the pool was composed.
    """
    groups = {
        str(device_type): part[column].to_numpy()
        for device_type, part in devices.groupby("device_type")
    }
    out = {
        "column": column,
        "by_type": {
            name: {
                "n_devices": int(len(values)),
                "median": round(float(np.median(values)), 3),
                "min": round(float(np.min(values)), 3),
                "max": round(float(np.max(values)), 3),
            }
            for name, values in sorted(groups.items())
        },
    }
    usable = [v for v in groups.values() if len(v) >= MIN_PER_GROUP]
    if len(usable) >= 3:
        statistic, p_value = kruskal(*usable)
        out["kruskal_p"] = round(float(p_value), 4)
        out["kruskal_statistic"] = round(float(statistic), 3)
    raw = {}
    names = sorted(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if len(groups[a]) < MIN_PER_GROUP or len(groups[b]) < MIN_PER_GROUP:
                continue
            _, p_value = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
            raw[f"{a}_vs_{b}"] = float(p_value)
    out["pairwise"] = {k: round(v, 4) for k, v in raw.items()}
    out["pairwise_holm"] = holm(raw)
    return out


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values: several pairwise tests are run per metric."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(ordered)
    adjusted, running = {}, 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (n - rank)))
        adjusted[name] = round(running, 4)
    return adjusted


def compare_device_types(devices: pd.DataFrame) -> dict:
    """Mann-Whitney comparison of |asymmetry| between device types, with an effect size."""
    hetero = devices[devices["heterogeneous"]]["median_abs_asymmetry"].to_numpy()
    homo = devices[~devices["heterogeneous"]]["median_abs_asymmetry"].to_numpy()
    out = {
        "n_heterogeneous": int(len(hetero)),
        "n_homogeneous": int(len(homo)),
        "median_heterogeneous": round(float(np.median(hetero)), 3) if len(hetero) else None,
        "median_homogeneous": round(float(np.median(homo)), 3) if len(homo) else None,
        "device_types_heterogeneous": sorted(set(devices[devices["heterogeneous"]]["device_type"])),
        "device_types_homogeneous": sorted(set(devices[~devices["heterogeneous"]]["device_type"])),
    }
    if len(hetero) < MIN_PER_GROUP or len(homo) < MIN_PER_GROUP:
        out["p_value"] = None
        out["rank_biserial"] = None
        out["note"] = f"too few devices to test (need {MIN_PER_GROUP} per group)"
        return out
    statistic, p_value = mannwhitneyu(hetero, homo, alternative="two-sided")
    out["p_value"] = round(float(p_value), 4)
    out["rank_biserial"] = round(float(2 * statistic / (len(hetero) * len(homo)) - 1), 3)
    out["test"] = "Mann-Whitney U, two-sided, one value per device"
    return out


def plot_device_types(devices: pd.DataFrame, column: str, path, title: str, ylabel: str) -> None:
    """One point per device, grouped by device type."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["CxCx", "HpHp", "CxHp"]
    colours = {"CxCx": "#0072B2", "HpHp": "#009E73", "CxHp": "#D55E00"}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, device_type in enumerate(order):
        values = devices[devices["device_type"] == device_type][column].to_numpy()
        if not len(values):
            continue
        ax.scatter(
            i + rng.uniform(-0.08, 0.08, len(values)),
            values,
            s=45,
            color=colours[device_type],
            label=f"{device_type} (n={len(values)})",
            zorder=3,
        )
        ax.hlines(np.median(values), i - 0.22, i + 0.22, color="black", linewidth=2, zorder=4)
    ax.set_xticks(range(len(order)), ["CxCx\n(same)", "HpHp\n(same)", "CxHp\n(different)"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run the device-type comparison and write metrics and a figure."""
    import argparse
    from pathlib import Path

    from neuromps.asset import compartment_records
    from neuromps.metrics import update_metrics
    from neuromps.paths import asset_path, figures_dir, for_reading

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asset = args.asset or for_reading(asset_path())
    figures = figures_dir()
    figures.mkdir(parents=True, exist_ok=True)

    records = pd.read_parquet(asset)
    compartments = compartment_records(records)
    # This analysis is about one device family - the cortical/hippocampal
    # three-chamber devices whose chamber contents the device_id encodes. Other
    # two-chamber sources carry compartments too, and pooling them here would
    # silently answer a different question with the same words.
    compartments = compartments[compartments["source"] == DEVICE_FAMILY]
    if compartments.empty:
        log.warning(
            "no compartment records in %s; skipping (the two-chamber source is fetched "
            "with --full). Any compartment_asymmetry in metrics.json is from an earlier run.",
            asset,
        )
        return 0
    devices = device_asymmetry(compartments)
    results = {
        "n_devices": int(len(devices)),
        "per_device": devices.to_dict("records"),
        "pooled_heterogeneous_vs_homogeneous": compare_device_types(devices),
        "by_device_type": {
            column: by_device_type(devices, column)
            for column in ("median_abs_asymmetry", "median_abs_lag_ms", "median_peak_correlation")
        },
    }
    plot_device_types(
        devices,
        "median_abs_lag_ms",
        figures / "compartment_lag_by_device_type.png",
        "Inter-compartment coupling lag by device type (real data)\n"
        "one point per device, median over its bicuculline conditions",
        "median |lag| between chambers (ms)",
    )
    update_metrics("compartment_asymmetry", results)
    log.info("wrote metrics and %s", figures / "compartment_lag_by_device_type.png")
    import json

    print(json.dumps(results["by_device_type"]["median_abs_lag_ms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
