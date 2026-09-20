"""Apply the trained posterior to real recordings.

The twin is only useful if it says something about measured data, so this takes
the two-chamber recordings from the asset, builds the **same** summary vector the
simulator produces, and asks the posterior what parameters could have generated
them.

Before any of that, it checks whether the real summaries even lie inside the
simulated distribution. A posterior evaluated on out-of-distribution data returns
confident nonsense, so the distribution check is reported first and gates how the
inferred values may be read.

The reference quantity is `gabaa_scale` under bicuculline: a GABA-A antagonist
should push it below 1 and further down with dose. HD14 found that parameter only
weakly identified (contraction 0.34), so the honest expectation is a direction,
not a value.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neuromps.metrics import update_metrics
from neuromps.twin.inference import (
    PARAMETER_NAMES,
    SIMULATION_CACHE,
    SUMMARY_NAMES,
    load_simulations,
    within_prior,
)

log = logging.getLogger("neuromps.twin")
REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET = REPO_ROOT / "data" / "processed" / "neuromps.parquet"
TWIN_DIR = REPO_ROOT / "outputs" / "twin"
POSTERIOR_FILE = TWIN_DIR / "hd14_posterior_simulated.pt"
CHAMBERS = ("red", "green")  # red -> compartment 0, green -> compartment 1
FEATURE_KEYS = (
    "mfr_hz",
    "burst_rate_per_min",
    "pct_spikes_in_bursts",
    "network_burst_rate_per_min",
    "synchrony_index",
    "isi_cv",
    "active_electrodes",
)
MIN_DISTINCT_DOSES = 3


def _coupling_between_chambers(row: pd.Series) -> dict:
    """The red -> green coupling entry of a record, if present."""
    raw = row.get("coupling")
    links = json.loads(raw) if isinstance(raw, str) else (raw or [])
    for link in links:
        if link.get("to_compartment") == CHAMBERS[1]:
            return link
    return {}


def real_summaries(records: pd.DataFrame) -> pd.DataFrame:
    """Build one simulator-shaped summary vector per device recording."""
    usable = records[
        (records["data_level"] == "spike_times")
        & records["device.compartment_id"].isin(CHAMBERS)
        & records["qc.usable"].astype(bool)
    ]
    rows = []
    keys = ["device.device_id", "compound.dose_value", "compound.dose_unit", "recording.phase"]
    for key, group in usable.groupby(keys, dropna=False):
        by_chamber = group.drop_duplicates("device.compartment_id").set_index(
            "device.compartment_id"
        )
        if not set(CHAMBERS) <= set(by_chamber.index):
            continue
        values = {}
        for compartment, chamber in enumerate(CHAMBERS):
            for feature in FEATURE_KEYS:
                value = by_chamber.at[chamber, f"features.{feature}"]
                values[f"c{compartment}_{feature}"] = (
                    float(value) if value is not None and np.isfinite(value) else 0.0
                )
        coupling = _coupling_between_chambers(by_chamber.loc[CHAMBERS[0]])
        values["log_rate_ratio"] = float(
            np.log2((values["c0_mfr_hz"] + 1e-3) / (values["c1_mfr_hz"] + 1e-3))
        )
        values["peak_correlation"] = float(coupling.get("peak_correlation") or 0.0)
        values["lag_ms"] = float(coupling.get("lag_ms") or 0.0)
        values["asymmetry"] = float(coupling.get("asymmetry") or 0.0)
        values["coupling_measured"] = 1.0 if coupling.get("asymmetry") is not None else 0.0
        device, dose, unit, phase = key
        rows.append(
            {
                "device_id": device,
                "dose_value": dose,
                "dose_unit": unit,
                "phase": phase,
                **{name: values[name] for name in SUMMARY_NAMES},
            }
        )
    return pd.DataFrame(rows)


def distribution_check(real_x: np.ndarray, simulated_x: np.ndarray) -> dict:
    """Per dimension, the share of real values inside the simulated 1-99 percentile range.

    Model misspecification shows up here: a dimension the simulator never
    produces means the posterior is extrapolating for every real recording.
    """
    low = np.percentile(simulated_x, 1, axis=0)
    high = np.percentile(simulated_x, 99, axis=0)
    by_dimension = {}
    for i, name in enumerate(SUMMARY_NAMES):
        inside = (real_x[:, i] >= low[i]) & (real_x[:, i] <= high[i])
        by_dimension[name] = {
            "fraction_inside": round(float(inside.mean()), 3),
            "simulated_range": [round(float(low[i]), 3), round(float(high[i]), 3)],
            "real_median": round(float(np.median(real_x[:, i])), 3),
        }
    outside = [n for n, v in by_dimension.items() if v["fraction_inside"] < 0.5]
    return {
        "n_real": int(len(real_x)),
        "mean_fraction_inside": round(
            float(np.mean([v["fraction_inside"] for v in by_dimension.values()])), 3
        ),
        "outside_dimensions": outside,
        "by_dimension": by_dimension,
    }


def dose_trend(
    doses: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict:
    """Monotone trend of an inferred parameter against dose.

    Recordings are not independent: one device contributes its whole dose series.
    When ``groups`` is given, the interval comes from a cluster bootstrap over
    devices, and that interval - not the naive p-value, which counts 113
    recordings as 113 independent observations - is what the claim rests on.
    """
    doses = np.asarray(doses, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(doses) & np.isfinite(values)
    if len(set(doses[mask])) < MIN_DISTINCT_DOSES:
        return {"spearman_rho": None, "p_value": None, "note": "too few distinct doses to test"}
    rho, p_value = spearmanr(doses[mask], values[mask])
    out = {
        "spearman_rho": round(float(rho), 3),
        "naive_p_value": round(float(p_value), 4),
        "n_recordings": int(mask.sum()),
        "direction": ("falls with dose" if rho < 0 else "rises with dose"),
    }
    if groups is None:
        out["note"] = "no grouping supplied; the naive p-value ignores repeated devices"
        return out
    from neuromps.models.bootstrap import group_bootstrap_metric

    def metric(idx: np.ndarray) -> float:
        drawn_doses, drawn_values = doses[idx], values[idx]
        keep = np.isfinite(drawn_doses) & np.isfinite(drawn_values)
        if len(set(drawn_doses[keep])) < MIN_DISTINCT_DOSES:
            return float("nan")
        return float(spearmanr(drawn_doses[keep], drawn_values[keep]).statistic)

    interval = group_bootstrap_metric(
        metric, np.asarray(groups)[mask], n_resamples=n_resamples, seed=seed
    )
    low, high = interval["ci95"]
    out["spearman_rho_ci95"] = [low, high]
    out["n_devices"] = int(len(set(np.asarray(groups)[mask])))
    out["bootstrap_resamples"] = interval["n_resamples"]
    out["excludes_zero"] = bool(low is not None and (low > 0 or high < 0))
    return out


def sensitivity_without_dimensions(
    posterior, x: np.ndarray, simulated_x: np.ndarray, dimensions: list[str], n_samples: int = 200
) -> list[dict]:
    """Re-infer with out-of-distribution dimensions replaced by the simulated median.

    If a conclusion survives neutralising the dimensions the simulator cannot
    reproduce, it does not rest on the posterior extrapolating.
    """
    patched = x.copy()
    for name in dimensions:
        i = SUMMARY_NAMES.index(name)
        patched[:, i] = float(np.median(simulated_x[:, i]))
    return infer_parameters(posterior, patched, n_samples=n_samples)


def infer_parameters(posterior, x: np.ndarray, n_samples: int = 500) -> list[dict]:
    """Posterior median and 90% interval per parameter, for each real recording."""
    import torch

    out = []
    for observation in x:
        samples = posterior.sample(
            (n_samples,),
            x=torch.as_tensor(observation, dtype=torch.float32),
            show_progress_bars=False,
        ).numpy()
        out.append(
            {
                name: {
                    "median": round(float(np.median(samples[:, i])), 4),
                    "ci90": [
                        round(float(np.percentile(samples[:, i], 5)), 4),
                        round(float(np.percentile(samples[:, i], 95)), 4),
                    ],
                }
                for i, name in enumerate(PARAMETER_NAMES)
            }
        )
    return out


def plot_dose_response(doses, values, trend: dict, path, title: str) -> None:
    """Inferred gabaa_scale against bicuculline dose."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    shown = np.where(np.asarray(doses) <= 0, 0.03, doses)  # baseline plotted at the left edge
    ax.scatter(shown, values, color="#D55E00", s=45, zorder=3)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="untreated (scale = 1)")
    ax.set_xscale("log")
    ax.set_xlabel("bicuculline (uM, baseline plotted at 0.03)")
    ax.set_ylabel("inferred gabaa_scale (posterior median)")
    rho = trend.get("spearman_rho")
    ax.set_title(f"{title}\nSpearman rho = {rho}, p = {trend.get('p_value')}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Infer parameters for the real two-chamber recordings and test the dose trend."""
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="fewer posterior samples")
    p.add_argument("--asset", type=Path, default=ASSET)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not POSTERIOR_FILE.exists():
        log.warning("no trained posterior at %s; run neuromps.twin.inference first", POSTERIOR_FILE)
        return 0
    TWIN_DIR.mkdir(parents=True, exist_ok=True)

    summaries = real_summaries(pd.read_parquet(args.asset))
    if summaries.empty:
        log.warning("no two-chamber recordings in the asset")
        return 0
    real_x = summaries[list(SUMMARY_NAMES)].to_numpy(dtype=float)
    cached = load_simulations(SIMULATION_CACHE)
    inside = within_prior(cached["theta"])
    check = distribution_check(real_x, cached["x"][inside])
    log.info(
        "distribution check: %.0f%% of real values inside the simulated range; outside: %s",
        100 * check["mean_fraction_inside"],
        check["outside_dimensions"],
    )

    posterior = torch.load(POSTERIOR_FILE, weights_only=False)
    inferred = infer_parameters(posterior, real_x, n_samples=200 if args.quick else 500)
    gabaa = np.array([row["gabaa_scale"]["median"] for row in inferred])
    doses = summaries["dose_value"].fillna(0.0).to_numpy(dtype=float)
    units = summaries["dose_unit"].fillna("uM").to_numpy()
    doses_um = np.where(units == "nM", doses / 1000.0, doses)
    devices = summaries["device_id"].to_numpy()
    trend = dose_trend(doses_um, gabaa, groups=devices, n_resamples=200 if args.quick else 1000)
    patched = sensitivity_without_dimensions(
        posterior,
        real_x,
        cached["x"][inside],
        check["outside_dimensions"],
        n_samples=200 if args.quick else 500,
    )
    gabaa_patched = np.array([row["gabaa_scale"]["median"] for row in patched])
    trend_patched = dose_trend(
        doses_um, gabaa_patched, groups=devices, n_resamples=200 if args.quick else 1000
    )

    results = {
        "label": "posterior trained on SIMULATIONS, applied to REAL recordings",
        "n_recordings": int(len(summaries)),
        "distribution_check": check,
        "gabaa_scale_dose_trend": trend,
        "gabaa_scale_dose_trend_without_ood_dimensions": trend_patched,
        "expected": "a GABA-A antagonist should lower gabaa_scale, and lower it further with dose",
        "per_recording": [
            {
                "device_id": row.device_id,
                "dose_uM": float(dose),
                "phase": row.phase,
                **{k: v for k, v in inferred[i].items()},
            }
            for i, (row, dose) in enumerate(zip(summaries.itertuples(), doses_um, strict=True))
        ],
        "caveat": (
            "HD14 found gabaa_scale only weakly identified (contraction 0.34) and no parameter "
            "passing the rank-uniformity check, so these medians are directional evidence at best"
        ),
    }
    (TWIN_DIR / "hd15_real_inference.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_dose_response(
        doses_um,
        gabaa,
        trend,
        TWIN_DIR / "hd15_real_dose_response.png",
        "Inferred GABA-A scale for REAL bicuculline recordings (posterior trained on simulations)",
    )
    update_metrics(
        "twin_hd15_real_inference", {k: v for k, v in results.items() if k != "per_recording"}
    )
    print(
        json.dumps(
            {
                "distribution_check": {
                    k: check[k] for k in ("n_real", "mean_fraction_inside", "outside_dimensions")
                },
                "gabaa_scale_dose_trend": trend,
                "gabaa_scale_dose_trend_without_ood_dimensions": trend_patched,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
