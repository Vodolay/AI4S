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

from neuromps.asset import compartment_records
from neuromps.metrics import update_metrics
from neuromps.paths import asset_path, for_reading, simulation_cache, twin_dir
from neuromps.twin.inference import (
    PARAMETER_NAMES,
    POSTERIOR_NAME,
    SUMMARY_NAMES,
    load_simulations,
    within_prior,
)

log = logging.getLogger("neuromps.twin")
REPO_ROOT = Path(__file__).resolve().parents[3]


def posterior_file() -> Path:
    """The trained posterior this run should use, its own or the full run's."""
    return for_reading(twin_dir() / POSTERIOR_NAME)


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
# Columns of the table :func:`real_summaries` returns, named so the empty case
# is still a frame consumers can index by column.
SUMMARY_TABLE_COLUMNS = ("device_id", "dose_value", "dose_unit", "phase", *SUMMARY_NAMES)


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
    chambers = compartment_records(records)
    if chambers.empty:  # a --quick asset carries no device block at all
        return pd.DataFrame(columns=list(SUMMARY_TABLE_COLUMNS))
    usable = chambers[
        (chambers["data_level"] == "spike_times")
        & chambers["device.compartment_id"].isin(CHAMBERS)
        & chambers["qc.usable"].astype(bool)
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
    return pd.DataFrame(rows, columns=list(SUMMARY_TABLE_COLUMNS))


def distribution_check(real_x: np.ndarray, simulated_x: np.ndarray) -> dict:
    """Per dimension, how well the simulations actually cover the real values.

    Two tests, because they fail differently. `fraction_inside` asks whether the
    real values fall in the simulated 1-99 percentile *range*; a dimension the
    simulator never produces means the posterior extrapolates for every real
    recording. `simulated_support_in_real_iqr` asks whether simulated draws are
    actually *dense* where the real values sit: a bimodal simulated dimension
    (silent networks at 0, saturated ones at the electrode count) passes the
    range test while leaving the real values in an empty gap between the modes.
    """
    low = np.percentile(simulated_x, 1, axis=0)
    high = np.percentile(simulated_x, 99, axis=0)
    by_dimension = {}
    for i, name in enumerate(SUMMARY_NAMES):
        inside = (real_x[:, i] >= low[i]) & (real_x[:, i] <= high[i])
        real_low, real_high = np.percentile(real_x[:, i], [25, 75])
        support = np.mean((simulated_x[:, i] >= real_low) & (simulated_x[:, i] <= real_high))
        by_dimension[name] = {
            "fraction_inside": round(float(inside.mean()), 3),
            "simulated_support_in_real_iqr": round(float(support), 3),
            "simulated_range": [round(float(low[i]), 3), round(float(high[i]), 3)],
            "real_median": round(float(np.median(real_x[:, i])), 3),
        }
    outside = [n for n, v in by_dimension.items() if v["fraction_inside"] < 0.5]
    unsupported = [
        n
        for n, v in by_dimension.items()
        if n not in outside and v["simulated_support_in_real_iqr"] < 0.05
    ]
    return {
        "n_real": int(len(real_x)),
        "unsupported_dimensions": unsupported,
        "support_note": (
            "unsupported_dimensions pass the range test but fewer than 5% of simulated draws "
            "land in the real interquartile range, so the posterior is still interpolating "
            "across a gap there"
        ),
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


def prior_acceptance(posterior, x: np.ndarray, n_draws: int = 2000) -> dict:
    """Share of raw posterior draws that land inside the prior box, per recording.

    This is a misspecification diagnostic, not a tuning knob. The posterior is
    only trained where the prior has support, so a low acceptance rate means the
    density estimator, shown a real recording, wants to put its mass outside the
    region it was ever trained on. Everything downstream is then a truncated
    tail of that density, and the medians must be read with that in mind.
    """
    import torch

    rates = []
    for observation in x:
        raw = posterior.sample(
            (n_draws,),
            x=torch.as_tensor(observation, dtype=torch.float32),
            show_progress_bars=False,
            reject_outside_prior=False,
        )
        raw = raw.detach().numpy() if hasattr(raw, "detach") else np.asarray(raw)
        rates.append(float(within_prior(raw).mean()))
    return {
        "median_acceptance": round(float(np.median(rates)), 5),
        "min_acceptance": round(float(np.min(rates)), 5),
        "max_acceptance": round(float(np.max(rates)), 5),
        "fraction_below_1_percent": round(float(np.mean(np.array(rates) < 0.01)), 3),
        "n_draws_per_recording": int(n_draws),
        "note": (
            "share of raw posterior draws inside the prior box; low values mean the "
            "posterior is extrapolating outside its training region for real inputs"
        ),
    }


def infer_parameters(
    posterior, x: np.ndarray, n_samples: int = 500, max_seconds: float = 15.0
) -> list[dict]:
    """Posterior median and 90% interval per parameter, for each real recording.

    Sampling is time-capped: when the posterior mass sits largely outside the
    prior (see :func:`prior_acceptance`) rejection sampling can run for minutes
    per recording, and a partial sample is more honest than an unbounded wait.
    """
    import torch

    out = []
    for observation in x:
        samples = posterior.sample(
            (n_samples,),
            x=torch.as_tensor(observation, dtype=torch.float32),
            show_progress_bars=False,
            max_sampling_time=max_seconds,
            return_partial_on_timeout=True,
        )
        samples = samples.detach().numpy() if hasattr(samples, "detach") else np.asarray(samples)
        row = {
            name: {
                "median": round(float(np.median(samples[:, i])), 4),
                "ci90": [
                    round(float(np.percentile(samples[:, i], 5)), 4),
                    round(float(np.percentile(samples[:, i], 95)), 4),
                ],
            }
            for i, name in enumerate(PARAMETER_NAMES)
        }
        row["n_effective_samples"] = int(len(samples))
        out.append(row)
    return out


def baseline_offset(
    values: np.ndarray,
    doses: np.ndarray,
    groups: np.ndarray | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict:
    """Where the posterior puts an *untreated* recording, which must be near 1.

    An untreated chip has no drug applied, so a well-behaved posterior should
    return the knob to its neutral value. If untreated recordings infer a value
    far from 1, the parameter is absorbing general model mismatch, and any dose
    trend measured on top of that offset is measured on a shifted baseline.
    """
    from neuromps.models.bootstrap import ci95, group_bootstrap_indices

    untreated = np.asarray(doses) <= 0
    if untreated.sum() < 3:
        return {"median": None, "note": "too few untreated recordings"}
    baseline_values = np.asarray(values)[untreated]
    median = float(np.median(baseline_values))
    out = {
        "median": round(median, 4),
        "n_untreated": int(untreated.sum()),
        "neutral_value": 1.0,
    }
    if groups is not None:
        base_groups = np.asarray(groups)[untreated]
        draws = [
            float(np.median(baseline_values[idx]))
            for idx in group_bootstrap_indices(
                base_groups, np.random.default_rng(seed), n_resamples
            )
        ]
        low, high = ci95(np.array(draws))
        out["median_ci95"] = [round(low, 4), round(high, 4)]
        out["includes_neutral"] = bool(low <= 1.0 <= high)
        out["bootstrap_resamples"] = int(n_resamples)
        out["n_devices"] = int(len(set(base_groups.tolist())))
    return out


def trend_caption(trend: dict) -> str:
    """One line describing the trend by its clustered interval, not a naive p-value.

    The p-value across recordings treats 113 recordings from 16 devices as
    independent, so it is not what a reader should be shown on a figure.
    """
    rho = trend.get("spearman_rho")
    if rho is None:
        return "no trend estimated: " + str(trend.get("note", ""))
    low, high = trend.get("spearman_rho_ci95", (None, None))
    if low is None:
        return f"Spearman rho = {rho} (no cluster interval)"
    verdict = "excludes zero" if trend.get("excludes_zero") else "includes zero"
    return (
        f"Spearman rho = {rho}, 95% CI [{low}, {high}] over "
        f"{trend.get('n_devices')} devices - {verdict}"
    )


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
    import textwrap

    wrapped = "\n".join(textwrap.wrap(title, 72))
    ax.set_title(f"{wrapped}\n{trend_caption(trend)}", fontsize=8)
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
    p.add_argument("--asset", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0, help="seeds posterior sampling, which is random")
    p.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="subsample recordings (quick mode uses 8 to stay inside the run_all budget)",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)  # posterior sampling is stochastic; without this, rho drifts
    posterior_path = posterior_file()  # resolved once: it may log a fallback
    if not posterior_path.exists():
        log.warning("no trained posterior at %s; run neuromps.twin.inference first", posterior_path)
        return 0
    twin_dir().mkdir(parents=True, exist_ok=True)

    summaries = real_summaries(pd.read_parquet(args.asset or for_reading(asset_path())))
    if summaries.empty:
        log.warning("no two-chamber recordings in the asset")
        return 0
    # The quick tier's posterior is trained on 45 draws for five epochs, and
    # rejection sampling from it accepts about 0.06% of proposals - so asking it
    # for samples costs far more than simulating did. Every count below is small
    # for that reason, not because 8 recordings would tell us anything; the
    # reported numbers all come from the full run.
    limit = args.max_recordings or (8 if args.quick else None)
    if limit and len(summaries) > limit:
        step = len(summaries) // limit
        summaries = summaries.iloc[::step].head(limit).reset_index(drop=True)
        log.info("quick mode: using %d of the available recordings", len(summaries))
    real_x = summaries[list(SUMMARY_NAMES)].to_numpy(dtype=float)
    if not simulation_cache().exists():
        log.warning("no simulations at %s; run neuromps.twin.inference first", simulation_cache())
        return 0
    cached = load_simulations(simulation_cache())
    inside = within_prior(cached["theta"])
    check = distribution_check(real_x, cached["x"][inside])
    suspect = check["outside_dimensions"] + check["unsupported_dimensions"]
    log.info(
        "distribution check: %.0f%% inside the simulated range; outside: %s; unsupported: %s",
        100 * check["mean_fraction_inside"],
        check["outside_dimensions"],
        check["unsupported_dimensions"],
    )

    posterior = torch.load(posterior_path, weights_only=False)
    acceptance = prior_acceptance(posterior, real_x, n_draws=200 if args.quick else 2000)
    log.info(
        "prior acceptance: median %.4f%%, %.0f%% of recordings below 1%%",
        100 * acceptance["median_acceptance"],
        100 * acceptance["fraction_below_1_percent"],
    )
    inferred = infer_parameters(posterior, real_x, n_samples=60 if args.quick else 500)
    gabaa = np.array([row["gabaa_scale"]["median"] for row in inferred])
    doses = summaries["dose_value"].fillna(0.0).to_numpy(dtype=float)
    units = summaries["dose_unit"].fillna("uM").to_numpy()
    doses_um = np.where(units == "nM", doses / 1000.0, doses)
    devices = summaries["device_id"].to_numpy()
    resamples = 100 if args.quick else 1000
    trend = dose_trend(doses_um, gabaa, groups=devices, n_resamples=resamples)
    baseline = baseline_offset(gabaa, doses_um, groups=devices, n_resamples=resamples)
    patched = sensitivity_without_dimensions(
        posterior,
        real_x,
        cached["x"][inside],
        suspect,
        n_samples=60 if args.quick else 500,
    )
    gabaa_patched = np.array([row["gabaa_scale"]["median"] for row in patched])
    trend_patched = dose_trend(
        doses_um, gabaa_patched, groups=devices, n_resamples=100 if args.quick else 1000
    )

    results = {
        "label": "posterior trained on SIMULATIONS, applied to REAL recordings",
        "n_recordings": int(len(summaries)),
        "subsampled": bool(limit),
        "seed": int(args.seed),
        "distribution_check": check,
        "prior_acceptance": acceptance,
        "gabaa_scale_dose_trend": trend,
        "gabaa_scale_at_baseline": baseline,
        "gabaa_scale_dose_trend_without_ood_dimensions": trend_patched,
        "neutralised_dimensions": suspect,
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
    (twin_dir() / "hd15_real_inference.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_dose_response(
        doses_um,
        gabaa,
        trend,
        twin_dir() / "hd15_real_dose_response.png",
        "Inferred GABA-A scale for REAL bicuculline recordings (posterior trained on simulations)",
    )
    update_metrics(
        "twin_hd15_real_inference", {k: v for k, v in results.items() if k != "per_recording"}
    )
    print(
        json.dumps(
            {
                "distribution_check": {
                    k: check[k]
                    for k in (
                        "n_real",
                        "mean_fraction_inside",
                        "outside_dimensions",
                        "unsupported_dimensions",
                    )
                },
                "prior_acceptance": acceptance,
                "gabaa_scale_dose_trend": trend,
                "gabaa_scale_at_baseline": baseline,
                "gabaa_scale_dose_trend_without_ood_dimensions": trend_patched,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
