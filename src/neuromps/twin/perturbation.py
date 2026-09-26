"""Do the twin's drug knobs move features the way the real drugs do?

Each knob is compared against the real pharmacology **read from the asset**, not
from memory: for a mechanism class, the median log2 feature change of the real
exposure/baseline pairs, with a cluster-bootstrap interval. A real direction
counts as established only when that interval excludes zero; otherwise the
comparison is recorded as "not established" rather than scored, because agreeing
with a direction the data does not support is not evidence.

The simulated change is the same log2 ratio against the twin's own untreated
baseline, computed with the same `features/mea.py` used on real recordings.

Caveat carried from the data: the real reference for each mechanism comes from
whichever source has that drug — bicuculline from the 60-electrode MCS chambers,
TTX and the glutamatergic drugs from the 16-electrode Axion wells — and HD8
showed feature deltas do not transfer unchanged across platforms. Agreement is
therefore evidence about direction, not about magnitude.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.features.mea import extract_features
from neuromps.metrics import update_metrics
from neuromps.models.bootstrap import group_bootstrap_metric, resample_count
from neuromps.models.dataset import DELTA_FEATURES, load_pairs, log_ratio
from neuromps.paths import twin_dir
from neuromps.twin.calibrate import calibrated_params
from neuromps.twin.mea import ElectrodeLayout, electrode_positions, place_neurons, record_electrodes
from neuromps.twin.network import NetworkParams, simulate

log = logging.getLogger("neuromps.twin")
REPO_ROOT = Path(__file__).resolve().parents[3]
# Features bounded below by zero: if the network falls silent these are 0, not
# undefined, so a ratio against the baseline still means "abolished". The real
# ETL already floors such a ratio at log2(0.05) = -4.32; using the same
# convention keeps simulated and real deltas on one scale.
ZERO_WHEN_SILENT = (
    "mfr_hz",
    "burst_rate_per_min",
    "pct_spikes_in_bursts",
    "network_burst_rate_per_min",
)
COMPARE_FEATURES = (
    "mfr_hz",
    "burst_rate_per_min",
    "pct_spikes_in_bursts",
    "network_burst_rate_per_min",
    "synchrony_index",
    "isi_cv",
)
KNOBS = {
    "gabaa_block": {
        "parameter": "gabaa_scale",
        "value": 0.2,
        "mechanism": "gabaa_antagonist",
        "reads_as": "GABA-A antagonist (bicuculline, gabazine): inhibition removed",
    },
    "gabaa_potentiate": {
        "parameter": "gabaa_scale",
        "value": 2.0,
        "mechanism": "gabaa_agonist_or_pam",
        "reads_as": "GABA-A positive modulator (diazepam): inhibition strengthened",
    },
    "ampa_block": {
        "parameter": "ampa_scale",
        "value": 0.4,
        "mechanism": "glutamatergic_antagonist",
        "reads_as": "glutamatergic antagonist (CNQX, D-AP5): excitation reduced",
    },
    "sodium_block": {
        "parameter": "threshold_scale",
        "value": 1.6,
        "mechanism": "na_channel_blocker",
        "reads_as": "sodium-channel blocker (TTX): spike threshold raised",
    },
}


def real_directions(
    pairs: pd.DataFrame, mechanism: str, n_resamples: int = 1000, seed: int = 0
) -> dict:
    """Median real log2 change per feature for one mechanism, with intervals."""
    subset = pairs[pairs["label"] == mechanism]
    if subset.empty:
        return {"mechanism": mechanism, "n_pairs": 0, "by_feature": {}}
    groups = subset["group"].to_numpy()
    by_feature = {}
    for feature in COMPARE_FEATURES:
        values = subset[f"delta.{feature}"].to_numpy(dtype=float)

        def metric(idx: np.ndarray, values=values) -> float:
            drawn = values[idx]
            drawn = drawn[np.isfinite(drawn)]
            return float(np.median(drawn)) if drawn.size else float("nan")

        interval = group_bootstrap_metric(metric, groups, n_resamples=n_resamples, seed=seed)
        low, high = interval["ci95"]
        finite = values[np.isfinite(values)]
        by_feature[feature] = {
            "median_delta": round(float(np.median(finite)), 3) if finite.size else None,
            "ci95": [low, high],
            "established": bool(low is not None and (low > 0 or high < 0)),
            "n": int(finite.size),
        }
    return {
        "mechanism": mechanism,
        "n_pairs": int(len(subset)),
        "sources": sorted(set(subset["source"])),
        "bootstrap_resamples": n_resamples,
        "by_feature": by_feature,
    }


def simulated_delta(treated: dict, baseline: dict) -> dict:
    """log2(treated / untreated) per feature, using the same rule as the real pairs."""
    out = {}
    for feature in COMPARE_FEATURES:
        value = log_ratio(treated.get(feature), baseline.get(feature))
        out[feature] = None if value is None or not np.isfinite(value) else round(float(value), 3)
    return out


def _direction(value: float | None) -> str | None:
    """'up', 'down', or None when the value is missing."""
    if value is None or not np.isfinite(value):
        return None
    return "up" if value > 0 else "down"


def agreement_table(simulated: dict, real: dict) -> dict:
    """Per feature: the two directions and whether they agree.

    ``agrees`` is None when the real direction is not established, so a feature
    the real data cannot resolve neither credits nor penalises the twin.
    """
    table = {}
    for feature, reference in real["by_feature"].items():
        simulated_value = simulated.get(feature)
        real_direction = _direction(reference["median_delta"])
        simulated_direction = _direction(simulated_value)
        agrees = None
        if reference["established"] and real_direction and simulated_direction:
            agrees = real_direction == simulated_direction
        table[feature] = {
            "real_delta": reference["median_delta"],
            "real_direction": real_direction,
            "real_established": reference["established"],
            "simulated_delta": simulated_value,
            "simulated_direction": simulated_direction,
            "agrees": agrees,
        }
    return table


def _features_for(params: NetworkParams, layout: ElectrodeLayout, seed: int) -> dict:
    """Per-compartment mean features of one simulated recording."""
    result = simulate(params)
    positions, compartment = place_neurons(params, layout, seed=seed)
    electrodes = electrode_positions(layout)
    heard = record_electrodes(
        result.spike_times, result.spike_neurons, positions, compartment, electrodes, layout
    )
    per_compartment = [
        extract_features(
            {k: v for k, v in heard.items() if k.startswith(f"c{c}_")}, result.duration_s
        )
        for c in (0, 1)
    ]
    merged = {}
    for feature in DELTA_FEATURES:
        values = [f.get(feature) for f in per_compartment]
        values = [v for v in values if v is not None and np.isfinite(v)]
        merged[feature] = float(np.mean(values)) if values else None
    if not merged.get("active_electrodes"):
        # Silent network: rate-like features are zero, not unknown. Synchrony and
        # ISI CV stay undefined, because there are no spikes to relate.
        for feature in ZERO_WHEN_SILENT:
            merged[feature] = 0.0
        merged["silent"] = True
    return merged


def run_knobs(duration_s: float, seed: int, n_per_compartment: int = 250) -> dict:
    """Simulate the untreated baseline and one recording per drug knob."""
    tuned = calibrated_params()
    network_keys = set(NetworkParams().as_record())
    layout_keys = set(ElectrodeLayout().as_record())
    layout = ElectrodeLayout(**{k: v for k, v in tuned.items() if k in layout_keys})
    base_kwargs = {
        "n_per_compartment": n_per_compartment,
        "duration_s": duration_s,
        "seed": seed,
        **{k: v for k, v in tuned.items() if k in network_keys},
    }
    baseline = _features_for(NetworkParams(**base_kwargs), layout, seed)
    log.info("simulated untreated baseline: mfr %.2f Hz", baseline.get("mfr_hz") or float("nan"))
    treated = {}
    for name, knob in KNOBS.items():
        params = NetworkParams(**{**base_kwargs, knob["parameter"]: knob["value"]})
        treated[name] = _features_for(params, layout, seed)
        log.info("  %s (%s=%s) done", name, knob["parameter"], knob["value"])
    return {
        "calibrated_parameters_used": bool(tuned),
        "baseline_features_simulated": baseline,
        "treated_features_simulated": treated,
        "layout": layout.as_record(),
        "duration_s": duration_s,
        # What was simulated, not what was asked for: the calibrated network
        # size overrides the argument above, so reporting the argument told the
        # reader 250 while the run used the calibrated value.
        "n_per_compartment": int(base_kwargs["n_per_compartment"]),
    }


def plot_validation(rows: list[dict], path, title: str) -> None:
    """Simulated vs real direction per knob and feature: a qualitative grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    knobs = [row["knob"] for row in rows]
    features = list(COMPARE_FEATURES)
    grid = np.full((len(knobs), len(features)), np.nan)
    for i, row in enumerate(rows):
        for j, feature in enumerate(features):
            entry = row["agreement"].get(feature, {})
            if entry.get("agrees") is True:
                grid[i, j] = 1.0
            elif entry.get("agrees") is False:
                grid[i, j] = 0.0
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.imshow(grid, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    for i in range(len(knobs)):
        for j, feature in enumerate(features):
            entry = rows[i]["agreement"].get(feature, {})
            mark = {True: "agree", False: "differ", None: "n/e"}[entry.get("agrees")]
            ax.text(j, i, mark, ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(features)), features, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(knobs)), knobs, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run every knob and write the qualitative validation table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="shorter simulations, fewer resamples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    twin_dir().mkdir(parents=True, exist_ok=True)
    duration = 12.0 if args.quick else 40.0
    n_resamples = resample_count(args.quick)

    simulated = run_knobs(duration, args.seed)
    pairs = load_pairs()
    rows = []
    for name, knob in KNOBS.items():
        real = real_directions(pairs, knob["mechanism"], n_resamples=n_resamples, seed=args.seed)
        delta = simulated_delta(
            simulated["treated_features_simulated"][name], simulated["baseline_features_simulated"]
        )
        table = agreement_table(delta, real)
        scored = [v["agrees"] for v in table.values() if v["agrees"] is not None]
        rows.append(
            {
                "knob": name,
                **knob,
                "real_reference": real,
                "simulated_delta": delta,
                "agreement": table,
                "n_features_scored": len(scored),
                "n_features_agreeing": int(sum(scored)),
            }
        )
    results = {
        "label": "SIMULATED twin vs REAL pharmacology (directions only)",
        "simulation": simulated,
        "knobs": rows,
        "bootstrap_resamples": n_resamples,
    }
    (twin_dir() / "hd13_perturbation_simulated.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    plot_validation(
        rows,
        twin_dir() / "hd13_drug_knob_validation_simulated.png",
        "SIMULATED drug knobs vs REAL pharmacology direction\n"
        "'n/e' = the real direction is not established (interval includes zero)",
    )
    update_metrics("twin_hd13_perturbation", results)
    for row in rows:
        print(
            f"{row['knob']:<18} {row['n_features_agreeing']}/{row['n_features_scored']} "
            f"features agree  ({row['reads_as']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
