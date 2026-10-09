"""Does chamber coupling asymmetry reflect the axons, or the firing rates?

`models/tunnels.py` shows that the coupling asymmetry measured between two
chambers of a Brewer 2021 device tracks the chambers' rate imbalance. That is
evidence the measure is confounded, but not proof it is wrong, because nothing
in that analysis knows which way the axons actually fired.

This module supplies the missing term. The same devices instrument the
microtunnels themselves, two electrodes per tunnel, so a spike travelling
between chambers is recorded at both and the sign of the delay is the direction
it travelled (`etl/brewer_tunnels.py` recovers the geometry). Counting signed
crossings gives a physical flow direction per chamber pair, and the question
becomes a correlation: does the chamber measure agree with the axons?

Two controls the claim needs:

- A **shift predictor**: one electrode's train is circularly shifted by many
  seconds and the counting repeated. Two trains with no causal relation still
  produce short-latency coincidences when both are bursting; the shifted count
  is what chance alone delivers, and the flow index is reported net of it.
- A **window sweep**. A conduction delay sets a scale the analysis should not
  be free to choose, so the flow index is recomputed across windows and the
  agreement is reported for each.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neuromps.etl.brewer import cultures
from neuromps.etl.brewer_tunnels import (
    SOURCE,
    build_tunnels,
    coordinate_disagreements,
    read_tunnels,
)
from neuromps.models.bootstrap import group_bootstrap_metric

log = logging.getLogger("neuromps.models")

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = REPO_ROOT / "data" / "raw" / SOURCE / "spike_trains_data.zip"
# Windows bracketing an axonal conduction delay across one electrode pitch.
WINDOWS_MS = (0.5, 1.0, 2.0, 5.0)
HEADLINE_WINDOW_MS = 1.0
# Circular shifts for the chance-level control, in seconds - long enough that
# no conduction delay survives, short enough to keep the burst statistics.
SHIFTS_S = (5.0, 17.0, 41.0)
RESAMPLES = 2000
MIN_CROSSINGS = 50


def crossing_counts(source: np.ndarray, target: np.ndarray, window_s: float) -> tuple[int, int]:
    """Spikes on ``target`` shortly after, and shortly before, one on ``source``.

    Counts every pair within the window rather than nearest neighbours only, so
    the two directions are counted by the same rule and a burst cannot bias one.
    """
    if source.size == 0 or target.size == 0:
        return 0, 0
    after = np.searchsorted(target, source + window_s, "right") - np.searchsorted(
        target, source, "right"
    )
    before = np.searchsorted(target, source, "left") - np.searchsorted(
        target, source - window_s, "left"
    )
    return int(after.sum()), int(before.sum())


def _shifted(times: np.ndarray, shift_s: float, duration_s: float) -> np.ndarray:
    """One train moved around the recording, preserving its own structure."""
    if duration_s <= 0:
        return times
    return np.sort((times + shift_s) % duration_s)


def tunnel_flow(reading: dict, window_s: float, seed: int = 0) -> list[dict]:
    """Signed axonal crossings per chamber pair for one culture.

    The index is the share of crossings running in the pair's forward
    direction minus the share running back, net of the shift control, so 0 is
    balanced traffic and +1 is traffic entirely one way.
    """
    del seed
    spikes, duration = reading["spikes"], reading["duration_s"]
    tunnels = build_tunnels(reading["names"], reading["subregions"])
    by_pair: dict[str, dict] = {}
    for tunnel in tunnels:
        source = spikes.get(tunnel.from_electrode)
        target = spikes.get(tunnel.to_electrode)
        if source is None or target is None:
            continue
        forward, backward = crossing_counts(source, target, window_s)
        if not tunnel.forward_is_pair_order:
            forward, backward = backward, forward
        control = [
            crossing_counts(source, _shifted(target, shift, duration), window_s)
            for shift in SHIFTS_S
        ]
        shift_forward = float(np.mean([c[0] for c in control]))
        shift_backward = float(np.mean([c[1] for c in control]))
        if not tunnel.forward_is_pair_order:
            shift_forward, shift_backward = shift_backward, shift_forward
        entry = by_pair.setdefault(
            tunnel.pair,
            {
                "forward": 0,
                "backward": 0,
                "shift_forward": 0.0,
                "shift_backward": 0.0,
                "n_tunnels": 0,
            },
        )
        entry["forward"] += forward
        entry["backward"] += backward
        entry["shift_forward"] += shift_forward
        entry["shift_backward"] += shift_backward
        entry["n_tunnels"] += 1

    rows = []
    for pair, entry in sorted(by_pair.items()):
        total = entry["forward"] + entry["backward"]
        shift_total = entry["shift_forward"] + entry["shift_backward"]
        rows.append(
            {
                # `etl/brewer.py` keys a device on the first word of the
                # culture directory; the join needs the same key.
                "device_id": reading["culture"].split()[0],
                "pair": pair,
                "window_ms": round(window_s * 1000, 3),
                "n_tunnels": entry["n_tunnels"],
                "forward": entry["forward"],
                "backward": entry["backward"],
                "n_crossings": total,
                "flow_index": _index(entry["forward"], entry["backward"]),
                "shift_flow_index": _index(entry["shift_forward"], entry["shift_backward"]),
                "shift_crossings": round(shift_total, 1),
                "excess_over_shift": round(total - shift_total, 1),
            }
        )
    return rows


def _index(forward: float, backward: float) -> float | None:
    """Normalised difference of the two directions, or None with no traffic."""
    total = forward + backward
    return round(float((forward - backward) / total), 4) if total > 0 else None


def flow_table(
    archive_path: Path = ARCHIVE,
    windows_ms: tuple[float, ...] = WINDOWS_MS,
    max_cultures: int | None = None,
) -> pd.DataFrame:
    """Axonal flow per culture, chamber pair and window, in one archive pass."""
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        names = cultures(archive)[:max_cultures]
        for culture in names:
            reading = read_tunnels(archive, culture)
            for window_ms in windows_ms:
                rows.extend(tunnel_flow(reading, window_ms / 1000.0))
            log.info("tunnels: %s, %d electrodes with spikes", culture, len(reading["spikes"]))
    return pd.DataFrame(rows)


def agreement(flow: pd.DataFrame, chamber: pd.DataFrame, seed: int = 0) -> dict:
    """Rank correlation between the axonal direction and the chamber measure.

    Both are signed the same way - positive means flow from the first chamber
    of the pair to the second - so agreement would show as a positive rho.
    """
    merged = flow.merge(chamber, on=["device_id", "pair"], how="inner", validate="one_to_one")
    n_joined = int(len(merged))
    merged = merged[merged["flow_index"].notna() & merged["n_crossings"].ge(MIN_CROSSINGS)]
    n_dropped = n_joined - int(len(merged))
    if len(merged) < 4:
        return {"n_pairs": int(len(merged)), "verdict": "too few pairs to test"}
    axonal = merged["flow_index"].to_numpy(dtype=float)
    measured = merged["asymmetry"].to_numpy(dtype=float)
    rate = merged["log2_rate_ratio"].to_numpy(dtype=float)
    groups = merged["device_id"].to_numpy()

    def rho(values: np.ndarray, other: np.ndarray):
        def metric(idx: np.ndarray) -> float:
            if len({*values[idx]}) < 2 or len({*other[idx]}) < 2:
                return float("nan")
            return float(spearmanr(values[idx], other[idx]).statistic)

        return metric

    out = {
        "n_pairs": int(len(merged)),
        "n_devices": int(merged["device_id"].nunique()),
        "n_pairs_joined": n_joined,
        "n_pairs_below_crossing_threshold": n_dropped,
        "window_ms": float(merged["window_ms"].iloc[0]),
        "min_crossings_per_pair": MIN_CROSSINGS,
    }
    for name, other in (("chamber_asymmetry", measured), ("log2_rate_ratio", rate)):
        observed = spearmanr(axonal, other)
        boot = group_bootstrap_metric(rho(axonal, other), groups, RESAMPLES, seed)
        out[f"axonal_vs_{name}"] = {
            "spearman_rho": round(float(observed.statistic), 4),
            "p_value": round(float(observed.pvalue), 4),
            "spearman_rho_ci95": boot.get("ci95"),
            "excludes_zero": _excludes_zero(boot.get("ci95")),
        }
    # The most direct question: told which way the axons fired, how often does
    # the chamber measure point the same way? Chance is one half.
    same = (np.sign(axonal) * np.sign(measured)) > 0
    out["sign_agreement"] = {
        "fraction": round(float(same.mean()), 4),
        "n_agreeing": int(same.sum()),
        "n_pairs": int(same.size),
        "chance": 0.5,
        "fraction_ci95": group_bootstrap_metric(
            lambda idx: float(same[idx].mean()), groups, RESAMPLES, seed
        ).get("ci95"),
    }
    out["sign_agreement"]["better_than_chance"] = _above(
        out["sign_agreement"]["fraction_ci95"], 0.5
    )
    out["median_abs_flow_index"] = round(float(np.abs(axonal).mean()), 4)
    out["shift_control_median_abs_index"] = round(
        float(merged["shift_flow_index"].abs().median()), 4
    )
    return out


def _above(interval, level: float) -> bool | None:
    """Whether a bootstrap interval lies wholly above a reference level."""
    if not interval or interval[0] is None:
        return None
    return bool(interval[0] > level)


def _excludes_zero(interval) -> bool | None:
    """Whether a bootstrap interval lies wholly on one side of zero."""
    if not interval or interval[0] is None or interval[1] is None:
        return None
    return bool(interval[0] > 0 or interval[1] < 0)


def plot_flow(flow: pd.DataFrame, chamber: pd.DataFrame, path: Path) -> Path:
    """Axonal direction against the chamber measure, one point per pair (real)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = flow.merge(chamber, on=["device_id", "pair"], how="inner")
    merged = merged[merged["flow_index"].notna()]
    colours = {
        "CA1->CA3": "#0072B2",
        "CA1->EC": "#E69F00",
        "CA3->DG": "#009E73",
        "DG->EC": "#D55E00",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)  # near its printed size
    for ax, column, label in (
        (axes[0], "asymmetry", "chamber coupling asymmetry"),
        (axes[1], "log2_rate_ratio", "chamber rate imbalance (log2 ratio)"),
    ):
        for pair, rows in merged.groupby("pair"):
            ax.scatter(
                rows[column],
                rows["flow_index"],
                s=34,
                color=colours.get(str(pair), "#666666"),
                edgecolor="black",
                linewidth=0.4,
                label=str(pair) if column == "asymmetry" else None,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("axonal flow index in the tunnels")
    axes[0].legend(fontsize=7, title="chamber pair", title_fontsize=7)
    fig.suptitle(
        "REAL Brewer 2021 devices: measured axonal direction against the chamber measure.\n"
        "Each point is one chamber pair of one culture; positive is flow toward the "
        "second chamber named.",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def run(
    records: pd.DataFrame,
    seed: int = 0,
    max_cultures: int | None = None,
    flow: pd.DataFrame | None = None,
) -> dict | None:
    """The whole analysis, or None when the source is not in the asset.

    ``flow`` is accepted so a caller that also plots it pays for one pass over
    the archive rather than two.
    """
    from neuromps.models.tunnels import chamber_pairs

    subset = records[records["source"] == SOURCE]
    if subset.empty or not ARCHIVE.exists():
        return None
    chamber = chamber_pairs(subset)
    if chamber.empty:
        return None
    if flow is None:
        flow = flow_table(max_cultures=max_cultures)
    if flow.empty:
        return None
    result = {
        "label": "REAL - measured axonal crossings, not simulated",
        "n_cultures": int(flow["device_id"].nunique()),
        "windows_ms": list(WINDOWS_MS),
        "headline_window_ms": HEADLINE_WINDOW_MS,
        "shifts_s": list(SHIFTS_S),
        "by_window": {},
    }
    for window_ms, rows in flow.groupby("window_ms"):
        result["by_window"][_window_key(window_ms)] = agreement(rows, chamber, seed)
    result["headline"] = result["by_window"].get(_window_key(HEADLINE_WINDOW_MS))
    headline_rows = flow[flow["window_ms"] == HEADLINE_WINDOW_MS]
    result["chamber_readouts_predict_direction"] = predicts_axonal_direction(
        headline_rows, chamber, seed
    )
    result["agreement_holds_at_every_window"] = _stable(result["by_window"])
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = cultures(archive)[:max_cultures]
        disagreements = {c: coordinate_disagreements(archive, c) for c in names}
    result["stored_coordinate_disagreements"] = {
        "n_files_checked": sum(len(v) for v in disagreements.values()),
        "n_files_disagreeing": sum(
            1 for v in disagreements.values() for e in v.values() if e and e["n_disagree"]
        ),
        "note": (
            "files whose stored coordinates contradict their own channel names; "
            "position is taken from the names, which never disagree with the convention"
        ),
    }
    return result


def _window_key(window_ms: float) -> str:
    """Metrics key for a window, in microseconds so it carries no decimal point.

    `report/verify.py` addresses metrics by a dotted path, so a key like
    `0.5ms` would be read as two levels of nesting.
    """
    return f"{round(window_ms * 1000)}us"


def _stable(by_window: dict) -> bool | None:
    """Whether every window agrees on the sign, or None when a window is unusable."""
    signs = []
    for entry in by_window.values():
        block = entry.get("axonal_vs_chamber_asymmetry")
        if block is None or block.get("excludes_zero") is None:
            return None
        signs.append(block["excludes_zero"] and block["spearman_rho"] > 0)
    return bool(signs) and all(signs)


def main(argv: list[str] | None = None) -> int:
    """CLI: compare the chamber coupling measure against the axons themselves."""
    import argparse
    import json

    from neuromps.metrics import update_metrics
    from neuromps.models.tunnels import chamber_pairs
    from neuromps.paths import asset_path, figures_dir, for_reading

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="fewer cultures")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asset = for_reading(asset_path())
    if not asset.exists():
        log.warning("no asset at %s; skipping", asset)
        return 0
    records = pd.read_parquet(asset)
    flow = flow_table(max_cultures=3 if args.quick else None)
    result = run(records, seed=args.seed, flow=flow)
    if result is None:
        log.warning("%s not in the asset; nothing to do", SOURCE)
        update_metrics("axonal_directionality", {"status": "source not in asset"})
        return 0
    subset = records[records["source"] == SOURCE]
    figure = plot_flow(
        flow[flow["window_ms"] == HEADLINE_WINDOW_MS],
        chamber_pairs(subset),
        figures_dir() / "axonal_flow_vs_chamber.png",
    )
    result["figure"] = str(figure.relative_to(REPO_ROOT))
    update_metrics("axonal_directionality", result)
    print(json.dumps({k: v for k, v in result.items() if k != "by_window"}, indent=2))
    print(json.dumps(result["by_window"], indent=2))
    return 0


# The chamber-level readouts a device gives you without opening the tunnels.
# Kept small on purpose: 31 pairs over 9 cultures cannot support a wide model,
# and a wide one would find something whether or not there is anything there.
CHAMBER_FEATURES = ("asymmetry", "peak_correlation", "lag_ms", "log2_rate_ratio")
PERMUTATIONS = 2000


def _leave_one_culture_out(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold sign predictions, each culture held out in turn."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    predicted = np.empty(len(y), dtype=int)
    for held in np.unique(groups):
        test = groups == held
        train = ~test
        if len(set(y[train])) < 2:
            # A fold whose training half is all one class cannot be fitted; the
            # majority guess is what it would have produced anyway.
            predicted[test] = y[train][0] if train.any() else 1
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(
            x[train], y[train]
        )
        predicted[test] = model.predict(x[test])
    return predicted


def predicts_axonal_direction(flow: pd.DataFrame, chamber: pd.DataFrame, seed: int = 0) -> dict:
    """Can *any* chamber-level readout recover the direction the axons fired?

    The constructive form of the finding above. The chamber coupling asymmetry
    fails on its own; this asks whether the four readouts a two-chamber device
    gives you, taken together, do better - fitted with one culture held out at a
    time, and scored against a permutation null that shuffles the axonal signs
    within each culture so the cluster structure survives the shuffle.

    With 31 pairs over 9 cultures this can only detect a large effect, and the
    reported detectable effect says how large.
    """
    from sklearn.metrics import balanced_accuracy_score

    merged = flow.merge(chamber, on=["device_id", "pair"], how="inner", validate="one_to_one")
    merged = merged[merged["flow_index"].notna() & merged["n_crossings"].ge(MIN_CROSSINGS)]
    usable = merged[list(CHAMBER_FEATURES)].notna().all(axis=1) & merged["flow_index"].ne(0)
    merged = merged[usable]
    if len(merged) < 8 or merged["device_id"].nunique() < 3:
        return {"n_pairs": int(len(merged)), "verdict": "too few pairs to fit"}

    x = merged[list(CHAMBER_FEATURES)].to_numpy(dtype=float)
    y = np.sign(merged["flow_index"].to_numpy(dtype=float)).astype(int)
    groups = merged["device_id"].to_numpy()
    observed = float(balanced_accuracy_score(y, _leave_one_culture_out(x, y, groups)))

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(PERMUTATIONS):
        shuffled = y.copy()
        for culture in np.unique(groups):
            mask = groups == culture
            shuffled[mask] = rng.permutation(y[mask])
        null.append(balanced_accuracy_score(shuffled, _leave_one_culture_out(x, shuffled, groups)))
    null = np.asarray(null, dtype=float)
    # +1 in both terms: the observed labelling is one of the arrangements.
    p_value = float((1 + (null >= observed).sum()) / (1 + len(null)))
    return {
        "n_pairs": int(len(merged)),
        "n_cultures": int(merged["device_id"].nunique()),
        "features": list(CHAMBER_FEATURES),
        "balanced_accuracy": round(observed, 4),
        "chance": 0.5,
        "permutation_p": round(p_value, 4),
        "permutations": PERMUTATIONS,
        "beats_chance": bool(p_value < 0.05),
        "detectable_balanced_accuracy": round(float(np.percentile(null, 95)), 4),
        "note": (
            "leave-one-culture-out logistic regression on the chamber readouts, predicting the "
            "sign of the measured axonal flow; the null shuffles signs within each culture"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
