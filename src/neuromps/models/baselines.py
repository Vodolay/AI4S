"""What untreated activity looks like, by species and by platform.

Everything else in `models/` works on pairs: an exposure minus its own
baseline. That is the right unit for a drug effect, and it is why adding 184
human organoid recordings to the asset changed not one modelling number - they
are untreated baselines with nothing to be paired against.

This module asks the question those recordings can answer. Do human cultures
sit inside the rat feature distribution at rest, or somewhere else? It matters
because the twin is calibrated against real chamber baselines, and because the
report's portability claim - rate-like measures travel between species,
synchrony-like measures do not - was established on paired deltas and has never
been checked on absolute activity.

Species and platform are confounded in this asset, badly: every MaxWell
recording is human and every Multi Channel Systems recording is rat. So the
headline comparison is made **within one vendor**, where both species exist,
and the cross-vendor comparison is reported beside it as the confounded
contrast it is. Resampling is over culture units, never rows: an organoid
recorded twenty times is one culture, not twenty independent observations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.metrics import update_metrics
from neuromps.models.asymmetry import holm
from neuromps.models.bootstrap import group_bootstrap_metric
from neuromps.paths import asset_path, figures_dir, for_reading

log = logging.getLogger("neuromps.models")

FEATURES = (
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
# Measures the report calls portable across species, kept apart from the rest so
# the claim can be read off the table rather than assembled by eye.
RATE_LIKE = ("mfr_hz", "burst_rate_per_min", "pct_spikes_in_bursts", "spikes_per_burst")
# A count of active electrodes cannot be compared across plate formats: a well
# with 64 electrodes has four times the room to be active that a 16-electrode
# well has, and the difference shows up as biology if nobody stops it. The
# fraction is the comparable quantity, and the raw count keeps a flag saying so.
COVERAGE = "active_electrodes"
COVERAGE_FRACTION = "active_fraction"
RESAMPLES = 2000
# Dissociated two-dimensional cultures, as opposed to organoids and slices.
MONOLAYER_SYSTEMS = ("primary_cortical", "primary_hippocampal", "hpsc_derived_cortical")


# Sources that re-record the same physical culture over time. For these the
# plate id is a date, so it must stay out of the culture identity or every
# session becomes its own "culture" - the pseudo-replication this project keeps
# finding in other people's analyses and in two of its own.
LONGITUDINAL_SOURCES = frozenset({"trujillo2019"})


def culture_id(row: pd.Series) -> str:
    """Identity of the physical culture a recording came from.

    Well identifiers repeat across plates: "A1" exists on the rat plate and on
    the human one. Grouping on the well alone therefore built clusters that
    spanned both species, which is worse than useless in a bootstrap meant to
    resample independent cultures.
    """
    if row["source"] in LONGITUDINAL_SOURCES:
        return f"{row['source']}:{row['unit_id']}"
    return f"{row['source']}:{row.get('plate_id') or ''}:{row['unit_id']}"


def baseline_records(records: pd.DataFrame) -> pd.DataFrame:
    """Usable, untreated, spike-time recordings, one row each."""
    df = records[records["data_level"] == "spike_times"].copy()
    df = df[df["qc.usable"].astype(bool)]
    df = df[df["compound.role"] == "untreated_control"]
    df["culture"] = df.apply(culture_id, axis=1)
    available = pd.to_numeric(df["platform.electrodes_per_unit"], errors="coerce")
    active = pd.to_numeric(df[f"features.{COVERAGE}"], errors="coerce")
    df[f"features.{COVERAGE_FRACTION}"] = active / available.where(available > 0)
    spanning = df.groupby("culture")["cell_source.species"].nunique()
    if (spanning > 1).any():
        raise ValueError(
            f"culture ids spanning more than one species: {sorted(spanning[spanning > 1].index)}"
        )
    return df


def _median_difference(df: pd.DataFrame, column: str, seed: int) -> dict:
    """Human-minus-rat median difference for one feature, resampling cultures."""
    values = df[column].to_numpy(dtype=float)
    species = df["cell_source.species"].to_numpy()
    cultures = df["culture"].to_numpy()

    def metric(idx: np.ndarray) -> float:
        drawn_species, drawn = species[idx], values[idx]
        rat = drawn[(drawn_species == "rat") & np.isfinite(drawn)]
        human = drawn[(drawn_species == "human") & np.isfinite(drawn)]
        if not rat.size or not human.size:
            return float("nan")
        return float(np.median(human) - np.median(rat))

    return group_bootstrap_metric(metric, cultures, n_resamples=RESAMPLES, seed=seed)


def _excludes_zero(interval: dict) -> bool:
    """Whether a bootstrap interval lies wholly on one side of zero."""
    low, high = interval.get("ci95", [float("nan"), float("nan")])
    return bool(np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0))


def compare(df: pd.DataFrame, label: str, seed: int = 0) -> dict:
    """Per-feature species comparison within one slice of the asset."""
    species = df["cell_source.species"]
    out = {
        "slice": label,
        "n_records": int(len(df)),
        "n_cultures": int(df["culture"].nunique()),
        "n_human_records": int((species == "human").sum()),
        "n_rat_records": int((species == "rat").sum()),
        "n_human_cultures": int(df.loc[species == "human", "culture"].nunique()),
        "n_rat_cultures": int(df.loc[species == "rat", "culture"].nunique()),
        "sources": sorted(df["source"].unique()),
        "features": {},
    }
    if out["n_human_cultures"] < 2 or out["n_rat_cultures"] < 2:
        out["note"] = "too few cultures on one side to resample; no intervals computed"
        return out
    formats = pd.to_numeric(df["platform.electrodes_per_unit"], errors="coerce").dropna()
    mixed_formats = formats.nunique() > 1
    out["electrodes_per_unit"] = sorted(int(v) for v in formats.unique())
    for feature in (*FEATURES, COVERAGE_FRACTION):
        column = f"features.{feature}"
        if column not in df:
            continue
        interval = _median_difference(df, column, seed)
        confounded = mixed_formats and feature == COVERAGE
        out["features"][feature] = {
            "median_human": _median(df, column, species == "human"),
            "median_rat": _median(df, column, species == "rat"),
            "difference_ci95": interval.get("ci95"),
            "kind": (
                "coverage"
                if feature in (COVERAGE, COVERAGE_FRACTION)
                else "rate_like"
                if feature in RATE_LIKE
                else "synchrony_like"
            ),
            "excludes_zero": _excludes_zero(interval) and not confounded,
            "format_confounded": confounded,
        }
    separated = [f for f, v in out["features"].items() if v["excludes_zero"]]
    out["features_separating_species"] = sorted(separated)
    out["n_features_separating_species"] = len(separated)
    return out


def _median(df: pd.DataFrame, column: str, mask: pd.Series) -> float | None:
    """Median of a feature over a subset, or None when it is undefined."""
    values = pd.to_numeric(df.loc[mask, column], errors="coerce").dropna()
    return round(float(values.median()), 4) if len(values) else None


def _mann_whitney(df: pd.DataFrame) -> dict:
    """Per-feature rank test between species, Holm-corrected across features.

    The bootstrap interval is the estimate; this is only a companion test, and
    it is corrected because nine features are asked the same question.
    """
    from scipy.stats import mannwhitneyu

    species = df["cell_source.species"]
    raw = {}
    for feature in FEATURES:
        column = f"features.{feature}"
        human = pd.to_numeric(df.loc[species == "human", column], errors="coerce").dropna()
        rat = pd.to_numeric(df.loc[species == "rat", column], errors="coerce").dropna()
        if len(human) < 3 or len(rat) < 3:
            continue
        raw[feature] = float(mannwhitneyu(human, rat, alternative="two-sided").pvalue)
    return {"raw_p": {k: round(v, 5) for k, v in raw.items()}, "holm_p": holm(raw)} if raw else {}


def preparation_group(row: pd.Series) -> str:
    """Species and preparation as one label, which is what the figure compares."""
    kind = "organoid" if row["cell_source.system"] not in MONOLAYER_SYSTEMS else "monolayer"
    return f"{row['cell_source.species']} {kind}"


def plot_comparison(df: pd.DataFrame, path: Path) -> Path:
    """Baseline feature distributions by species and preparation, one vendor.

    Three groups rather than two, because the species contrast and the
    preparation contrast are both present in this slice and a two-group figure
    would let a reader attribute the whole gap to species.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["rat monolayer", "human monolayer", "human organoid"]
    colours = {
        "rat monolayer": "#D55E00",
        "human monolayer": "#0072B2",
        "human organoid": "#009E73",
    }
    df = df.copy()
    df["group"] = df.apply(preparation_group, axis=1)
    # Drawn near its printed width. At 11.5 in, nine panels printed at half a column
    # had 2 pt tick labels; two-line group labels leave room for the data.
    short = {
        "rat monolayer": "rat mono",
        "human monolayer": "human mono",
        "human organoid": "organoid",
    }
    fig, axes = plt.subplots(3, 3, figsize=(5.6, 4.4))
    panels = [f if f != COVERAGE else COVERAGE_FRACTION for f in FEATURES]
    for ax, feature in zip(axes.ravel(), panels, strict=False):
        column = f"features.{feature}"
        data, labels, used = [], [], []
        for group in order:
            values = pd.to_numeric(df.loc[df["group"] == group, column], errors="coerce").dropna()
            if len(values):
                data.append(values.to_numpy())
                labels.append(f"{short[group]}\nn={len(values)}")
                used.append(group)
        if not data:
            ax.set_visible(False)
            continue
        parts = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, group in zip(parts["boxes"], used, strict=False):
            patch.set_facecolor(colours[group])
            patch.set_alpha(0.55)
        for median in parts["medians"]:
            median.set_color("black")
        ax.set_title(feature, fontsize=6.5)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.tick_params(axis="x", labelsize=5.5, length=0)
    fig.suptitle(
        "Untreated baseline activity on Axion platforms, by species and preparation "
        "(REAL recordings)",
        fontsize=7.5,
    )
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def compare_groups(
    df: pd.DataFrame, column: str, left: str, right: str, label: str, seed: int = 0
) -> dict:
    """Per-feature comparison of two groups that differ in `column`, not species."""
    subset = df[df[column].isin([left, right])].copy()
    # Reuse the species machinery by mapping the contrast onto its two sides:
    # the bootstrap asks "left minus right", whatever the sides mean.
    subset["cell_source.species"] = np.where(subset[column] == left, "human", "rat")
    out = compare(subset, label, seed)
    out["contrast"] = {"column": column, "left_as_human": left, "right_as_rat": right}
    return out


def run(records: pd.DataFrame, seed: int = 0) -> dict:
    """Baseline comparisons, each holding fixed what the previous one confounded."""
    df = baseline_records(records)
    axion = df[df["platform.vendor"] == "Axion Biosystems"]
    monolayer = axion[axion["cell_source.system"].isin(MONOLAYER_SYSTEMS)]
    human_axion = axion[axion["cell_source.species"] == "human"]
    result = {
        "label": "untreated baseline activity, human vs rat",
        "within_vendor": compare(axion, "Axion Biosystems only", seed),
        # The slice above mixes 3D organoids with 2D monolayers, so on its own it
        # cannot tell a species difference from a preparation difference. These
        # two hold one of those fixed at a time.
        "matched_preparation": compare(
            monolayer, "Axion Biosystems, dissociated monolayers only", seed
        ),
        "preparation_within_human": compare_groups(
            human_axion,
            "cell_source.system",
            "brain_organoid",
            "hpsc_derived_cortical",
            "human only, Axion: 3D organoid vs 2D monolayer",
            seed,
        ),
        "all_platforms": compare(df, "every platform (species and vendor confounded)", seed),
        "n_baseline_records": int(len(df)),
        "records_by_source": {k: int(v) for k, v in df["source"].value_counts().items()},
        "human_records_by_source": {
            k: int(v)
            for k, v in df[df["cell_source.species"] == "human"]["source"].value_counts().items()
        },
        "vendor_by_species": {
            species: sorted(group["platform.vendor"].dropna().unique())
            for species, group in df.groupby("cell_source.species")
        },
    }
    result["within_vendor"]["mann_whitney"] = _mann_whitney(axion)
    result["matched_preparation"]["mann_whitney"] = _mann_whitney(monolayer)
    # An untreated control well recorded during a plate's exposure phase is
    # still untreated, so it is kept - but it is a later timepoint, and activity
    # drifts. If the headline count depended on that choice it would be a
    # finding about session order rather than about species.
    result["matched_preparation_baseline_phase_only"] = compare(
        monolayer[monolayer["recording.phase"] == "baseline"],
        "Axion monolayers, baseline-phase recordings only",
        seed,
    )
    # `compare` returns early, without a feature list, when a slice has too few
    # cultures to resample - which is what a blocked source looks like from
    # here. Comparing two absent lists would report agreement where there was no
    # comparison, so the answer is null rather than True.
    full_slice = result["matched_preparation"].get("features_separating_species")
    baseline_only = result["matched_preparation_baseline_phase_only"].get(
        "features_separating_species"
    )
    result["headline_survives_dropping_exposure_phase_controls"] = (
        None if full_slice is None or baseline_only is None else full_slice == baseline_only
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI: compare untreated baselines across species and write the metrics."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="accepted for pipeline symmetry")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asset = for_reading(asset_path())
    if not asset.exists():
        log.warning("no asset at %s; skipping baseline comparison", asset)
        return 0
    records = pd.read_parquet(asset)
    result = run(records, seed=args.seed)
    figure = plot_comparison(
        baseline_records(records).pipe(lambda d: d[d["platform.vendor"] == "Axion Biosystems"]),
        figures_dir() / "baseline_species_comparison.png",
    )
    result["figure"] = (
        str(figure.relative_to(Path.cwd())) if figure.is_relative_to(Path.cwd()) else str(figure)
    )
    update_metrics("baseline_activity", result)
    log.info(json.dumps(result["within_vendor"], indent=2, default=str)[:900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
