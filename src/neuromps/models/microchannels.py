"""Does microchannel geometry change how one-sided propagation is? (REAL data)

HD16 asked whether MEA features can tell symmetric from asymmetric connectivity
and could only answer in simulation, where the ground truth is something we set.
The Mateus deposit has the geometry fabricated into the device - straight
``control`` channels against ``arrows``, ``rams`` and ``tesla`` valves - so the
same question can be put to real recordings.

Three things decide whether the answer means anything, and all three are
computed here rather than assumed:

- **One chip is one measurement.** A chip recorded at DIV 12, 17 and 21 supplies
  three recordings and one independent unit. Everything is aggregated per chip
  before any comparison, and intervals resample chips.
- **Direction is unanchored.** Which end of a device is "source" depends on how
  it was seated on the array, which the deposit does not record, so the
  statistic is |asymmetry| - how one-sided propagation is, not which way it
  points.
- **Few events make |asymmetry| large by chance.** A chip with 30 propagation
  events will look more one-sided than one with 900 even if both are balanced,
  so the relationship between event count and |asymmetry| is reported beside the
  comparison, and the comparison is repeated on the better-sampled half.

The deposit's own design confounds condition with batch: control chips appear
only in experiments 9-11. That cannot be analysed away and is reported with the
result.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.metrics import update_metrics
from neuromps.paths import RAW_ROOT, figures_dir, interim_dir

log = logging.getLogger("neuromps.models")

ARCHIVE = "1_MEA_Recordings.zip"
SOURCE = "mateus2024"
CHIP = re.compile(r"chip[_ ]?(\d+)", re.IGNORECASE)
DIV = re.compile(r"DIV[_ ]?(\d+)", re.IGNORECASE)
REFERENCE = "control"


def chip_identity(experiment: str, chip: str) -> str:
    """A chip is identified by its experiment as well as its number.

    ``chip_2`` in EXP1 and ``chip_2`` in EXP2 are different cultures on different
    days; treating them as one unit would merge unrelated devices.
    """
    return f"{experiment}/{chip}"


def parse_filename(name: str) -> dict[str, str | int | None]:
    """Chip number and DIV, as recorded in the recording's filename."""
    chip = CHIP.search(name)
    div = DIV.search(name)
    return {
        "chip": f"chip_{chip.group(1)}" if chip else "chip_unknown",
        "div": int(div.group(1)) if div else None,
    }


# The deposit records neither the device's orientation on the array nor which
# electrodes were dead, so both are assumptions. Each variant re-runs the whole
# detector under a different one, in the same pass over the archive.
GEOMETRY_VARIANTS = (
    ("column_strict", "column", None),
    ("row_strict", "row", None),
    ("column_one_missing", "column", 4),
    ("row_one_missing", "row", 4),
)
HEADLINE_VARIANT = "column_strict"


def recordings_table(archive: Path, variants=GEOMETRY_VARIANTS) -> pd.DataFrame:
    """Run the propagation detector over every recording, under each geometry.

    One pass over the archive, several detectors: reading the 109 MB deposit
    four times to answer a robustness question would be four times the cost for
    the same answer.
    """
    import tempfile
    import zipfile

    from neuromps.etl.mateus2024 import chip_summary, read_recording

    rows = []
    with zipfile.ZipFile(archive) as zf, tempfile.TemporaryDirectory() as tmp:
        members = sorted(n for n in zf.namelist() if n.endswith(".h5"))
        log.info("reading %d recordings from %s", len(members), archive.name)
        for member in members:
            parts = member.split("/")
            target = Path(tmp) / "recording.h5"
            target.write_bytes(zf.read(member))
            spikes, duration_s, unreadable = read_recording(target)
            for name, orientation, min_electrodes in variants:
                summary = chip_summary(spikes, duration_s, orientation, min_electrodes)
                rows.append(
                    {
                        "variant": name,
                        "experiment": parts[2],
                        "condition": parts[3].lower(),
                        "file": parts[-1],
                        "duration_s": duration_s,
                        "n_electrodes": len(spikes),
                        "n_unreadable_channels": len(unreadable),
                        "n_channels_detected": summary["n_channels_detected"],
                        "forward_events": summary["forward_events"],
                        "backward_events": summary["backward_events"],
                        "asymmetry": summary["asymmetry"],
                        **parse_filename(parts[-1]),
                    }
                )
    return pd.DataFrame(rows)


def per_chip(recordings: pd.DataFrame) -> pd.DataFrame:
    """One row per chip: the median of its recordings, and how many it had."""
    frame = recordings.copy()
    frame["chip_id"] = [
        chip_identity(e, c) for e, c in zip(frame["experiment"], frame["chip"], strict=True)
    ]
    frame["abs_asymmetry"] = frame["asymmetry"].abs()
    frame["total_events"] = frame["forward_events"] + frame["backward_events"]
    usable = frame[frame["abs_asymmetry"].notna()]
    # Only columns this aggregation needs: a recording count comes from the group
    # size, not from a particular column, so a caller with a narrower table works.
    aggregation = {
        "abs_asymmetry": ("abs_asymmetry", "median"),
        "total_events": ("total_events", "median"),
        "n_recordings": ("abs_asymmetry", "size"),
    }
    if "n_channels_detected" in usable.columns:
        aggregation["n_channels"] = ("n_channels_detected", "median")
    grouped = usable.groupby(["chip_id", "condition"], as_index=False).agg(**aggregation)
    return grouped.rename(columns={"chip_id": "chip"})


def condition_comparison(chips: pd.DataFrame, reference: str = REFERENCE) -> dict:
    """Compare |asymmetry| across geometries, over chips rather than recordings."""
    from scipy.stats import kruskal, mannwhitneyu

    from neuromps.models.bootstrap import ci95

    groups = {c: part["abs_asymmetry"].to_numpy() for c, part in chips.groupby("condition")}
    by_condition = {
        name: {
            "n_chips": int(values.size),
            "median_abs_asymmetry": round(float(np.median(values)), 4),
            "median_ci95": _median_ci(values),
            "median_total_events": round(
                float(chips.loc[chips["condition"] == name, "total_events"].median()), 1
            ),
        }
        for name, values in groups.items()
    }
    if len(groups) < 2:
        return {
            "n_chips": int(len(chips)),
            "by_condition": by_condition,
            "kruskal_p": None,
            "note": "too few conditions to compare",
        }
    statistic, p_value = kruskal(*groups.values())
    if not np.isfinite(p_value):
        # Identical values in every group leave the test undefined. NaN in a
        # metrics file reads as a number that failed; None plus a reason does not.
        return {
            "n_chips": int(len(chips)),
            "by_condition": by_condition,
            "kruskal_p": None,
            "note": "no variation between conditions, so the test is undefined",
        }
    against_reference = {}
    if reference in groups:
        for name, values in groups.items():
            if name == reference:
                continue
            _, p = mannwhitneyu(values, groups[reference], alternative="two-sided")
            against_reference[name] = round(float(p), 4)
        against_reference = _holm(against_reference)
    return {
        "n_chips": int(len(chips)),
        "statistic": "median |asymmetry| per chip; sign is unanchored",
        "by_condition": by_condition,
        "kruskal_statistic": round(float(statistic), 4),
        "kruskal_p": round(float(p_value), 5),
        "vs_reference_holm": against_reference,
        "reference": reference,
        "ci_method": f"percentile bootstrap over chips, {ci95.__module__} conventions",
    }


def _median_ci(values: np.ndarray, n_resamples: int = 1000, seed: int = 0) -> list[float] | None:
    """Bootstrap interval for a median, resampling chips."""
    if values.size < 3:
        return None
    rng = np.random.default_rng(seed)
    draws = [
        float(np.median(rng.choice(values, values.size, replace=True))) for _ in range(n_resamples)
    ]
    low, high = np.percentile(draws, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment, so three comparisons are not three chances."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for index, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (len(ordered) - index)))
        adjusted[name] = round(running, 4)
    return adjusted


def count_confound(chips: pd.DataFrame) -> dict:
    """Whether |asymmetry| is riding on how many events a chip produced."""
    from scipy.stats import spearmanr

    usable = chips.dropna(subset=["abs_asymmetry", "total_events"])
    if len(usable) < 5:
        return {"spearman_rho": None, "note": "too few chips"}
    rho, p = spearmanr(usable["total_events"], usable["abs_asymmetry"])
    return {
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(p), 5),
        "n_chips": int(len(usable)),
        "reads_as": (
            "a negative correlation means chips with fewer propagation events look more "
            "one-sided, which is what a counting artefact looks like"
        ),
    }


def _naive_by_recording(recordings: pd.DataFrame) -> dict:
    """The same comparison with recordings as the unit, which they are not.

    Kept because the difference between this and the per-chip result is the
    whole point: a chip recorded at three ages contributes three rows and one
    device, and treating the rows as independent is how a geometry effect
    appears that the chips do not support.
    """
    frame = recordings.copy()
    frame["abs_asymmetry"] = frame["asymmetry"].abs()
    frame["total_events"] = frame["forward_events"] + frame["backward_events"]
    frame = frame[frame["abs_asymmetry"].notna()]
    frame["chip"] = frame["file"]  # one "unit" per recording, deliberately wrong
    return condition_comparison(frame[["chip", "condition", "abs_asymmetry", "total_events"]])


def plot_by_condition(chips: pd.DataFrame, path: Path, title: str) -> None:
    """One point per chip, so the reader sees the sample the claim rests on."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [
        c
        for c in (REFERENCE, "arrows", "rams", "tesla", "tesla_v2")
        if c in set(chips["condition"])
    ]
    colours = {
        "control": "#0072B2",
        "arrows": "#E69F00",
        "rams": "#009E73",
        "tesla": "#CC79A7",
        "tesla_v2": "#D55E00",
    }
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for index, condition in enumerate(order):
        values = chips.loc[chips["condition"] == condition, "abs_asymmetry"].to_numpy()
        ax.scatter(
            index + rng.normal(0, 0.06, values.size),
            values,
            s=42,
            alpha=0.85,
            color=colours.get(condition, "#555555"),
            linewidths=0,
            label=f"{condition} (n={values.size})",
        )
        ax.hlines(np.median(values), index - 0.24, index + 0.24, color="black", lw=2)
    ax.set_xticks(range(len(order)), order, fontsize=9)
    ax.set_ylabel("|propagation asymmetry| per chip", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def geometry_robustness(recordings: pd.DataFrame) -> dict:
    """Does the null in the headline variant survive the other geometry assumptions?

    §7.7 reports no separation between straight and asymmetric microchannels.
    That rests on a detector that assumes the channels run down the columns and
    that every one of their five electrodes fired. If either assumption is
    wrong the events are simply missed, and a null found by missing the data is
    not a null. Each variant is the same comparison under a different
    assumption.
    """
    out = {}
    for name in sorted(recordings["variant"].unique()):
        subset = recordings[recordings["variant"] == name]
        chips = per_chip(subset)
        if chips.empty:
            out[name] = {"n_chips": 0, "note": "no chip carried a detectable channel"}
            continue
        comparison = condition_comparison(chips)
        out[name] = {
            "n_chips": int(len(chips)),
            "recordings_with_a_channel": int((~subset["asymmetry"].isna()).sum()),
            "median_channels_detected": float(subset["n_channels_detected"].median()),
            "total_events": int(subset[["forward_events", "backward_events"]].to_numpy().sum()),
            "kruskal_p": comparison.get("kruskal_p"),
            "median_abs_asymmetry_by_condition": comparison.get("median_abs_asymmetry"),
        }
    passing = [
        name
        for name, entry in out.items()
        if entry.get("kruskal_p") is not None and entry["kruskal_p"] < 0.05
    ]
    return {
        "variants": out,
        "headline_variant": HEADLINE_VARIANT,
        "variants_separating_conditions": sorted(passing),
        "null_survives_every_variant": not passing,
        "note": (
            "orientation and dead-electrode tolerance are assumptions the deposit does not "
            "settle; 'one_missing' takes the best four-of-five ordered chain, which is "
            "optimistic by construction and bounds the sensitivity rather than improving it"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the detector over the deposit and compare geometries, chip by chip."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="fewer bootstrap resamples")
    p.add_argument("--rebuild", action="store_true", help="ignore the cached per-recording table")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    archive = RAW_ROOT / SOURCE / ARCHIVE
    if not archive.exists():
        log.warning("%s not fetched; run the fetch stage at full tier", archive)
        return 0
    cache = interim_dir() / "mateus2024_propagation.parquet"
    if cache.exists() and not args.rebuild:
        recordings = pd.read_parquet(cache)
        log.info("reusing %d cached recording summaries", len(recordings))
    else:
        recordings = recordings_table(archive)
        cache.parent.mkdir(parents=True, exist_ok=True)
        recordings.to_parquet(cache, index=False)

    if "variant" not in recordings.columns:  # a cache from before the variants existed
        recordings = recordings.assign(variant=HEADLINE_VARIANT)
    robustness = geometry_robustness(recordings)
    recordings = recordings[recordings["variant"] == HEADLINE_VARIANT]
    chips = per_chip(recordings)
    comparison = condition_comparison(chips)
    confound = count_confound(chips)
    well_sampled = chips[chips["total_events"] >= chips["total_events"].median()]
    results = {
        "label": "REAL recordings: microchannel propagation asymmetry by device geometry",
        "source": SOURCE,
        "n_recordings": int(len(recordings)),
        "n_chips": int(len(chips)),
        "n_unreadable_channels": int(recordings["n_unreadable_channels"].sum()),
        "recordings_without_detected_channels": int((recordings["asymmetry"].isna()).sum()),
        "criterion": (
            "the deposit's own: an event on all 5 electrodes of a microchannel with <=1 ms "
            "between consecutive electrodes"
        ),
        "comparison": comparison,
        "geometry_robustness": robustness,
        "per_recording_naive": _naive_by_recording(recordings),
        "event_count_confound": confound,
        "well_sampled_half": condition_comparison(well_sampled),
        "batch_confound": (
            "control chips appear only in experiments 9-11 and the asymmetric geometries only in "
            "1-8 and 12, so condition is confounded with culture batch by the deposit's design"
        ),
        "direction_note": (
            "sign is unanchored: the deposit does not record how each device was seated on the "
            "array, so only the magnitude of the asymmetry is comparable across chips"
        ),
    }
    figures = figures_dir()
    figures.mkdir(parents=True, exist_ok=True)
    plot_by_condition(
        chips,
        figures / "microchannel_asymmetry_real.png",
        "REAL data: propagation asymmetry per chip, by microchannel geometry\n"
        "one point per chip; bars are medians; sign unanchored, so magnitude only",
    )
    update_metrics("microchannel_directionality", results)
    print(
        json.dumps(
            {
                k: results[k]
                for k in ("n_recordings", "n_chips", "comparison", "event_count_confound")
            },
            indent=2,
        )[:1800]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
