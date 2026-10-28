"""Does `features/mea.py` reproduce the numbers the depositors themselves published?

The asset's central claim is that one fixed extractor makes sources comparable.
That claim was asserted rather than measured: nothing in the pipeline checked
our nine features against an independent implementation of the same quantities.

Kapucu 2022 makes the check possible. Its three maturation plates deposit both
the spike times and the authors' own meaRtools output - the same recordings,
scored by different software - for seven of our nine features. Running our
extractor over their spike files and joining on (plate, well, DIV) turns "our
definitions are standardized" into a number per feature.

**Disagreement is the expected result for some features, and is the point.**
Two implementations of "burst rate" differ because the burst detector and the
normalisation differ, not because one is wrong. What this module produces is a
map of where the two agree to within a few percent and where a named definitional
choice separates them - which is exactly what a reader pooling MEA data across
studies needs and almost never has.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.features.mea import FeatureParams, extract_features
from neuromps.parallel import worker_processes

log = logging.getLogger("neuromps.models")

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = "kapucu2022"
DATA_ROOT = REPO_ROOT / "data" / "raw" / SOURCE / "Data"
# The three maturation plates; the pharmacology plates carry no meaRtools output.
PLATES = ("hPSC_MEA1", "hPSC_MEA2", "Rat_MEA1")

# Our feature name -> the deposit's file stem. `meanfiringrate_by_active_
# electordes` is the deposit's spelling and is kept verbatim so the path exists.
MEARTOOLS_FILES = {
    "mfr_hz": "meanfiringrate_by_active_electordes",
    "burst_rate_per_min": "bursts_per_min",
    "burst_duration_s": "mean_dur",
    "spikes_per_burst": "mean_spikes_in_burst",
    "pct_spikes_in_bursts": "per_spikes_in_burst",
    "synchrony_index": "STTC",
    "active_electrodes": "nae",
}
# `isi_cv` and `network_burst_rate_per_min` have no meaRtools counterpart in this
# deposit and are not checked here.
UNCHECKABLE = ("isi_cv", "network_burst_rate_per_min")

# A feature "reproduces" when the two implementations order the wells the same
# way and agree on scale. Both thresholds are conventions, stated so a reader can
# move them.
RANK_AGREEMENT = 0.95
SCALE_TOLERANCE = 0.05
RESAMPLES = 1000


def plate_prefix(plate_dir: Path) -> str | None:
    """The file-name prefix this plate's files share, read off the files."""
    for path in sorted(plate_dir.glob("*_spikes.csv")):
        match = re.match(r"(.+)_DIV\d+_spikes\.csv$", path.name)
        if match:
            return match.group(1)
    return None


def read_meartools(output_dir: Path, prefix: str) -> pd.DataFrame:
    """The depositors' own values, long-form: one row per well, DIV and feature."""
    rows = []
    for feature, stem in MEARTOOLS_FILES.items():
        path = output_dir / f"{prefix}_{stem}.csv"
        if not path.exists():
            log.warning("no meaRtools file for %s at %s", feature, path)
            continue
        table = pd.read_csv(path)
        div_columns = [c for c in table.columns if c.startswith("divDIV")]
        melted = table.melt(
            id_vars=[c for c in ("well", "treatment") if c in table.columns],
            value_vars=div_columns,
            var_name="div_column",
            value_name="theirs",
        )
        melted["div"] = melted["div_column"].str.removeprefix("divDIV").astype(int)
        melted["feature"] = feature
        rows.append(melted[["well", "div", "feature", "theirs"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


NOISY_MODES = ("kept", "dropped_per_div", "dropped_all_divs")


@dataclass(frozen=True)
class Definition:
    """One set of feature definitions to run over the deposit's spike files.

    ``params`` go to the shared extractor unchanged. ``noisy`` says what happens to
    the deposit's noisy-electrode list before extraction: ``kept`` (our default,
    the ETL flags them and drops nothing), ``dropped_per_div`` (the list in that
    recording's DIV column) or ``dropped_all_divs`` (any electrode listed on any
    DIV of the plate, the alternative reading of the file).
    """

    name: str
    params: FeatureParams = field(default_factory=FeatureParams)
    noisy: str = "kept"

    def __post_init__(self) -> None:
        if self.noisy not in NOISY_MODES:
            raise ValueError(f"noisy must be one of {NOISY_MODES}, got {self.noisy!r}")


# The published agreement: our extractor exactly as the asset uses it.
DEFAULT_DEFINITION = Definition("S0_ours")


def noisy_lists(path: Path | None, div: int) -> dict[str, frozenset[str]]:
    """The deposit's noisy electrodes for one DIV, and over all its DIVs.

    The developmental files hold one column per DIV. A DIV with no column of its
    own drops nothing rather than borrowing another DIV's list.
    """
    empty: dict[str, frozenset[str]] = {
        "dropped_per_div": frozenset(),
        "dropped_all_divs": frozenset(),
    }
    if path is None:
        return empty
    table = pd.read_csv(path, dtype=str)
    table.columns = [str(c).strip() for c in table.columns]

    def listed(columns: list[str]) -> frozenset[str]:
        values = table[columns].stack().dropna() if columns else pd.Series(dtype=str)
        return frozenset(v.strip() for v in values if v.strip())

    column = f"DIV{div}"
    return {
        "dropped_per_div": listed([column]) if column in table.columns else frozenset(),
        "dropped_all_divs": listed(list(table.columns)),
    }


def _channels_for(
    channels: dict[str, np.ndarray], noisy: dict[str, frozenset[str]], definition: Definition
) -> dict[str, np.ndarray]:
    """The well's electrodes a definition keeps."""
    if definition.noisy == "kept":
        return channels
    drop = noisy.get(definition.noisy, frozenset())
    return {name: times for name, times in channels.items() if name not in drop}


def _features_job(args) -> list[dict]:
    """Our features for every well of one recording, under each definition, in its own process."""
    path, div, noisy, definitions = args
    from neuromps.etl.kapucu import _load_spikes

    try:
        wells, duration_s = _load_spikes(path)
    except Exception:  # noqa: BLE001 - one unreadable file must not stop the sweep
        log.exception("could not read %s", path)
        return []
    rows = []
    for well, channels in wells.items():
        computed_for: dict[tuple, dict] = {}
        for definition in definitions:
            key = (definition.params, definition.noisy)
            if key not in computed_for:
                kept = _channels_for(channels, noisy, definition)
                computed_for[key] = extract_features(kept, duration_s, definition.params)
            computed = computed_for[key]
            for feature in MEARTOOLS_FILES:
                value = computed.get(feature)
                if value is not None and np.isfinite(value):
                    rows.append(
                        {
                            "well": well,
                            "div": div,
                            "feature": feature,
                            "ours": float(value),
                            "variant": definition.name,
                        }
                    )
    return rows


def _definition_groups(definitions: tuple[Definition, ...]) -> list[tuple[Definition, ...]]:
    """Definitions that compute the same thing travel together, so each runs once."""
    groups: dict[tuple, list[Definition]] = {}
    for definition in definitions:
        groups.setdefault((definition.params, definition.noisy), []).append(definition)
    return [tuple(group) for group in groups.values()]


def our_features(
    spikes_dir: Path,
    prefix: str,
    processes: int | None = None,
    max_recordings: int | None = None,
    definitions: tuple[Definition, ...] = (DEFAULT_DEFINITION,),
) -> pd.DataFrame:
    """Run the shared extractor over one plate's recordings, under each definition.

    ``max_recordings`` exists for the quick tier, which only has to reach every
    branch; a path test does not need all twenty DIVs of a plate. One job per
    recording and group of identical definitions, so a ladder of definitions
    spreads over the pool instead of lengthening each job.
    """
    import multiprocessing as mp

    recordings = []
    for path in sorted(spikes_dir.glob(f"{prefix}_DIV*_spikes.csv")):
        match = re.search(r"_DIV(\d+)_spikes\.csv$", path.name)
        if match:
            recordings.append((path, int(match.group(1))))
    recordings = recordings[:max_recordings] if max_recordings else recordings
    if not recordings:
        return pd.DataFrame()
    # Only a definition that drops noisy electrodes reads the list; the default never does.
    noisy_path = (
        next(iter(sorted(spikes_dir.glob("noisy_electrodes*.csv"))), None)
        if any(d.noisy != "kept" for d in definitions)
        else None
    )
    jobs = [
        (path, div, noisy_lists(noisy_path, div), group)
        for path, div in recordings
        for group in _definition_groups(definitions)
    ]
    processes = processes or worker_processes()
    log.info(
        "extracting %s: %d recordings x %d definitions on %d processes",
        prefix,
        len(recordings),
        len(definitions),
        processes,
    )
    with mp.Pool(processes) as pool:
        results = pool.map(_features_job, jobs, chunksize=1)
    return pd.DataFrame([row for batch in results for row in batch])


def paired_table(
    plates: tuple[str, ...] = PLATES,
    root: Path = DATA_ROOT,
    max_recordings: int | None = None,
    definitions: tuple[Definition, ...] | None = None,
    processes: int | None = None,
) -> pd.DataFrame:
    """Our value beside theirs for every well, DIV and feature both computed.

    With ``definitions`` the table gains a ``variant`` column, one block of rows
    per definition; without, it is the published agreement's table, unchanged.
    """
    frames = []
    for plate in plates:
        plate_dir = root / plate
        spikes_dir = plate_dir / f"{plate}_spikes_noise_explogs"
        output_dir = plate_dir / f"{plate}_meaRtools_output"
        if not spikes_dir.is_dir() or not output_dir.is_dir():
            log.info("plate %s not present; skipped", plate)
            continue
        prefix = plate_prefix(spikes_dir)
        if prefix is None:
            log.warning("no spike files found for %s", plate)
            continue
        ours = our_features(
            spikes_dir,
            prefix,
            processes=processes,
            max_recordings=max_recordings,
            definitions=definitions or (DEFAULT_DEFINITION,),
        )
        if definitions is None:
            ours = ours.drop(columns="variant", errors="ignore")
        theirs = read_meartools(output_dir, prefix)
        if ours.empty or theirs.empty:
            continue
        joined = ours.merge(theirs, on=["well", "div", "feature"], how="inner")
        joined["plate"] = plate
        frames.append(joined)
    if not frames:
        columns = ["plate", "well", "div", "feature", "ours", "theirs"]
        return pd.DataFrame(columns=columns + (["variant"] if definitions else []))
    table = pd.concat(frames, ignore_index=True)
    return table[table["theirs"].notna() & np.isfinite(table["theirs"])]


def _ratio(ours: np.ndarray, theirs: np.ndarray) -> np.ndarray:
    """Our value over theirs, where theirs is non-zero."""
    usable = theirs != 0
    return ours[usable] / theirs[usable]


def agreement_by_feature(table: pd.DataFrame, seed: int = 0) -> dict:
    """Rank agreement and scale agreement per feature, clustered over wells.

    Two numbers, because they answer different questions. Spearman says whether
    the two implementations order the recordings the same way - which is what a
    model trained on either would see. The median ratio says whether they are on
    the same scale - which is what matters when values from two studies are
    pooled without refitting.
    """
    from scipy.stats import spearmanr

    from neuromps.models.bootstrap import group_bootstrap_metric

    out = {}
    for feature, rows in table.groupby("feature"):
        ours = rows["ours"].to_numpy(dtype=float)
        theirs = rows["theirs"].to_numpy(dtype=float)
        # A well is the cluster: one well contributes ~20 correlated DIVs.
        groups = (rows["plate"] + ":" + rows["well"]).to_numpy()
        ratios = _ratio(ours, theirs)
        if len(ours) < 4 or len(set(theirs)) < 2:
            out[str(feature)] = {"n": int(len(ours)), "note": "too few paired points"}
            continue

        def rank(idx: np.ndarray, a=ours, b=theirs) -> float:
            if len(set(a[idx])) < 2 or len(set(b[idx])) < 2:
                return float("nan")
            return float(spearmanr(a[idx], b[idx]).statistic)

        observed = float(spearmanr(ours, theirs).statistic)
        boot = group_bootstrap_metric(rank, groups, RESAMPLES, seed)
        median_ratio = float(np.median(ratios)) if ratios.size else float("nan")
        within = float(np.mean(np.abs(ratios - 1.0) <= 0.10)) if ratios.size else float("nan")
        reproduces = bool(observed >= RANK_AGREEMENT and abs(median_ratio - 1.0) <= SCALE_TOLERANCE)
        out[str(feature)] = {
            "n_paired": int(len(ours)),
            "n_wells": int(len(set(groups))),
            "spearman_rho": round(observed, 4),
            "spearman_rho_ci95": boot.get("ci95"),
            "median_ratio_ours_over_theirs": round(median_ratio, 4),
            "fraction_within_10pct": round(within, 4),
            "reproduces": reproduces,
        }
    return out


def summarise(by_feature: dict) -> dict:
    """Which features reproduce, which are rescaled, and which disagree outright."""
    reproduce, rescaled, disagree = [], [], []
    for feature, entry in sorted(by_feature.items()):
        if "spearman_rho" not in entry:
            continue
        if entry["reproduces"]:
            reproduce.append(feature)
        elif entry["spearman_rho"] >= RANK_AGREEMENT:
            rescaled.append(feature)
        else:
            disagree.append(feature)
    return {
        "reproduce": reproduce,
        "same_ordering_different_scale": rescaled,
        "disagree_on_ordering_too": disagree,
        "n_checked": len(reproduce) + len(rescaled) + len(disagree),
        "n_uncheckable": len(UNCHECKABLE),
        "uncheckable": list(UNCHECKABLE),
        "rank_agreement_threshold": RANK_AGREEMENT,
        "scale_tolerance": SCALE_TOLERANCE,
    }


def plot_agreement(table: pd.DataFrame, by_feature: dict, path: Path) -> Path:
    """Our value against the depositors', one panel per feature (real data)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = [f for f in MEARTOOLS_FILES if f in set(table["feature"])]
    if not features:
        return path
    # Nothing has made figures/ yet on a fresh quick run.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 4
    rows_n = int(np.ceil(len(features) / columns))
    # Drawn at about the width it prints at; drawn at 11 in and shown at under 4,
    # its tick labels printed at about 2 pt.
    fig, axes = plt.subplots(rows_n, columns, figsize=(7.6, 2.5 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    colours = {"hPSC_MEA1": "#0072B2", "hPSC_MEA2": "#E69F00", "Rat_MEA1": "#009E73"}
    for ax, feature in zip(axes, features, strict=False):
        rows = table[table["feature"] == feature]
        for plate, subset in rows.groupby("plate"):
            ax.scatter(
                subset["theirs"],
                subset["ours"],
                s=9,
                alpha=0.6,
                color=colours.get(str(plate), "#666666"),
                label=str(plate),
                linewidth=0,
            )
        # Clipped at a high percentile rather than the maximum: one well with a
        # 700-spike "burst" otherwise compresses a whole panel into the corner.
        limit = float(np.nanpercentile(np.concatenate([rows["ours"], rows["theirs"]]), 99)) * 1.05
        clipped = int(((rows["ours"] > limit) | (rows["theirs"] > limit)).sum())
        ax.plot([0, limit], [0, limit], "k--", linewidth=0.8)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        if clipped:
            ax.text(
                0.97,
                0.06,
                f"{clipped} off scale",
                transform=ax.transAxes,
                fontsize=6,
                ha="right",
                color="#555555",
            )
        entry = by_feature.get(feature, {})
        ratio = entry.get("median_ratio_ours_over_theirs")
        rho = entry.get("spearman_rho")
        ax.set_title(
            f"{feature}\nrho {rho}, ratio {ratio}" if rho is not None else str(feature),
            fontsize=7.5,
        )
        ax.tick_params(labelsize=7.5)
    for ax in axes[len(features) :]:
        ax.set_visible(False)
    axes[0].legend(fontsize=6)
    fig.supxlabel("meaRtools value, as the depositors published it", fontsize=8.5)
    fig.supylabel("our features/mea.py value", fontsize=8.5)
    fig.suptitle(
        "REAL recordings, two implementations of the same features:\nours against the "
        "depositors' meaRtools output. Dashed line is equality.",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def default_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The published agreement's rows (S0) out of a table carrying several definitions."""
    s0 = table[table["variant"] == DEFAULT_DEFINITION.name]
    return s0.drop(columns="variant").reset_index(drop=True)


def _previous_by_feature() -> dict | None:
    """The agreement already in this run's metrics.json, before it is overwritten."""
    import json

    from neuromps.paths import metrics_path

    path = metrics_path()
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("extractor_agreement", {}).get("by_feature")


def agreement_record(table: pd.DataFrame, by_feature: dict) -> dict:
    """The published agreement's metrics section, before its figure and ladder are added."""
    return {
        "label": "REAL recordings, our extractor against the depositors' meaRtools output",
        "question": (
            "does features/mea.py reproduce the numbers an independent implementation "
            "computed from the same spike files"
        ),
        "n_plates": int(table["plate"].nunique()),
        "n_recordings": int(table.groupby(["plate", "well", "div"]).ngroups),
        "n_paired_values": int(len(table)),
        "by_feature": by_feature,
        **summarise(by_feature),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: measure the extractor against the depositors' own numbers."""
    import argparse
    import json

    from neuromps.metrics import update_metrics
    from neuromps.paths import figures_dir

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="one plate, for the path test")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from neuromps.models import extractor_ladder, extractor_logisi

    plates = PLATES[:1] if args.quick else PLATES
    max_recordings = 4 if args.quick else None
    # One sweep computes the published agreement (S0), the post hoc settings ladder
    # and the registered logISI variant (S4, D34) together; the S0 rows are the
    # agreement's table, row for row, and the ladder sees its own rows only.
    swept = paired_table(
        plates,
        max_recordings=max_recordings,
        definitions=(*extractor_ladder.ALL_DEFINITIONS, extractor_logisi.LOGISI),
    )
    ladder_table = swept[swept["variant"] != extractor_logisi.LOGISI.name]
    table = default_rows(ladder_table)
    if table.empty:
        log.warning("no paired recordings found; is %s fetched?", DATA_ROOT)
        update_metrics(
            "extractor_agreement", {"status": "kapucu2022 maturation plates not present"}
        )
        return 0
    by_feature = agreement_by_feature(table, seed=args.seed)
    result = agreement_record(table, by_feature)
    figure = plot_agreement(table, by_feature, figures_dir() / "extractor_agreement_real.png")
    result["figure"] = figure.name
    result["settings_ladder"] = extractor_ladder.run(
        ladder_table, by_feature, _previous_by_feature(), plates
    )
    result["logisi"] = extractor_logisi.run(swept, plates, max_recordings)
    update_metrics("extractor_agreement", result)
    print(json.dumps({k: v for k, v in result.items() if k != "by_feature"}, indent=2))
    for feature, entry in sorted(by_feature.items()):
        print(f"  {feature:28s} {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
