"""POST HOC: our extractor under the Kapucu 2022 depositors' stated settings, one change at a time.

`extractor_agreement` measures how far `features/mea.py` sits from the meaRtools
output the depositors published for the same spike files; burst rate differs
about twofold. This module asks where that distance comes from, by rerunning the
*same* extractor with the depositors' stated choices switched on one by one:

- ``S0_ours``: our definitions, exactly as the asset uses them.
- ``S1_active_gt10``: an electrode is active above 10 spikes/min, not at 5 or more.
- ``S2_noisy_dropped``: S1, and the deposit's noisy electrodes for that DIV are
  removed before extraction (we flag them and keep them).
- ``S3_rate_per_bursting_electrode``: S2, and bursts per minute are divided by the
  electrodes that burst, not by every active electrode.

plus each change alone from S0, and one assumption check. The burst detector
itself (their adaptive logISI against our fixed ISI rule) is not changed: that
difference is named, not tested.

**This is post hoc.** A scratch analysis ran these variants before this module
was written, so the numbers here were seen before the code that produces them.
It is not registered, makes no prediction, and is labelled so in every output.
It says what the definitional choices cost on one deposit; it does not say which
definitions are correct.
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.features.mea import FeatureParams
from neuromps.models.extractor_agreement import DEFAULT_DEFINITION, PLATES, Definition, _ratio

log = logging.getLogger("neuromps.models")

LABEL = (
    "POST HOC (exploratory; the result was seen in a scratch analysis before this code was "
    "written; not registered): REAL recordings (Kapucu 2022), our extractor rerun on the "
    "depositors' spike files under their stated settings, one change at a time, against the "
    "meaRtools output they published"
)
ASSUMPTION = (
    "noisy_electrodes_<plate>.csv lists noisy electrodes per DIV column, and a DIV's list "
    "applies to that DIV's recording only; checked against the alternative reading that any "
    "electrode listed on any DIV is noisy throughout (check_noisy_dropped_all_divs)"
)
FIGURE_NAME = "extractor_settings_ladder_real.png"

# The depositors' choices as their Methods state them (Kapucu et al. 2022, copy at
# data/raw/kapucu2022/paper/PMC8969177.xml). Recorded here so the report can cite them.
DEPOSITORS_STATED = {
    "active_min_spikes_per_min": 10.0,
    "active_rule": "more than 10 spikes per minute",
    "noisy_electrodes": "discarded by the analysis code",
    "burst_detector": "logISI (adaptive), at least 5 spikes, bursts merged when under 100 ms apart",
    "burst_min_spikes": 5,
    "burst_merge_interval_ms": 100,
    "burst_rate_over": "electrodes exhibiting bursts",
    "sttc_dt_ms": 50,
    "source": "data/raw/kapucu2022/paper/PMC8969177.xml, Methods",
}

_GT10 = FeatureParams(active_min_spikes_per_min=10.0, active_strict=True)
LADDER = (
    DEFAULT_DEFINITION,
    Definition("S1_active_gt10", _GT10),
    Definition("S2_noisy_dropped", _GT10, noisy="dropped_per_div"),
    Definition(
        "S3_rate_per_bursting_electrode",
        replace(_GT10, burst_rate_denominator="bursting"),
        noisy="dropped_per_div",
    ),
)
ONE_AT_A_TIME = (
    Definition("only_active_gt10", _GT10),
    Definition("only_noisy_dropped", noisy="dropped_per_div"),
    Definition(
        "only_rate_per_bursting_electrode", FeatureParams(burst_rate_denominator="bursting")
    ),
)
ASSUMPTION_CHECK = Definition("check_noisy_dropped_all_divs", _GT10, noisy="dropped_all_divs")
ALL_DEFINITIONS = (*LADDER, *ONE_AT_A_TIME, ASSUMPTION_CHECK)
START, END = LADDER[0].name, LADDER[-1].name


def changes_from(base: Definition, other: Definition) -> list[str]:
    """The settings ``other`` changes relative to ``base``, extractor fields first."""
    changed = [
        f.name
        for f in fields(FeatureParams)
        if getattr(base.params, f.name) != getattr(other.params, f.name)
    ]
    return changed + (["noisy"] if base.noisy != other.noisy else [])


def _stats(rows: pd.DataFrame, feature: str) -> dict:
    """Pairs, median ratio ours/theirs, Spearman rho and share within 10%, as the agreement does."""
    from scipy.stats import spearmanr

    ours = rows["ours"].to_numpy(dtype=float)
    theirs = rows["theirs"].to_numpy(dtype=float)
    ratios = _ratio(ours, theirs)
    rho = (
        round(float(spearmanr(ours, theirs).statistic), 4)
        if len(set(ours)) > 1 and len(set(theirs)) > 1
        else None
    )
    out = {
        "n": int(len(ours)),
        "median_ratio": round(float(np.median(ratios)), 4) if ratios.size else None,
        "spearman_rho": rho,
        "fraction_within_10pct": (
            round(float(np.mean(np.abs(ratios - 1.0) <= 0.10)), 4) if ratios.size else None
        ),
    }
    if feature == "active_electrodes":
        out["fraction_exact"] = round(float(np.mean(ours == theirs)), 4)
    return out


def ladder_stats(table: pd.DataFrame) -> dict:
    """Variant -> ``pooled`` and each plate -> feature -> statistics."""
    out: dict[str, dict] = {}
    for variant, by_variant in table.groupby("variant", sort=False):
        scopes = {"pooled": by_variant, **dict(tuple(by_variant.groupby("plate")))}
        out[str(variant)] = {
            str(scope): {
                str(feature): _stats(rows, str(feature))
                for feature, rows in subset.groupby("feature")
            }
            for scope, subset in scopes.items()
        }
    return out


def _median_ratio(rows: pd.DataFrame) -> float | None:
    ratios = _ratio(rows["ours"].to_numpy(dtype=float), rows["theirs"].to_numpy(dtype=float))
    return float(np.median(ratios)) if ratios.size else None


def _gap(start: float | None, end: float | None) -> float | None:
    """Share of |log ratio| removed going from ``start`` to ``end``; None if nothing to close."""
    if start is None or end is None or start <= 0 or end <= 0:
        return None
    before = abs(np.log(start))
    if before < 1e-9:
        return None
    return round(float((before - abs(np.log(end))) / before), 4)


def gap_closed(table: pd.DataFrame, start: str = START, end: str = END) -> dict:
    """Feature -> ``pooled`` and each plate -> share of the log-ratio gap closed, start to end."""
    out: dict[str, dict] = {}
    first, last = table[table["variant"] == start], table[table["variant"] == end]
    for feature in sorted(set(first["feature"]) & set(last["feature"])):
        a, b = first[first["feature"] == feature], last[last["feature"] == feature]
        entry = {"pooled": _gap(_median_ratio(a), _median_ratio(b))}
        for plate in sorted(set(a["plate"]) & set(b["plate"])):
            entry[str(plate)] = _gap(
                _median_ratio(a[a["plate"] == plate]), _median_ratio(b[b["plate"] == plate])
            )
        out[str(feature)] = entry
    return out


def bursting_fraction(table: pd.DataFrame) -> dict:
    """Median share of active electrodes that burst, per plate and pooled (S2 rate / S3 rate).

    The two rates count the same bursts on the same electrodes and differ only in the
    denominator, so their ratio is bursting electrodes over active electrodes.
    """
    keys = ["plate", "well", "div"]
    rate = table[table["feature"] == "burst_rate_per_min"]
    s2 = rate[rate["variant"] == LADDER[2].name].set_index(keys)["ours"]
    s3 = rate[rate["variant"] == LADDER[3].name].set_index(keys)["ours"]
    both = pd.concat({"s2": s2, "s3": s3}, axis=1, join="inner")
    both = both[both["s3"] > 0]
    share = (both["s2"] / both["s3"]).rename("share").reset_index()
    out = {"pooled": round(float(share["share"].median()), 4) if len(share) else None}
    for plate, rows in share.groupby("plate"):
        out[str(plate)] = round(float(rows["share"].median()), 4)
    return out


def s0_reproduces(ladder_pooled: dict, by_feature: dict) -> bool:
    """Whether the ladder's S0 gives every published agreement statistic exactly."""
    pairs = {
        "n": "n_paired",
        "spearman_rho": "spearman_rho",
        "median_ratio": "median_ratio_ours_over_theirs",
        "fraction_within_10pct": "fraction_within_10pct",
    }
    checked = [f for f, entry in by_feature.items() if "spearman_rho" in entry]
    if not checked:
        return False
    for feature in checked:
        mine = ladder_pooled.get(feature)
        if mine is None:
            return False
        if any(mine.get(a) != by_feature[feature].get(b) for a, b in pairs.items()):
            return False
    return True


def _same_record(current: dict, previous: dict | None) -> bool | None:
    """Whether this run's by_feature equals the one already in metrics.json (None if absent)."""
    if not previous:
        return None
    return json.loads(json.dumps(current)) == json.loads(json.dumps(previous))


def noisy_column_coverage(plates: tuple[str, ...], root: Path) -> dict:
    """Spike-file DIVs the deposit's noisy list has a column for, per plate (the assumption)."""
    import re

    out = {}
    for plate in plates:
        folder = root / plate / f"{plate}_spikes_noise_explogs"
        noisy = next(iter(sorted(folder.glob("noisy_electrodes*.csv"))), None)
        divs = sorted(
            int(m.group(1))
            for p in folder.glob("*_DIV*_spikes.csv")
            if (m := re.search(r"_DIV(\d+)_spikes\.csv$", p.name))
        )
        header = pd.read_csv(noisy, nrows=0).columns if noisy else []
        columns = {str(c).strip() for c in header}
        out[plate] = {
            "spike_file_divs": len(divs),
            "divs_with_a_noisy_column": sum(f"DIV{d}" in columns for d in divs),
        }
    return out


def definitions_record() -> dict:
    """Each variant's settings and what it changes from S0, for metrics.json."""
    return {
        d.name: {
            "changes_from_S0": changes_from(DEFAULT_DEFINITION, d),
            "active_min_spikes_per_min": d.params.active_min_spikes_per_min,
            "active_strict": d.params.active_strict,
            "noisy": d.noisy,
            "burst_rate_denominator": d.params.burst_rate_denominator,
        }
        for d in ALL_DEFINITIONS
    }


def our_settings() -> dict:
    """The asset's own definitions, as `FeatureParams` defaults hold them (D33)."""
    p = FeatureParams()
    return {
        "active_min_spikes_per_min": p.active_min_spikes_per_min,
        "active_rule": "at least 5 spikes per minute",
        "noisy_electrodes": "flagged in qc, kept",
        "burst_detector": "fixed: ISI <= 100 ms, at least 5 spikes",
        "burst_max_isi_ms": round(p.burst_max_isi_s * 1000),
        "burst_min_spikes": p.burst_min_spikes,
        "burst_rate_over": "active electrodes",
        "sttc_dt_ms": round(p.sttc_dt_s * 1000),
    }


def settings_ladder(
    table: pd.DataFrame,
    by_feature: dict,
    previous_by_feature: dict | None,
    coverage: dict,
) -> dict:
    """The metrics payload: every variant's agreement, the gap closed, and the checks."""
    stats = ladder_stats(table)
    pooled_nae = {name: stats.get(name, {}).get("pooled", {}) for name in stats}
    exact = {
        name: pooled_nae.get(name, {}).get("active_electrodes", {}).get("fraction_exact")
        for name in (LADDER[2].name, ASSUMPTION_CHECK.name)
    }
    this_run = s0_reproduces(stats.get(START, {}).get("pooled", {}), by_feature)
    previous = _same_record(by_feature, previous_by_feature)
    return {
        "label": LABEL,
        "status": "post hoc, exploratory, not registered",
        "assumption": ASSUMPTION,
        "noisy_dropped": "before extraction, so before the active-electrode threshold",
        "not_tested": "the burst detector: the depositors' adaptive logISI against our fixed rule",
        "depositors_stated_settings": DEPOSITORS_STATED,
        "our_settings": our_settings(),
        "definitions": definitions_record(),
        "ladder": [d.name for d in LADDER],
        "one_at_a_time": [d.name for d in ONE_AT_A_TIME],
        "by_variant": stats,
        "gap_closed_S0_to_S3": gap_closed(table),
        "bursting_fraction_of_active_S2": bursting_fraction(table),
        "assumption_check": {
            "active_electrodes_fraction_exact": exact,
            "noisy_column_coverage": coverage,
        },
        "s0_matches_this_runs_agreement": this_run,
        "s0_agreement_matches_previous_record": previous,
        "check_s0_reproduces_published_agreement": bool(this_run and previous is not False),
    }


OKABE_ITO = {
    "hPSC_MEA1": "#0072B2",
    "hPSC_MEA2": "#E69F00",
    "Rat_MEA1": "#009E73",
    "pooled": "#000000",
}


def _ratio_of(stats: dict, variant: str, scope: str, feature: str) -> float:
    value = stats.get(variant, {}).get(scope, {}).get(feature, {}).get("median_ratio")
    return float(value) if value is not None else float("nan")


def _burst_rate_panel(ax, stats: dict) -> None:
    """Burst-rate median ratio per plate across S0 -> S3."""
    x = np.arange(len(LADDER))
    for scope, colour in OKABE_ITO.items():
        y = [_ratio_of(stats, d.name, scope, "burst_rate_per_min") for d in LADDER]
        if np.all(np.isnan(y)):
            continue
        style = {"linewidth": 2.2} if scope == "pooled" else {"linewidth": 1.4}
        ax.plot(x, y, marker="o", color=colour, label=scope, **style)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, ["S0\nours", "S1\nactive >10", "S2\n+noisy dropped", "S3\n+per bursting"])
    ax.set_ylabel("burst rate, median ratio ours / theirs")
    ax.set_ylim(0, 1.2)
    ax.set_title("Burst rate per plate", fontsize=8.5)
    ax.legend(fontsize=7, loc="lower right")


def _scale_panel(ax, stats: dict) -> None:
    """Active electrodes, firing rate and STTC (pooled) at S0, S1 and S2."""
    features = ("active_electrodes", "mfr_hz", "synchrony_index")
    labels = ("active\nelectrodes", "mean firing\nrate", "STTC")
    shades = ("#D55E00", "#56B4E9", "#CC79A7")
    width = 0.26
    for k, (definition, shade) in enumerate(zip(LADDER[:3], shades, strict=True)):
        y = [_ratio_of(stats, definition.name, "pooled", f) for f in features]
        ax.bar(np.arange(3) + (k - 1) * width, y, width, color=shade, label=definition.name)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("median ratio ours / theirs (pooled)")
    ax.set_title("Electrode set: S0 against S1 and S2", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper right")


def plot_ladder(stats: dict, path: Path) -> Path:
    """The ladder figure (real recordings, post hoc)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.4), width_ratios=(1.5, 1))
    _burst_rate_panel(left, stats)
    _scale_panel(right, stats)
    for ax in (left, right):
        ax.tick_params(labelsize=7)
        ax.yaxis.label.set_size(7.5)
    fig.suptitle(
        "REAL recordings (Kapucu 2022): our extractor with the depositors' settings,\n"
        "one change at a time (post hoc). Dashed line is equality.",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def run(
    table: pd.DataFrame,
    by_feature: dict,
    previous_by_feature: dict | None,
    plates: tuple[str, ...] = PLATES,
    root: Path | None = None,
) -> dict:
    """The ladder payload with its figure, from a table carrying every definition."""
    from neuromps.models.extractor_agreement import DATA_ROOT
    from neuromps.paths import figures_dir

    payload = settings_ladder(
        table, by_feature, previous_by_feature, noisy_column_coverage(plates, root or DATA_ROOT)
    )
    figure = plot_ladder(payload["by_variant"], figures_dir() / FIGURE_NAME)
    payload["figure"] = figure.name
    if not payload["check_s0_reproduces_published_agreement"]:
        log.error("the ladder's S0 does not reproduce the published agreement")
    return payload
