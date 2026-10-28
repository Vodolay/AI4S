"""Settling or steady noise? Within-recording firing-rate trends of untreated wells (REAL data).

D32 (``hpsc_split_half``) found the firing rate of untreated human iPSC wells about twice
as noisy as rat between the two halves of one recording, and noted that human wells fall
more within the recording. This study, registered in ``preregistration/hpsc_settling.json``
(decision D35) before any window shorter than half a recording was computed, separates a
well's own straight-line trend over the recording from its scatter around that line:

* the D32 primary pool (the two 48-well untreated baselines), the wells D32 judged usable,
  read from D32's own results file;
* per well, the electrodes active over the whole recording (``features/mea.py``'s active
  rule at its defaults), the same set in each of 6 equal windows;
* y_k = log2 of the mean electrode rate in window k, floored at one spike over the set;
* per well, OLS of y_k on (k - 2.5): slope b and residual SD s (ddof 2); the split-half
  change d = mean(y_3..5) - mean(y_0..2) = 3b (trend part) + the residual part;
* T1 (spread of b), T2 (median s), T3 (share of var(d) carried by 3b) with a 2,000-resample
  within-species well bootstrap, seed 20261028; a quadratic-term assumption check.

Research decision support only: a measurement property of two public culture sets from one
lab, not a statement about any compound or about human biology.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.etl.kapucu import _load_spikes
from neuromps.features.mea import FeatureParams, _active_trains
from neuromps.models import hpsc_split_half as sh
from neuromps.models.toxcast_noise_floors import provenance
from neuromps.paths import REPO_ROOT, figures_dir, outputs_dir

log = logging.getLogger("neuromps.models")
REGISTRATION = REPO_ROOT / "preregistration" / "hpsc_settling.json"
ROOT = "hpsc_settling"  # the metrics.json section
RESULTS_NAME = "hpsc_settling_REAL.json"
FIGURE_NAME = "hpsc_settling_real.png"
N_WINDOWS = 6
SPECIES = sh.SPECIES
RESAMPLES = 2000
SEED = 20261028
MIN_WELLS = 10
T3_SHARE = 0.5
QUAD_DROP = 1 / 3
QUAD_MAJORITY = 0.5
X = np.arange(N_WINDOWS) - (N_WINDOWS - 1) / 2  # k - 2.5
HALF = np.r_[-np.ones(N_WINDOWS // 2), np.ones(N_WINDOWS // 2)] / (N_WINDOWS // 2)
PREDICTIONS = (
    "T1_human_settling_more_uneven",
    "T2_human_steady_noise_larger",
    "T3_settling_carries_the_human_spread",
)


# ------------------------------------------------------------------ per well


def electrode_set(channels: dict[str, np.ndarray], duration_s: float) -> list[str]:
    """Electrodes active over the whole recording under the extractor's default active rule."""
    params = FeatureParams()
    return sorted(
        name
        for name, times in channels.items()
        if _active_trains([np.asarray(times, dtype=float)], duration_s, params)
    )


def window_counts(
    channels: dict[str, np.ndarray], electrodes: list[str], duration_s: float
) -> np.ndarray:
    """Spikes per electrode (rows) and equal window (columns).

    A spike exactly on a window edge starts the next window; the recording's last
    spike (at ``duration_s``) ends the last one.
    """
    edges = duration_s / N_WINDOWS * np.arange(1, N_WINDOWS)
    out = np.zeros((len(electrodes), N_WINDOWS), dtype=int)
    for i, name in enumerate(electrodes):
        k = np.searchsorted(edges, np.asarray(channels[name], dtype=float), side="right")
        out[i] = np.bincount(k, minlength=N_WINDOWS)[:N_WINDOWS]
    return out


def log_rates(counts: np.ndarray, window_s: float) -> np.ndarray:
    """y_k = log2(max(mean electrode rate, 1 / (n_electrodes x window length)))."""
    n_electrodes = counts.shape[0]
    rate = counts.sum(axis=0) / n_electrodes / window_s
    return np.log2(np.maximum(rate, 1.0 / (n_electrodes * window_s)))


def trend(y: np.ndarray) -> tuple[float, float]:
    """OLS slope of y on (k - 2.5) and the residual SD (ddof 2)."""
    y = np.asarray(y, dtype=float)
    b = float(X @ (y - y.mean()) / (X @ X))
    resid = y - y.mean() - b * X
    return b, float(np.sqrt(resid @ resid / (N_WINDOWS - 2)))


def quadratic_sd(y: np.ndarray) -> float:
    """Residual SD of y on 1, (k - 2.5) and (k - 2.5)^2, with its own ddof (3)."""
    design = np.column_stack([np.ones(N_WINDOWS), X, X**2])
    coef, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)
    resid = np.asarray(y, dtype=float) - design @ coef
    return float(np.sqrt(resid @ resid / (N_WINDOWS - 3)))


def decompose(y: np.ndarray) -> dict:
    """Slope, residual SD, the split-half change and its trend and residual parts."""
    y = np.asarray(y, dtype=float)
    b, s = trend(y)
    d = float(HALF @ y)
    s_quad = quadratic_sd(y)
    s_quad_ddof2 = s_quad * np.sqrt((N_WINDOWS - 3) / (N_WINDOWS - 2))
    return {
        "b": b,
        "s": s,
        "d": d,
        "d_trend": 3 * b,
        "d_resid": d - 3 * b,
        "s_quad": s_quad,
        "quad_lowers": bool(s_quad < (1 - QUAD_DROP) * s),
        "quad_lowers_ddof2": bool(s_quad_ddof2 < (1 - QUAD_DROP) * s),
    }


def well_record(channels: dict[str, np.ndarray], duration_s: float) -> dict:
    """The electrode set, window rates and decomposition of one well."""
    electrodes = electrode_set(channels, duration_s)
    window_s = duration_s / N_WINDOWS
    row = {"n_electrodes": len(electrodes), "window_s": window_s, "defined": bool(electrodes)}
    if not electrodes:
        return row
    counts = window_counts(channels, electrodes, duration_s)
    y = log_rates(counts, window_s)
    row["rate_hz"] = [float(v) for v in counts.sum(axis=0) / len(electrodes) / window_s]
    row["y"] = [float(v) for v in y]
    row["defined"] = bool(np.all(np.isfinite(y)))
    return {**row, **decompose(y)}


# ------------------------------------------------------------------ inputs


def usable_units(path: Path) -> set[str]:
    """The primary-pool wells D32 judged usable, from its results file."""
    wells = json.loads(Path(path).read_text())["wells"]
    return {w["unit"] for w in wells if w["pool"] == "primary" and w["usable"]}


def load_usable_units() -> set[str]:
    """D32's usable wells: its results file of this run, else D32's own table rebuilt."""
    path = outputs_dir() / sh.RESULTS_NAME
    if path.exists():
        return usable_units(path)
    log.info("%s absent; rebuilding D32's usable-well table", path)
    table = sh.build_table(sh.primary_recordings())
    return set(table.loc[table["usable"].astype(bool), "unit"])


def recording_records(rec: sh.Recording, units: set[str]) -> list[dict]:
    """Every usable well of one recording, in well order."""
    wells, duration = _load_spikes(rec.spikes)
    rows = []
    for well in sorted(wells):
        unit = f"{rec.plate}:{well}"
        if unit in units:
            meta = {"species": rec.species, "plate": rec.plate, "well": well, "unit": unit}
            rows.append({**meta, "duration_s": duration, **well_record(wells[well], duration)})
    return rows


def build_table(recordings: list[sh.Recording], units: set[str]) -> pd.DataFrame:
    """One row per usable well of the given recordings."""
    return pd.DataFrame([row for rec in recordings for row in recording_records(rec, units)])


# ------------------------------------------------------------------ statistics


def white_noise_share() -> float:
    """var(d_trend) / var(d) when the six window values are independent with equal variance."""
    return float((3 / (X @ X)) ** 2 * (X @ X) / (HALF @ HALF))


def noise_slope_var(s: np.ndarray) -> float:
    """Variance of a fitted slope expected from the scatter around each well's line alone."""
    s = np.asarray(s, dtype=float)
    return float(np.mean(s**2) / (X @ X))


def _share(d_trend: np.ndarray, d: np.ndarray) -> float:
    return float(np.var(d_trend, ddof=1) / np.var(d, ddof=1))


def _values(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows = table[table["defined"].astype(bool)]
    return {s: rows[rows["species"] == s] for s in SPECIES}


def species_summary(rows: pd.DataFrame) -> dict:
    """Slope, residual SD and variance parts of the split-half change for one species."""
    r4 = sh._r4
    var = {k: float(np.var(rows[k], ddof=1)) for k in ("d", "d_trend", "d_resid")}
    cov = float(np.cov(rows["d_trend"], rows["d_resid"], ddof=1)[0, 1])
    return {
        "n_wells": len(rows),
        "median_electrodes": (
            r4(float(rows["n_electrodes"].median())) if "n_electrodes" in rows else None
        ),
        "mean_b": r4(float(rows["b"].mean())),
        "median_b": r4(float(rows["b"].median())),
        "sd_b": r4(sh.sd(rows["b"])),
        "median_s": r4(float(rows["s"].median())),
        "sd_d": r4(sh.sd(rows["d"])),
        "sd_d_resid": r4(sh.sd(rows["d_resid"])),
        "var_d": r4(var["d"]),
        "var_d_trend": r4(var["d_trend"]),
        "var_d_resid": r4(var["d_resid"]),
        "cov_trend_resid": r4(cov),
        "share_trend": r4(var["d_trend"] / var["d"]),
    }


def _interval(draws: np.ndarray) -> list[float | None]:
    finite = draws[np.isfinite(draws)]
    low, high = np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan)
    return [sh._r4(low), sh._r4(high)]


_COLUMNS = ("b", "s", "d", "d_trend")


def _arrays(rows: pd.DataFrame) -> np.ndarray:
    """Columns b, s, d, d_trend of one species as a (wells x 4) array."""
    return rows[list(_COLUMNS)].to_numpy(dtype=float)


def _statistics(h: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    """SD(b) ratio, median(s) ratio (human / rat) and the human trend share."""
    sd_r, med_r = sh.sd(r[:, 0]), float(np.median(r[:, 1]))
    return (
        sh.sd(h[:, 0]) / sd_r if sd_r > 0 else np.nan,
        float(np.median(h[:, 1])) / med_r if med_r > 0 else np.nan,
        _share(h[:, 3], h[:, 2]),
    )


def bootstrap(table: pd.DataFrame, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """SD(b) and median(s) ratios (human / rat) and the human trend share, with 95% intervals.

    One generator; each resample draws human wells, then rat wells, with replacement, as
    the D32 interval does. The share's interval is descriptive (T3 is judged on its point).
    """
    by = _values(table)
    h, r = _arrays(by["human"]), _arrays(by["rat"])
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, 3))
    for i in range(resamples):
        hi = rng.integers(0, len(h), len(h))
        ri = rng.integers(0, len(r), len(r))
        draws[i] = _statistics(h[hi], r[ri])
    point = _statistics(h, r)
    out: dict = {"resamples": int(resamples), "seed": int(seed)}
    for j, name in enumerate(("sd_b_ratio", "median_s_ratio", "share_human")):
        out[name] = {
            "point": sh._r4(point[j]),
            "ci95": _interval(draws[:, j]),
            "nonfinite": int(np.sum(~np.isfinite(draws[:, j]))),
        }
    return out


# ------------------------------------------------------------------ predictions


def _above_one(got: dict) -> bool:
    low = got["ci95"][0]
    return bool(low is not None and low > 1)


def predict(summary: dict, boot: dict | None, judgeable: bool, curved: bool) -> dict:
    """T1-T3 with their registered rules; every one uninterpretable below 10 wells a species."""
    t3_pass = bool(judgeable and summary["human"]["share_trend"] >= T3_SHARE)
    t1_pass = bool(judgeable and _above_one(boot["sd_b_ratio"]))
    t2_pass = bool(judgeable and _above_one(boot["median_s_ratio"]))
    wells = {s: summary[s]["n_wells"] for s in SPECIES}
    t1 = {**_ratio("sd_b", summary, boot, "sd_b_ratio"), "wells": wells}
    t2 = {**_ratio("median_s", summary, boot, "median_s_ratio"), "wells": wells}
    t3 = {
        "share_human": summary["human"]["share_trend"],
        "share_rat": summary["rat"]["share_trend"],
        "threshold": T3_SHARE,
        "white_noise_reference": sh._r4(white_noise_share()),
        "share_human_ci95_descriptive": boot["share_human"]["ci95"] if boot else None,
        "wells": wells,
    }
    t1["read_with_nonlinearity_caveat"] = t3["read_with_nonlinearity_caveat"] = curved
    out = {}
    for name, rec, passed in zip(
        PREDICTIONS, (t1, t2, t3), (t1_pass, t2_pass, t3_pass), strict=True
    ):
        out[name] = {**rec, "pass": passed, "outcome": sh.outcome(passed, judgeable)}
    return out


def _ratio(stat: str, summary: dict, boot: dict | None, key: str) -> dict:
    got = boot[key] if boot else {"point": None, "ci95": None}
    return {
        f"{stat}_human": summary["human"][stat],
        f"{stat}_rat": summary["rat"][stat],
        "ratio": got["point"],
        "ci95": got["ci95"],
    }


# ------------------------------------------------------------------ reported, not tested


def wells_needed_record(summary: dict) -> dict:
    """D32's wells-needed formula on SD(d_resid), and on SD(d) of this rate, per species."""
    out = {
        "note": (
            "ILLUSTRATIVE, and a LOWER BOUND as in D32: what would remain if each well's own "
            "trend were removed (for example by a longer settling period before dosing). "
            "Removing a fitted line also removes the part of steady noise the line absorbs: "
            f"with no trend at all the residual part keeps {1 - white_noise_share():.4f} of "
            "the variance of d."
        ),
        "false_alarm_rate": 0.05,
        "power": 0.8,
    }
    for s in SPECIES:
        resid, total = summary[s]["sd_d_resid"], summary[s]["sd_d"]
        out[s] = {
            "sd_d_resid": resid,
            "sd_d": total,
            **{k: sh.wells_needed(resid, v) for k, v in sh.FALLS.items()},
            "from_sd_d": {k: sh.wells_needed(total, v) for k, v in sh.FALLS.items()},
        }
    return out


def per_window(by: dict[str, pd.DataFrame]) -> dict:
    """The median firing rate (Hz) per window and species: the settling curve."""
    out = {"window_minutes": None}
    for s in SPECIES:
        rates = np.array(by[s]["rate_hz"].tolist(), dtype=float)
        out[s] = [sh._r4(v) for v in np.median(rates, axis=0)]
    if len(by["human"]):
        out["window_minutes"] = sh._r4(float(by["human"]["window_s"].iloc[0]) / 60)
    return out


def assumption_check(by: dict[str, pd.DataFrame]) -> dict:
    """Share of wells in which a quadratic term lowers the residual SD by more than a third."""
    share = {s: sh._r4(float(by[s]["quad_lowers"].mean())) for s in SPECIES}
    share2 = {s: sh._r4(float(by[s]["quad_lowers_ddof2"].mean())) for s in SPECIES}
    linear = all(share[s] <= QUAD_MAJORITY for s in SPECIES)
    return {
        "share_lowered": share,
        "threshold_drop": sh._r4(QUAD_DROP),
        "threshold_share": QUAD_MAJORITY,
        "linear": bool(linear),
        "verdict": "linear" if linear else "not linear: T1 and T3 read with that caveat",
        "share_lowered_both_ddof2_sensitivity": share2,
    }


def slope_noise_check(by: dict[str, pd.DataFrame]) -> dict:
    """Post hoc, descriptive: how much of SD(b) the window scatter alone would produce."""
    out = {"note": "post hoc, not registered: var(b) expected from s alone is mean(s^2)/17.5"}
    for s in SPECIES:
        noise = noise_slope_var(by[s]["s"])
        total = float(np.var(by[s]["b"], ddof=1))
        out[s] = {
            "var_b": sh._r4(total),
            "var_b_from_scatter": sh._r4(noise),
            "share_from_scatter": sh._r4(noise / total) if total > 0 else None,
            "sd_b_beyond_scatter": sh._r4(float(np.sqrt(max(total - noise, 0.0)))),
        }
    h, r = out["human"]["sd_b_beyond_scatter"], out["rat"]["sd_b_beyond_scatter"]
    out["ratio_beyond_scatter"] = sh._r4(h / r) if r else None
    return out


# ------------------------------------------------------------------ assembly


def analyse(table: pd.DataFrame, resamples: int = RESAMPLES) -> dict:
    """Every registered statistic (and the post hoc slope check) from a table of wells."""
    by = _values(table)
    judgeable = all(len(by[s]) >= MIN_WELLS for s in SPECIES)
    enough = all(len(by[s]) >= 2 for s in SPECIES)
    summary = {s: species_summary(by[s]) if len(by[s]) >= 2 else None for s in SPECIES}
    boot = bootstrap(table, resamples) if enough else None
    check = assumption_check(by)
    preds = (
        predict(summary, boot, judgeable, not check["linear"])
        if enough
        else {p: {"pass": False, "outcome": "uninterpretable"} for p in PREDICTIONS}
    )
    return {
        "label": "REAL Kapucu 2022 untreated wells: settling vs steady noise, human iPSC vs rat",
        "registration": provenance(REGISTRATION),
        "resamples": resamples,
        "seed": SEED,
        "n_windows": N_WINDOWS,
        "wells_defined": {s: len(by[s]) for s in SPECIES},
        "wells_in_table": {s: int((table["species"] == s).sum()) for s in SPECIES},
        "species": summary,
        "bootstrap": boot,
        "predictions": preds,
        "wells_needed_residual_only": wells_needed_record(summary) if enough else None,
        "per_window": per_window(by) if "rate_hz" in table else None,
        "assumption_check": check,
        "slope_noise_check_post_hoc": slope_noise_check(by) if enough else None,
        "white_noise_share": sh._r4(white_noise_share()),
        "implementation_notes": IMPLEMENTATION_NOTES,
    }


IMPLEMENTATION_NOTES = [
    "Usable wells are read from D32's results file of the same run (hpsc_split_half_REAL.json, "
    "primary pool, usable true); if it is absent, D32's own table is rebuilt.",
    "Recording duration and window length follow D32: the plate file's last spike time, cut "
    "into 6 equal windows; a spike on a window edge starts the next window and the last spike "
    "ends window 5.",
    "The electrode set is the extractor's active rule (FeatureParams defaults, >= 5 spikes/min "
    "over the whole recording) applied to each electrode; noisy electrodes are not removed "
    "(D32 did not remove them either).",
    "A rate is defined when the set has at least one electrode; the floor makes an empty window "
    "defined, so 'defined in all six windows' means a non-empty set.",
    "SD(b) and the variances in the T3 share are sample statistics (ddof 1); the median of s is "
    "numpy's median.",
    "T1 and T2 use one bootstrap: one generator (seed 20261028), each resample drawing human "
    "then rat wells, the same draw order as D32's interval; the ratios are human / rat; a "
    "resample with a zero rat denominator would be dropped and counted as nonfinite.",
    "'Lies above 1' is read strictly: the interval's lower end must exceed 1.",
    "The quadratic check compares the quadratic fit's residual SD with its own ddof (3) against "
    "the line's (ddof 2), and 'more than a third' strictly (s_quad < 2/3 s); the share with both "
    "at ddof 2 is reported as a sensitivity.",
    "The wells-needed figures use D32's formula (5% two-sided, 80% power, 20% and 50% falls) on "
    "the SD of d_resid; the same formula on the SD of d of this windowed rate is given beside it.",
    "Not registered, reported for reading T3: with independent, equal-variance window values and "
    "no trend, var(d_trend)/var(d) = 27/35 (about 0.77), above T3's 0.5 threshold.",
]


def main(argv: list[str] | None = None) -> int:
    """Build the well table, compute every registered statistic, write results and figure."""
    from neuromps.metrics import update_metrics

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="same data (the quick tier holds both)")
    p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    recordings = sh.primary_recordings()
    if len(recordings) < len(sh.PRIMARY):
        log.warning("Kapucu baseline plates not present under %s; skipping", sh.DATA_ROOT)
        return 0
    table = build_table(recordings, load_usable_units())
    payload = analyse(table)
    figures_dir().mkdir(parents=True, exist_ok=True)
    plot(payload, table, figures_dir() / FIGURE_NAME)
    payload["results_file"] = f"outputs/{RESULTS_NAME}"
    payload["figure"] = f"outputs/figures/{FIGURE_NAME}"
    results = {**payload, "wells": json.loads(table.to_json(orient="records"))}
    (outputs_dir() / RESULTS_NAME).write_text(json.dumps(results, indent=1) + "\n")
    update_metrics(ROOT, payload)
    print(json.dumps(payload["predictions"], indent=1))
    return 0


# ------------------------------------------------------------------ figure

COLOURS = sh.COLOURS
LABELS = {"human": "human iPSC (DIV29)", "rat": "rat (DIV22)"}


def _curve_panel(ax, rows: pd.DataFrame, median: list, species: str) -> None:
    for y in rows["y"]:
        ax.plot(np.arange(N_WINDOWS), y, color=COLOURS[species], lw=0.6, alpha=0.25)
    ax.plot(
        np.arange(N_WINDOWS),
        np.log2(median),
        "o-",
        color="#000000",
        lw=1.6,
        ms=3,
        label="median rate per window",
    )
    ax.set_xticks(np.arange(N_WINDOWS))
    ax.set_xlabel("5-min window", fontsize=7)
    ax.set_ylabel("log2 firing rate (Hz), fixed electrode set", fontsize=7)
    ax.set_title(f"{LABELS[species]}: settling curve, {len(rows)} wells", fontsize=7.5)
    ax.legend(fontsize=6, frameon=False)


def _slope_panel(ax, by: dict[str, pd.DataFrame]) -> None:
    rng = np.random.default_rng(0)
    for i, s in enumerate(SPECIES):
        b = by[s]["b"].to_numpy(dtype=float)
        ax.plot(i + rng.uniform(-0.15, 0.15, b.size), b, "o", ms=3, color=COLOURS[s], alpha=0.8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks([0, 1], [LABELS[s].replace(" (", "\n(") for s in SPECIES], fontsize=7)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("within-recording slope b (log2 per window)", fontsize=7)
    ax.set_title("slope per well", fontsize=7.5)


def _decomposition_panel(ax, by: dict[str, pd.DataFrame]) -> None:
    for s in SPECIES:
        ax.plot(by[s]["d_trend"], by[s]["d"], "o", ms=3, color=COLOURS[s], label=LABELS[s])
    lim = max(float(np.abs(by[s][["d", "d_trend"]]).to_numpy().max()) for s in SPECIES) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.6, ls="--", label="d = 3b")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("trend part d_trend = 3b (log2)", fontsize=7)
    ax.set_ylabel("split-half change d (log2)", fontsize=7)
    ax.set_title("split-half change against its trend part", fontsize=7.5)
    ax.legend(fontsize=6, frameon=False)


def plot(payload: dict, table: pd.DataFrame, path: Path) -> str:
    """Settling curves per species, the slope per well, and d against its trend part."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = _values(table)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    for ax, s in zip(axes[:2], SPECIES, strict=True):
        _curve_panel(ax, by[s], payload["per_window"][s], s)
    _slope_panel(axes[2], by)
    _decomposition_panel(axes[3], by)
    for ax in axes:
        ax.tick_params(labelsize=7)
    title = (
        "REAL data: Kapucu 2022 untreated wells, one lab: settling vs steady noise of human "
        "iPSC and rat cortical MEA firing rate over 6 windows (registered, D35)"
    )
    fig.suptitle(title, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return title


if __name__ == "__main__":
    raise SystemExit(main())
