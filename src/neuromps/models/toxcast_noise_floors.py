"""How large a change can an MEA chip see? Noise floors from real ToxCast vehicle wells.

The EPA's acute MEA screen (CCTE_Shafer_MEA_acute, rat cortical cultures on
48-well Axion plates, invitrodb v4.3, CC0) records every well for 40 minutes
before and 40 minutes after exposure, and stores each well's percent change
from its own baseline (``rval``). Its vehicle wells (DMSO or water) are therefore
a direct measurement of how much an untreated well's readout moves on its own,
for each of the 15 electrophysiology components, across about a hundred
exposure dates.

This module turns them into what a lab designing a chip assay needs:

* per component, the change an untreated well exceeds 5% of the time (``abs_q95``);
* the share of the variance carried by the exposure date (a one-way
  random-effects decomposition, method of moments), because wells from one date
  share it and averaging more of them does not remove it;
* the change detectable at 5% false alarms with 80% power as a function of the
  number of wells, for three designs: independent wells, all wells from one
  date against a historical reference, and all wells from one date against the
  same number of vehicle wells from that date; and from that, the wells needed
  for a 20% or 50% change.

Design registered in ``preregistration/toxcast_noise_floors.json`` before any
response statistic was computed. Research decision support only: these are the
measurement properties of one public screen, not a statement about any compound.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.paths import INTERIM_ROOT, REPO_ROOT, figures_dir, outputs_dir
from neuromps.qc.provenance import sha256_file

log = logging.getLogger("neuromps.models")
WELLS = INTERIM_ROOT / "toxcast_mea" / "ephys_wells_recovered.parquet"
COMPONENTS = INTERIM_ROOT / "toxcast_mea" / "components.parquet"
REGISTRATION = REPO_ROOT / "preregistration" / "toxcast_noise_floors.json"
PREFIX = "CCTE_Shafer_MEA_acute_"
EPHYS_ACIDS = tuple(range(2447, 2462))
HEADLINE = "firing_rate_mean"
HEADLINE_ACID = 2448
ROOT = "toxcast_noise_floors"  # the metrics.json section
FLOOR = 0.05  # the asset's floor for an abolished readout (models.dataset.FLOOR)
N_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96)
DESIGNS = ("independent", "one_date", "same_date_vehicle")
TARGETS = {
    "down_20pct": ("detectable_down", float(np.log2(0.8))),
    "down_50pct": ("detectable_down", float(np.log2(0.5))),
    "up_20pct": ("detectable_up", float(np.log2(1.2))),
    "up_50pct": ("detectable_up", float(np.log2(1.5))),
}
DRAWS = 20_000
SHARE_RESAMPLES = 2_000
SHARE_SEED = 20261026
DETECTION_SEED = 20261027
CHECK_MAX_N = 6


def log2_change(rval: np.ndarray) -> np.ndarray:
    """log2(1 + rval/100), floored at log2(0.05); NaN stays NaN."""
    ratio = 1.0 + np.asarray(rval, dtype=float) / 100.0
    with np.errstate(invalid="ignore"):
        return np.log2(np.where(np.isnan(ratio), np.nan, np.maximum(ratio, FLOOR)))


def variance_components(values: np.ndarray, groups: np.ndarray) -> dict:
    """One-way random-effects decomposition by method of moments.

    sigma_b^2 = (MSB - MSW) / n0, truncated at zero; ``share`` is the between-group
    share of the total variance (the intraclass correlation).
    """
    values = np.asarray(values, dtype=float)
    _, codes = np.unique(np.asarray(groups), return_inverse=True)
    sizes = np.bincount(codes)
    k, total = sizes.size, values.size
    means = np.bincount(codes, weights=values) / sizes
    grand = values.mean()
    msb = float((sizes * (means - grand) ** 2).sum() / (k - 1))
    msw = float(((values - means[codes]) ** 2).sum() / (total - k))
    n0 = (total - (sizes**2).sum() / total) / (k - 1)
    sigma_b2 = max(0.0, (msb - msw) / n0)
    share = sigma_b2 / (sigma_b2 + msw) if sigma_b2 + msw > 0 else 0.0
    return {"sigma_b2": sigma_b2, "sigma_w2": msw, "share": share, "groups": int(k), "n0": n0}


def share_interval(values: np.ndarray, groups: np.ndarray, n_resamples: int, seed: int) -> list:
    """95% percentile interval of the between-group share, resampling whole groups."""
    rng = np.random.default_rng(seed)
    labels, codes = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(codes == g) for g in range(labels.size)]
    shares = []
    for _ in range(n_resamples):
        drawn = rng.integers(0, labels.size, labels.size)
        idx = np.concatenate([members[g] for g in drawn])
        new_groups = np.repeat(np.arange(drawn.size), [members[g].size for g in drawn])
        shares.append(variance_components(values[idx], new_groups)["share"])
    low, high = np.percentile(shares, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def null_model(values: np.ndarray, groups: np.ndarray) -> dict:
    """Shrunken date effects and rescaled within-date residuals for resampling.

    Raw date means carry within-date noise (sigma_w^2 / n_j); each is shrunk by
    lambda_j so its spread matches the estimated sigma_b^2. Residuals from dates
    with two or more wells are rescaled by sqrt(n_j / (n_j - 1)) to undo the
    shrinkage that subtracting the date's own mean causes.
    """
    values = np.asarray(values, dtype=float)
    comp = variance_components(values, groups)
    _, codes = np.unique(np.asarray(groups), return_inverse=True)
    sizes = np.bincount(codes)
    means = np.bincount(codes, weights=values) / sizes
    centre = float(values.mean())
    sb2, sw2 = comp["sigma_b2"], comp["sigma_w2"]
    lam = np.sqrt(sb2 / (sb2 + sw2 / sizes)) if sb2 > 0 else np.zeros(sizes.size)
    keep = sizes[codes] >= 2
    scale = np.sqrt(sizes[codes] / np.maximum(sizes[codes] - 1, 1))
    residuals = ((values - means[codes]) * scale)[keep]
    return {
        "centre": centre,
        "effects": lam * (means - centre),
        "weights": sizes / sizes.sum(),
        "residuals": residuals,
    }


def null_draws(model: dict, n: int, design: str, draws: int, rng) -> np.ndarray:
    """The test statistic under no effect: mean of n wells minus the reference."""
    res, eff = model["residuals"], model["effects"]

    def residual_means() -> np.ndarray:
        return rng.choice(res, size=(draws, n)).mean(axis=1)

    if design == "independent":
        dates = rng.choice(eff.size, size=(draws, n), p=model["weights"])
        return eff[dates].mean(axis=1) + residual_means()
    if design == "one_date":
        return eff[rng.choice(eff.size, size=draws, p=model["weights"])] + residual_means()
    if design == "same_date_vehicle":
        return residual_means() - residual_means()
    raise ValueError(f"unknown design {design!r}")


def detection(null: np.ndarray) -> dict:
    """Two-sided 5% threshold, and the shifts detected at it with 80% power."""
    threshold = float(np.quantile(np.abs(null), 0.95))
    return {
        "threshold": threshold,
        "detectable_down": threshold + float(np.quantile(null, 0.8)),
        "detectable_up": threshold - float(np.quantile(null, 0.2)),
    }


def wells_needed(curve: pd.DataFrame, target_log2: float, column: str) -> int | None:
    """Smallest grid n whose detectable change is within the target, else None."""
    ok = curve[curve[column] <= abs(target_log2)]
    return int(ok["n"].min()) if len(ok) else None


def resampled_independent(values: np.ndarray, n: int, draws: int, rng) -> np.ndarray:
    """Model-free: means of n real wells drawn with replacement from all wells."""
    return rng.choice(np.asarray(values, dtype=float), size=(draws, n)).mean(axis=1)


def resampled_one_date(values: np.ndarray, groups: np.ndarray, n: int, draws: int, rng):
    """Model-free: means of n real wells drawn without replacement from one date.

    Only dates with at least n wells can supply them; they are drawn in
    proportion to their wells, as in the model.
    """
    values = np.asarray(values, dtype=float)
    labels, codes = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(codes == g) for g in range(labels.size)]
    eligible = [m for m in members if m.size >= n]
    weights = np.array([m.size for m in eligible], dtype=float)
    picks = rng.choice(len(eligible), size=draws, p=weights / weights.sum())
    return np.array([values[rng.choice(eligible[p], n, replace=False)].mean() for p in picks])


def provenance(path: Path) -> dict:
    """Hash of a registration file and the first commit that added it."""
    try:
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H %cI", "--", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        out = []
    return {
        "file": str(Path(path).relative_to(REPO_ROOT)),
        "sha256": sha256_file(Path(path)),
        "registered_in_commit": out[-2] if len(out) >= 2 else None,
        "committed_at": out[-1] if len(out) >= 2 else None,
    }


def component_names(path: Path = COMPONENTS) -> dict[int, str]:
    """acid -> short component name (without the assay prefix)."""
    table = pd.read_parquet(path)
    return {
        int(a): str(n).removeprefix(PREFIX)
        for a, n in zip(table["acid"], table["assay_component_name"], strict=True)
    }


def vehicle_wells(wells: pd.DataFrame, acid: int, dmso_only: bool = False) -> pd.DataFrame:
    """Vehicle wells of one component with their log2 change; NaN rval dropped."""
    rows = wells[(wells["acid"] == acid) & (wells["wllt"] == "n")]
    if dmso_only:
        rows = rows[rows["spid"] == "DMSO"]
    rows = rows.assign(x=log2_change(rows["rval"].to_numpy()))
    return rows[np.isfinite(rows["x"])]


def describe(rows: pd.DataFrame, n_missing: int) -> dict:
    """Counts, location, spread and the 95th percentile of |change|."""
    x = rows["x"].to_numpy()
    q95 = float(np.quantile(np.abs(x), 0.95))
    median, q05, q95_signed = (float(np.quantile(x, q)) for q in (0.5, 0.05, 0.95))
    return {
        "wells": int(len(rows)),
        "dates": int(rows["apid"].nunique()),
        "plates": int(rows["plate"].nunique()),
        "rval_missing": int(n_missing),
        "at_floor": int((x <= np.log2(FLOOR) + 1e-12).sum()),
        "median_log2": round(median, 4),
        "mean_log2": round(float(x.mean()), 4),
        "sd_log2": round(float(x.std(ddof=1)), 4),
        "q05_log2": round(q05, 4),
        "q95_log2": round(q95_signed, 4),
        "abs_q95_log2": round(q95, 4),
        # signed fractions (2^x - 1), so the report's "%" displays are checked
        # against value x 100
        "median_fraction": round(2**median - 1, 4),
        "q05_fraction": round(2**q05 - 1, 4),
        "q95_fraction": round(2**q95_signed - 1, 4),
    }


def shares(rows: pd.DataFrame, n_resamples: int, seed: int) -> dict:
    """Date share with its interval, and the plate share for context."""
    x = rows["x"].to_numpy()
    by_date = variance_components(x, rows["apid"].to_numpy())
    by_plate = variance_components(x, rows["plate"].to_numpy())
    return {
        "date_share": round(by_date["share"], 4),
        "date_share_ci95": share_interval(x, rows["apid"].to_numpy(), n_resamples, seed),
        "sigma_date_log2": round(float(np.sqrt(by_date["sigma_b2"])), 4),
        "sigma_within_date_log2": round(float(np.sqrt(by_date["sigma_w2"])), 4),
        "plate_share": round(by_plate["share"], 4),
    }


def curves(rows: pd.DataFrame, draws: int, seed: int) -> dict[str, pd.DataFrame]:
    """Threshold and detectable changes against n, per design."""
    model = null_model(rows["x"].to_numpy(), rows["apid"].to_numpy())
    rng = np.random.default_rng(seed)
    out = {}
    for design in DESIGNS:
        out[design] = pd.DataFrame(
            [{"n": n, **detection(null_draws(model, n, design, draws, rng))} for n in N_GRID]
        )
    return out


def as_fraction(log2_value: float, column: str) -> float:
    """A log2 detectable change as a fraction: a decrease for 'down', a rise for 'up'."""
    return 1 - 2**-log2_value if column == "detectable_down" else 2**log2_value - 1


def curve_record(curve: pd.DataFrame) -> dict:
    """Per-n thresholds and detectable changes, in log2 and as fractions."""
    out = {}
    for row in curve.itertuples(index=False):
        down, up = row.detectable_down, row.detectable_up
        out[str(row.n)] = {
            "threshold_log2": round(row.threshold, 4),
            "detectable_down_log2": round(down, 4),
            "detectable_up_log2": round(up, 4),
            "detectable_down_fraction": round(as_fraction(down, "detectable_down"), 4),
            "detectable_up_fraction": round(as_fraction(up, "detectable_up"), 4),
        }
    return out


def needed_record(by_design: dict[str, pd.DataFrame]) -> dict:
    """Wells needed per design and target; None where no grid n reaches it."""
    return {
        design: {name: wells_needed(curve, t, col) for name, (col, t) in TARGETS.items()}
        for design, curve in by_design.items()
    }


def model_checks(rows: pd.DataFrame, by_design: dict[str, pd.DataFrame], seed: int) -> dict:
    """Largest relative difference in threshold between the model and plain resampling."""
    x, dates = rows["x"].to_numpy(), rows["apid"].to_numpy()
    centre = float(x.mean())
    rng = np.random.default_rng(seed)
    out = {}
    for design, max_n in (("independent", max(N_GRID)), ("one_date", CHECK_MAX_N)):
        diffs = {}
        for n in [n for n in N_GRID if n <= max_n]:
            if design == "independent":
                null = resampled_independent(x, n, DRAWS, rng) - centre
            else:
                null = resampled_one_date(x, dates, n, DRAWS // 4, rng) - centre
            model = float(by_design[design].set_index("n").loc[n, "threshold"])
            diffs[str(n)] = round(detection(null)["threshold"] / model - 1, 4)
        worst = max(diffs.values(), key=abs)
        out[design] = {"relative_difference_by_n": diffs, "largest_relative_difference": worst}
    return out


def analyse_component(wells: pd.DataFrame, acid: int, quick: bool) -> tuple[dict, dict]:
    """Every registered statistic for one component; also returns the curves."""
    all_rows = wells[(wells["acid"] == acid) & (wells["wllt"] == "n")]
    rows = vehicle_wells(wells, acid)
    draws = DRAWS // 10 if quick else DRAWS
    resamples = SHARE_RESAMPLES // 10 if quick else SHARE_RESAMPLES
    by_design = curves(rows, draws, DETECTION_SEED + acid)
    record = {
        "acid": acid,
        **describe(rows, len(all_rows) - len(rows)),
        **shares(rows, resamples, SHARE_SEED + acid),
        "curves": {d: curve_record(c) for d, c in by_design.items()},
        "wells_needed": needed_record(by_design),
    }
    dmso = vehicle_wells(wells, acid, dmso_only=True)
    record["dmso_only"] = {
        "wells": int(len(dmso)),
        "abs_q95_log2": describe(dmso, 0)["abs_q95_log2"],
        "date_share": round(variance_components(dmso["x"], dmso["apid"])["share"], 4),
    }
    if record["acid"] == HEADLINE_ACID:
        dmso_curves = curves(dmso, draws, DETECTION_SEED + acid)
        record["dmso_only"]["wells_needed"] = needed_record(dmso_curves)
    record["model_checks"] = model_checks(rows, by_design, DETECTION_SEED + 1000 + acid)
    return record, by_design


COLOURS = {"independent": "#0072B2", "one_date": "#D55E00", "same_date_vehicle": "#009E73"}
LABELS = {
    "independent": "independent wells",
    "one_date": "all wells from one date",
    "same_date_vehicle": "one date vs n same-date vehicle wells",
}


def plot(components: dict, headline_curves: dict, path: Path) -> None:
    """abs_q95 and date share per component; detectable firing-rate change against n."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(components, key=lambda k: components[k]["abs_q95_log2"])
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.2, 1, 1.3]})
    low = np.array([components[k]["q05_log2"] for k in names])
    high = np.array([components[k]["q95_log2"] for k in names])
    axes[0].barh(y, high - low, left=low, color="#56B4E9", height=0.6)
    axes[0].plot([components[k]["median_log2"] for k in names], y, "o", ms=3.5, color="#000000")
    axes[0].axvline(0, color="grey", lw=0.6)
    ticks = np.log2([0.25, 0.5, 1, 2, 4])
    axes[0].set_xticks(ticks, ["-75%", "-50%", "0", "+100%", "+300%"])
    axes[0].set_yticks(y, [k.replace("_", " ") for k in names], fontsize=7)
    axes[0].set_xlabel(
        "untreated well's change from its own baseline\n(bar: 5th-95th percentile; dot: median)",
        fontsize=7,
    )
    share = [components[k]["date_share"] for k in names]
    ci = np.array([components[k]["date_share_ci95"] for k in names])
    axes[1].barh(y, share, color="#E69F00")
    axes[1].errorbar(
        share,
        y,
        xerr=[np.subtract(share, ci[:, 0]), ci[:, 1] - share],
        fmt="none",
        ecolor="black",
        lw=0.8,
    )
    axes[1].set_yticks(y, [])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("share of vehicle variance carried by exposure date", fontsize=7)
    for design, curve in headline_curves.items():
        frac = [100 * as_fraction(v, "detectable_down") for v in curve["detectable_down"]]
        axes[2].plot(curve["n"], frac, "o-", ms=3, color=COLOURS[design], label=LABELS[design])
    for level in (20, 50):
        axes[2].axhline(level, color="grey", lw=0.6, ls="--")
    axes[2].set_xscale("log")
    axes[2].set_xticks(list(N_GRID[::2]) + [96], [str(n) for n in list(N_GRID[::2]) + [96]])
    axes[2].set_xlabel("wells per condition", fontsize=7)
    axes[2].set_ylabel(
        "firing-rate decrease detectable (%)\n5% false alarms, 80% power", fontsize=7
    )
    axes[2].legend(fontsize=6.5, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.suptitle(
        "REAL data: ToxCast acute MEA vehicle wells (rat cortex, invitrodb v4.3): "
        "noise floor per feature, date share, and wells needed (firing rate)",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _cell(value, key: str) -> str:
    shown = ">96" if value is None else str(value)
    pointer = "n_max" if value is None else key
    return f"{shown} <!-- metrics.json: toxcast_noise_floors.{pointer} -->"


def _pct(c: dict, base: str, key: str) -> str:
    shown = f"{100 * c[key]:+.0f}%".replace("-", "−")
    return f"{shown} <!-- metrics.json: toxcast_noise_floors.{base}.{key} -->"


def markdown_table(components: dict) -> str:
    """The report's wells-needed table, each number carrying its metrics.json key."""
    head = (
        "| readout | untreated change, median (5th to 95th pct.) | abs. q95 (log2) | date share "
        "| wells for a 20% fall (A / B / C) | wells for a 50% fall (A / B / C) |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = []
    for name, c in sorted(components.items(), key=lambda kv: kv[1]["abs_q95_log2"]):
        base = f"components.{name}"
        spread = (
            f"{_pct(c, base, 'median_fraction')} ({_pct(c, base, 'q05_fraction')} to "
            f"{_pct(c, base, 'q95_fraction')})"
        )
        cells = [
            name.replace("_", " "),
            spread,
            f"{c['abs_q95_log2']:.2f} <!-- metrics.json: {ROOT}.{base}.abs_q95_log2 -->",
            f"{c['date_share']:.2f} <!-- metrics.json: {ROOT}.{base}.date_share -->",
        ]
        for target in ("down_20pct", "down_50pct"):
            cells.append(
                " / ".join(
                    _cell(c["wells_needed"][d][target], f"{base}.wells_needed.{d}.{target}")
                    for d in DESIGNS
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Compute every registered statistic, the figure and the report table."""
    from neuromps.metrics import update_metrics

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="fewer draws and resamples")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not WELLS.exists():
        log.warning(
            "%s not built; run fetch.toxcast_mysql, etl.toxcast_wells, "
            "etl.toxcast_recover (outside the fetch tiers)",
            WELLS,
        )
        return 0
    wells = pd.read_parquet(WELLS, columns=["acid", "wllt", "spid", "apid", "plate", "rval"])
    names = component_names()
    components, headline = {}, {}
    for acid in EPHYS_ACIDS:
        record, by_design = analyse_component(wells, acid, args.quick)
        components[names[acid]] = record
        if acid == HEADLINE_ACID:
            headline = by_design
        log.info(
            "%s: abs_q95 %.3f log2, date share %.3f",
            names[acid],
            record["abs_q95_log2"],
            record["date_share"],
        )
    figures_dir().mkdir(parents=True, exist_ok=True)
    plot(components, headline, figures_dir() / "toxcast_noise_floors_real.png")
    tables = outputs_dir() / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "toxcast_wells_needed.md").write_text(markdown_table(components))
    results = {
        "label": "REAL ToxCast acute MEA vehicle wells: noise floors and wells needed",
        "registration": provenance(REGISTRATION),
        "source": "invitrodb v4.3 MySQL dump (CC0), level-0 wells recovered (DECISION_LOG D29)",
        "n_grid": list(N_GRID),
        "n_max": max(N_GRID),
        "false_alarm_rate": 0.05,  # two-sided, per test: the 95th percentile of |null|
        "power": 0.8,  # the q80 / q20 terms of `detection`
        "draws_per_n": DRAWS // 10 if args.quick else DRAWS,
        "targets_log2": {k: round(v[1], 4) for k, v in TARGETS.items()},
        "headline_component": HEADLINE,
        "components": components,
    }
    update_metrics("toxcast_noise_floors", results)
    print(
        json.dumps(
            {
                k: {
                    "abs_q95": v["abs_q95_log2"],
                    "date_share": v["date_share"],
                    "needed": v["wells_needed"],
                }
                for k, v in components.items()
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
