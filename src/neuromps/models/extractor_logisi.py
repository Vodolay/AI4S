"""REGISTERED (D34): does the depositors' adaptive burst detector close the rest of the gap?

The post hoc settings ladder (`extractor_ladder`) closed the electrode-set and denominator
differences between `features/mea.py` and the meaRtools output the Kapucu 2022 depositors
published; what remained (burst duration, spikes per burst, 10-15% of the hPSC burst rate)
was attributed to the burst detector without a test. This module is that test, as
`preregistration/extractor_logisi.json` fixes it:

- ``S4_logisi``: the ladder's S3 (active above 10 spikes/min, the deposit's noisy
  electrodes dropped per DIV, burst rate per bursting electrode) with the logISI detector
  (`FeatureParams.burst_detector = "logisi"`), reported beside S0 and S3;
- the ladder's statistics per plate and pooled, the four predictions L1-L4 with their
  pass, fail and uninterpretable rules, the per-plate fallback share, the per-electrode
  threshold distribution, and (post hoc, labelled) the pairs that move by more than 10%
  between S3 and S4.

It says how far one more stated choice of the depositors moves our numbers towards theirs;
it does not say which detector is right.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.features.mea import logisi_thresholds
from neuromps.models.extractor_agreement import (
    DATA_ROOT,
    PLATES,
    _channels_for,
    noisy_lists,
    plate_prefix,
)
from neuromps.models.extractor_ladder import LADDER, S4_LOGISI, ladder_stats
from neuromps.models.toxcast_noise_floors import provenance
from neuromps.paths import REPO_ROOT

log = logging.getLogger("neuromps.models")

REGISTRATION = REPO_ROOT / "preregistration" / "extractor_logisi.json"
RESULTS_NAME = "extractor_logisi_REAL.json"
FIGURE_NAME = "extractor_logisi_real.png"
S3 = LADDER[3]
LOGISI = S4_LOGISI
VARIANTS = (LADDER[0].name, S3.name, LOGISI.name)
FEATURES = ("burst_duration_s", "spikes_per_burst", "burst_rate_per_min")
HPSC = ("hPSC_MEA1", "hPSC_MEA2")
MAX_FALLBACK_SHARE = 0.5
MOVE = 0.10
LABEL = (
    "REGISTERED (preregistration/extractor_logisi.json, D34): REAL recordings (Kapucu 2022), "
    "our extractor with the depositors' stated settings and the logISI burst detector (S4), "
    "against the meaRtools output they published"
)

# Each prediction: the plates it concerns, and its checks as (scope, feature, ratio band,
# minimum Spearman rho or None), exactly as the registration states them.
SPEC = {
    "L1_burst_duration_closes": (PLATES, [("pooled", "burst_duration_s", (0.90, 1.10), 0.90)]),
    "L2_spikes_per_burst_closes": (PLATES, [("pooled", "spikes_per_burst", (0.95, 1.05), 0.95)]),
    "L3_hpsc_burst_rate_closes": (
        HPSC,
        [(p, "burst_rate_per_min", (0.95, 1.05), None) for p in HPSC],
    ),
    "L4_rat_not_worse": (
        ("Rat_MEA1",),
        [
            ("Rat_MEA1", "burst_rate_per_min", (0.95, 1.05), None),
            ("Rat_MEA1", "burst_duration_s", (0.90, 1.10), None),
        ],
    ),
}


def _check(stats: dict, scope: str, feature: str, band: tuple, min_rho: float | None) -> dict:
    """One registered check on the S4 statistics: values and whether they meet it."""
    entry = stats.get(scope, {}).get(feature, {})
    ratio, rho = entry.get("median_ratio"), entry.get("spearman_rho")
    readable = ratio is not None and (min_rho is None or rho is not None)
    met = bool(readable and band[0] <= ratio <= band[1] and (min_rho is None or rho >= min_rho))
    return {
        "scope": scope,
        "feature": feature,
        "median_ratio": ratio,
        "spearman_rho": rho,
        "ratio_band": list(band),
        "min_spearman_rho": min_rho,
        "readable": readable,
        "met": met,
    }


def _fallback_share(fallback: dict, plates: tuple[str, ...]) -> float | None:
    """Fallback electrodes over active electrodes, pooled over ``plates``."""
    active = sum(fallback.get(p, {}).get("n_active", 0) for p in plates)
    falls = sum(fallback.get(p, {}).get("n_fallback", 0) for p in plates)
    return falls / active if active else None


def predictions(stats: dict, fallback: dict, present: tuple[str, ...]) -> dict:
    """L1-L4 from the S4 statistics, with the registered uninterpretable rule.

    A prediction cannot be read when a plate it concerns is not in this run, a statistic it
    needs is undefined, or the fallback applies to more than half of the active electrodes
    on its plates (pooled over them).
    """
    registered = json.loads(REGISTRATION.read_text())["predictions"]
    out = {}
    for name, (plates, checks) in SPEC.items():
        got = [_check(stats, *c) for c in checks]
        share = _fallback_share(fallback, plates)
        missing = [p for p in plates if p not in present]
        judgeable = (
            not missing
            and all(c["readable"] for c in got)
            and share is not None
            and share <= MAX_FALLBACK_SHARE
        )
        passed = all(c["met"] for c in got)
        out[name] = {
            "statement": registered[name]["statement"],
            "test": registered[name]["test"],
            "checks": got,
            "plates": list(plates),
            "plates_missing": missing,
            "fallback_share": round(share, 4) if share is not None else None,
            "pass": bool(passed and judgeable),
            "outcome": ("pass" if passed else "fail") if judgeable else "uninterpretable",
        }
    return out


def fallback_record(rows: pd.DataFrame) -> dict:
    """Per plate and pooled: active electrode-recordings, how many fell back, and why."""
    scopes = {**dict(tuple(rows.groupby("plate"))), "pooled": rows}
    out = {}
    for scope, sub in scopes.items():
        n, falls = int(len(sub)), int(sub["fallback"].astype(bool).sum())
        out[str(scope)] = {
            "n_active": n,
            "n_fallback": falls,
            "share": round(falls / n, 4) if n else None,
            "by_reason": {str(k): int(v) for k, v in sub["reason"].value_counts().items()},
        }
    return out


def _quantiles_ms(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median_ms": None, "p10_ms": None, "p25_ms": None, "p75_ms": None, "p90_ms": None}
    q = np.percentile(values * 1000.0, [10, 25, 50, 75, 90])
    return {
        "median_ms": round(float(q[2]), 2),
        "p10_ms": round(float(q[0]), 2),
        "p25_ms": round(float(q[1]), 2),
        "p75_ms": round(float(q[3]), 2),
        "p90_ms": round(float(q[4]), 2),
    }


def threshold_distribution(rows: pd.DataFrame, cutoff_s: float = 0.1) -> dict:
    """Per plate and pooled: the valid per-electrode logISI thresholds, in ms."""
    scopes = {**dict(tuple(rows.groupby("plate"))), "pooled": rows}
    out = {}
    for scope, sub in scopes.items():
        valid = sub.loc[~sub["fallback"].astype(bool), "threshold_s"].to_numpy(dtype=float)
        out[str(scope)] = {
            "n_valid": int(valid.size),
            **_quantiles_ms(valid),
            "share_valid_above_cutoff": (
                round(float(np.mean(valid > cutoff_s)), 4) if valid.size else None
            ),
            # Descriptive, added after the quick run showed thresholds of a fraction of a
            # millisecond (a peak of near-coincident spike pairs taken as the intraburst peak).
            "share_valid_below_1ms": (
                round(float(np.mean(valid < 0.001)), 4) if valid.size else None
            ),
        }
    return out


def moved_pairs(table: pd.DataFrame) -> dict:
    """POST HOC: pairs whose value moves by more than 10% from S3 to S4, per plate and feature."""
    keys = ["plate", "well", "div", "feature"]
    s3 = table[table["variant"] == S3.name].set_index(keys)["ours"]
    s4 = table[table["variant"] == LOGISI.name].set_index(keys)["ours"]
    both = pd.concat({"s3": s3, "s4": s4}, axis=1, join="inner").reset_index()
    both = both[both["s3"] != 0]
    both["moved"] = np.abs(both["s4"] / both["s3"] - 1.0) > MOVE
    scopes = {**dict(tuple(both.groupby("plate"))), "pooled": both}
    by_plate = {
        str(scope): {
            str(feature): {
                "n": int(len(rows)),
                "moved": int(rows["moved"].sum()),
                "share": round(float(rows["moved"].mean()), 4),
            }
            for feature, rows in sub.groupby("feature")
        }
        for scope, sub in scopes.items()
    }
    return {
        "label": "POST HOC (named in the registration's assumption check, no threshold set)",
        "definition": "|S4 / S3 - 1| > 0.10 for the same plate, well, DIV and feature",
        "by_plate": by_plate,
    }


def _threshold_job(args) -> list[dict]:
    """Per active electrode, the logISI threshold under S4, for one recording."""
    path, plate, div, noisy = args
    from neuromps.etl.kapucu import _load_spikes

    wells, duration_s = _load_spikes(path)
    rows = []
    for well, channels in wells.items():
        kept = _channels_for(channels, noisy, LOGISI)
        for row in logisi_thresholds(kept, duration_s, LOGISI.params):
            rows.append({"plate": plate, "well": well, "div": div, **row})
    return rows


def thresholds_table(
    plates: tuple[str, ...] = PLATES,
    root: Path = DATA_ROOT,
    max_recordings: int | None = None,
    processes: int | None = None,
) -> pd.DataFrame:
    """Every active electrode-recording's logISI threshold under S4, on every plate."""
    import multiprocessing as mp

    from neuromps.parallel import worker_processes

    jobs = []
    for plate in plates:
        spikes_dir = root / plate / f"{plate}_spikes_noise_explogs"
        prefix = plate_prefix(spikes_dir) if spikes_dir.is_dir() else None
        if prefix is None:
            continue
        noisy_path = next(iter(sorted(spikes_dir.glob("noisy_electrodes*.csv"))), None)
        files = sorted(spikes_dir.glob(f"{prefix}_DIV*_spikes.csv"))
        for path in files[:max_recordings] if max_recordings else files:
            div = int(re.search(r"_DIV(\d+)_spikes\.csv$", path.name).group(1))
            jobs.append((path, plate, div, noisy_lists(noisy_path, div)))
    if not jobs:
        return pd.DataFrame(columns=["plate", "well", "div", "electrode", "threshold_s"])
    with mp.Pool(processes or worker_processes()) as pool:
        results = pool.map(_threshold_job, jobs, chunksize=1)
    return pd.DataFrame([row for batch in results for row in batch])


def analyse(table: pd.DataFrame, thresholds: pd.DataFrame, plates: tuple[str, ...]) -> dict:
    """Every registered statistic, from the paired table (S0, S3, S4 rows) and the thresholds."""
    rows = table[table["variant"].isin(VARIANTS) & table["feature"].isin(FEATURES)]
    stats = ladder_stats(rows)
    fallback = fallback_record(thresholds)
    present = tuple(p for p in plates if p in set(rows["plate"]))
    return {
        "label": LABEL,
        "status": "registered; predictions judged as the registration states them",
        "registration": provenance(REGISTRATION),
        "variants": list(VARIANTS),
        "features": list(FEATURES),
        "s4_method_record": LOGISI.params.as_method_record(),
        "by_variant": stats,
        "predictions": predictions(stats.get(LOGISI.name, {}), fallback, present),
        "fallback": fallback,
        "threshold_distribution": threshold_distribution(thresholds, LOGISI.params.logisi_cutoff_s),
        "moved_s3_to_s4_post_hoc": moved_pairs(rows),
        "plates": list(present),
    }


OKABE_ITO = {
    "S0_ours": "#E69F00",
    "S3_rate_per_bursting_electrode": "#56B4E9",
    "S4_logisi": "#009E73",
}
SHORT = {
    "S0_ours": "S0 ours",
    "S3_rate_per_bursting_electrode": "S3 settings",
    "S4_logisi": "S4 logISI",
}
TITLES = {
    "burst_duration_s": "Burst duration",
    "spikes_per_burst": "Spikes per burst",
    "burst_rate_per_min": "Burst rate",
}


def _panel(ax, stats: dict, scopes: list[str], feature: str) -> None:
    """Median ratio ours/theirs per plate for S0, S3 and S4, one feature."""
    width = 0.26
    x = np.arange(len(scopes))
    for k, variant in enumerate(VARIANTS):
        y = [stats.get(variant, {}).get(s, {}).get(feature, {}).get("median_ratio") for s in scopes]
        y = [float(v) if v is not None else np.nan for v in y]
        ax.bar(x + (k - 1) * width, y, width, color=OKABE_ITO[variant], label=SHORT[variant])
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [s.replace("_", "\n") for s in scopes])
    ax.set_title(TITLES[feature], fontsize=8.5)
    ax.tick_params(labelsize=7)


def plot_logisi(stats: dict, plates: tuple[str, ...], path: Path) -> Path:
    """Burst duration, spikes per burst and burst rate per plate: S0, S3, S4 (real data)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scopes = [*plates, "pooled"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), sharey=True)
    for ax, feature in zip(axes, FEATURES, strict=True):
        _panel(ax, stats, scopes, feature)
    axes[0].set_ylabel("median ratio ours / theirs", fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "REAL recordings (Kapucu 2022): our extractor against the depositors' meaRtools output,\n"
        "S0 (ours), S3 (their stated settings) and S4 (S3 + logISI, registered). "
        "Dashed line is equality.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


THRESHOLD_COLUMNS = ("plate", "div", "electrode", "n_spikes", "threshold_s", "reason")


def results_text(payload: dict, thresholds: pd.DataFrame) -> str:
    """The results file: the payload, then every electrode's threshold, one row per line."""
    rows = [
        json.dumps([r[0], int(r[1]), r[2], int(r[3]), _sig(r[4]), r[5]])
        for r in thresholds.reindex(columns=list(THRESHOLD_COLUMNS)).itertuples(index=False)
    ]
    head = json.dumps({**payload, "threshold_columns": list(THRESHOLD_COLUMNS)}, indent=1)
    body = ",\n  ".join(rows)
    return f'{head[:-2]},\n "thresholds": [\n  {body}\n ]\n}}\n' if rows else head + "\n"


def _sig(value: object) -> float | None:
    """A threshold to 6 significant digits; None when there is none."""
    if value is None or not np.isfinite(float(value)):
        return None
    return float(f"{float(value):.6g}")


def run(
    table: pd.DataFrame,
    plates: tuple[str, ...] = PLATES,
    max_recordings: int | None = None,
    root: Path = DATA_ROOT,
) -> dict:
    """The registered payload, its results file (with every threshold) and its figure."""
    from neuromps.paths import figures_dir, outputs_dir

    thresholds = thresholds_table(plates, root, max_recordings)
    payload = analyse(table, thresholds, plates)
    figure = plot_logisi(
        payload["by_variant"], tuple(payload["plates"]), figures_dir() / FIGURE_NAME
    )
    payload["results_file"] = f"outputs/{RESULTS_NAME}"
    payload["figure"] = f"outputs/figures/{figure.name}"
    outputs_dir().mkdir(parents=True, exist_ok=True)
    (outputs_dir() / RESULTS_NAME).write_text(results_text(payload, thresholds))
    for name, p in payload["predictions"].items():
        log.info("%s: %s", name, p["outcome"])
    return payload
