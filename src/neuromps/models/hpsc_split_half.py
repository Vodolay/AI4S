"""Is an untreated human iPSC MEA recording noisier than a rat one? Split-half noise (REAL data).

The final-round deck gives wells needed for rat cortical chips only (the ToxCast
vehicle wells). The Kapucu et al. 2022 deposit (CC-BY-4.0) has human hPSC-derived
and rat primary cortical cultures recorded by one lab with one protocol, but too
few untreated wells recorded twice. So each untreated recording is cut in half
and the two halves are compared: how much a well's readout moves between the
halves of one recording, with nothing done to it, per species.

Design registered in ``preregistration/hpsc_split_half.json`` (decision D32)
before any half-recording feature or change was computed:

* each well's recording is cut at half its duration (the plate file's last spike
  time); the second half is shifted to start at 0; both halves go through
  ``features/mea.py`` ``extract_features`` with duration = half the recording;
* a well is usable when it is not ExcludedWell and has at least 10 active
  electrodes in each half;
* per feature, d = log2(max(f2 / f1, 0.05)); its SD (and 1.4826 x MAD) per species;
* R = SD_human / SD_rat with a 2,000-resample within-species well bootstrap;
* P1-P3 (firing rate, across features, on the developmental plates); the wells
  needed from the primary SD, a LOWER BOUND; and a firing-rate-overlap check.

Research decision support only: a measurement property of two public culture
sets from one lab, not a statement about any compound or about human biology.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.etl.kapucu import _files, _load_spikes, parse_explog, parse_noisy_electrodes
from neuromps.features.mea import extract_features
from neuromps.models.toxcast_noise_floors import provenance
from neuromps.paths import RAW_ROOT, REPO_ROOT, figures_dir, outputs_dir

log = logging.getLogger("neuromps.models")
REGISTRATION = REPO_ROOT / "preregistration" / "hpsc_split_half.json"
DATA_ROOT = RAW_ROOT / "kapucu2022" / "Data"
ROOT = "hpsc_split_half"  # the metrics.json section
RESULTS_NAME = "hpsc_split_half_REAL.json"
FIGURE_NAME = "hpsc_split_half_real.png"
FEATURES = (
    "mfr_hz",
    "burst_rate_per_min",
    "burst_duration_s",
    "spikes_per_burst",
    "pct_spikes_in_bursts",
    "network_burst_rate_per_min",
    "synchrony_index",
    "isi_cv",
)
HEADLINE = "mfr_hz"
SPECIES = ("human", "rat")
FLOOR = 0.05
MIN_ACTIVE = 10
RESAMPLES = 2000
SEED = 20261028
MIN_WELLS = {"primary": 10, "replication": 6}
MIN_FEATURES = 5
MAJORITY = 5
MAD_SCALE = 1.4826
Z_ALPHA = 1.959964  # 5% two-sided
Z_POWER = 0.841621  # 80% power
FALLS = {"down_20pct": 0.2, "down_50pct": 0.5}
PERCENTILES = (10, 90)
PRIMARY = (
    ("human", "hPSC_MEA3_Pharmacology/hPSC_MEA3Baseline_DIV29", 29),
    ("rat", "Rat_MEA2_Pharmacology/Rat_MEA2Baseline_DIV22", 22),
)
# (species, plate directory, age floor in days in vitro)
REPLICATION = (("human", "hPSC_MEA1", 28), ("human", "hPSC_MEA2", 28), ("rat", "Rat_MEA1", 21))
_DIV = re.compile(r"_DIV(\d+)_spikes\.csv$")


@dataclass(frozen=True)
class Recording:
    """One plate recording that enters a pool."""

    pool: str
    species: str
    plate: str
    div: int
    spikes: Path
    explog: Path
    noisy: Path | None


# ------------------------------------------------------------------ inputs


def primary_recordings(data_root: Path = DATA_ROOT) -> list[Recording]:
    """The two matched 48-well untreated baselines (empty if not fetched)."""
    out = []
    for species, directory, div in PRIMARY:
        plate_dir = data_root / directory
        if not plate_dir.exists():
            continue
        found = _files(plate_dir)
        name = Path(directory).name
        out.append(
            Recording(
                "primary", species, name, div, found["spikes"], found["explog"], found.get("noisy")
            )
        )
    return out


def replication_recordings(data_root: Path = DATA_ROOT) -> list[Recording]:
    """Every developmental recording at or after its species' age floor."""
    out = []
    for species, plate, floor in REPLICATION:
        folder = data_root / plate / f"{plate}_spikes_noise_explogs"
        explogs = sorted(folder.glob("*_expLog.csv"))
        if not explogs:
            continue
        noisy = next(iter(sorted(folder.glob("noisy_electrodes*.csv"))), None)
        for path in sorted(folder.glob("*_DIV*_spikes.csv")):
            div = int(_DIV.search(path.name).group(1))
            if div >= floor:
                out.append(Recording("replication", species, plate, div, path, explogs[0], noisy))
    return sorted(out, key=lambda r: (r.plate, r.div))


def noisy_electrodes(path: Path | None, div: int) -> dict[str, list[str]]:
    """Well -> noisy electrodes for one recording.

    The pharmacology files hold one DIV column; the developmental files one column
    per DIV. The column for ``div`` (else the only one) is handed to the ETL parser.
    """
    if path is None:
        return {}
    table = pd.read_csv(path, dtype=str)
    column = f"DIV{div}" if f"DIV{div}" in table.columns else table.columns[0]
    values = [v.strip() for v in table[column].dropna() if v.strip()]
    return parse_noisy_electrodes("\n".join([column, *values]))


# ------------------------------------------------------------------ split and features


def split_halves(
    channels: dict[str, np.ndarray], duration_s: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Cut at half the duration; t < half is the first half, the rest is shifted by -half."""
    half = duration_s / 2.0
    first, second = {}, {}
    for name, times in channels.items():
        t = np.asarray(times, dtype=float)
        first[name] = t[t < half]
        second[name] = t[t >= half] - half
    return first, second, half


def well_features(channels: dict[str, np.ndarray], duration_s: float) -> dict[str, dict]:
    """Shared-extractor features of the full recording and of each half."""
    first, second, half = split_halves(channels, duration_s)
    return {
        "full": extract_features(channels, duration_s),
        "first": extract_features(first, half),
        "second": extract_features(second, half),
    }


def is_usable(treatment: str | None, active_1: int, active_2: int, untreated_only: bool) -> bool:
    """Not ExcludedWell (and NoTreatment where required); >= 10 active electrodes per half."""
    if treatment == "excludedwell":
        return False
    if untreated_only and treatment != "notreatment":
        return False
    return bool(active_1 >= MIN_ACTIVE and active_2 >= MIN_ACTIVE)


def change(f1: float | None, f2: float | None) -> float | None:
    """d = log2(max(f2 / f1, 0.05)); None when a half is undefined or f1 <= 0."""
    if f1 is None or f2 is None or not (np.isfinite(f1) and np.isfinite(f2)) or f1 <= 0:
        return None
    return float(math.log2(max(f2 / f1, FLOOR)))


def well_row(channels: dict[str, np.ndarray], duration_s: float, meta: dict) -> dict:
    """One well-recording: active counts, half features, changes and the usable flag."""
    got = well_features(channels, duration_s)
    first, second = got["first"], got["second"]
    row = {
        **meta,
        "duration_s": duration_s,
        "active_full": got["full"]["active_electrodes"],
        "active_1": first["active_electrodes"],
        "active_2": second["active_electrodes"],
        "mfr_full": got["full"]["mfr_hz"],
    }
    for f in FEATURES:
        row[f"{f}_1"], row[f"{f}_2"] = first[f], second[f]
        row[f"d_{f}"] = change(first[f], second[f])
    row["usable"] = is_usable(
        meta["treatment"], row["active_1"], row["active_2"], meta["pool"] == "replication"
    )
    return row


def recording_rows(rec: Recording) -> list[dict]:
    """Every well of one recording: the union of the spike file and the expLog."""
    wells, duration = _load_spikes(rec.spikes)
    treatments = parse_explog(rec.explog).set_index("well")["compound"].to_dict()
    noisy = noisy_electrodes(rec.noisy, rec.div)
    rows = []
    for well in sorted(set(wells) | set(treatments)):
        meta = {
            "pool": rec.pool,
            "species": rec.species,
            "plate": rec.plate,
            "well": well,
            "unit": f"{rec.plate}:{well}",
            "div": rec.div,
            "treatment": treatments.get(well),
            "n_noisy": len(noisy.get(well, [])),
        }
        rows.append(well_row(wells.get(well, {}), duration, meta))
    return rows


def build_table(recordings: list[Recording], processes: int = 1) -> pd.DataFrame:
    """All well-recordings of the given recordings, one process per recording."""
    if processes > 1 and len(recordings) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(processes, len(recordings))) as pool:
            chunks = pool.map(recording_rows, recordings)
    else:
        chunks = [recording_rows(r) for r in recordings]
    return pd.DataFrame([row for chunk in chunks for row in chunk])


# ------------------------------------------------------------------ statistics


def sd(x: np.ndarray) -> float:
    """Sample SD (ddof 1); NaN below two values."""
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1)) if x.size >= 2 else float("nan")


def robust_sd(x: np.ndarray) -> float:
    """1.4826 x the median absolute deviation from the median."""
    x = np.asarray(x, dtype=float)
    return float(MAD_SCALE * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def _resampled_sd(groups: list[np.ndarray], rng) -> float:
    drawn = rng.integers(0, len(groups), len(groups))
    return sd(np.concatenate([groups[i] for i in drawn]))


def ratio_interval(
    human: list[np.ndarray], rat: list[np.ndarray], resamples: int, seed: int
) -> dict:
    """R = SD_human / SD_rat over all values, and its 95% percentile bootstrap interval.

    Each list holds one array per well (a well's recordings move together); the
    bootstrap draws wells with replacement within each species, human then rat.
    """
    rng = np.random.default_rng(seed)
    ratios = np.empty(resamples)
    for i in range(resamples):
        h, r = _resampled_sd(human, rng), _resampled_sd(rat, rng)
        ratios[i] = h / r if r > 0 else np.nan
    finite = ratios[np.isfinite(ratios)]
    point = sd(np.concatenate(human)) / sd(np.concatenate(rat))
    low, high = np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan)
    return {
        "R": _r4(point),
        "ci95": [_r4(low), _r4(high)],
        "resamples": int(resamples),
        "nonfinite": int(resamples - finite.size),
    }


def wells_needed(sd_log2: float, fall: float) -> int:
    """Wells per arm for a fall against same-session controls: 5% two-sided, 80% power."""
    return int(math.ceil(2 * (Z_ALPHA + Z_POWER) ** 2 * sd_log2**2 / math.log2(1 - fall) ** 2))


def overlap_range(human: np.ndarray, rat: np.ndarray) -> tuple[float, float] | None:
    """Intersection of the two species' 10th-90th percentile ranges; None if they miss."""
    lo_h, hi_h = np.percentile(np.asarray(human, dtype=float), PERCENTILES)
    lo_r, hi_r = np.percentile(np.asarray(rat, dtype=float), PERCENTILES)
    low, high = max(lo_h, lo_r), min(hi_h, hi_r)
    return (float(low), float(high)) if low <= high else None


def in_range(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Mask of values inside [low, high], ends included."""
    x = np.asarray(x, dtype=float)
    return (x >= low) & (x <= high)


def _r4(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


# ------------------------------------------------------------------ per pool


def _groups(rows: pd.DataFrame, column: str) -> list[np.ndarray]:
    """One array of defined values per well (unit), in unit order."""
    defined = rows[rows[column].notna()]
    return [g[column].to_numpy(dtype=float) for _, g in defined.groupby("unit", sort=True)]


def species_spread(rows: pd.DataFrame, feature: str) -> dict:
    """SD, robust SD and counts of d for one species and feature."""
    column = f"d_{feature}"
    values = rows[column].dropna().to_numpy(dtype=float)
    return {
        "n_wells": int(rows.loc[rows[column].notna(), "unit"].nunique()),
        "n_values": int(values.size),
        "unscored": int(rows[column].isna().sum()),
        "sd": _r4(sd(values)),
        "robust_sd": _r4(robust_sd(values)),
    }


def compare_feature(rows: pd.DataFrame, feature: str, min_wells: int, resamples: int) -> dict:
    """Both species' spread, and R with its interval when the feature is scored."""
    by = {s: rows[rows["species"] == s] for s in SPECIES}
    out = {s: species_spread(by[s], feature) for s in SPECIES}
    out["scored"] = all(out[s]["n_wells"] >= min_wells for s in SPECIES)
    out["R"], out["ci95"] = None, None
    if out["scored"]:
        column = f"d_{feature}"
        got = ratio_interval(
            _groups(by["human"], column), _groups(by["rat"], column), resamples, SEED
        )
        out.update(got)
    return out


def compare_pool(table: pd.DataFrame, pool: str, resamples: int) -> dict | None:
    """Every feature of one pool, over its usable well-recordings; None if the pool is absent."""
    rows = table[table["pool"] == pool]
    if rows.empty:
        return None
    usable = rows[rows["usable"].astype(bool)]
    by = {s: usable[usable["species"] == s] for s in SPECIES}
    return {
        "min_wells_per_feature": MIN_WELLS[pool],
        "wells": {s: int(rows.loc[rows["species"] == s, "unit"].nunique()) for s in SPECIES},
        "usable_wells": {s: int(by[s]["unit"].nunique()) for s in SPECIES},
        "usable_recordings": {s: len(by[s]) for s in SPECIES},
        "recordings": {s: int(rows.loc[rows["species"] == s, "div"].nunique()) for s in SPECIES},
        "noisy_electrodes_in_usable": {s: int(by[s]["n_noisy"].sum()) for s in SPECIES},
        "features": {f: compare_feature(usable, f, MIN_WELLS[pool], resamples) for f in FEATURES},
    }


# ------------------------------------------------------------------ predictions


def outcome(passed: bool, judgeable: bool) -> str:
    """The audit's outcome word: below a registered minimum the test cannot be read."""
    if not judgeable:
        return "uninterpretable"
    return "pass" if passed else "fail"


def _enough(pool: dict | None, minimum: int) -> bool:
    return pool is not None and all(pool["usable_wells"][s] >= minimum for s in SPECIES)


def predict_p1(primary: dict) -> dict:
    """P1: the 95% interval of R for firing rate lies above 1 (strictly)."""
    head = primary["features"][HEADLINE]
    ci = head.get("ci95")
    passed = bool(ci is not None and ci[0] is not None and ci[0] > 1)
    judgeable = _enough(primary, MIN_WELLS["primary"])
    return {
        "R": head.get("R"),
        "ci95": ci,
        "usable_wells": dict(primary["usable_wells"]),
        "pass": passed and judgeable,
        "outcome": outcome(passed, judgeable),
    }


def predict_p2(primary: dict) -> dict:
    """P2: SD_human > SD_rat for at least 5 features, among at least 5 scored."""
    features = primary["features"]
    scored = [f for f, v in features.items() if v["scored"]]
    noisier = [f for f in scored if features[f]["human"]["sd"] > features[f]["rat"]["sd"]]
    judgeable = len(scored) >= MIN_FEATURES
    passed = judgeable and len(noisier) >= MAJORITY
    return {
        "features_scored": len(scored),
        "human_noisier": len(noisier),
        "noisier_features": noisier,
        "scored_features": scored,
        "pass": bool(passed),
        "outcome": outcome(passed, judgeable),
    }


def predict_p3(replication: dict | None) -> dict:
    """P3: the point estimate of R for firing rate on the replication pool is above 1."""
    if replication is None:
        return {
            "R": None,
            "status": "replication plates not present (full-tier download)",
            "pass": False,
            "outcome": "uninterpretable",
        }
    head = replication["features"][HEADLINE]
    judgeable = _enough(replication, MIN_WELLS["replication"])
    passed = bool(head.get("R") is not None and head["R"] > 1)
    return {
        "R": head.get("R"),
        "ci95": head.get("ci95"),
        "usable_wells": dict(replication["usable_wells"]),
        "pass": passed and judgeable,
        "outcome": outcome(passed, judgeable),
    }


# ------------------------------------------------------------------ reported, not tested


def wells_needed_record(primary: dict) -> dict:
    """Wells per arm for a 20% and a 50% firing-rate fall, from each species' primary SD."""
    out = {
        "note": (
            "LOWER BOUND: split-half noise leaves out handling, compound addition and "
            "day-to-day variation, which the ToxCast vehicle wells include"
        ),
        "false_alarm_rate": 0.05,
        "power": 0.8,
    }
    for s in SPECIES:
        sd_log2 = primary["features"][HEADLINE][s]["sd"]
        out[s] = {"sd_log2": sd_log2, **{k: wells_needed(sd_log2, v) for k, v in FALLS.items()}}
    return out


def assumption_check(table: pd.DataFrame, primary: dict, resamples: int) -> dict:
    """R for firing rate on the wells inside both species' 10th-90th pct firing-rate range."""
    rows = table[(table["pool"] == "primary") & table["usable"].astype(bool)]
    rates = {s: rows.loc[rows["species"] == s, "mfr_full"].to_numpy(dtype=float) for s in SPECIES}
    out = {"percentiles": list(PERCENTILES), "pool": "primary", "feature": HEADLINE}
    for s in SPECIES:
        lo, hi = np.percentile(rates[s], PERCENTILES)
        out[f"{s}_range_hz"] = [_r4(lo), _r4(hi)]
    found = overlap_range(rates["human"], rates["rat"])
    out["overlap_hz"] = None if found is None else [_r4(found[0]), _r4(found[1])]
    kept = {
        s: rows[(rows["species"] == s) & in_range(rows["mfr_full"], *found)] if found else rows[:0]
        for s in SPECIES
    }
    out["kept"] = {s: int(kept[s]["unit"].nunique()) for s in SPECIES}
    out["R"], out["ci95"], out["other_side_of_one"] = None, None, None
    column = f"d_{HEADLINE}"
    groups = {s: _groups(kept[s], column) for s in SPECIES}
    if all(len(groups[s]) >= 2 for s in SPECIES):
        out.update(ratio_interval(groups["human"], groups["rat"], resamples, SEED))
        primary_r = primary["features"][HEADLINE]["R"]
        out["other_side_of_one"] = bool((out["R"] - 1) * (primary_r - 1) < 0)
    return out


# ------------------------------------------------------------------ assembly


def analyse(table: pd.DataFrame, resamples: int = RESAMPLES) -> dict:
    """Every registered statistic from a table of well-recordings."""
    primary = compare_pool(table, "primary", resamples)
    replication = compare_pool(table, "replication", resamples)
    predictions = {
        "P1_human_firing_rate_noisier": predict_p1(primary),
        "P2_human_noisier_across_features": predict_p2(primary),
        "P3_replicates_on_developmental_plates": predict_p3(replication),
    }
    return {
        "label": "REAL Kapucu 2022 untreated wells: split-half noise, human iPSC vs rat",
        "registration": provenance(REGISTRATION),
        "resamples": resamples,
        "seed": SEED,
        "primary": primary,
        "replication": replication,
        "predictions": predictions,
        "wells_needed": wells_needed_record(primary),
        "assumption_check": assumption_check(table, primary, resamples),
        "implementation_notes": IMPLEMENTATION_NOTES,
    }


IMPLEMENTATION_NOTES = [
    "A spike exactly at the cut (half the plate file's last spike time) starts the second "
    "half; the plate's last spike ends it.",
    "Wells are the union of the spike file and the expLog, as in the asset; a well with no "
    "spikes has 0 active electrodes and is not usable.",
    "Replication noisy electrodes are read from the DIV column of the series' table.",
    "A feature's change is also unscored when a half's value is non-finite; for "
    "synchrony_index (STTC), a non-positive second half is floored at 0.05 like any fall.",
    "Usable and per-feature well counts are distinct wells (plate:well); on the replication "
    "pool SD pools all of a species' usable well-recordings and the bootstrap draws wells.",
    "P2 is judged on the primary pool, with SD (not the robust SD).",
    "P1 needs the interval's lower end strictly above 1; P3 needs R strictly above 1.",
    "Each R interval uses a fresh generator with seed 20261028, drawing human then rat per "
    "resample; resamples with SD_rat = 0 would be dropped and are counted as nonfinite.",
    "The overlap check uses the usable primary wells: each species' 10th and 90th percentile "
    "(numpy linear) of the full-recording firing rate; wells inside both ranges, ends included.",
    "Wells needed use the primary pool's firing-rate SD of d for each species.",
]


def main(argv: list[str] | None = None) -> int:
    """Build the well table, compute every registered statistic, write results and figure."""
    from neuromps.metrics import update_metrics
    from neuromps.parallel import worker_processes

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="primary pool only (quick data tier)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    recordings = primary_recordings()
    if len(recordings) < len(PRIMARY):
        log.warning("Kapucu baseline plates not present under %s; skipping", DATA_ROOT)
        return 0
    if not args.quick:
        recordings += replication_recordings()
    table = build_table(recordings, worker_processes(cap=12))
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

COLOURS = {"human": "#D55E00", "rat": "#0072B2"}
POOL_MARKERS = {"primary": "o", "replication": "s"}


def _ratio_panel(ax, payload: dict) -> None:
    y = np.arange(len(FEATURES))
    for k, pool in enumerate(("primary", "replication")):
        record = payload.get(pool)
        if record is None:
            continue
        for i, f in enumerate(FEATURES):
            got = record["features"][f]
            if not got["scored"] or got["R"] is None:
                continue
            low, high = got["ci95"]
            yy = y[i] + (0.15 if k else -0.15)
            ax.plot([low, high], [yy, yy], color="#000000", lw=0.8)
            ax.plot(got["R"], yy, POOL_MARKERS[pool], ms=4, color="#009E73" if k else "#CC79A7")
    ax.plot([], [], "o", color="#CC79A7", label="48-well baselines (primary)")
    ax.plot([], [], "s", color="#009E73", label="12-well developmental (replication)")
    ax.axvline(1, color="grey", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y, [f.replace("_", " ") for f in FEATURES], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("R = SD human / SD rat of split-half change (95% interval)", fontsize=7)
    ax.legend(fontsize=6, frameon=False, loc="lower right")


def _change_panel(ax, table: pd.DataFrame) -> None:
    rows = table[(table["pool"] == "primary") & table["usable"].astype(bool)]
    rng = np.random.default_rng(0)
    for i, s in enumerate(SPECIES):
        d = rows.loc[rows["species"] == s, f"d_{HEADLINE}"].dropna().to_numpy(dtype=float)
        ax.plot(i + rng.uniform(-0.15, 0.15, d.size), d, "o", ms=3, color=COLOURS[s], alpha=0.8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks([0, 1], ["human iPSC\n(DIV29)", "rat\n(DIV22)"], fontsize=7)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("firing-rate change, second vs first half (log2)", fontsize=7)


def _rate_panel(ax, table: pd.DataFrame, check: dict) -> None:
    rows = table[(table["pool"] == "primary") & table["usable"].astype(bool)]
    for s in SPECIES:
        sel = rows[rows["species"] == s]
        ax.plot(sel["mfr_full"], sel[f"d_{HEADLINE}"].abs(), "o", ms=3, color=COLOURS[s], label=s)
    if check.get("overlap_hz"):
        ax.axvspan(*check["overlap_hz"], color="#F0E442", alpha=0.35, lw=0, label="rate overlap")
    ax.set_xscale("log")
    ax.set_xlabel("full-recording mean firing rate (Hz)", fontsize=7)
    ax.set_ylabel("|firing-rate change| (log2)", fontsize=7)
    ax.legend(fontsize=6, frameon=False)


def plot(payload: dict, table: pd.DataFrame, path: Path) -> str:
    """R per feature and pool; the firing-rate changes; the firing-rate overlap check."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.4, 0.8, 1]})
    _ratio_panel(axes[0], payload)
    _change_panel(axes[1], table)
    _rate_panel(axes[2], table, payload["assumption_check"])
    for ax in axes:
        ax.tick_params(labelsize=7)
    title = (
        "REAL data: Kapucu 2022 untreated wells, one lab: split-half noise of human iPSC vs "
        "rat cortical MEA recordings (registered, D32)"
    )
    fig.suptitle(title, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return title


if __name__ == "__main__":
    raise SystemExit(main())
