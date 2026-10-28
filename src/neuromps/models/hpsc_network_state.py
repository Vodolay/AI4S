"""Does early network bursting predict how an untreated well settles? A replication (REAL data).

An exploratory, post hoc look at D35's human wells (the 48-well baseline) found two
settling behaviours, wells that keep falling after the first window and wells that
recover, and that the network-burst rate of the first window tracked which. This study,
registered in ``preregistration/hpsc_network_state.json`` (decision D36) before any
windowed feature was computed on the developmental plates, tests that on D32's
replication pool:

* the recordings D32 judged usable on hPSC_MEA1/MEA2 (DIV >= 28) and Rat_MEA1 (DIV >= 21),
  read from D32's own results file;
* each recording cut into 6 equal windows as in D32 and D35; the early state is
  ``features/mea.py`` ``extract_features`` at defaults on window 0 alone (nb0, sync0, mfr0);
* the late trend late_b is the OLS slope over windows 1..5 of D35's log2 firing rate on the
  electrodes active over the whole recording, with D35's floor;
* pooled Spearman rho per species with a 2,000-resample well bootstrap (seed 20261028);
  N1-N3 with their registered rules; an age (DIV) check; the exploratory result recomputed on
  D35's wells and labelled post hoc.

Research decision support only: a measurement property of public culture sets from one lab,
not a statement about any compound or about human biology.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from neuromps.etl.kapucu import _load_spikes
from neuromps.features.mea import extract_features
from neuromps.models import hpsc_settling as st
from neuromps.models import hpsc_split_half as sh
from neuromps.models.toxcast_noise_floors import provenance
from neuromps.paths import REPO_ROOT, figures_dir, outputs_dir

log = logging.getLogger("neuromps.models")
REGISTRATION = REPO_ROOT / "preregistration" / "hpsc_network_state.json"
ROOT = "hpsc_network_state"  # the metrics.json section
RESULTS_NAME = "hpsc_network_state_REAL.json"
FIGURE_NAME = "hpsc_network_state_real.png"
N_WINDOWS = st.N_WINDOWS
SPECIES = sh.SPECIES
RESAMPLES = 2000
SEED = 20261028
EARLY = {
    "nb0": "network_burst_rate_per_min",
    "sync0": "synchrony_index",
    "mfr0": "mfr_hz",
}
COLUMNS = ("nb0", "sync0", "mfr0", "late_b")  # the per-well arrays of the bootstrap
LATE_K = np.arange(1, N_WINDOWS)
MIN_RECORDINGS = 30
MIN_WELLS = 8
MAX_ZERO_SHARE = 0.8
AGE_PARTIAL = -0.1
PREDICTIONS = (
    "N1_early_network_bursting_predicts_settling_in_human",
    "N2_not_in_rat",
    "N3_network_state_not_firing_rate",
)
NOT_IN_RUN = (
    "replication pool not in this run: D32's results hold no usable replication recordings "
    "(a --quick run takes the two 48-well baselines only)"
)


# ------------------------------------------------------------------ per recording


def window_zero(
    channels: dict[str, np.ndarray], duration_s: float
) -> tuple[dict[str, np.ndarray], float]:
    """The first of the 6 equal windows: t < duration / 6 (an edge spike starts window 1)."""
    length = duration_s / N_WINDOWS
    out = {k: np.asarray(v, dtype=float) for k, v in channels.items()}
    return {k: v[v < length] for k, v in out.items()}, length


def late_slope(y: np.ndarray) -> float:
    """Ordinary least-squares slope of y_k on k for k = 1..5 (window 0 left out)."""
    late = np.asarray(y, dtype=float)[LATE_K]
    x = LATE_K - LATE_K.mean()
    return float(x @ (late - late.mean()) / (x @ x))


def recording_state(channels: dict[str, np.ndarray], duration_s: float) -> dict:
    """Early state (window-0 features) and late trend (D35's rate, windows 1..5) of one well."""
    w0, length = window_zero(channels, duration_s)
    got = extract_features(w0, length)
    row: dict = {"window_s": length, "active0": got["active_electrodes"]}
    row.update({short: got[feature] for short, feature in EARLY.items()})
    electrodes = st.electrode_set(channels, duration_s)
    row["n_electrodes"] = len(electrodes)
    row["y"], row["late_b"] = None, None
    if electrodes:
        y = st.log_rates(st.window_counts(channels, electrodes, duration_s), length)
        row["y"] = [float(v) for v in y]
        row["late_b"] = late_slope(y)
    return row


# ------------------------------------------------------------------ inputs


def usable_recordings(path: Path) -> set[tuple[str, int, str]]:
    """(plate, DIV, well) of every replication recording D32 judged usable, from its results."""
    wells = json.loads(Path(path).read_text())["wells"]
    return {
        (w["plate"], int(w["div"]), w["well"])
        for w in wells
        if w["pool"] == "replication" and w["usable"]
    }


def load_usable_recordings(quick: bool, processes: int) -> set[tuple[str, int, str]] | None:
    """D32's usable replication recordings: its results file of this run, else rebuilt (full).

    None on a quick run whose D32 results hold no replication rows, or without the plates.
    """
    path = outputs_dir() / sh.RESULTS_NAME
    keys = usable_recordings(path) if path.exists() else set()
    if keys or quick:
        return keys or None
    recordings = sh.replication_recordings()
    if not recordings:
        return None
    log.info("no replication rows in %s; rebuilding D32's replication table", path)
    table = sh.build_table(recordings, processes)
    rows = table[table["usable"].astype(bool)]
    return {(r.plate, int(r.div), r.well) for r in rows.itertuples()} or None


def _meta(pool: str, rec: sh.Recording, well: str) -> dict:
    unit = f"{rec.plate}:{well}"
    return {"pool": pool, "species": rec.species, "plate": rec.plate, "well": well, "unit": unit}


def replication_rows(rec: sh.Recording, keys: set[tuple[str, int, str]]) -> list[dict]:
    """Every usable well of one developmental recording, in well order."""
    wells, duration = _load_spikes(rec.spikes)
    rows = []
    for well in sorted(wells):
        if (rec.plate, int(rec.div), well) in keys:
            meta = {**_meta("replication", rec, well), "div": rec.div, "duration_s": duration}
            rows.append({**meta, **recording_state(wells[well], duration)})
    return rows


def _replication_job(job: tuple[sh.Recording, set]) -> list[dict]:
    return replication_rows(*job)


def _primary_job(job: tuple[dict, dict, float]) -> dict:
    meta, channels, duration = job
    return {**meta, **recording_state(channels, duration)}


def _map(fn, jobs: list, processes: int) -> list:
    if processes > 1 and len(jobs) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(processes, len(jobs))) as pool:
            return pool.map(fn, jobs)
    return [fn(j) for j in jobs]


def replication_table(
    recordings: list[sh.Recording], keys: set[tuple[str, int, str]], processes: int = 1
) -> pd.DataFrame:
    """One row per usable replication recording (well x DIV), one process per plate file."""
    chunks = _map(_replication_job, [(r, keys) for r in recordings], processes)
    return pd.DataFrame([row for chunk in chunks for row in chunk])


def primary_table(
    recordings: list[sh.Recording], units: set[str], processes: int = 1
) -> pd.DataFrame:
    """One row per D35 well of the 48-well baselines (the post hoc pool), one process per well."""
    jobs = []
    for rec in recordings:
        wells, duration = _load_spikes(rec.spikes)
        for well in sorted(wells):
            meta = _meta("primary", rec, well)
            if meta["unit"] in units:
                meta.update(div=rec.div, duration_s=duration)
                jobs.append((meta, wells[well], duration))
    return pd.DataFrame(_map(_primary_job, jobs, processes))


# ------------------------------------------------------------------ statistics


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho (Pearson of average ranks); NaN below 3 pairs or for a constant side."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan")
    rx, ry = rankdata(x) - (x.size + 1) / 2, rankdata(y) - (y.size + 1) / 2
    denom = float(np.sqrt((rx @ rx) * (ry @ ry)))
    return float(rx @ ry / denom) if denom > 0 else float("nan")


def partial_spearman(x: np.ndarray, y: np.ndarray, div: np.ndarray) -> float:
    """Pearson of the residuals of the ranks of x and of y, each regressed (OLS) on 1 and DIV."""
    design = np.column_stack([np.ones(len(div)), np.asarray(div, dtype=float)])

    def resid(v: np.ndarray) -> np.ndarray:
        r = rankdata(np.asarray(v, dtype=float))
        coef, *_ = np.linalg.lstsq(design, r, rcond=None)
        return r - design @ coef

    rx, ry = resid(x), resid(y)
    denom = float(np.sqrt((rx @ rx) * (ry @ ry)))
    return float(rx @ ry / denom) if denom > 0 else float("nan")


def _num(rows: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)


def well_arrays(rows: pd.DataFrame) -> list[np.ndarray]:
    """One (recordings x 4) array per well (unit order): nb0, sync0, mfr0, late_b; NaN unscored."""
    out = []
    for _, g in rows.groupby("unit", sort=True):
        out.append(np.column_stack([_num(g, c) for c in COLUMNS]))
    return out


def draw(groups: list[np.ndarray], idx: np.ndarray) -> np.ndarray:
    """The recordings of the drawn wells, each well's recordings together."""
    return np.concatenate([groups[i] for i in idx])


def _rho(values: np.ndarray, j: int) -> float:
    ok = np.isfinite(values[:, j]) & np.isfinite(values[:, 3])
    return spearman(values[ok, j], values[ok, 3])


def statistics(values: np.ndarray) -> np.ndarray:
    """rho(nb0, late_b), rho(sync0, late_b), rho(mfr0, late_b) and rho(mfr0) - rho(nb0)."""
    nb, sync, mfr = (_rho(values, j) for j in range(3))
    both = np.isfinite(values[:, 0]) & np.isfinite(values[:, 2])
    diff = _rho(values[both], 2) - _rho(values[both], 0)
    return np.array([nb, sync, mfr, diff])


STATS = ("nb0", "sync0", "mfr0", "mfr0_minus_nb0")


def bootstrap(rows: pd.DataFrame, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Point and 95% percentile interval of each statistic, drawing wells with replacement."""
    groups = well_arrays(rows)
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(STATS)))
    for i in range(resamples):
        draws[i] = statistics(draw(groups, rng.integers(0, len(groups), len(groups))))
    point = statistics(np.concatenate(groups))
    out = {}
    for j, name in enumerate(STATS):
        out[name] = {
            "point": sh._r4(point[j]),
            "ci95": st._interval(draws[:, j]),
            "nonfinite": int(np.sum(~np.isfinite(draws[:, j]))),
        }
    return out


# ------------------------------------------------------------------ per species


def _counts(rows: pd.DataFrame, columns: tuple[str, ...]) -> dict:
    ok = np.all([np.isfinite(_num(rows, c)) for c in (*columns, "late_b")], axis=0)
    return {
        "n_recordings": int(ok.sum()),
        "n_wells": int(rows.loc[ok, "unit"].nunique()),
    }


def species_record(rows: pd.DataFrame, resamples: int) -> dict:
    """Counts, unscored recordings, the nb0-zero share and every rho with its interval."""
    boot = bootstrap(rows, resamples) if rows["unit"].nunique() >= 2 else None
    nb_scored = np.isfinite(_num(rows, "nb0")) & np.isfinite(_num(rows, "late_b"))
    late = _num(rows, "late_b")
    late = late[np.isfinite(late)]
    rho = {}
    for f in EARLY:
        got = boot[f] if boot else {"point": None, "ci95": [None, None], "nonfinite": None}
        rho[f] = {**got, **_counts(rows, (f,))}
    diff = boot["mfr0_minus_nb0"] if boot else {"point": None, "ci95": [None, None]}
    return {
        "n_recordings": len(rows),
        "n_wells": int(rows["unit"].nunique()),
        "divs": sorted(int(d) for d in rows["div"].unique()),
        "unscored": {c: int((~np.isfinite(_num(rows, c))).sum()) for c in COLUMNS},
        "nb0_zero_share": sh._r4(float(np.mean(_num(rows, "nb0")[nb_scored] == 0)))
        if nb_scored.any()
        else None,
        "median_late_b": sh._r4(float(np.median(late))) if late.size else None,
        "late_b_negative_share": sh._r4(float(np.mean(late < 0))) if late.size else None,
        "rho": rho,
        "rho_mfr0_minus_nb0": {**diff, **_counts(rows, ("nb0", "mfr0"))},
    }


def per_species(table: pd.DataFrame, resamples: int) -> dict:
    """species_record for each species present in the table."""
    return {
        s: species_record(table[table["species"] == s], resamples)
        for s in SPECIES
        if (table["species"] == s).any()
    }


# ------------------------------------------------------------------ age check


def age_check(rows: pd.DataFrame) -> dict:
    """rho(nb0, late_b), its partial on DIV, and the per-DIV rhos and their median."""
    nb, late, div = _num(rows, "nb0"), _num(rows, "late_b"), _num(rows, "div")
    ok = np.isfinite(nb) & np.isfinite(late)
    nb, late, div = nb[ok], late[ok], div[ok]
    by_div = {}
    for d in np.unique(div):
        sel = div == d
        by_div[str(int(d))] = {
            "n": int(sel.sum()),
            "rho": sh._r4(spearman(nb[sel], late[sel])),
        }
    rhos = [v["rho"] for v in by_div.values() if v["rho"] is not None]
    return {
        "n_recordings": int(ok.sum()),
        "rho": sh._r4(spearman(nb, late)),
        "partial_rho": sh._r4(partial_spearman(nb, late, div)) if ok.sum() >= 3 else None,
        "per_div": {
            "median_rho": sh._r4(float(np.median(rhos))) if rhos else None,
            "n_divs": len(rhos),
            "n_divs_total": len(by_div),
            # Post hoc, descriptive: how many of the per-DIV rhos are below 0.
            "n_divs_negative": int(sum(r < 0 for r in rhos)),
            "by_div": by_div,
        },
        # Post hoc, descriptive: how each side moves with age (why partial and raw differ).
        "rho_nb0_div_post_hoc": sh._r4(spearman(nb, div)),
        "rho_late_b_div_post_hoc": sh._r4(spearman(late, div)),
    }


def age_record(table: pd.DataFrame) -> dict:
    """The registered age check for each species and the human flag."""
    out = {s: age_check(table[table["species"] == s]) for s in SPECIES}
    partial = out["human"]["partial_rho"]
    out["threshold_partial"] = AGE_PARTIAL
    out["age_effect_possible"] = None if partial is None else bool(partial > AGE_PARTIAL)
    out["verdict"] = (
        "not judged"
        if partial is None
        else "the N1 relation may be an age effect"
        if partial > AGE_PARTIAL
        else "registered age flag not raised: the human partial rho is not above -0.1"
    )
    return out


def nonzero_check(table: pd.DataFrame, resamples: int) -> dict:
    """Post hoc: rho(nb0, late_b) over the recordings with at least one window-0 network burst."""
    out = {
        "note": "post hoc, not registered, added after the first real-data run: is a missing "
        "relation only the recordings with no network burst in window 0 (ties at zero)?"
    }
    for s in SPECIES:
        rows = table[(table["species"] == s) & (_num(table, "nb0") > 0)]
        out[s] = species_record(rows, resamples)["rho"]["nb0"] if len(rows) else None
    return out


# ------------------------------------------------------------------ predictions


def _judgeable(counts: dict, zero_share: float | None = None) -> bool:
    enough = counts["n_recordings"] >= MIN_RECORDINGS and counts["n_wells"] >= MIN_WELLS
    if zero_share is not None and zero_share > MAX_ZERO_SHARE:
        return False
    return bool(enough)


def _prediction(record: dict, passed: bool, judgeable: bool) -> dict:
    return {**record, "pass": bool(passed and judgeable), "outcome": sh.outcome(passed, judgeable)}


def predict(species: dict | None) -> dict:
    """N1-N3 with their registered rules; uninterpretable without the replication pool."""
    if species is None or not {"human", "rat"} <= set(species):
        return {
            p: {"status": NOT_IN_RUN, "pass": False, "outcome": "uninterpretable"}
            for p in PREDICTIONS
        }
    h, r = species["human"], species["rat"]
    zero = h["nb0_zero_share"]
    n1, n2, n3 = h["rho"]["nb0"], r["rho"]["nb0"], h["rho_mfr0_minus_nb0"]
    hi1, hi2, lo3 = n1["ci95"][1], n2["ci95"][1], n3["ci95"][0]
    rules = (
        (n1, hi1 is not None and hi1 < 0, _judgeable(n1, zero)),
        (n2, not (hi2 is not None and hi2 < 0), _judgeable(n2)),
        (n3, lo3 is not None and lo3 > 0, _judgeable(n3, zero)),
    )
    out = {}
    for name, (rec, passed, judgeable) in zip(PREDICTIONS, rules, strict=True):
        record = {**_flat(rec), "nb0_zero_share_human": zero}
        out[name] = _prediction(record, passed, judgeable)
    out[PREDICTIONS[0]]["nb0_zero_share"] = zero
    return out


def _flat(rec: dict) -> dict:
    """The tested value, its interval and the counts it rests on, for citation."""
    keys = ("n_recordings", "n_wells", "nonfinite")
    return {"value": rec["point"], "ci95": rec["ci95"], **{k: rec.get(k) for k in keys}}


# ------------------------------------------------------------------ assembly


def post_hoc_record(primary: pd.DataFrame | None, resamples: int) -> dict | None:
    """The exploratory result on D35's 48-well wells, recomputed and labelled post hoc."""
    if primary is None or primary.empty:
        return None
    return {
        "post_hoc": True,
        "label": "post hoc, not registered: the exploratory observation that motivated D36, "
        "recomputed on D35's usable wells of the two 48-well baselines (one recording a well)",
        "window_minutes": (
            sh._r4(float(primary["window_s"].iloc[0]) / 60) if "window_s" in primary else None
        ),
        **per_species(primary, resamples),
    }


def analyse(
    replication: pd.DataFrame | None, primary: pd.DataFrame | None, resamples: int = RESAMPLES
) -> dict:
    """Every registered statistic, the age check and the post hoc primary-pool result."""
    have = replication is not None and not replication.empty
    species = per_species(replication, resamples) if have else None
    return {
        "label": "REAL Kapucu 2022 untreated wells: early network state and later settling, "
        "human iPSC vs rat (registered replication, D36)",
        "registration": provenance(REGISTRATION),
        "resamples": resamples,
        "seed": SEED,
        "n_windows": N_WINDOWS,
        "replication": species,
        "replication_status": "scored" if have else NOT_IN_RUN,
        "predictions": predict(species),
        "reported_not_tested": {
            "sync0": {s: species[s]["rho"]["sync0"] for s in species} if species else None,
        },
        "age_check": age_record(replication) if have else None,
        "nonzero_nb0_check_post_hoc": nonzero_check(replication, resamples) if have else None,
        "primary_pool_post_hoc": post_hoc_record(primary, resamples),
        "implementation_notes": IMPLEMENTATION_NOTES,
    }


IMPLEMENTATION_NOTES = [
    "The replication recordings are read from D32's results file of the same run "
    "(hpsc_split_half_REAL.json, pool replication, usable true) as (plate, DIV, well); on a full "
    "run without those rows D32's replication table is rebuilt; a --quick run has none and "
    "N1-N3 are then uninterpretable with a status.",
    "Windows follow D32 and D35: the plate file's last spike time is the duration, cut into 6 "
    "equal windows; window 0 is t < duration / 6 and goes to extract_features with that length; "
    "a spike on the edge starts window 1.",
    "nb0, sync0 and mfr0 are extract_features' network_burst_rate_per_min, synchrony_index and "
    "mfr_hz on window 0 at FeatureParams defaults; window 0's own active rule selects electrodes.",
    "late_b uses D35's electrode set (active over the whole recording), D35's per-window rate "
    "and its one-spike floor; the slope is fitted on y_1..y_5 against k = 1..5.",
    "A recording is scored for a feature when the feature and late_b are both defined; "
    "synchrony_index can be undefined with active electrodes (no STTC pair), which leaves only "
    "sync0 unscored.",
    "Spearman rho is the Pearson correlation of average ranks; a constant side gives no rho; "
    "a bootstrap resample with no rho is dropped from the interval and counted as nonfinite.",
    "Each species' interval uses a fresh generator (seed 20261028); each resample draws that "
    "species' wells (plate:well) with replacement and keeps all of a drawn well's recordings; "
    "rho(nb0), rho(sync0), rho(mfr0) and the N3 difference come from the same draws.",
    "The N3 difference is computed on the recordings scored for both nb0 and mfr0.",
    "'Lies below 0' is strict (upper end < 0) and 'lies above 0' strict (lower end > 0); N2 "
    "passes unless the rat interval's upper end is below 0.",
    "Uninterpretable is read literally: fewer than 30 scored recordings, or fewer than 8 wells "
    "among them, for the tested statistic's species; N1 and N3 also when nb0 is exactly 0 in "
    "more than 80% of scored human recordings.",
    "The partial Spearman regresses the average ranks of nb0 and of late_b (over the scored "
    "recordings) by OLS on an intercept and DIV as a number, and correlates the residuals; the "
    "per-DIV rhos pool both plates of a species at the same DIV, and a DIV with fewer than 3 "
    "recordings or a constant side gives no rho and is left out of the median.",
    "The post hoc pool is D35's usable wells (from D32's results file), one 30-min recording a "
    "well cut into D35's 5-min windows; its intervals use the same well bootstrap.",
    "Added after the first real-data run and labelled post hoc: rho(nb0, late_b) over the "
    "recordings with nb0 > 0, the count of negative per-DIV rhos, and the rank correlations "
    "of nb0 and of late_b with DIV; the age-check verdict text was reworded to state only "
    "whether the registered flag is raised.",
    "Before the code was committed, a timing check computed window-0 features on one "
    "developmental plate file (Rat_MEA1 DIV21) and printed one well's values; no late trend "
    "or correlation was computed before the commit.",
]


def main(argv: list[str] | None = None) -> int:
    """Score the replication and post hoc pools, test N1-N3, write results and figure."""
    from neuromps.metrics import update_metrics
    from neuromps.parallel import worker_processes

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="post hoc pool only (quick data tier)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    primary_recs = sh.primary_recordings()
    if len(primary_recs) < len(sh.PRIMARY):
        log.warning("Kapucu baseline plates not present under %s; skipping", sh.DATA_ROOT)
        return 0
    procs = worker_processes(cap=12)
    primary = primary_table(primary_recs, st.load_usable_units(), procs)
    keys = load_usable_recordings(args.quick, procs)
    rep = replication_table(sh.replication_recordings(), keys, procs) if keys else None
    payload = analyse(rep, primary)
    figures_dir().mkdir(parents=True, exist_ok=True)
    plot(payload, rep, primary, figures_dir() / FIGURE_NAME)
    payload["results_file"] = f"outputs/{RESULTS_NAME}"
    payload["figure"] = f"outputs/figures/{FIGURE_NAME}"
    tables = [t for t in (rep, primary) if t is not None]
    rows = json.loads(pd.concat(tables).to_json(orient="records"))
    (outputs_dir() / RESULTS_NAME).write_text(
        json.dumps({**payload, "recordings": rows}, indent=1) + "\n"
    )
    update_metrics(ROOT, payload)
    print(json.dumps(payload["predictions"], indent=1))
    return 0


# ------------------------------------------------------------------ figure

PLATE_COLOURS = {
    "hPSC_MEA1": "#E69F00",
    "hPSC_MEA2": "#CC79A7",
    "Rat_MEA1": "#009E73",
}


def _rho_text(record: dict | None) -> str:
    if not record:
        return ""
    got = record["rho"]["nb0"]
    ci = got["ci95"]
    interval = f" [{ci[0]}, {ci[1]}]" if ci and ci[0] is not None else ""
    return f"rho(nb0, late_b) = {got['point']}{interval}, n = {got['n_recordings']}"


def _scatter(ax, rows: pd.DataFrame, colour_by_plate: bool, species: str) -> None:
    for plate, g in rows.groupby("plate", sort=True):
        colour = PLATE_COLOURS.get(plate, sh.COLOURS[species]) if colour_by_plate else None
        ax.plot(
            _num(g, "nb0"),
            _num(g, "late_b"),
            "o",
            ms=3,
            alpha=0.75,
            color=colour or sh.COLOURS[species],
            label=plate,
        )
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("network bursts per min in window 0 (nb0)", fontsize=7)
    ax.set_ylabel("late trend late_b (log2 rate per window, windows 1-5)", fontsize=7)
    ax.legend(fontsize=6, frameon=False)


def _panel(ax, rows, record, species: str, title: str, by_plate: bool) -> None:
    if rows is None or not (rows["species"] == species).any():
        ax.text(0.5, 0.5, "not in this run", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        ax.set_title(title, fontsize=7.5)
        return
    _scatter(ax, rows[rows["species"] == species], by_plate, species)
    ax.set_title(f"{title}\n{_rho_text(record)}", fontsize=7)


def plot(
    payload: dict, replication: pd.DataFrame | None, primary: pd.DataFrame | None, path: Path
) -> str:
    """late_b against nb0 per species: the registered replication and the post hoc pool."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = payload["replication"] or {}
    post = payload["primary_pool_post_hoc"] or {}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    panels = (
        (
            replication,
            rep.get("human"),
            "human",
            "REGISTERED replication: human iPSC, 12-well, 100-s windows",
        ),
        (replication, rep.get("rat"), "rat", "REGISTERED replication: rat, 12-well, 100-s windows"),
        (primary, post.get("human"), "human", "POST HOC: human iPSC, 48-well DIV29, 5-min windows"),
        (primary, post.get("rat"), "rat", "POST HOC: rat, 48-well DIV22, 5-min windows"),
    )
    for ax, (rows, record, species, title) in zip(axes, panels, strict=True):
        _panel(ax, rows, record, species, title, by_plate=rows is replication)
        ax.tick_params(labelsize=7)
    title = (
        "REAL data: Kapucu 2022 untreated wells, one lab: first-window network bursting against "
        "the later firing-rate trend (registered replication D36, and the post hoc 48-well pool)"
    )
    fig.suptitle(title, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return title


if __name__ == "__main__":
    raise SystemExit(main())
