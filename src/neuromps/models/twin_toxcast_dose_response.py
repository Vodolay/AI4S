"""Does the twin's knob dose-response have the shape of real concentration-response curves?

Section 33 checked the twin's drug knobs against the EPA acute MEA screen's three
positive controls, each at one saturating concentration: every scored direction
agreed, but a rule that follows firing rate did as well, and section 34 recorded
what was left open - nothing there tested the twin's *dose response*.

The same screen runs about five hundred compounds over seven concentrations, with
a viability readout on the same plates. This module takes the compounds whose
established primary target is one of the twin's knobs (the class labels are fixed
in the registration), simulates each knob over a seven-level grid on twelve
chips, matches concentrations to levels by rank - the two axes share no unit - and
asks whether the twin's curve carries the real curve's rank order, and which
readout starts to move first. Two nulls must be beaten: the compound's own real
firing-rate curve, and the twin's curve under a knob the compound is not labelled
with.

Registered in ``preregistration/twin_toxcast_dose_response.json`` (decision D30)
before any treated-well value of these compounds was computed; the twin's curves
are committed (``--predict``) before the real side is computed. Firing rate is
treated as already seen and is excluded from the shape tests.

Research decision support only: this scores a simulator against one public
screen's measurements and states nothing about any compound.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neuromps.models.toxcast_noise_floors import WELLS, log2_change, provenance
from neuromps.paths import INTERIM_ROOT, REPO_ROOT, twin_dir

log = logging.getLogger("neuromps.models")
REGISTRATION = REPO_ROOT / "preregistration" / "twin_toxcast_dose_response.json"
PREDICTIONS_NAME = "toxcast_dose_response_predictions_simulated.json"
SAMPLES = INTERIM_ROOT / "toxcast_mea" / "samples.parquet"
VIABILITY_WELLS = INTERIM_ROOT / "toxcast_mea" / "mc0_mea_acute.parquet"
VIABILITY_ACID = 2496
ROOT = "twin_toxcast_dose_response"  # the metrics.json section

KNOB_GRIDS = {
    "gabaa_block": ("gabaa_scale", (0.9, 0.8, 0.65, 0.5, 0.35, 0.2, 0.1)),
    "gabaa_potentiate": ("gabaa_scale", (1.1, 1.25, 1.5, 1.8, 2.2, 2.6, 3.0)),
    "sodium_block": ("threshold_scale", (1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8)),
    "ampa_block": ("ampa_scale", (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)),
}
LABEL_KNOB = {
    "gabaa_antagonist": "gabaa_block",
    "gabaa_positive_modulator": "gabaa_potentiate",
    "sodium_channel_blocker": "sodium_block",
}
MAPPING = {
    "mfr_hz": 2448,
    "burst_rate_per_min": 2449,
    "burst_duration_s": 2450,
    "spikes_per_burst": 2451,
    "pct_spikes_in_bursts": 2453,
    "synchrony_index": 2461,
}
SEEN_FEATURE = "mfr_hz"  # a private analysis looked at it; not part of P1-P3
LEVELS = 7
BUILD_SEEDS = tuple(range(9201, 9213))
DURATION_S = 40.0
MIN_MOVE = 0.1
MIN_SIGN_SHARE = 7 / 8
MIN_TWIN_CHIPS = 8
MIN_BINS = 4
HIT_RHO = 0.6
MARGIN = 0.1
HIT_RATE = 0.5
MIN_PAIRS = 8
NULL_DRAWS = 20_000
REAL_SEED = 20261029
VIABILITY_SEED = 20261030
ALPHA = 0.05
TRUNCATED_LEVELS = 5  # the assumption check: the grid without its two strongest levels


def compounds() -> dict[str, str]:
    """Compound -> class label, read from the registration (fixed before any data)."""
    registration = json.loads(REGISTRATION.read_text())["compound_labels"]
    return {
        name: label
        for label, spec in registration.items()
        if isinstance(spec, dict) and "members" in spec
        for name in spec["members"]
    }


# --------------------------------------------------------------------------- twin


def _simulate(job: tuple) -> dict:
    """One chip's paired change at one knob level: untreated and treated, fresh noise."""
    from neuromps.models.twin_prior import _features, pair_deltas

    knob, level, build = job
    parameter, grid = KNOB_GRIDS[knob]
    noise = build + 200_000 + 1_000 * list(KNOB_GRIDS).index(knob) + level
    baseline = _features({}, build, build + 100_000)
    treated = _features({parameter: grid[level]}, build, noise)
    deltas = pair_deltas(treated, baseline)
    return {
        "knob": knob,
        "level": level,
        "value": grid[level],
        "build_seed": build,
        **{f: deltas.get(f"delta.{f}") for f in MAPPING},
    }


def simulate_changes(processes: int, seeds: tuple = BUILD_SEEDS) -> pd.DataFrame:
    """Every registered (knob, level, chip) triple, simulated in parallel."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(knob, level, seed) for knob in KNOB_GRIDS for level in range(LEVELS) for seed in seeds]
    with ProcessPoolExecutor(processes) as pool:
        return pd.DataFrame(list(pool.map(_simulate, jobs, chunksize=1)))


def level_summary(changes: np.ndarray) -> dict:
    """The registered move rule over chips at one knob level and feature."""
    defined = np.asarray(changes, dtype=float)
    defined = defined[np.isfinite(defined)]
    if defined.size == 0:
        return {"median": None, "n_defined": 0, "moves": False}
    median = float(np.median(defined))
    same = int((np.sign(defined) == np.sign(median)).sum())
    moves = (
        defined.size >= MIN_TWIN_CHIPS
        and abs(median) >= MIN_MOVE
        and same >= MIN_SIGN_SHARE * defined.size
    )
    return {"median": round(median, 4), "n_defined": int(defined.size), "moves": bool(moves)}


def twin_curves(changes: pd.DataFrame) -> dict:
    """knob -> feature -> per-level summaries, in level order."""
    out: dict[str, dict] = {}
    for knob, rows in changes.groupby("knob"):
        out[str(knob)] = {
            feature: [
                level_summary(rows.loc[rows["level"] == level, feature].to_numpy(dtype=float))
                for level in range(LEVELS)
            ]
            for feature in MAPPING
        }
    return out


# ------------------------------------------------------------------- rank matching


def concentration_bins(conc: np.ndarray) -> np.ndarray:
    """Half-decade bin index of each concentration in uM; zero or less gives NaN."""
    conc = np.asarray(conc, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        binned = np.round(2 * np.log10(np.where(conc > 0, conc, np.nan)))
    return binned


def matched_positions(n_bins: int) -> list[float]:
    """Positions on the knob's level grid for a compound's bins, weakest first."""
    if n_bins < 2:
        return [0.0]
    return [i * (LEVELS - 1) / (n_bins - 1) for i in range(n_bins)]


def at_position(summaries: list[dict], position: float, levels: int = LEVELS) -> dict:
    """The twin's value at a matched position: interpolated median, bracketing moves."""
    position = float(np.clip(position, 0, levels - 1))
    low, high = int(np.floor(position)), int(np.ceil(position))
    values = [summaries[low]["median"], summaries[high]["median"]]
    if any(v is None for v in values):
        median = None
    elif low == high:
        median = float(values[0])
    else:
        weight = position - low
        median = float(values[0] * (1 - weight) + values[1] * weight)
    return {
        "median": None if median is None else round(median, 4),
        "moves": bool(summaries[low]["moves"] or summaries[high]["moves"]),
    }


def matched_curve(summaries: list[dict], n_bins: int, levels: int = LEVELS) -> list[dict]:
    """One feature's twin curve matched onto a compound's bins."""
    return [
        at_position(summaries[:levels], p, levels)
        for p in [i * (levels - 1) / max(n_bins - 1, 1) for i in range(n_bins)]
    ]


# --------------------------------------------------------------------------- real


def vehicle_groups(wells: pd.DataFrame, acid: int) -> list[np.ndarray]:
    """Per exposure date, the vehicle wells' log2 changes for one component."""
    rows = wells[(wells["acid"] == acid) & (wells["wllt"] == "n")]
    x = log2_change(rows["rval"].to_numpy())
    frame = pd.DataFrame({"apid": rows["apid"].to_numpy(), "x": x})
    frame = frame[np.isfinite(frame["x"])]
    return [group["x"].to_numpy() for _, group in frame.groupby("apid")]


def null_draws(groups: list[np.ndarray], n: int, draws: int, rng) -> np.ndarray:
    """Vehicle-only draws of the statistic: n wells' median minus the date's others."""
    usable = [g for g in groups if g.size >= n + 3]
    if not usable:
        return np.array([])
    counts = rng.multinomial(draws, np.full(len(usable), 1 / len(usable)))
    out = []
    for group, count in zip(usable, counts, strict=True):
        if count == 0:
            continue
        order = np.argsort(rng.random((int(count), group.size)), axis=1)
        picked = group[order[:, :n]]
        rest = group[order[:, n:]]
        out.append(np.median(picked, axis=1) - np.median(rest, axis=1))
    return np.concatenate(out) if out else np.array([])


def null_threshold(
    groups: list[np.ndarray], n: int, k: int, seed: int, lower: bool = False
) -> dict:
    """The registered cut: |draws| quantile 1 - alpha/k, or the signed lower quantile."""
    rng = np.random.default_rng(seed)
    draws = null_draws(groups, n, NULL_DRAWS, rng)
    if draws.size == 0:
        return {"threshold": None, "draws": 0}
    if lower:
        value = float(np.quantile(draws, ALPHA / max(k, 1)))
    else:
        value = float(np.quantile(np.abs(draws), 1 - ALPHA / max(k, 1)))
    return {"threshold": round(value, 4), "draws": int(draws.size)}


def treated_effects(wells: pd.DataFrame, acid: int, spids: list[str]) -> pd.DataFrame:
    """Each treated well's log2 change minus the median of its date's vehicle wells."""
    rows = wells[wells["acid"] == acid].assign(x=lambda d: log2_change(d["rval"].to_numpy()))
    rows = rows[np.isfinite(rows["x"])]
    vehicle = rows[rows["wllt"] == "n"].groupby("apid")["x"].median().rename("vehicle")
    treated = rows[(rows["wllt"] == "t") & (rows["spid"].isin(spids))].join(vehicle, on="apid")
    treated = treated[np.isfinite(treated["vehicle"])]
    treated = treated.assign(
        effect=treated["x"] - treated["vehicle"], bin=concentration_bins(treated["conc"].to_numpy())
    )
    return treated[np.isfinite(treated["bin"])][["apid", "bin", "effect"]]


def real_curve(effects: pd.DataFrame, bins: list[int]) -> list[dict]:
    """Per bin: the median effect over the compound's wells, and how many there are."""
    grouped = effects.groupby("bin")["effect"]
    out = []
    for b in bins:
        if b not in grouped.groups:
            out.append({"bin": int(b), "wells": 0, "median": None})
            continue
        values = grouped.get_group(b).to_numpy(dtype=float)
        out.append(
            {"bin": int(b), "wells": int(values.size), "median": round(float(np.median(values)), 4)}
        )
    return out


def cached_threshold(
    cache: dict, groups: list[np.ndarray], n: int, k: int, seed: int, lower: bool = False
) -> dict:
    """``null_threshold`` memoised on its arguments: the same cut costs 20,000 draws once.

    The cut depends only on the component's vehicle wells, the well count, the
    family size and the seed, all of which repeat across compounds, so a shared
    cache changes no number and keeps the step inside the quick run's budget.
    """
    key = (seed, n, k, lower)
    if key not in cache:
        cache[key] = null_threshold(groups, n, k, seed, lower=lower)
    return cache[key]


def flag_moving(
    curve: list[dict], groups: list[np.ndarray], k: int, seed: int, cache: dict | None = None
) -> list[dict]:
    """Mark the bins whose median effect clears the vehicle null at their well count."""
    cache = {} if cache is None else cache
    out = []
    for point in curve:
        n = point["wells"]
        if n == 0 or point["median"] is None:
            out.append({**point, "threshold": None, "moves": False})
            continue
        cut = cached_threshold(cache, groups, n, k, seed + n)["threshold"]
        moves = cut is not None and abs(point["median"]) > cut
        out.append({**point, "threshold": cut, "moves": bool(moves)})
    return out


def viability_curve(viability: pd.DataFrame, spids: list[str], bins: list[int]) -> list[dict]:
    """Per bin, the median log2 viability signal against the same date's vehicle wells."""
    rows = viability[np.isfinite(viability["rval"])]
    vehicle = rows[rows["wllt"] == "n"].groupby("apid")["rval"].median().rename("vehicle")
    treated = rows[(rows["wllt"] == "t") & (rows["spid"].isin(spids))].join(vehicle, on="apid")
    treated = treated[(treated["vehicle"] > 0) & (treated["rval"] > 0)]
    treated = treated.assign(
        effect=np.log2(treated["rval"] / treated["vehicle"]),
        bin=concentration_bins(treated["conc"].to_numpy()),
    )
    return real_curve(treated[np.isfinite(treated["bin"])][["apid", "bin", "effect"]], bins)


def excluded_bins(
    curve: list[dict], groups: list[np.ndarray], k: int, cache: dict | None = None
) -> dict:
    """The lowest bin the viability readout flags, and every bin above it."""
    cache = {} if cache is None else cache
    flagged, untestable = [], []
    for point in curve:
        n = point["wells"]
        if n == 0 or point["median"] is None:
            continue
        cut = cached_threshold(cache, groups, n, k, VIABILITY_SEED + n, lower=True)["threshold"]
        if cut is None:
            untestable.append(point["bin"])
        elif point["median"] < cut:
            flagged.append(point["bin"])
    first = min(flagged) if flagged else None
    return {
        "first_flagged_bin": first,
        "excluded": [] if first is None else [p["bin"] for p in curve if p["bin"] >= first],
        "untestable_bins": untestable,
        "viability": curve,
    }


# ------------------------------------------------------------------------ scoring


def pair_score(real: list[dict], twin: list[dict], retained: list[int]) -> dict:
    """Spearman of the twin's matched curve against the real one over retained bins."""
    rows = [
        (r["median"], t["median"])
        for r, t in zip(real, twin, strict=True)
        if r["bin"] in retained and r["median"] is not None and t["median"] is not None
    ]
    if len(rows) < MIN_BINS:
        return {"bins": len(rows), "spearman": None}
    x, y = (np.array(v, dtype=float) for v in zip(*rows, strict=True))
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"bins": len(rows), "spearman": None}
    rho = float(spearmanr(y, x).statistic)
    return {"bins": len(rows), "spearman": round(rho, 4), "hit": bool(rho >= HIT_RHO)}


def first_movers(curves: dict, retained: list[int], key: str = "moves") -> list[str]:
    """The features whose lowest moving retained bin is lowest; ties kept as ties."""
    lowest: dict[str, int] = {}
    for feature, curve in curves.items():
        moving = [p["bin"] for p in curve if p["bin"] in retained and p[key]]
        if moving:
            lowest[feature] = min(moving)
    if not lowest:
        return []
    best = min(lowest.values())
    return sorted(f for f, b in lowest.items() if b == best)


def summarise(scores: list[dict]) -> dict:
    """Pairs, hit rate and mean Spearman over a list of scored pairs."""
    rhos = [s["spearman"] for s in scores if s.get("spearman") is not None]
    hits = [s["hit"] for s in scores if s.get("spearman") is not None]
    return {
        "pairs": len(rhos),
        "hits": int(sum(hits)),
        "hit_rate": round(sum(hits) / len(rhos), 4) if rhos else None,
        "mean_spearman": round(float(np.mean(rhos)), 4) if rhos else None,
    }


def margin_over(twin: dict, other: dict) -> float | None:
    """How far the twin's mean Spearman is above a null's; None when either is missing."""
    if twin["mean_spearman"] is None or other["mean_spearman"] is None:
        return None
    return round(twin["mean_spearman"] - other["mean_spearman"], 4)


def evaluate(twin: dict, nulls: dict, first_mover: dict) -> dict:
    """The registered tests P1-P4 (P5 is reported, not tested)."""
    follow, shuffled = nulls["follow_firing_rate"], nulls["shuffled_mechanism"]

    def gap(other: dict) -> float | None:
        return margin_over(twin, other)

    matched = [v for v in first_mover.values() if v["match"] is not None]
    agree = [v["match"] for v in matched]
    return {
        "P1_shape": {
            "pairs": twin["pairs"],
            "hit_rate": twin["hit_rate"],
            "pass": bool(twin["pairs"] >= MIN_PAIRS and (twin["hit_rate"] or 0) >= HIT_RATE),
        },
        "P2_beats_follow_firing_rate": {
            "margin": gap(follow),
            "null_mean_spearman": follow["mean_spearman"],
            "pass": bool(gap(follow) is not None and gap(follow) >= MARGIN),
        },
        "P3_beats_shuffled_mechanism": {
            "margin": gap(shuffled),
            "null_mean_spearman": shuffled["mean_spearman"],
            "pass": bool(gap(shuffled) is not None and gap(shuffled) >= MARGIN),
        },
        "P4_first_mover": {
            "compounds": len(agree),
            "agreeing": int(sum(agree)),
            "rate": round(sum(agree) / len(agree), 4) if agree else None,
            "descriptive_only": len(agree) < MIN_BINS,
            "pass": bool(agree and sum(agree) / len(agree) >= HIT_RATE and len(agree) >= MIN_BINS),
        },
    }


# ------------------------------------------------------------ per-compound assembly


def tested_bins(
    wells: pd.DataFrame, spids: list[str], acid: int = MAPPING[SEEN_FEATURE]
) -> list[int]:
    """The compound's tested half-decade bins, ascending, from its treated wells."""
    rows = wells[(wells["acid"] == acid) & (wells["wllt"] == "t") & wells["spid"].isin(spids)]
    binned = concentration_bins(rows["conc"].to_numpy())
    return sorted({int(b) for b in binned if np.isfinite(b)})


def real_side(
    wells: pd.DataFrame,
    spids: list[str],
    bins: list[int],
    groups: dict,
    cache: dict | None = None,
) -> dict[str, list[dict]]:
    """Per feature, the compound's real curve over its tested bins, moving bins flagged."""
    out = {}
    for feature, acid in MAPPING.items():
        curve = real_curve(treated_effects(wells, acid, spids), bins)
        out[feature] = flag_moving(curve, groups[feature], len(bins), REAL_SEED + acid, cache)
    return out


def score_compound(real: dict, twin: dict, knob: str, retained: list[int], levels: int) -> dict:
    """Every unseen feature of one compound: the twin's score and the two nulls'."""
    n_bins = len(next(iter(real.values())))
    pairs, follow, shuffled = {}, {}, {}
    for feature in MAPPING:
        if feature == SEEN_FEATURE:
            continue
        matched = matched_curve(twin[knob][feature], n_bins, levels)
        moves = any(
            m["moves"] for m, r in zip(matched, real[feature], strict=True) if r["bin"] in retained
        )
        responsive = any(p["moves"] for p in real[feature] if p["bin"] in retained)
        if not (moves and responsive):
            pairs[feature] = {
                "scored": False,
                "twin_moves": bool(moves),
                "responsive": bool(responsive),
            }
            continue
        score = pair_score(real[feature], matched, retained)
        pairs[feature] = {"scored": score.get("spearman") is not None, **score}
        if score.get("spearman") is None:
            continue
        follow[feature] = pair_score(real[feature], real[SEEN_FEATURE], retained)
        shuffled[feature] = [
            pair_score(real[feature], matched_curve(twin[other][feature], n_bins, levels), retained)
            for other in KNOB_GRIDS
            if other != knob
        ]
    return {"features": pairs, "follow_firing_rate": follow, "shuffled_mechanism": shuffled}


def _matched_with_bins(twin: dict, knob: str, feature: str, bins: list[int], levels: int) -> list:
    """One feature's matched twin curve, carrying the real bins so it can be flagged."""
    matched = matched_curve(twin[knob][feature], len(bins), levels)
    return [{**m, "bin": b} for m, b in zip(matched, bins, strict=True)]


def compound_first_mover(
    real: dict, twin: dict, knob: str, retained: list[int], levels: int
) -> dict:
    """Which readout starts to move first, in the real curves and in the twin's."""
    bins = [p["bin"] for p in real[SEEN_FEATURE]]
    unseen = [f for f in MAPPING if f != SEEN_FEATURE]
    got_real = first_movers({f: real[f] for f in unseen}, retained)
    got_twin = first_movers(
        {f: _matched_with_bins(twin, knob, f, bins, levels) for f in unseen}, retained
    )
    match = None if not (got_real and got_twin) else bool(set(got_real) & set(got_twin))
    return {"real": got_real, "twin": got_twin, "match": match}


def firing_rate_sign(real: dict, twin: dict, knob: str, retained: list[int], levels: int) -> dict:
    """SEEN readout, reported not tested: the sign at the highest retained bin."""
    curve = real[SEEN_FEATURE]
    matched = matched_curve(twin[knob][SEEN_FEATURE], len(curve), levels)
    rows = [
        (r["median"], m["median"])
        for r, m in zip(curve, matched, strict=True)
        if r["bin"] in retained and r["median"] is not None and m["median"] is not None
    ]
    if not rows:
        return {"real": None, "twin": None, "agrees": None}
    real_value, twin_value = rows[-1]
    return {
        "real": round(real_value, 4),
        "twin": round(twin_value, 4),
        "agrees": bool(np.sign(real_value) == np.sign(twin_value)),
    }


def analyse_compound(
    label: str,
    wells: pd.DataFrame,
    viability: pd.DataFrame,
    twin: dict,
    groups: dict,
    spids: list[str],
    levels: int = LEVELS,
    cache: dict | None = None,
) -> dict:
    """One compound end to end: bins, exclusion, curves, scores and first mover."""
    knob = LABEL_KNOB[label]
    bins = tested_bins(wells, spids)
    real = real_side(wells, spids, bins, groups, cache)
    exclusion = excluded_bins(
        viability_curve(viability, spids, bins), groups["viability"], len(bins), cache
    )
    retained = [b for b in bins if b not in exclusion["excluded"]]
    out = {
        "label": label,
        "knob": knob,
        "spids": spids,
        "bins": bins,
        "retained_bins": retained,
        "exclusion": exclusion,
        "real": real,
        "scored": len(retained) >= MIN_BINS,
    }
    if len(retained) < MIN_BINS:
        out["dropped"] = f"only {len(retained)} of {len(bins)} bins retained"
        return out
    out.update(score_compound(real, twin, knob, retained, levels))
    out["first_mover"] = compound_first_mover(real, twin, knob, retained, levels)
    out["firing_rate_sign"] = firing_rate_sign(real, twin, knob, retained, levels)
    return out


# --------------------------------------------------------------------- aggregation


def collect(results: dict) -> dict:
    """The twin's scored pairs and the two nulls', over every scored compound."""
    twin, follow, shuffled, pairs = [], [], [], []
    for name, result in results.items():
        if not result.get("scored"):
            continue
        for feature, score in result["features"].items():
            if score.get("spearman") is None:
                continue
            twin.append(score)
            follow.append(result["follow_firing_rate"][feature])
            shuffled.extend(result["shuffled_mechanism"][feature])
            pairs.append({"compound": name, "feature": feature, **score})
    return {
        "twin": summarise(twin),
        "nulls": {
            "follow_firing_rate": summarise(follow),
            "shuffled_mechanism": summarise(shuffled),
        },
        "pairs": pairs,
    }


def rescore(results: dict, twin: dict, levels: int) -> dict:
    """Score the stored real curves again against a shorter knob grid (the check)."""
    out = {}
    for name, result in results.items():
        if not result.get("scored"):
            continue
        knob, retained, real = result["knob"], result["retained_bins"], result["real"]
        out[name] = {
            "scored": True,
            "knob": knob,
            **score_compound(real, twin, knob, retained, levels),
            "first_mover": compound_first_mover(real, twin, knob, retained, levels),
        }
    return out


def rescore_dropping(results: dict, twin: dict, which: str) -> dict:
    """POST HOC (decided after seeing the curves): drop one bin per compound.

    The registered viability rule excluded nothing, so the highest concentration
    stayed in every curve - and that is where a compound can lose activity for
    reasons no mechanism knob represents. ``which='top'`` asks what the scores
    look like without it; ``'bottom'`` and ``'middle'`` are the controls that
    say whether dropping *any* one bin would do as much. Decided after the real
    curves were seen, so none of this is a test.
    """
    trimmed = {}
    for name, result in results.items():
        if not result.get("scored"):
            continue
        bins = list(result["retained_bins"])
        index = {"top": len(bins) - 1, "bottom": 0, "middle": len(bins) // 2}[which]
        kept = bins[:index] + bins[index + 1 :]
        if len(kept) < MIN_BINS:
            continue
        trimmed[name] = {**result, "retained_bins": kept}
    return rescore(trimmed, twin, LEVELS)


def analyse(wells: pd.DataFrame, viability: pd.DataFrame, twin: dict, spids: dict) -> dict:
    """Every registered compound that the screen tested, in registration order."""
    groups = {f: vehicle_groups(wells, acid) for f, acid in MAPPING.items()}
    groups["viability"] = viability_groups(viability)
    cache: dict = {}  # vehicle-null cuts repeat across compounds; drawn once each
    out = {}
    for name, label in compounds().items():
        if not spids.get(name):
            out[name] = {"label": label, "scored": False, "dropped": "not in the screen"}
            continue
        log.info("  %s (%s)", name, label)
        out[name] = analyse_compound(
            label, wells, viability, twin, groups, spids[name], cache=cache
        )
    return out


def viability_groups(viability: pd.DataFrame) -> list[np.ndarray]:
    """Per exposure date, log2 of the vehicle wells' viability signal."""
    rows = viability[(viability["wllt"] == "n") & (viability["rval"] > 0)]
    frame = pd.DataFrame(
        {"apid": rows["apid"].to_numpy(), "x": np.log2(rows["rval"].to_numpy(dtype=float))}
    )
    frame = frame[np.isfinite(frame["x"])]
    return [group["x"].to_numpy() for _, group in frame.groupby("apid")]


def first_mover_table(results: dict) -> dict:
    """Per scored compound, the first-moving readout in both curves."""
    return {n: r["first_mover"] for n, r in results.items() if r.get("scored")}


def untestable_bins(results: dict) -> dict:
    """Bins the vehicle null cannot judge: no exposure date holds enough vehicle wells.

    The registered null draws ``n`` pseudo-treated wells from a date that also
    keeps three others, so a bin pooling more wells than any single date's
    vehicle set can supply gets no threshold and cannot be called moving. Counted
    here rather than left silent, because it only ever makes the test stricter.
    """
    out = {}
    for name, result in results.items():
        if "real" not in result:
            continue
        missing = {
            feature: sorted({p["bin"] for p in curve if p["wells"] > 0 and p["threshold"] is None})
            for feature, curve in result["real"].items()
        }
        missing = {f: b for f, b in missing.items() if b}
        if missing:
            out[name] = missing
    return out


# ------------------------------------------------------------------------- figure


COLOURS = {"twin": "#D55E00", "real": "#0072B2", "follow": "#009E73", "shuffled": "#999999"}


def plot(scored: dict, results: dict, path: Path) -> None:
    """Spearman per scored pair, and the best-scoring pair's two curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = sorted(scored["pairs"], key=lambda p: p["spearman"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, max(3.4, 0.22 * len(pairs) + 1.6)))
    y = np.arange(len(pairs))
    left.barh(y, [p["spearman"] for p in pairs], color=COLOURS["twin"], height=0.7)
    left.set_yticks(
        y, [f"{p['compound'][:22]} · {p['feature'].replace('_', ' ')}" for p in pairs], fontsize=6
    )
    left.axvline(0, color="grey", lw=0.6)
    left.axvline(HIT_RHO, color="black", lw=0.8, ls=":", label=f"registered hit line {HIT_RHO}")
    for key, name in (("follow_firing_rate", "follow"), ("shuffled_mechanism", "shuffled")):
        mean = scored["nulls"][key]["mean_spearman"]
        if mean is not None:
            left.axvline(
                mean, color=COLOURS[name], lw=1.0, ls="--", label=f"{key.replace('_', ' ')} mean"
            )
    left.set_xlabel(
        "Spearman, SIMULATED twin curve vs REAL curve across concentrations", fontsize=7
    )
    left.tick_params(labelsize=7)
    left.legend(fontsize=6, frameon=False, loc="lower right")
    _plot_example(right, pairs, results)
    fig.suptitle(
        "SIMULATED twin knob dose-response vs REAL ToxCast concentration-response wells "
        "(rat cortex, same-date vehicle reference)",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_example(axis, pairs: list[dict], results: dict) -> None:
    """The highest-scoring pair's real and matched twin curves against concentration."""
    if not pairs:
        axis.set_axis_off()
        return
    best = pairs[-1]
    result = results[best["compound"]]
    real = result["real"][best["feature"]]
    retained = result["retained_bins"]
    matched = _matched_with_bins(
        {result["knob"]: {best["feature"]: _levels_of(result, best["feature"])}},
        result["knob"],
        best["feature"],
        [p["bin"] for p in real],
        LEVELS,
    )
    x = [p["bin"] / 2 for p in real]
    axis.axhline(0, color="grey", lw=0.6)
    axis.plot(
        x,
        [p["median"] for p in real],
        "o-",
        ms=4,
        color=COLOURS["real"],
        label="REAL wells (median vs same-date vehicle)",
    )
    axis.plot(
        x,
        [m["median"] for m in matched],
        "s--",
        ms=4,
        color=COLOURS["twin"],
        label="SIMULATED twin, knob levels matched by rank",
    )
    _style_example(axis, best, real, retained)


def _style_example(axis, best: dict, real: list[dict], retained: list[int]) -> None:
    """Mark the excluded bins and label the example panel."""
    for point in real:
        if point["bin"] not in retained:
            axis.axvspan(
                point["bin"] / 2 - 0.25, point["bin"] / 2 + 0.25, color="#CCCCCC", alpha=0.5
            )
    dropped = [p["bin"] for p in real if p["bin"] not in retained]
    note = (
        "grey bands excluded by the viability rule"
        if dropped
        else "the viability rule excluded no concentration"
    )
    axis.set_xlabel(f"log10 concentration (uM); {note}", fontsize=7)
    axis.set_ylabel("log2 change", fontsize=7)
    axis.set_title(
        f"{best['compound']} · {best['feature'].replace('_', ' ')} "
        f"(Spearman {best['spearman']:.2f})",
        fontsize=8,
    )
    axis.tick_params(labelsize=7)
    axis.legend(fontsize=6, frameon=False)


def _levels_of(result: dict, feature: str) -> list[dict]:
    """The per-level twin summaries this result was scored against."""
    return result["twin_levels"][feature]


# ---------------------------------------------------------------------------- main


def load_or_simulate(predict: bool) -> dict:
    """The registered twin curves: the committed file, or a fresh simulation."""
    from neuromps.parallel import worker_processes
    from neuromps.paths import for_reading

    path = twin_dir() / PREDICTIONS_NAME
    if not predict and for_reading(path).exists():
        return json.loads(for_reading(path).read_text())
    changes = simulate_changes(worker_processes(cap=5))
    record = {
        "label": "SIMULATED: the calibrated twin's paired changes over each knob's level grid",
        "registration": provenance(REGISTRATION),
        "knob_grids": {k: {"parameter": p, "levels": list(g)} for k, (p, g) in KNOB_GRIDS.items()},
        "build_seeds": list(BUILD_SEEDS),
        "duration_s": DURATION_S,
        "per_chip": json.loads(changes.to_json(orient="records")),
        "curves": twin_curves(changes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_wells() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The screen's electrophysiology wells, its viability wells, and sample ids per compound."""
    wells = pd.read_parquet(WELLS, columns=["acid", "wllt", "spid", "apid", "conc", "rval"])
    viability = pd.read_parquet(
        VIABILITY_WELLS, columns=["acid", "wllt", "spid", "apid", "conc", "rval", "wllq"]
    )
    viability = viability[(viability["acid"] == VIABILITY_ACID) & (viability["wllq"] == 1)]
    samples = pd.read_parquet(SAMPLES, columns=["spid", "chnm"])
    spids = {
        name: sorted(samples.loc[samples["chnm"] == name, "spid"].unique()) for name in compounds()
    }
    return wells, viability, spids


def _headline(scored: dict, predictions: dict, results: dict) -> dict:
    """The few numbers the report cites."""
    return {
        "compounds_scored": sum(1 for r in results.values() if r.get("scored")),
        "compounds_registered": len(results),
        "pairs_scored": scored["twin"]["pairs"],
        "hits": scored["twin"]["hits"],
        "hit_rate": scored["twin"]["hit_rate"],
        "mean_spearman": scored["twin"]["mean_spearman"],
        "predictions_passing": sum(1 for p in predictions.values() if p["pass"]),
        "predictions_total": len(predictions),
    }


def _posthoc_top_bin(results: dict, curves: dict) -> dict:
    """The post-hoc rescore without each compound's highest concentration.

    With the controls that drop one other bin instead: if they lift the score as
    much, the top concentration is not what the registered test tripped over.
    """
    out = {"note": "POST HOC, decided after the registered viability rule excluded no bin"}
    for which in ("top", "bottom", "middle"):
        got = collect(rescore_dropping(results, curves, which))
        out[f"dropping_{which}_bin"] = {"twin": got["twin"], "nulls": got["nulls"]}
    return out


def build_payload(results: dict, scored: dict, curves: dict) -> dict:
    """Everything this check writes to metrics.json, registered tests included."""
    first_mover = first_mover_table(results)
    predictions = evaluate(scored["twin"], scored["nulls"], first_mover)
    check = collect(rescore(results, curves, TRUNCATED_LEVELS))
    return {
        "label": "SIMULATED twin knob dose-response vs REAL ToxCast concentration-response wells",
        "registration": provenance(REGISTRATION),
        "predictions_file": str(Path("outputs/twin") / PREDICTIONS_NAME),
        "headline": _headline(scored, predictions, results),
        "twin": scored["twin"],
        "nulls": scored["nulls"],
        "predictions": predictions,
        "first_mover": first_mover,
        "firing_rate_sign": {
            n: r["firing_rate_sign"] for n, r in results.items() if r.get("scored")
        },
        "sensitivity_truncated_grid": {
            "levels": TRUNCATED_LEVELS,
            "twin": check["twin"],
            "nulls": check["nulls"],
        },
        "posthoc_without_top_bin": _posthoc_top_bin(results, curves),
        "dropped": {n: r["dropped"] for n, r in results.items() if r.get("dropped")},
        "untestable_bins": untestable_bins(results),
        "viability_untestable_bins": {
            n: r["exclusion"]["untestable_bins"]
            for n, r in results.items()
            if r.get("exclusion", {}).get("untestable_bins")
        },
        "compounds": results,
        "pairs": scored["pairs"],
    }


def main(argv: list[str] | None = None) -> int:
    """--predict: simulate and write the twin's curves only; else score them."""
    from neuromps.metrics import update_metrics
    from neuromps.paths import figures_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", action="store_true", help="simulate the twin's curves only")
    parser.add_argument(
        "--quick", action="store_true", help="accepted for run_all; nothing changes"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.predict and not WELLS.exists():
        log.warning("%s not built; skipping the dose-response check", WELLS)
        return 0
    record = load_or_simulate(args.predict)
    if args.predict:
        print(json.dumps({k: v["mfr_hz"] for k, v in record["curves"].items()}, indent=1))
        return 0
    curves = record["curves"]
    wells, viability, spids = load_wells()
    results = analyse(wells, viability, curves, spids)
    for result in results.values():
        if result.get("scored"):
            result["twin_levels"] = {f: curves[result["knob"]][f] for f in MAPPING}
    scored = collect(results)
    payload = build_payload(results, scored, curves)
    figures_dir().mkdir(parents=True, exist_ok=True)
    plot(scored, results, figures_dir() / "twin_toxcast_dose_response.png")
    # The twin's per-level curves live in the committed predictions file; the
    # figure needed them here, but metrics.json keeps the results, not a copy.
    for result in results.values():
        result.pop("twin_levels", None)
    update_metrics(ROOT, payload)
    print(json.dumps({k: payload[k] for k in ("headline", "nulls", "predictions")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
