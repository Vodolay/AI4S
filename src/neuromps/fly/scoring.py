"""Turn the sweep's raw summaries into tables, and score each registered prediction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neuromps.fly import analysis, design, direction
from neuromps.paths import metrics_path
from neuromps.twin.perturbation import (
    COMPARE_FEATURES,
    agreement_table,
    real_directions,
    simulated_delta,
)


def _mean(values) -> float | None:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def _chip(raw: dict) -> dict[str, dict]:
    by: dict[str, list] = {}
    for cond, _, got in raw["spontaneous"]:
        by.setdefault(cond.name, []).append(got["features"])
    return {
        name: {k: _mean([f.get(k) for f in fs]) for k in fs[0] if k != "silent"}
        for name, fs in by.items()
    }


def _behaviour(raw: dict) -> dict[str, dict]:
    rows = pd.DataFrame(
        [{"condition": c.name, "protocol": p, **got} for c, p, _, got in raw["behaviour"]]
    )
    m = rows.groupby(["condition", "protocol"]).mean(numeric_only=True)
    return {
        name: {
            "mn9_sugar": float(m.loc[(name, "sugar"), "mn9"]),
            "mn9_sugar_bitter": float(m.loc[(name, "sugar_bitter"), "mn9"]),
            "abn1_jon_ce": float(m.loc[(name, "jon_ce"), "abn1"]),
            "adn1_jon_ce": float(m.loc[(name, "jon_ce"), "adn1"]),
            "dn_recruited": float(m.loc[(name, "sugar"), "dn_recruited"]),
        }
        for name in rows["condition"].unique()
    }


def tables(raw: dict, context) -> dict:
    """Per-condition chip and behaviour table, and the direction measurements."""
    chip, animal = _chip(raw), _behaviour(raw)
    rows = []
    for cond in design.conditions(context.spec):
        if cond.name not in chip:
            continue
        delta = simulated_delta(chip[cond.name], chip["vehicle"])
        rows.append(
            {
                "condition": cond.name,
                "family": cond.family,
                "level": cond.level,
                "dose": cond.dose,
                **{f"chip.{k}": v for k, v in chip[cond.name].items()},
                **{f"delta.{k}": v for k, v in delta.items()},
                **animal[cond.name],
                "disruption": analysis.disruption(animal[cond.name], animal["vehicle"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["abs_delta_mfr"] = frame["delta.mfr_hz"].abs()
    return {
        "conditions": frame,
        "chip": chip,
        "direction": _direction(raw, context),
        "replication": {"783": raw["replication_783"], "630": raw["replication_630"]},
    }


def _direction(raw: dict, context) -> dict:
    pairs, strong = context.pairs, context.spec["design"]["pairs"]["strong_asymmetry"]
    out = {"n_pairs": int(len(pairs))}
    vehicle = [got for c, _, got in raw["spontaneous"] if c.family == "vehicle"]
    for label in ("whole_region", "array"):
        spont = np.nanmean([g[f"asymmetry_{label}"] for g in vehicle], axis=0)
        response: dict[str, dict[str, list]] = {}
        for region, _, got in raw["evoked"]:
            for target, value in got[label].items():
                response.setdefault(region, {}).setdefault(target, []).append(value)
        mean_resp = {a: {b: float(np.mean(v)) for b, v in d.items()} for a, d in response.items()}
        evoked = direction.evoked_asymmetry(mean_resp, pairs)
        out[label] = {
            "spontaneous": direction.score(pairs, spont, strong),
            "evoked": direction.score(pairs, evoked, strong),
            "points": {
                "truth": pairs["asymmetry"].round(4).tolist(),
                "spontaneous": np.round(spont, 4).tolist(),
                "evoked": np.round(evoked, 4).tolist(),
            },
        }
    return out


def _outcome(passed: bool | None) -> str:
    return "n/a" if passed is None else ("pass" if passed else "fail")


def _family(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    return frame[frame["family"].isin(["vehicle", family])].sort_values("dose")


def _trend(frame: pd.DataFrame, family: str, column: str) -> float | None:
    rows = _family(frame, family)
    return analysis.trend(rows["dose"], rows[column])


def _behaviour_predictions(frame: pd.DataFrame) -> dict:
    b1 = _trend(frame, "gaba_block", "mn9_sugar_bitter")
    b2 = [_trend(frame, "ach_block", c) for c in ("mn9_sugar", "abn1_jon_ce")]
    b3 = _trend(frame, "gaba_block", "dn_recruited")
    return {
        "B1_disinhibition_releases_feeding_from_bitter": (b1, b1 is not None and b1 >= 0.8),
        "B2_excitation_block_silences_motor_output": (
            {"mn9_sugar": b2[0], "abn1_jon_ce": b2[1]},
            all(v is not None and v <= -0.8 for v in b2),
        ),
        "B3_disinhibition_recruits_off_target_descending_neurons": (
            b3,
            b3 is not None and b3 >= 0.8,
        ),
    }


def _chip_predictions(frame: pd.DataFrame, d: dict) -> dict:
    drugs = frame[frame["family"] != "vehicle"]
    rho = spearmanr(drugs["abs_delta_mfr"], drugs["disruption"])[0]
    sentinel = analysis.sentinel(
        drugs, d["chip_detection"]["abs_log2_threshold"], d["behaviour_impairment_threshold"]
    )
    scored = [v["chip_first"] for v in sentinel.values() if v["chip_first"] is not None]
    return {
        "C1_chip_tracks_disruption": (float(rho), bool(rho >= 0.6)),
        "C2_chip_is_the_earlier_sentinel": (sentinel, bool(scored) and all(scored)),
    }


def agreement(frame: pd.DataFrame, chip: dict, spec: dict, twin: dict | None) -> dict:
    """Fly array vs real chips per matched mechanism, beside the chip twin's own score."""
    from neuromps.models.dataset import load_pairs

    pairs = load_pairs()
    rows, fly_total, twin_total = {}, 0, 0
    for family, entry in spec["design"]["conditions"].items():
        mechanism = entry["chip_mechanism"]
        if mechanism is None:
            continue
        name = design.Condition(family, entry["reference_level"]).name
        delta = simulated_delta(chip[name], chip["vehicle"])
        table = agreement_table(delta, real_directions(pairs, mechanism, 1000, seed=0))
        fly = int(sum(v["agrees"] is True for v in table.values()))
        twin_n = (twin or {}).get(mechanism)
        rows[family] = {
            "mechanism": mechanism,
            "condition": name,
            "fly_agreeing": fly,
            "fly_scored": int(sum(v["agrees"] is not None for v in table.values())),
            "twin_agreeing": twin_n,
            "table": table,
        }
        fly_total += fly
        twin_total += twin_n or 0
    return {"by_family": rows, "fly_total": fly_total, "twin_total": twin_total}


def twin_agreement(path=None) -> dict | None:
    """The chip twin's HD13 agreements per mechanism, from the metrics the twin stage wrote."""
    try:
        knobs = json.loads((path or metrics_path()).read_text())["twin_hd13_perturbation"]["knobs"]
    except (OSError, KeyError):
        return None
    return {k["mechanism"]: int(k["n_features_agreeing"]) for k in knobs}


def _direction_predictions(direction_tables: dict) -> dict:
    whole, array = direction_tables["whole_region"], direction_tables["array"]
    spont = whole["spontaneous"]["sign_agreement_strong"]
    evoked = whole["evoked"]["sign_agreement_strong"]
    arr = array["evoked"]["sign_agreement_strong"]
    return {
        "D1_spontaneous_direction_is_near_chance": (spont, spont is not None and spont < 0.65),
        "D2_pulsed_stimulation_recovers_direction": (
            {"evoked": evoked, "spontaneous": spont},
            evoked is not None and spont is not None and evoked >= 0.70 and evoked - spont >= 0.10,
        ),
        "D3_an_electrode_array_sees_it_too": (arr, arr is not None and arr >= 0.65),
    }


def score(tables_: dict, context, spec: dict, scored: bool) -> dict:
    """Every registered prediction with its value and outcome (n/a on a quick run)."""
    frame, d = tables_["conditions"], spec["design"]
    rep = tables_["replication"]
    ratio = float(np.mean(rep["783"]) / np.mean(rep["630"]))
    found = {
        "R1_replication": (
            {
                "mn9_783_hz": float(np.mean(rep["783"])),
                "mn9_630_hz": float(np.mean(rep["630"])),
                "ratio": ratio,
            },
            abs(ratio - 1) <= 0.20,
        )
    }
    if scored:
        found.update(_behaviour_predictions(frame))
        found.update(_chip_predictions(frame, d))
        agree = agreement(frame, tables_["chip"], spec, twin_agreement())
        found["A1_direction_agreement_with_real_chips"] = (
            {"fly": agree["fly_total"], "twin": agree["twin_total"]},
            agree["fly_total"] >= agree["twin_total"],
        )
        found.update(_direction_predictions(tables_["direction"]))
    predictions = {
        name: {
            "test": spec["predictions"][name]["test"],
            "value": value,
            "outcome": _outcome(bool(ok) if scored or name == "R1_replication" else None),
        }
        for name, (value, ok) in found.items()
    }
    out = {
        "label": "SIMULATED fly brain (FlyWire 783) vs chip readouts",
        "quick": not scored,
        "predictions": predictions,
        "direction": tables_["direction"],
        "exploratory": exploratory(frame, spec if scored else None),
    }
    if scored:
        out["agreement"] = agree
        out["sensitivity"] = sensitivity(frame, spec)
    return out


def sensitivity(frame: pd.DataFrame, spec: dict) -> dict:
    """Registered: the sentinel test at the two sensitivity thresholds of the chip readout."""
    d = spec["design"]
    drugs = frame[frame["family"] != "vehicle"]
    return analysis.sentinel_sensitivity(
        drugs, d["chip_detection"]["sensitivity_thresholds"], d["behaviour_impairment_threshold"]
    )


def _leave_one_out(frame: pd.DataFrame, spec: dict) -> dict:
    """The sentinel test with each behaviour readout left out of the disruption index."""
    d = spec["design"]
    vehicle = frame[frame["family"] == "vehicle"].iloc[0]
    out = {}
    for readout in analysis.READOUTS:
        rows = frame[frame["family"] != "vehicle"].copy()
        rows["disruption"] = [
            analysis.disruption(r, vehicle, leave_out=readout) for _, r in rows.iterrows()
        ]
        verdict = analysis.sentinel(
            rows, d["chip_detection"]["abs_log2_threshold"], d["behaviour_impairment_threshold"]
        )
        out[f"without_{readout}"] = {k: v["chip_first"] for k, v in verdict.items()}
    return out


def _earliest_feature(frame: pd.DataFrame, threshold: float) -> dict:
    """Per family and chip feature, the lowest dose whose |log2 change| reaches the threshold."""
    out = {}
    for family, rows in frame[frame["family"] != "vehicle"].groupby("family"):
        out[family] = {}
        for feature in COMPARE_FEATURES:
            hit = rows[rows[f"delta.{feature}"].abs() >= threshold]
            out[family][feature] = None if hit.empty else float(hit["dose"].min())
    return out


def exploratory(frame: pd.DataFrame, spec: dict | None = None) -> dict:
    """Not registered: which chip feature tracks disruption, and how robust the sentinel is."""
    drugs = frame[frame["family"] != "vehicle"]
    out = {}
    for feature in COMPARE_FEATURES:
        values = drugs[f"delta.{feature}"].abs()
        ok = values.notna()
        if ok.sum() > 3:
            rho = spearmanr(values[ok], drugs.loc[ok, "disruption"])[0]
            out[feature] = None if not np.isfinite(rho) else round(float(rho), 3)
    result = {"spearman_abs_delta_vs_disruption": out}
    if spec is not None:
        threshold = spec["design"]["chip_detection"]["abs_log2_threshold"]
        result["sentinel_chip_first_leaving_one_readout_out"] = _leave_one_out(frame, spec)
        result["first_detecting_dose_by_chip_feature"] = _earliest_feature(frame, threshold)
    return result


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return clean(value.item())
    return value


def metrics_payload(result: dict) -> dict:
    """What lands in metrics.json: predictions, scores and provenance, not the point clouds."""
    payload = {k: v for k, v in result.items() if k != "direction"}
    payload["direction"] = {
        label: {m: result["direction"][label][m] for m in ("spontaneous", "evoked")}
        for label in ("whole_region", "array")
    }
    payload["direction"]["n_pairs"] = result["direction"]["n_pairs"]
    if "agreement" in payload:
        payload["agreement"] = {k: v for k, v in payload["agreement"].items()}
    return clean(payload)


def model_sizes() -> dict:
    """Neurons, connections and synapses of the two fly-brain models, as built."""
    from neuromps.fly import connectome, malecns

    out = {}
    for name, con in (("flywire783", connectome.cached()), ("malecns", malecns.cached()[0])):
        out[name] = {
            "neurons": con.n_neurons,
            "connections": int(con.pre.size),
            "synapses": int(con.synapses.sum()),
            "inhibitory_connections": int((con.sign < 0).sum()),
        }
    out["flywire783"]["release_coverage"] = connectome.release_coverage()
    return out
