"""Embodied fly, stage 4e: the walking readout as the fraction of rhythmic draws.

``python -m neuromps.fly.embodied_stage4e`` runs the registered design
(`preregistration/embodied_stage4e_walking_fraction.json`): stage 4d on fresh draws, walking
scored per condition by the fraction of draws with a rhythm (corrected score >= 0.3), with
stage 4d's median readout reported beside it. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from neuromps.fly import design, scoring
from neuromps.fly import embodied_stage4a as s4a
from neuromps.fly import embodied_stage4c as s4c
from neuromps.fly import embodied_stage4d as s4d
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage4e_walking_fraction.json"
SEEDS = list(range(1500, 1532))
RHYTHM = 0.3
PREDICTIONS = (
    "E0_the_walking_rhythm_is_there",
    "E1_acetylcholine_block_is_caught_first",
    "E2_gaba_and_glucl_block_break_walking_first",
    "E3_gaba_potentiation_spares_walking",
)


def fractions(trials: pd.DataFrame) -> pd.Series:
    """Per condition: the fraction of feasible draws with a rhythm."""
    use = trials[trials["feasible"]]
    return use.groupby("condition")["score"].apply(lambda s: float((s >= RHYTHM).mean()))


def onsets(conds: pd.DataFrame, readout: pd.Series) -> dict:
    """Per family: the walking onset by a readout's disruption."""
    index = s4d.walking_disruption(readout).loc[conds["condition"]].to_numpy()
    return s4a.onset(conds.assign(wd=index), "wd", s4a.IMPAIR)


def score(
    conds: pd.DataFrame, frac: pd.Series, feasible: int, chip: dict, spec: dict
) -> tuple[dict, dict]:
    """E0-E3 and the resolutions they rest on."""
    walking = onsets(conds, frac)
    res = {f: s4c.resolution(walking.get(f), chip.get(f)) for f in s4d.FAMILIES}
    e0 = feasible >= 16 and frac["vehicle"] >= 0.3
    found = {
        PREDICTIONS[0]: ({"feasible": feasible, "fraction": round(frac["vehicle"], 4)}, e0),
        PREDICTIONS[1]: (
            {"walking_onset": walking.get("ach_block"), "chip_onset": chip.get("ach_block")},
            res["ach_block"] == "chip earlier",
        ),
        PREDICTIONS[2]: (
            {f: res[f] for f in ("gaba_block", "glucl_block")},
            res["gaba_block"] == res["glucl_block"] == "chip later",
        ),
        PREDICTIONS[3]: (
            {"walking_onset": walking.get("gaba_potentiate")},
            walking.get("gaba_potentiate") is None,
        ),
    }
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        if name != PREDICTIONS[0] and not e0:
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    return out, {"walking_onsets": walking, "resolutions": res}


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-4e design."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    conds = s4d.conditions()
    trials, ok, inputs, runs = s4d.run_trials(conds, SEEDS)
    frac = fractions(trials)
    median = trials[trials["feasible"]].groupby("condition")["score"].median()
    chip_table = s4d._chip_series()
    chip = s4a.onset(chip_table, "abs_delta_mfr", s4a.CHIP)
    predictions, details = score(conds, frac, int(ok.sum()), chip, spec)
    by_median = onsets(conds, median)
    moves = s4d._movement_medians()
    four = s4a.disruption(moves.assign(walking=frac.loc[moves.index]), s4a.EPS_MM)
    four_onset = s4a.onset(
        conds.assign(md=four.loc[conds["condition"]].to_numpy()), "md", s4a.IMPAIR
    )
    result = {
        "label": "SIMULATED walking rhythm under drug knobs, fraction of rhythmic draws",
        "predictions": predictions,
        **details,
        "chip_onsets": chip,
        "median_readout": {
            "walking_onsets": by_median,
            "resolutions": {f: s4c.resolution(by_median.get(f), chip.get(f)) for f in s4d.FAMILIES},
        },
        "four_movement_with_fraction": {
            "onsets": four_onset,
            "resolutions": {
                f: s4c.resolution(four_onset.get(f), chip.get(f)) for f in s4d.FAMILIES
            },
        },
        "levels": {
            c: {"fraction": round(float(frac[c]), 4), "median_score": round(float(median[c]), 4)}
            for c in conds["condition"]
        },
        "vehicle_inputs": {"feasible": int(ok.sum()), "runs": int(runs)},
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    trials.to_csv(out / "embodied_stage4e_SIMULATED.csv", index=False)
    (out / "embodied_stage4e_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage4e", scoring.clean(result))
    for name, entry in predictions.items():
        print(f"{name:48s} {entry['outcome']:>15}  {entry['value']}")
    print("resolutions", details["resolutions"])
    print("median readout", json.dumps(scoring.clean(result["median_readout"])))
    print("four movement", json.dumps(scoring.clean(result["four_movement_with_fraction"])))
    print("levels", json.dumps(result["levels"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
