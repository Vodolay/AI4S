"""Embodied fly, stage 4b: stage 4a's headline on fresh seeds.

``python -m neuromps.fly.embodied_stage4b`` runs the registered design
(`preregistration/embodied_stage4b_movement_replication.json`): stage 4a's movements on 16
fresh seeds, their onsets against the chip detections of the parent design's fresh-seed
replication, and the onsets with each movement left out of the disruption index.
Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from neuromps.fly import design, scoring
from neuromps.fly import embodied_stage4a as s4a
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage4b_movement_replication.json"
SEEDS = list(range(1360, 1376))
MIN_FAMILIES = 4
PREDICTIONS = (
    "R0_the_movements_are_there",
    "R1_the_chip_is_late_in_every_family",
    "R2_no_single_movement_carries_it",
    "R3_movement_onsets_reproduce",
)


def chip_onsets(name: str) -> dict:
    """Per family: the chip's first detection dose in a recorded conditions table."""
    table = pd.read_csv(outputs_dir() / "fly" / f"{name}_SIMULATED.csv")
    return s4a.onset(table, "abs_delta_mfr", s4a.CHIP)


def movement_onsets(
    conds: pd.DataFrame, medians: pd.DataFrame, keep: list[str], eps: float = s4a.EPS_MM
) -> dict:
    """Per family: the movement onset with the disruption index over `keep` only."""
    index = s4a.disruption(medians[keep], eps).loc[conds["condition"]].to_numpy()
    return s4a.onset(conds.assign(md=index), "md", s4a.IMPAIR)


def late(movement: dict, chip: dict) -> dict:
    """Per family with a movement onset: is the chip's first detection strictly later?"""
    return {
        f: chip.get(f) is None or chip[f] > d + 1e-9 for f, d in movement.items() if d is not None
    }


def dose_steps(conds: pd.DataFrame) -> dict:
    """Per family: its doses in increasing order."""
    rows = conds[conds["family"] != "vehicle"]
    return {f: sorted(round(float(d), 6) for d in g["dose"]) for f, g in rows.groupby("family")}


def within_one_step(doses: list[float], a: float | None, b: float | None) -> bool:
    """Are two onsets equal or adjacent in a family's dose list (both None counts as equal)?"""
    if a is None or b is None:
        return a is None and b is None
    k = [round(d, 6) for d in doses]
    return abs(k.index(round(a, 6)) - k.index(round(b, 6))) <= 1


def score(
    conds: pd.DataFrame, medians: pd.DataFrame, spec: dict, chip: dict, previous: dict
) -> tuple[dict, dict]:
    """R0-R3 and the onsets they rest on."""
    families = sorted(set(conds["family"]) - {"vehicle"})
    first = movement_onsets(conds, medians, list(s4a.MOVEMENTS))
    left_out = {
        m: movement_onsets(conds, medians, [k for k in s4a.MOVEMENTS if k != m])
        for m in s4a.MOVEMENTS
    }
    vehicle = medians.loc["vehicle"]
    r0 = vehicle["proboscis_sugar"] >= 0.1 and vehicle["reach_depth"] >= 0.3
    r1 = late(first, chip)
    r2 = {m: late(o, chip) for m, o in left_out.items()}
    steps = dose_steps(conds)
    r3 = {f: within_one_step(steps[f], first.get(f), previous.get(f)) for f in families}
    found = {
        PREDICTIONS[0]: (
            {k: round(float(vehicle[k]), 4) for k in ("proboscis_sugar", "reach_depth")},
            r0,
        ),
        PREDICTIONS[1]: (r1, len(r1) == len(families) and all(r1.values())),
        PREDICTIONS[2]: (
            r2,
            all(len(v) >= MIN_FAMILIES and all(v.values()) for v in r2.values()),
        ),
        PREDICTIONS[3]: (r3, all(r3.values())),
    }
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        if name != PREDICTIONS[0] and not r0:
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    details = {
        "onsets": {"movement": first, "stage4a_movement": previous, "chip_replication": chip},
        "onsets_leave_one_out": left_out,
        "movement_onset_eps_005mm": movement_onsets(
            conds, medians, list(s4a.MOVEMENTS), s4a.EPS_SENSITIVITY_MM
        ),
    }
    return out, details


def _reported(conds: pd.DataFrame, medians: pd.DataFrame) -> dict:
    """Stage 4a's N1 and N2 Spearman values on these seeds, and the disruption index."""
    index = s4a.disruption(medians, s4a.EPS_MM)
    return {
        "N1_spearman_gaba_block_sugar_bitter": round(
            s4a._spearman(
                *s4a._family_series(conds, medians, "gaba_block", "proboscis_sugar_bitter")
            ),
            4,
        ),
        "N2_spearman_ach_block": {
            m: round(s4a._spearman(*s4a._family_series(conds, medians, "ach_block", m)), 4)
            for m in ("proboscis_sugar", "reach_depth")
        },
        "movement_disruption": {c: round(float(index[c]), 4) for c in conds["condition"]},
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-4b design."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    conds = s4a.conditions()
    previous = json.loads((outputs_dir() / "fly" / "embodied_stage4a_SIMULATED.json").read_text())
    trials = s4a.run_trials(conds, SEEDS)
    medians = s4a.movement_medians(trials)
    chip = chip_onsets("confirm_conditions")
    predictions, details = score(conds, medians, spec, chip, previous["onsets"]["movement"])
    details["onsets"]["chip_first_run"] = chip_onsets("conditions")
    result = {
        "label": "SIMULATED drug knobs with movement as the readout, fresh seeds",
        "predictions": predictions,
        **details,
        "reported": _reported(conds, medians),
        "medians": {
            c: {m: round(float(medians.loc[c, m]), 4) for m in s4a.MOVEMENTS} for c in medians.index
        },
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    trials.to_csv(out / "embodied_stage4b_SIMULATED.csv", index=False)
    (out / "embodied_stage4b_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage4b", scoring.clean(result))
    for name, entry in predictions.items():
        print(f"{name:40s} {entry['outcome']:>15}  {entry['value']}")
    print("onsets", json.dumps(scoring.clean(result["onsets"])))
    print("leave one out", json.dumps(scoring.clean(result["onsets_leave_one_out"])))
    print("reported", json.dumps(scoring.clean(result["reported"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
