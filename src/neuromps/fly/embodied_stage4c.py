"""Embodied fly, stage 4c: a finer dose grid inside stage 4b's one-step margins.

``python -m neuromps.fly.embodied_stage4c`` runs the registered design
(`preregistration/embodied_stage4c_fine_dose_grid.json`): GABA-A potentiation and
excitability block at finer levels, the movements on stage 4b's seeds and the chip on both
of the parent design's seed sets, with the coarse levels rerun as a reproduction check.
Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import connectome, design, jobs, scoring
from neuromps.fly import embodied_stage4a as s4a
from neuromps.fly import embodied_stage4b as s4b
from neuromps.fly.study import _context, memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage4c_fine_dose_grid.json"
CHIP_RUNS = {"replication": ("confirm_conditions", 700), "first_run": ("conditions", 100)}
CHIP_SEEDS = 3
PREDICTIONS = (
    "F0_coarse_levels_reproduce",
    "F1_gaba_potentiation_chip_later",
    "F2_without_the_bitter_gate_chip_later",
    "F3_excitability_block_chip_later_for_both_chip_runs",
)


def levels(spec: dict) -> pd.DataFrame:
    """Vehicle, the coarse levels rerun as a check, then the fine levels, with their knobs."""
    d = spec["design"]
    knobs = {f: e["knob"] for f, e in d["fine_levels"].items()}
    out = [design.Condition("vehicle", 1.0, {})]
    for family, entry in d["fine_levels"].items():
        coarse = d["reproducibility_levels"][family]
        for level in [*coarse, *entry["levels"]]:
            out.append(design.Condition(family, level, {knobs[family]: float(level)}))
    return pd.DataFrame(
        {
            "condition": [c.name for c in out],
            "family": [c.family for c in out],
            "level": [c.level for c in out],
            "dose": [c.dose for c in out],
            "knob": [c.overrides for c in out],
        }
    )


def resolution(movement: float | None, chip: float | None) -> str:
    """Which came first: 'chip later', 'chip earlier' or 'still tied'."""
    if movement is not None and (chip is None or movement < chip - 1e-9):
        return "chip later"
    if chip is not None and (movement is None or chip < movement - 1e-9):
        return "chip earlier"
    return "still tied"


def combine(coarse: pd.DataFrame, fine: pd.DataFrame) -> pd.DataFrame:
    """Coarse rows as recorded, with the fine rows whose condition is new."""
    return pd.concat([coarse, fine.loc[~fine.index.isin(coarse.index)]])


def resolve_family(
    conds: pd.DataFrame, medians: pd.DataFrame, chip: pd.DataFrame, family: str, keep: list[str]
) -> dict:
    """One family's movement onset (index over `keep`), chip onset and resolution."""
    m = s4b.movement_onsets(conds, medians, keep).get(family)
    c = s4a.onset(chip, "abs_delta_mfr", s4a.CHIP).get(family)
    return {
        "movement_onset": None if m is None else round(m, 4),
        "chip_onset": None if c is None else round(c, 4),
        "resolution": resolution(m, c),
    }


def _indices() -> dict[str, list[str]]:
    moves = list(s4a.MOVEMENTS)
    return {"all_three": moves} | {f"without_{m}": [k for k in moves if k != m] for m in moves}


def score(
    spec: dict, conds: pd.DataFrame, medians: pd.DataFrame, chips: dict, f0: float
) -> tuple[dict, dict]:
    """F0-F3 and every resolution they rest on."""
    res = {
        index: {
            run: {
                f: resolve_family(conds, medians, chips[run], f, keep)
                for f in ("gaba_potentiate", "excitability_block")
            }
            for run in chips
        }
        for index, keep in _indices().items()
    }
    later = lambda index, run, f: res[index][run][f]["resolution"] == "chip later"  # noqa: E731
    found = {
        PREDICTIONS[0]: (f0, f0 <= 1e-9),
        PREDICTIONS[1]: (
            res["all_three"]["replication"]["gaba_potentiate"],
            later("all_three", "replication", "gaba_potentiate"),
        ),
        PREDICTIONS[2]: (
            res["without_proboscis_sugar_bitter"]["replication"]["gaba_potentiate"],
            later("without_proboscis_sugar_bitter", "replication", "gaba_potentiate"),
        ),
        PREDICTIONS[3]: (
            {run: res["all_three"][run]["excitability_block"] for run in chips},
            all(later("all_three", run, "excitability_block") for run in chips),
        ),
    }
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        if name != PREDICTIONS[0] and f0 > 1e-9:
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    return out, {"resolutions": res}


def _spont(job) -> dict:
    overrides, seed = job
    return jobs.spontaneous(overrides, seed, False)


def chip_tables(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per chip run: |delta log2 mfr| at every non-vehicle level, against that run's vehicle."""
    from neuromps.twin.perturbation import simulated_delta

    spec = design.registration()
    context, _ = _context(spec, connectome.cached())
    conds = [
        design.Condition(f, lv, k)
        for f, lv, k in zip(table["family"], table["level"], table["knob"], strict=True)
        if f != "vehicle"
    ]
    jobs_ = [
        (c, run, s)
        for run, (_, start) in CHIP_RUNS.items()
        for c in conds
        for s in range(start, start + CHIP_SEEDS)
    ]
    with ProcessPoolExecutor(
        memory_bounded_workers(), initializer=jobs.init, initargs=(context, "783")
    ) as pool:
        got = list(pool.map(_spont, [(c.overrides, s) for c, _, s in jobs_]))
    log.info("chip recordings done: %d", len(got))
    out = {}
    for run, (name, _) in CHIP_RUNS.items():
        recorded = pd.read_csv(outputs_dir() / "fly" / f"{name}_SIMULATED.csv")
        v = recorded[recorded["condition"] == "vehicle"].iloc[0]
        vehicle = {c.removeprefix("chip."): v[c] for c in recorded.columns if c.startswith("chip.")}
        mine = [(c, s, g) for (c, r, s), g in zip(jobs_, got, strict=True) if r == run]
        chip = scoring._chip({"spontaneous": mine})
        rows = []
        for c in conds:
            delta = simulated_delta(chip[c.name], vehicle)["mfr_hz"]
            rows.append(
                {
                    "condition": c.name,
                    "family": c.family,
                    "dose": c.dose,
                    "abs_delta_mfr": np.nan if delta is None else abs(delta),
                }
            )
        out[run] = pd.DataFrame(rows)
    return out


def reproduction(
    table: pd.DataFrame,
    medians: pd.DataFrame,
    chips: dict,
    coarse_medians: pd.DataFrame,
    coarse_chips: dict,
) -> float:
    """Largest difference between the rerun coarse levels and their recorded values."""
    names = [c for c in table["condition"] if c in coarse_medians.index]
    diffs = [float((medians.loc[names] - coarse_medians.loc[names]).abs().to_numpy().max())]
    for run, frame in chips.items():
        rec = coarse_chips[run].set_index("condition")["abs_delta_mfr"]
        new = frame.set_index("condition")["abs_delta_mfr"]
        shared = [c for c in new.index if c in rec.index]
        diffs.append(float((new.loc[shared] - rec.loc[shared]).abs().max()))
    return max(diffs)


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-4c design."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    table = levels(spec)
    chips = chip_tables(table)
    trials = s4a.run_trials(table, s4b.SEEDS)
    medians = s4a.movement_medians(trials)
    coarse = s4a.movement_medians(
        pd.read_csv(outputs_dir() / "fly" / "embodied_stage4b_SIMULATED.csv")
    )
    coarse_chips = {
        run: pd.read_csv(outputs_dir() / "fly" / f"{name}_SIMULATED.csv")[
            ["condition", "family", "dose", "abs_delta_mfr"]
        ]
        for run, (name, _) in CHIP_RUNS.items()
    }
    f0 = reproduction(table, medians, chips, coarse, coarse_chips)
    all_conds = pd.concat(
        [s4a.conditions(), table[~table["condition"].isin(s4a.conditions()["condition"])]]
    )
    combined = {
        run: pd.concat([coarse_chips[run], chips[run]]).drop_duplicates("condition")
        for run in chips
    }
    series = combine(coarse, medians)
    predictions, details = score(spec, all_conds, series, combined, f0)
    index = s4a.disruption(series, s4a.EPS_MM)
    result = {
        "label": "SIMULATED finer dose grid inside stage 4b's one-step margins",
        "predictions": predictions,
        **details,
        "levels": {
            c: {
                **{m: round(float(medians.loc[c, m]), 4) for m in s4a.MOVEMENTS},
                "disruption": round(float(index[c]), 4),
                **{
                    f"chip_{run}": round(float(f.set_index("condition").loc[c, "abs_delta_mfr"]), 3)
                    for run, f in chips.items()
                    if c in set(f["condition"])
                },
            }
            for c in table["condition"]
        },
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    trials.to_csv(out / "embodied_stage4c_SIMULATED.csv", index=False)
    (out / "embodied_stage4c_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage4c", scoring.clean(result))
    for name, entry in predictions.items():
        print(f"{name:52s} {entry['outcome']:>15}  {entry['value']}")
    print("resolutions", json.dumps(scoring.clean(details["resolutions"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
