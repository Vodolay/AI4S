"""Run the registered male CNS replication and score it.

``python -m neuromps.fly.replication`` runs the conditions of the parent
registration on MaleCNS v1.0, as `preregistration/fly_malecns_replication.json`
defines it: spontaneous array recordings and the published behaviour
protocols, no directionality (MaleCNS has no per-neuron region table here).
Everything written is **simulated**.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neuromps.fly import analysis, behaviour, connectome, design, jobs, malecns, scoring
from neuromps.fly.study import PROTOCOLS, memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir
from neuromps.twin.perturbation import simulated_delta

log = logging.getLogger("neuromps.fly")
ADDENDUM = REPO_ROOT / "preregistration" / "fly_malecns_replication.json"
MEMORY_PER_WORKER_GB = 12.0


def _context(spec: dict) -> jobs.Context:
    con, ann = malecns.cached()
    flywire = connectome.read_annotations()[["root_id", "cell_type", "side"]]
    sets = malecns.match_sets(behaviour.neuron_sets(), flywire, ann)
    descending = np.flatnonzero(ann["superclass"].to_numpy(dtype=object) == "descending_neuron")
    log.info("MaleCNS sets: %s", {k: len(v) for k, v in sets.items()})
    return jobs.Context(spec=spec, pairs=pd.DataFrame(), descending=descending, sets=sets)


def run(context: jobs.Context, addendum: dict, workers: int) -> dict:
    """Spontaneous recordings and behaviour trials of every condition."""
    from concurrent.futures import ProcessPoolExecutor

    d = addendum["design"]
    conds = design.conditions(context.spec)
    s0 = int(d["seeds"]["spontaneous"].rstrip("+"))
    b0 = int(d["seeds"]["behaviour"].rstrip("+"))
    spont = [(c, s0 + k) for c in conds for k in range(d["spontaneous"]["seeds"])]
    trials = [
        (c, p, b0 + k) for c in conds for p in PROTOCOLS for k in range(d["behaviour"]["trials"])
    ]
    started = time.time()
    with ProcessPoolExecutor(workers, initializer=jobs.init, initargs=(context, "malecns")) as pool:
        f_s = [pool.submit(jobs.spontaneous, c.overrides, s, False) for c, s in spont]
        f_b = [pool.submit(jobs.behaviour_trial, c.overrides, p, s) for c, p, s in trials]
        raw = {
            "spontaneous": [(c, s, f.result()) for (c, s), f in zip(spont, f_s, strict=True)],
            "behaviour": [(c, p, s, f.result()) for (c, p, s), f in zip(trials, f_b, strict=True)],
        }
    log.info("MaleCNS sweep: %d jobs in %.0f s", len(spont) + len(trials), time.time() - started)
    return raw


def table(raw: dict, spec: dict) -> pd.DataFrame:
    """Per-condition chip deltas, behaviour readouts and disruption."""
    chip, animal = scoring._chip(raw), scoring._behaviour(raw)
    rows = []
    for cond in design.conditions(spec):
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
    return frame


def score(frame: pd.DataFrame, addendum: dict) -> dict:
    """M0 gates the rest: without the feeding circuit the bridge tests mean nothing."""
    v = frame[frame["family"] == "vehicle"].iloc[0]
    m0 = bool(v["mn9_sugar"] >= 20 and v["mn9_sugar_bitter"] <= 0.5 * v["mn9_sugar"])
    found = {
        "M0_feeding_circuit_exists": (
            {"mn9_sugar_hz": v["mn9_sugar"], "mn9_sugar_bitter_hz": v["mn9_sugar_bitter"]},
            m0,
        )
    }
    parent = scoring._behaviour_predictions(frame)
    for short, name in (
        ("M_B1", "B1_disinhibition_releases_feeding_from_bitter"),
        ("M_B2", "B2_excitation_block_silences_motor_output"),
        ("M_B3", "B3_disinhibition_recruits_off_target_descending_neurons"),
    ):
        found[short] = parent[name]
    drugs = frame[frame["family"] != "vehicle"]
    rho = float(spearmanr(drugs["abs_delta_mfr"], drugs["disruption"])[0])
    found["M_C1"] = (rho, rho >= 0.6)
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        if name != "M0_feeding_circuit_exists" and not m0:
            outcome = "uninterpretable"
        out[name] = {
            "test": addendum["predictions"][name]["test"],
            "value": value,
            "outcome": outcome,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: run and score the replication."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec, addendum = design.registration(), json.loads(ADDENDUM.read_text())
    context = _context(spec)
    workers = args.workers or memory_bounded_workers(per_worker_gb=MEMORY_PER_WORKER_GB)
    frame = table(run(context, addendum, workers), spec)
    result = {
        "label": "SIMULATED male CNS (MaleCNS v1.0) replication of the fly study",
        "predictions": score(frame, addendum),
        "neuron_set_sizes": {k: len(v) for k, v in context.sets.items()},
        "registration": design.provenance(ADDENDUM),
        "deviations": malecns.DEVIATIONS,
    }
    out_dir = outputs_dir() / "fly"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "malecns_conditions_SIMULATED.csv", index=False)
    (out_dir / "malecns_replication_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("fly_malecns_replication", scoring.clean(result))
    for name, entry in result["predictions"].items():
        print(f"{name:30s} {entry['outcome']:>15}  {entry['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
