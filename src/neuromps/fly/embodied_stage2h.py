"""Embodied fly, stage 2h: stage 2g's front-leg network with a fine input search.

``python -m neuromps.fly.embodied_stage2h`` runs the registered design
(`preregistration/embodied_stage2h_front_leg_bisection.json`): everything as stage
2g, except that once a draw has an input recruiting too few neurons and one
recruiting too many, the next input is their geometric mean (at most 30 runs).
Fresh draws. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import design, scoring
from neuromps.fly import embodied_stage2g as s2g
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage2h_front_leg_bisection.json"
SEEDS = list(range(980, 1012))
MAX_RUNS = 30


def next_input(inputs: np.ndarray, recruited: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Geometric mean of the bracket where a draw has one; stage 2g's rule otherwise."""
    bracket = np.isfinite(lo) & np.isfinite(hi)
    return np.where(bracket, np.sqrt(lo * hi), s2g.next_inputs(inputs, recruited))


def bisect_inputs(rule: str, drive: str, seeds: list[int]):
    """Per draw: final run, inputs, feasibility, runs used, and the bracket ratio."""
    inputs = np.full(len(seeds), s2g.START_INPUT)
    lo, hi = np.full(len(seeds), np.nan), np.full(len(seeds), np.nan)
    for runs in range(1, MAX_RUNS + 1):
        run = s2g._simulate(rule, drive, inputs, seeds=seeds)
        n = s2g._recruited(run, drive)
        ok = (n >= s2g.FEWEST) & (n <= s2g.MOST)
        if ok.all() or runs == MAX_RUNS:
            break
        lo = np.where(n < s2g.FEWEST, np.fmax(lo, inputs), lo)
        hi = np.where(n > s2g.MOST, np.fmin(hi, inputs), hi)
        inputs = np.where(ok, inputs, next_input(inputs, n, lo, hi))
    return run, inputs, ok, runs, {"bracket_ratio": hi / lo}


def run_job(job) -> dict:
    """One (size rule, drive), stage 2g's measures with the bisection search."""
    return s2g.run_job(job, seeds=SEEDS, search=bisect_inputs)


def score(table: pd.DataFrame, spec: dict) -> dict:
    """H0-H6: stage 2g's P0-P6 computed by its code, with this registration's names."""
    got = s2g.score(table, json.loads(s2g.REGISTRATION.read_text()))
    return {
        h: got[p] | {"test": spec["predictions"][h]["test"]}
        for p, h in zip(got, spec["predictions"], strict=True)
    }


def reported(table: pd.DataFrame) -> dict:
    """Stage 2g's report, plus the bracket ratios of infeasible forward-walking draws."""
    out = s2g.reported(table)
    fw = table[(table["condition"] == "forward_walking") & ~table["feasible"]]
    ratios = fw["bracket_ratio"].dropna()
    out["forward_walking"]["infeasible_bracket_ratio_median"] = (
        float(ratios.median()) if len(ratios) else None
    )
    out["forward_walking"]["infeasible_never_bracketed"] = int(fw["bracket_ratio"].isna().sum())
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-2h design."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    jobs = [(rule, drive) for rule in s2g.SIZE_RULES for drive in s2g.DRIVES]
    with ProcessPoolExecutor(len(jobs), initializer=s2g._init) as pool:
        got = list(pool.map(run_job, jobs))
    s2g._init()
    table = pd.DataFrame([row for g in got for row in g["rows"]])
    examples = {
        rule: g.get("example")
        for (rule, drive), g in zip(jobs, got, strict=True)
        if drive == "forward_walking"
    }
    result = {
        "label": "SIMULATED left front-leg network of the male CNS, fine input search",
        "runs_used": {f"{r}/{d}": g["runs"] for (r, d), g in zip(jobs, got, strict=True)},
    }
    for rule in s2g.SIZE_RULES:
        sub = table[table["size_rule"] == rule]
        result[rule] = {"predictions": score(sub, spec), "reported": reported(sub)}
    result["registration"] = design.provenance(REGISTRATION)
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage2h_SIMULATED.csv", index=False)
    (out / "embodied_stage2h_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage2h", scoring.clean(result))
    s2g.plot(examples, table, figures_dir() / "embodied_stage2h_front_leg_SIMULATED.png")
    print("runs", result["runs_used"])
    for rule in s2g.SIZE_RULES:
        print("size rule:", rule)
        for name, entry in result[rule]["predictions"].items():
            print(f"   {name:48s} {entry['outcome']:>15}  {entry['value']}")
        print("   reported", json.dumps(scoring.clean(result[rule]["reported"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
