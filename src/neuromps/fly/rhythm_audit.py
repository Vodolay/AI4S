"""The registered rhythm-measure audit (`preregistration/rhythm_measure_audit.json`).

``python -m neuromps.fly.rhythm_audit`` re-simulates stages 1d, 2a, 2c, 2d and 2e with
their own code and seeds, measures every trial with the registered rhythm measure and
with the corrected one (`kinematics.rhythm`, which requires an autocorrelation peak),
applies each stage's registered tests to both, and checks that the registered measure
reproduces the recorded outcomes. Registered outcomes are not replaced; this reports
where the corrected measure agrees. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import design, embodied_stage1c, fictive, scoring
from neuromps.fly import embodied_stage1d as s1d
from neuromps.fly import embodied_stage2c as s2c
from neuromps.fly import embodied_stage2d as s2d
from neuromps.fly import embodied_stage2e as s2e
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "rhythm_measure_audit.json"
METHODS = ("registered", "audited")


def _recorded(stage: str) -> dict:
    return json.loads((outputs_dir() / "fly" / f"{stage}_SIMULATED.json").read_text())


def same(a, b) -> bool:
    """Equal after the cleaning every stage applies before writing its results."""
    clean = scoring.clean
    return json.dumps(clean(a), sort_keys=True) == json.dumps(clean(b), sort_keys=True)


def compare(got: dict[str, dict], recorded: dict, names: list[str]) -> dict:
    """Per prediction: registered and audited value and outcome; reproducibility of the first."""
    out = {"predictions": {}, "reproducible": True}
    for n in names:
        reg, aud = got["registered"][n], got["audited"][n]
        out["predictions"][n] = {
            "registered": {"value": reg["value"], "outcome": reg["outcome"]},
            "audited": {"value": aud["value"], "outcome": aud["outcome"]},
            "changed": reg["outcome"] != aud["outcome"],
        }
        rec = recorded["predictions"][n]
        if rec["outcome"] != reg["outcome"] or not same(rec["value"], reg["value"]):
            out["reproducible"] = False
    return out


def _pairs(tables: dict[str, pd.DataFrame], column: str) -> list[list[float]]:
    old, new = tables["registered"][column], tables["audited"][column]
    return [[float(a), float(b)] for a, b in zip(old, new, strict=True)]


def audit_1d() -> tuple[dict, list]:
    spec = json.loads(s1d.REGISTRATION.read_text())
    trials, moves = embodied_stage1c.run(
        [(d, s, embodied_stage1c.W_SYN) for d in s1d.DRIVES for s in s1d.SEEDS]
    )
    pools = pd.DataFrame(
        [{"drive": t["drive"], "seed": t["seed"], **t["pool_rates"]} for t in trials]
    )
    tables = {m: s1d.readouts(moves, m) for m in METHODS}
    got = {m: s1d.score(tables[m], pools, spec) for m in METHODS}
    names = list(spec["predictions"])
    result = compare(got, _recorded("embodied_stage1d"), names)
    main = {m: t[t["r_full"] == 20.0] for m, t in tables.items()}
    return result, _pairs(main, "rhythmicity")


def _both_2a(job) -> list[dict]:
    trial = fictive._trial(job)
    return [fictive.measures(trial, m) for m in METHODS]


def audit_2a() -> tuple[dict, list]:
    spec = json.loads(fictive.REGISTRATION.read_text())
    jobs = [(d, s) for d in fictive.DRIVES for s in fictive.SEEDS]
    with ProcessPoolExecutor(
        memory_bounded_workers(per_worker_gb=8.0), initializer=fictive._init
    ) as pool:
        both = list(pool.map(_both_2a, jobs))
    tables = {m: pd.DataFrame([b[k] for b in both]) for k, m in enumerate(METHODS)}
    got = {m: fictive.score(tables[m], spec) for m in METHODS}
    result = compare(got, _recorded("embodied_stage2a"), list(spec["predictions"]))
    return result, _pairs(tables, "rhythmicity_L1_flexor")


def _both_sweep(job) -> list[dict]:
    spikes = s2c.simulate(job)
    counts = np.bincount(spikes.neurons, minlength=s2c._STATE[job[0]].n_neurons)
    active = int(np.sum(counts / fictive.DURATION_S >= fictive.MIN_RATE_HZ))
    rows = []
    for m in METHODS:
        row = s2c.measure(job, spikes, method=m)
        row.pop("signals", None)
        rows.append(row | {"active_neurons": active})
    return rows


def _sweep(jobs, initializer) -> dict[str, pd.DataFrame]:
    with ProcessPoolExecutor(
        memory_bounded_workers(per_worker_gb=8.0), initializer=initializer
    ) as pool:
        both = list(pool.map(_both_sweep, jobs))
    return {m: pd.DataFrame([b[k] for b in both]) for k, m in enumerate(METHODS)}


def _cells_passing(cells: list[dict]) -> dict:
    return {w: sum(c["tests"][w] == "pass" for c in cells) for w in s2c.WALKING}


def audit_2c() -> tuple[dict, list]:
    spec2a = json.loads(fictive.REGISTRATION.read_text())
    grid = [(0.0, 100.0)] + [(b, t) for b in s2c.B_MV for t in s2c.TAU_MS]
    jobs = [
        ("intact", d, s, b, t) for b, t in grid for d in s2c.SWEEP_DRIVES for s in s2c.SWEEP_SEEDS
    ]
    tables = _sweep(jobs, s2c._init)
    out = {"reproducible": True}
    recorded = _recorded("embodied_stage2c")
    for m in METHODS:
        cells = s2c.sweep_cells(tables[m], spec2a)
        selected = s2c.select([c for c in cells if c["b_mv"] > 0])
        grid_cells = [c for c in cells if c["b_mv"] > 0]
        out[m] = {
            "cells_passing": _cells_passing(grid_cells),
            "selected": None if selected is None else [selected["b_mv"], selected["tau_ms"]],
        }
    rec = recorded["sweep"]["cells_passing"]
    out["reproducible"] = same(rec, out["registered"]["cells_passing"]) and (
        recorded["selected_cell"] is None
    ) == (out["registered"]["selected"] is None)
    return out, _pairs(tables, "rhythmicity_L1_flexor")


def audit_2d() -> tuple[dict, list]:
    spec2a = json.loads(fictive.REGISTRATION.read_text())
    spec = json.loads(s2d.REGISTRATION.read_text())
    grid = [(0.0, 100.0)] + [(b, t) for b in s2c.B_MV for t in s2c.TAU_MS]
    jobs = [
        (rule, d, s, b, t)
        for rule in s2d.RULES
        for b, t in grid
        for d in s2c.SWEEP_DRIVES
        for s in s2d.SWEEP_SEEDS
    ]
    tables = _sweep(jobs, s2d._init)
    recorded = _recorded("embodied_stage2d")
    out = {}
    for m in METHODS:
        cells = s2d.sweep_cells(tables[m], spec2a)
        selected = s2d.select(cells)
        out[m] = {
            "cells_passing": {
                r: _cells_passing([c for c in cells if c["rule"] == r]) for r in s2d.RULES
            },
            "selected": None
            if selected is None
            else [selected["rule"], selected["b_mv"], selected["tau_ms"]],
        }
    sel = recorded["selected_cell"]
    rule, b, t = sel["rule"], sel["b_mv"], sel["tau_ms"]
    jobs = [
        (n, d, s, b, t)
        for n in (rule, f"{rule}_rewired")
        for d in s2c.CONFIRM_DRIVES
        for s in s2d.CONFIRM_SEEDS
    ]
    conf = _sweep(jobs, s2d._init)
    got = {}
    for m in METHODS:
        main = conf[m][conf[m]["drive"] != "backward_walking"]
        got[m] = s2c.confirm(
            main[main["network"] == rule], main[main["network"] == f"{rule}_rewired"], spec, spec2a
        )
    confirmation = compare(got, recorded, list(spec["predictions"]))
    out["confirmation_at_the_registered_cell"] = confirmation
    out["reproducible"] = confirmation["reproducible"] and same(
        recorded["sweep"]["cells_passing"], out["registered"]["cells_passing"]
    )
    return out, _pairs(tables, "rhythmicity_L1_flexor")


def _both_2e(name: str) -> list[list[dict]]:
    run = s2e.simulate_condition(name, s2e.SEEDS, s2e.DT_S)
    both = []
    for m in METHODS:
        rows = s2e.condition_rows(run, name, s2e.SEEDS, m)
        both.append([{k: v for k, v in r.items() if k != "signals"} for r in rows])
    return both


def audit_2e() -> tuple[dict, list]:
    spec = json.loads(s2e.REGISTRATION.read_text())
    with ProcessPoolExecutor(len(s2e.CONDITIONS), initializer=s2e._init) as pool:
        per = list(pool.map(_both_2e, list(s2e.CONDITIONS)))
    tables = {m: pd.DataFrame([r for p in per for r in p[k]]) for k, m in enumerate(METHODS)}
    got = {m: s2e.score(tables[m], spec) for m in METHODS}
    result = compare(got, _recorded("embodied_stage2e"), list(spec["predictions"]))
    result["reported"] = {m: s2e.reported(tables[m]) for m in METHODS}
    return result, _pairs(tables, "rhythmicity_L1_flexor")


def plot(pairs: dict[str, list], path) -> None:
    """Every trial's rhythmicity under the registered measure against the corrected one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "stage 1d (claw distance)": "#D55E00",
        "stage 2a": "#E69F00",
        "stage 2c sweep": "#999999",
        "stage 2d sweep": "#0072B2",
        "stage 2e": "#009E73",
    }
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for (name, values), colour in zip(pairs.items(), colours.values(), strict=True):
        v = np.asarray(values)
        ax.scatter(v[:, 0], v[:, 1], s=5, color=colour, alpha=0.6, label=name, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="black", lw=0.6)
    ax.axhline(0.3, color="black", lw=0.5, ls="--")
    ax.axvline(0.3, color="black", lw=0.5, ls="--")
    ax.set_xlabel("registered measure (no peak required)", fontsize=7)
    ax.set_ylabel("corrected measure (autocorrelation peak)", fontsize=7)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.0)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax.set_title("SIMULATED: every audited trial's rhythmicity, two measures", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _line(name: str, p: dict) -> str:
    mark = "CHANGED" if p["changed"] else ""
    return f"   {name:46s} {p['registered']['outcome']:>5} -> {p['audited']['outcome']:<5} {mark}"


def main(argv: list[str] | None = None) -> int:
    """CLI: run the registered audit."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stages = {
        "embodied_stage1d": ("stage 1d (claw distance)", audit_1d),
        "embodied_stage2a": ("stage 2a", audit_2a),
        "embodied_stage2c": ("stage 2c sweep", audit_2c),
        "embodied_stage2d": ("stage 2d sweep", audit_2d),
        "embodied_stage2e": ("stage 2e", audit_2e),
    }
    result, pairs = {"label": "SIMULATED re-analysis: the rhythm-measure audit"}, {}
    for stage, (label, fn) in stages.items():
        log.info("auditing %s", stage)
        result[stage], pairs[label] = fn()
    result["registration"] = design.provenance(REGISTRATION)
    out = outputs_dir() / "fly" / "rhythm_audit_SIMULATED.json"
    out.write_text(json.dumps(scoring.clean(result), indent=2) + "\n")
    update_metrics("rhythm_audit", scoring.clean(result))
    plot(pairs, figures_dir() / "rhythm_audit_SIMULATED.png")
    for stage in stages:
        r = result[stage]
        print(stage, "reproducible:", r["reproducible"])
        for n, p in r.get("predictions", {}).items():
            print(_line(n, p))
        for m in METHODS:
            if m in r:
                print(f"   {m}: {r[m]}")
        if "confirmation_at_the_registered_cell" in r:
            for n, p in r["confirmation_at_the_registered_cell"]["predictions"].items():
                print(_line(n, p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
