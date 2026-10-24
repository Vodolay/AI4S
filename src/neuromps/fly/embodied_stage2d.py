"""Embodied fly, stage 2d: the glutamate sign rule, two bounds, each with the adaptation sweep.

``python -m neuromps.fly.embodied_stage2d`` runs the registered design
(`preregistration/embodied_stage2d_glutamate_sign.json`): the stage-1c nerve cord
with glutamate made excitatory onto motor neurons only (``glut_motor``) or
everywhere (``glut_all``), each without adaptation and over stage 2c's grid, on
exploration seeds (every cell reported); one cell chosen by the registered rule;
the predictions tested there on fresh seeds beside the rule's rewired network.
Measures are stage 2a's, through stage 2c's code. Everything is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import connectome, design, embodied_stage1c, fictive, muscle_map, scoring
from neuromps.fly import embodied_stage2c as s2c
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage2d_glutamate_sign.json"
RULES = ("glut_motor", "glut_all")
SWEEP_SEEDS = range(840, 844)
CONFIRM_SEEDS = range(850, 858)
ACTIVE_HZ = 1.0


def networks(con: connectome.Connectome, motor: np.ndarray) -> dict:
    """The two rules' networks and their rewired controls."""
    glut = con.pre_class == "glut"
    nets = {
        "glut_motor": connectome.make_excitatory(con, glut & np.isin(con.post, motor)),
        "glut_all": connectome.make_excitatory(con, glut),
    }
    return nets | {f"{k}_rewired": connectome.rewire(v, seed=0) for k, v in nets.items()}


def _init() -> None:
    fictive._init()
    con = embodied_stage1c._STATE["con"]
    table = embodied_stage1c._annotations(con.root_ids)
    motor = np.flatnonzero(table["superclass"].isin(muscle_map.MOTOR).to_numpy())
    s2c._STATE.update(networks(con, motor))


def _trial(job) -> dict:
    """Stage 2c's measures plus the number of nerve-cord neurons firing at >= 1 Hz."""
    spikes = s2c.simulate(job)
    row = s2c.measure(job, spikes, keep_signals_seed=CONFIRM_SEEDS[0])
    counts = np.bincount(spikes.neurons, minlength=s2c._STATE[job[0]].n_neurons)
    row["active_neurons"] = int(np.sum(counts / fictive.DURATION_S >= ACTIVE_HZ))
    return row


def sweep_cells(table: pd.DataFrame, spec2a: dict) -> list[dict]:
    """Every cell of both rules, with stage 2c's summary and the activity level."""
    cells = []
    for rule in RULES:
        sub = table[table["network"] == rule]
        for cell in s2c.sweep_cells(sub, spec2a):
            g = sub[(sub["b_mv"] == cell["b_mv"]) & (sub["tau_ms"] == cell["tau_ms"])]
            fw = g[g["drive"] == "forward_walking"]
            cells.append(
                {"rule": rule} | cell | {"active_neurons": int(fw["active_neurons"].median())}
            )
    return cells


def select(cells: list[dict]) -> dict | None:
    """Most W1-W4 passed; then distance, glut_motor first, smaller b, smaller tau."""
    best = min(
        cells,
        key=lambda c: (
            -c["passed"],
            c["tie_distance"],
            RULES.index(c["rule"]),
            c["b_mv"],
            c["tau_ms"],
        ),
    )
    return best if best["passed"] > 0 else None


def plot_sweep(cells: list[dict], selected: dict | None, path) -> None:
    """Per rule: walking tests passed, flexor rate and active neurons in every grid cell."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("passed", "walking tests passed (of W1-W4)", "{:.0f}"),
        ("flexor_hz", "tibia flexor pools (Hz)", "{:.1f}"),
        ("active_neurons", "neurons >= 1 Hz", "{:.0f}"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.8))
    for r, rule in enumerate(RULES):
        mine = [c for c in cells if c["rule"] == rule]
        none = next(c for c in mine if c["b_mv"] == 0)
        for k, (key, title, fmt) in enumerate(panels):
            ax = axes[r, k]
            data = np.zeros((len(s2c.B_MV), len(s2c.TAU_MS)))
            for c in mine:
                if c["b_mv"] > 0:
                    data[s2c.B_MV.index(c["b_mv"]), s2c.TAU_MS.index(c["tau_ms"])] = c[key]
            vmax = 4 if key == "passed" else max(data.max(), 1.0)
            ax.imshow(data, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
            for i, j in np.ndindex(data.shape):
                colour = "white" if data[i, j] < 0.6 * vmax else "black"
                ax.text(
                    j,
                    i,
                    fmt.format(data[i, j]),
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    color=colour,
                )
            if selected is not None and selected["rule"] == rule and selected["b_mv"] > 0:
                ax.scatter(
                    s2c.TAU_MS.index(selected["tau_ms"]),
                    s2c.B_MV.index(selected["b_mv"]),
                    s=200,
                    facecolors="none",
                    edgecolors="#E69F00",
                    linewidths=1.5,
                )
            ax.set_xticks(range(len(s2c.TAU_MS)), [f"{t:g}" for t in s2c.TAU_MS], fontsize=5.5)
            ax.set_yticks(range(len(s2c.B_MV)), [f"{b:g}" for b in s2c.B_MV], fontsize=5.5)
            ax.set_xlabel("adaptation time constant (ms)", fontsize=6.5)
            ax.set_ylabel("adaptation per spike (mV)", fontsize=6.5)
            ax.set_title(f"{rule}: {title}; no adaptation: {fmt.format(none[key])}", fontsize=6.5)
    fig.suptitle(
        "SIMULATED male nerve cord with glutamate excitatory: the whole registered sweep "
        "(seeds 840-843)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _confirm(pool, selected: dict, spec: dict, spec2a: dict):
    rule, b, t = selected["rule"], selected["b_mv"], selected["tau_ms"]
    jobs = [
        (n, d, s, b, t)
        for n in (rule, f"{rule}_rewired")
        for d in s2c.CONFIRM_DRIVES
        for s in CONFIRM_SEEDS
    ]
    rows = list(pool.map(_trial, jobs))
    table = s2c._frame(rows)
    main_rows = table[table["drive"] != "backward_walking"]
    predictions = s2c.confirm(
        main_rows[main_rows["network"] == rule],
        main_rows[main_rows["network"] == f"{rule}_rewired"],
        spec,
        spec2a,
    )
    signals = next(r["signals"] for r in rows if r["network"] == rule and "signals" in r)
    return table, predictions, signals


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered sweep of both rules, selection and confirmation."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    spec2a = json.loads(fictive.REGISTRATION.read_text())
    grid = [(0.0, 100.0)] + [(b, t) for b in s2c.B_MV for t in s2c.TAU_MS]
    jobs = [
        (rule, d, s, b, t)
        for rule in RULES
        for b, t in grid
        for d in s2c.SWEEP_DRIVES
        for s in SWEEP_SEEDS
    ]
    out = outputs_dir() / "fly"
    confirmation, signals = pd.DataFrame(), None
    with ProcessPoolExecutor(memory_bounded_workers(per_worker_gb=8.0), initializer=_init) as pool:
        sweep = s2c._frame(list(pool.map(_trial, jobs)))
        cells = sweep_cells(sweep, spec2a)
        selected = select(cells)
        log.info("selected cell %s", selected)
        if selected is not None:
            confirmation, predictions, signals = _confirm(pool, selected, spec, spec2a)
    if selected is None:
        predictions = s2c.stopped(spec)
    result = {
        "label": "SIMULATED glutamate sign rules with the adaptation sweep, male nerve cord",
        "predictions": predictions,
        "selected_cell": selected,
        "sweep": {
            "cells": cells,
            "cells_passing": {
                rule: {
                    w: sum(c["tests"][w] == "pass" for c in cells if c["rule"] == rule)
                    for w in s2c.WALKING
                }
                for rule in RULES
            },
            "cells_per_rule": len(grid),
        },
        "registration": design.provenance(REGISTRATION),
    }
    sweep.to_csv(out / "embodied_stage2d_sweep_SIMULATED.csv", index=False)
    if not confirmation.empty:
        confirmation.to_csv(out / "embodied_stage2d_confirmation_SIMULATED.csv", index=False)
    (out / "embodied_stage2d_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage2d", scoring.clean(result))
    plot_sweep(cells, selected, figures_dir() / "embodied_stage2d_sweep_SIMULATED.png")
    if signals is not None:
        s2c.plot_confirmation(
            signals,
            selected,
            figures_dir() / "embodied_stage2d_confirmation_SIMULATED.png",
            seed=CONFIRM_SEEDS[0],
        )
    print("selected", selected)
    print("cells passing", result["sweep"]["cells_passing"])
    for name, entry in predictions.items():
        print(f"{name:44s} {entry['outcome']:>5}  {entry['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
