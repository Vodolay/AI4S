"""Embodied fly, stage 2c: spike-frequency adaptation in the male nerve cord, swept.

``python -m neuromps.fly.embodied_stage2c`` runs the registered design
(`preregistration/embodied_stage2c_adaptation_sweep.json`): stage 2a's fictive
walking test over a 6 x 6 grid of adaptation strength and time constant on
exploration seeds (every cell reported), one cell chosen by the registered rule,
and the predictions tested there on fresh seeds, beside a degree-preserving rewired
nerve cord. Measures are stage 2a's, unchanged. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import (
    connectome,
    design,
    embodied_stage1c,
    fictive,
    model,
    muscle_map,
    scoring,
)
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage2c_adaptation_sweep.json"
B_MV = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
TAU_MS = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0)
SWEEP_SEEDS = range(820, 824)
SWEEP_DRIVES = ("forward_walking", "grooming")
CONFIRM_SEEDS = range(830, 838)
CONFIRM_DRIVES = ("forward_walking", "grooming", "none", "backward_walking")
WALKING = (
    "W1_the_walking_command_makes_a_rhythm",
    "W2_antagonists_alternate",
    "W3_left_and_right_alternate",
    "W4_a_tripod",
)
PERIOD_MS = (60.0, 200.0)
_STATE: dict = {}


def _init() -> None:
    fictive._init()
    _STATE["intact"] = embodied_stage1c._STATE["con"]
    _STATE["rewired"] = connectome.rewire(_STATE["intact"], seed=0)


def period_s(signal: np.ndarray, dt_s: float = fictive.DT_S) -> float | None:
    """Lag of the largest autocorrelation within 50-500 ms; None for a flat signal."""
    x = np.asarray(signal, float) - np.mean(signal)
    if np.dot(x, x) <= 1e-12:
        return None
    lo, hi = (int(round(v / dt_s)) for v in fictive.LAGS_S)
    hi = min(hi, x.size - 1)
    auto = [np.dot(x[:-k], x[k:]) for k in range(lo, hi + 1)]
    return float((lo + int(np.argmax(auto))) * dt_s)


def _trial(job) -> dict:
    """One trial of stage 2a's measures, with adaptation, on the intact or rewired cord."""
    network, drive, seed, b_mv, tau_ms = job
    params = model.FlyParams(
        duration_s=fictive.DURATION_S,
        stim_rate_hz=150.0,
        seed=seed,
        adapt_mv=b_mv,
        adapt_tau_ms=tau_ms,
    )
    spikes = model.simulate(_STATE[network], params, excite=fictive._STATE["drive"][drive])
    start = int(round(fictive.START_S / fictive.DT_S))
    signals, rates = {}, {}
    for key, idx in fictive._STATE["pools"].items():
        trains = [
            muscle_map.filtered_rate(
                spikes.times[spikes.neurons == n], fictive.DURATION_S, fictive.DT_S, fictive.TAU_S
            )
            for n in idx
        ]
        signals[f"{key[0]}_{key[1]}"] = np.mean(trains, axis=0)[start:]
        rates[f"{key[0]}_{key[1]}"] = float(
            np.isin(spikes.neurons, idx).sum() / idx.size / fictive.DURATION_S
        )
    trial = {"drive": drive, "seed": seed, "signals": signals, "rates": rates}
    row = fictive.measures(trial)
    row["period_L1_flexor_s"] = (
        period_s(signals["L1_flexor"]) if rates["L1_flexor"] >= fictive.MIN_RATE_HZ else None
    )
    row |= {f"rate_{k}": v for k, v in rates.items()}
    row |= {"network": network, "b_mv": b_mv, "tau_ms": tau_ms}
    return row | ({"signals": signals} if drive == "forward_walking" and seed == 830 else {})


def _distance(value) -> float:
    return 0.5 if value is None else float(value)


def walking_tests(table: pd.DataFrame, spec2a: dict) -> dict:
    """Stage 2a's W1-W4 on one cell's trials: outcomes, values and the tie-break distance."""
    got = fictive.score(table, spec2a)
    w4 = got["W4_a_tripod"]["value"]
    tie = (
        _distance(got["W2_antagonists_alternate"]["value"])
        + _distance(got["W3_left_and_right_alternate"]["value"])
        + _distance(w4["L1_L2_from_half"])
        + _distance(w4["L1_L3_from_zero"])
    )
    return {
        "passed": int(sum(got[w]["outcome"] == "pass" for w in WALKING)),
        "tests": {w: got[w]["outcome"] for w in WALKING},
        "values": {w: got[w]["value"] for w in WALKING},
        "rhythmicity_fw": got[WALKING[0]]["value"]["forward_walking"],
        "tie_distance": round(tie, 4),
    }


def select(cells: list[dict]) -> dict | None:
    """The registered rule: most W1-W4 passed, then smallest distance, smaller b, smaller tau."""
    best = min(cells, key=lambda c: (-c["passed"], c["tie_distance"], c["b_mv"], c["tau_ms"]))
    return best if best["passed"] > 0 else None


def _period_ms(table: pd.DataFrame) -> float | None:
    fw = table.loc[table["drive"] == "forward_walking", "period_L1_flexor_s"].dropna()
    return None if fw.empty else float(fw.median() * 1000)


def sweep_cells(table: pd.DataFrame, spec2a: dict) -> list[dict]:
    """Every grid cell (and the reference, b = 0) with its walking tests and period."""
    cells = []
    for (b, tau), g in table.groupby(["b_mv", "tau_ms"], sort=True):
        cell = {"b_mv": float(b), "tau_ms": float(tau)} | walking_tests(g, spec2a)
        cell["period_ms"] = _period_ms(g)
        fw = g[g["drive"] == "forward_walking"]
        cell["flexor_hz"] = round(
            float(
                fw[[c for c in g if c.endswith("_flexor") and c.startswith("rate_")]].mean().mean()
            ),
            2,
        )
        cell["extensor_hz"] = round(
            float(
                fw[[c for c in g if c.endswith("_extensor") and c.startswith("rate_")]]
                .mean()
                .mean()
            ),
            2,
        )
        cells.append(cell)
    return cells


def confirm(intact: pd.DataFrame, rewired: pd.DataFrame, spec: dict, spec2a: dict) -> dict:
    """A0-A6 at the selected cell, on the confirmation seeds."""
    got = fictive.score(intact, spec2a)
    control = fictive.score(rewired, spec2a)
    names = dict(
        zip(("W0_silent_without_drive", *WALKING), list(spec["predictions"])[:5], strict=True)
    )
    out = {names[w]: (got[w]["value"], got[w]["outcome"] == "pass") for w in names}
    coord = WALKING[1:]
    n_intact = sum(got[w]["outcome"] == "pass" for w in coord)
    n_rewired = sum(control[w]["outcome"] == "pass" for w in coord)
    out["A5_the_coordination_comes_from_the_wiring"] = (
        {"intact_passes": n_intact, "rewired_passes": n_rewired},
        n_intact >= 1 and n_rewired < n_intact,
    )
    period = _period_ms(intact)
    out["A6_a_fly_like_step_period"] = (
        period,
        period is not None and PERIOD_MS[0] <= period <= PERIOD_MS[1],
    )
    return {
        n: {"test": spec["predictions"][n]["test"], "value": v, "outcome": "pass" if ok else "fail"}
        for n, (v, ok) in out.items()
    }


def stopped(spec: dict) -> dict:
    """A0-A6 when no cell passes any walking test: the registered stop, every prediction failing."""
    return {
        n: {"test": p["test"], "value": None, "outcome": "fail"}
        for n, p in spec["predictions"].items()
    }


def plot_sweep(cells: list[dict], selected: dict | None, path) -> None:
    """Walking tests passed and rhythmicity in every grid cell."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = [c for c in cells if c["b_mv"] > 0]
    passed = np.zeros((len(B_MV), len(TAU_MS)))
    rhythm = np.zeros_like(passed)
    for c in grid:
        i, j = B_MV.index(c["b_mv"]), TAU_MS.index(c["tau_ms"])
        passed[i, j], rhythm[i, j] = c["passed"], c["rhythmicity_fw"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, data, title, vmax in (
        (axes[0], passed, "walking tests passed (of W1-W4)", 4),
        (axes[1], rhythm, "rhythmicity, L1 flexor, walking drive", 1),
    ):
        im = ax.imshow(data, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        for i, j in np.ndindex(data.shape):
            text = f"{int(data[i, j])}" if vmax == 4 else f"{data[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=5.5, color="white")
        if selected is not None:
            ax.scatter(
                TAU_MS.index(selected["tau_ms"]),
                B_MV.index(selected["b_mv"]),
                s=260,
                facecolors="none",
                edgecolors="#E69F00",
                linewidths=1.5,
            )
        ax.set_xticks(range(len(TAU_MS)), [f"{t:g}" for t in TAU_MS], fontsize=6)
        ax.set_yticks(range(len(B_MV)), [f"{b:g}" for b in B_MV], fontsize=6)
        ax.set_xlabel("adaptation time constant (ms)", fontsize=7)
        ax.set_ylabel("adaptation per spike (mV)", fontsize=7)
        ax.set_title(title, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=6)
    fig.suptitle(
        "SIMULATED male nerve cord with adaptation: the whole registered sweep (seeds 820-823)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confirmation(signals: dict, cell: dict, path) -> None:
    """The selected cell: one confirmation trial's tibia pools under the walking command."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    colours = {
        "L1_flexor": "#D55E00",
        "L1_extensor": "#E69F00",
        "R1_flexor": "#0072B2",
        "L2_flexor": "#009E73",
        "L3_flexor": "#CC79A7",
    }
    t = fictive.START_S + np.arange(signals["L1_flexor"].size) * fictive.DT_S
    for key, colour in colours.items():
        ax.plot(t, signals[key], color=colour, lw=0.9, label=key.replace("_", " "))
    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("pool rate (Hz)", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=5)
    ax.tick_params(labelsize=6)
    ax.set_title(
        f"SIMULATED intact nerve cord, selected cell (b {cell['b_mv']:g} mV, "
        f"tau {cell['tau_ms']:g} ms), walking drive, seed 830",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _run(pool, jobs: list[tuple]) -> list[dict]:
    return list(pool.map(_trial, jobs))


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k != "signals"} for r in rows])


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered sweep, selection and confirmation."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    spec2a = json.loads(fictive.REGISTRATION.read_text())
    grid = [(0.0, 100.0)] + [(b, t) for b in B_MV for t in TAU_MS]
    sweep_jobs = [
        ("intact", d, s, b, t) for b, t in grid for d in SWEEP_DRIVES for s in SWEEP_SEEDS
    ]
    out = outputs_dir() / "fly"
    with ProcessPoolExecutor(memory_bounded_workers(per_worker_gb=8.0), initializer=_init) as pool:
        sweep = _frame(_run(pool, sweep_jobs))
        cells = sweep_cells(sweep, spec2a)
        selected = select([c for c in cells if c["b_mv"] > 0])
        log.info("selected cell %s", selected)
        confirmation = pd.DataFrame()
        signals = None
        if selected is not None:
            b, t = selected["b_mv"], selected["tau_ms"]
            jobs = [
                (n, d, s, b, t)
                for n in ("intact", "rewired")
                for d in CONFIRM_DRIVES
                for s in CONFIRM_SEEDS
            ]
            rows = _run(pool, jobs)
            confirmation = _frame(rows)
            signals = next(
                r["signals"] for r in rows if r["network"] == "intact" and "signals" in r
            )
    if selected is None:
        predictions = stopped(spec)
    else:
        main_rows = confirmation[confirmation["drive"] != "backward_walking"]
        predictions = confirm(
            main_rows[main_rows["network"] == "intact"],
            main_rows[main_rows["network"] == "rewired"],
            spec,
            spec2a,
        )
    result = {
        "label": "SIMULATED adaptation sweep and confirmation in the male nerve cord",
        "predictions": predictions,
        "selected_cell": selected,
        "sweep": {
            "cells": cells,
            "cells_passing": {
                w: sum(c["tests"][w] == "pass" for c in cells if c["b_mv"] > 0) for w in WALKING
            },
            "grid_size": len(B_MV) * len(TAU_MS),
        },
        "backward_walking_reported": None
        if confirmation.empty
        else fictive.score(
            confirmation[
                (confirmation["network"] == "intact")
                & confirmation["drive"].isin(["backward_walking", "grooming", "none"])
            ].replace({"backward_walking": "forward_walking"}),
            spec2a,
        ),
        "registration": design.provenance(REGISTRATION),
    }
    sweep.to_csv(out / "embodied_stage2c_sweep_SIMULATED.csv", index=False)
    if not confirmation.empty:
        confirmation.to_csv(out / "embodied_stage2c_confirmation_SIMULATED.csv", index=False)
    (out / "embodied_stage2c_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage2c", scoring.clean(result))
    plot_sweep(cells, selected, figures_dir() / "embodied_stage2c_sweep_SIMULATED.png")
    if signals is not None:
        plot_confirmation(
            signals, selected, figures_dir() / "embodied_stage2c_confirmation_SIMULATED.png"
        )
    print("selected", {k: selected[k] for k in ("b_mv", "tau_ms", "passed")} if selected else None)
    print("cells passing", result["sweep"]["cells_passing"])
    for name, entry in predictions.items():
        print(f"{name:44s} {entry['outcome']:>5}  {entry['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
