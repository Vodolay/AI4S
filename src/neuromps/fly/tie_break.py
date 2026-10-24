"""Resolve the motor-level ties with a finer dose grid (registered follow-up).

The parent test (`fly.motor`, MR5) found the chip later than motor output in
three drug families and tied in two, because both onsets fell inside one coarse
dose step. This runs finer levels inside those two steps, with the parent's
seeds, readouts and untreated references, and says which came first. MR5 stays
failed as registered; this is reported beside it. See
`preregistration/fly_motor_tie_break.json`. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from neuromps.fly import design, jobs, motor, scoring
from neuromps.fly.study import PROTOCOLS, memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "fly_motor_tie_break.json"
SPONTANEOUS_SEEDS = (500, 501, 502)
THRESHOLD = 0.5


def fine_conditions(fine: dict, tied: dict) -> list[design.Condition]:
    """The fine levels of each tied family, then its tied level."""
    out = []
    for family, entry in fine.items():
        for level in [*entry["levels"], tied[family]]:
            out.append(design.Condition(family, level, {entry["knob"]: level}))
    return out


def resolve(series: pd.DataFrame, motor_threshold: float, chip_threshold: float) -> dict:
    """First dose at which each readout reaches its threshold, and which came first."""

    def onset(column, threshold):
        hit = series[series[column] >= threshold].sort_values("dose")
        return None if hit.empty else round(float(hit.iloc[0]["dose"]), 4)

    m, c = onset("motor_disruption", motor_threshold), onset("abs_delta_mfr", chip_threshold)
    if m is not None and (c is None or m < c):
        resolution = "chip later"
    elif c is not None and (m is None or c < m):
        resolution = "chip earlier"
    else:
        resolution = "still tied"
    return {"motor_onset": m, "chip_onset": c, "resolution": resolution}


def _spont(args) -> dict:
    overrides, seed = args
    return jobs.spontaneous(overrides, seed, False)


def run(context, conds: list[design.Condition]) -> dict:
    """Spontaneous recordings and behaviour trials of the fine and tied levels."""
    spont = [(c, s) for c in conds for s in SPONTANEOUS_SEEDS]
    trials = [(c, p, s) for c in conds for p in PROTOCOLS for s in motor.SEEDS]
    workers = memory_bounded_workers(per_worker_gb=motor.WORKERS_GB)
    with ProcessPoolExecutor(workers, initializer=motor._init, initargs=(context,)) as pool:
        f_s = [pool.submit(_spont, (c.overrides, s)) for c, s in spont]
        f_t = [pool.submit(motor._trial, (c.overrides, p, s)) for c, p, s in trials]
        return {
            "spontaneous": [(c, s, f.result()) for (c, s), f in zip(spont, f_s, strict=True)],
            "trials": [(c, p, f.result()) for (c, p, _), f in zip(trials, f_t, strict=True)],
        }


def table(raw: dict, conds: list[design.Condition]) -> pd.DataFrame:
    """Per level: chip |delta log2 mfr| and motor disruption, against the parent references."""
    from neuromps.twin.perturbation import simulated_delta

    fly = outputs_dir() / "fly"
    replication = pd.read_csv(fly / "malecns_conditions_SIMULATED.csv")
    parent = pd.read_csv(fly / "malecns_motor_SIMULATED.csv")
    vehicle_row = replication[replication["condition"] == "vehicle"].iloc[0]
    vehicle_chip = {
        c.removeprefix("chip."): vehicle_row[c]
        for c in replication.columns
        if c.startswith("chip.")
    }
    vehicle_motor = parent[parent["condition"] == "vehicle"].iloc[0]
    chip = scoring._chip({"spontaneous": raw["spontaneous"]})
    trials = pd.DataFrame([{"condition": c.name, "protocol": p, **r} for c, p, r in raw["trials"]])
    m = trials.groupby(["condition", "protocol"]).mean(numeric_only=True)
    rows = []
    for c in conds:
        readouts = {
            "proboscis_sugar": float(m.loc[(c.name, "sugar"), "proboscis"]),
            "proboscis_sugar_bitter": float(m.loc[(c.name, "sugar_bitter"), "proboscis"]),
            "front_leg_jon_ce": float(m.loc[(c.name, "jon_ce"), "front_leg"]),
            "motor_recruited_sugar": float(m.loc[(c.name, "sugar"), "motor_recruited"]),
        }
        delta = simulated_delta(chip[c.name], vehicle_chip)["mfr_hz"]
        rows.append(
            {
                "condition": c.name,
                "family": c.family,
                "level": c.level,
                "dose": c.dose,
                "abs_delta_mfr": abs(delta),
                **readouts,
                "motor_disruption": motor.disruption(readouts, vehicle_motor),
            }
        )
    return pd.DataFrame(rows)


def score(frame: pd.DataFrame, spec: dict) -> dict:
    """TB0 and TB1, and each tied family's resolution on the combined coarse and fine series."""
    parent = pd.read_csv(outputs_dir() / "fly" / "malecns_motor_SIMULATED.csv")
    tied = spec["design"]["reproducibility_levels"]
    diffs = []
    for family, level in tied.items():
        new = frame[(frame["family"] == family) & (frame["level"] == level)].iloc[0]
        old = parent[(parent["family"] == family) & (parent["level"] == level)].iloc[0]
        diffs += [abs(new[c] - old[c]) for c in ("abs_delta_mfr", "motor_disruption")]
    tb0 = float(max(diffs))
    resolutions = {}
    for family, level in tied.items():
        tie_dose = float(
            frame[(frame["family"] == family) & (frame["level"] == level)]["dose"].iloc[0]
        )
        below = parent[(parent["family"] == family) & (parent["dose"] < tie_dose - 1e-9)]
        series = pd.concat(
            [
                below[["dose", "motor_disruption", "abs_delta_mfr"]],
                frame[frame["family"] == family][["dose", "motor_disruption", "abs_delta_mfr"]],
            ]
        )
        resolutions[family] = resolve(series, THRESHOLD, THRESHOLD)
    tb1 = all(r["resolution"] == "chip later" for r in resolutions.values())
    preds = spec["predictions"]
    return {
        "TB0_tied_levels_reproduce": {
            "test": preds["TB0_tied_levels_reproduce"]["test"],
            "value": tb0,
            "outcome": "pass" if tb0 <= 1e-9 else "fail",
        },
        "TB1_both_ties_resolve_chip_later": {
            "test": preds["TB1_both_ties_resolve_chip_later"]["test"],
            "value": {f: r["resolution"] for f, r in resolutions.items()},
            "outcome": "pass" if tb1 else "fail",
        },
    }, resolutions


def plot(frame: pd.DataFrame, path) -> None:
    """Chip |delta| and motor disruption across each tie step, against the 0.5 thresholds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neuromps.fly.figures import LABELS

    parent = pd.read_csv(outputs_dir() / "fly" / "malecns_motor_SIMULATED.csv")
    families = list(dict.fromkeys(frame["family"]))
    fig, axes = plt.subplots(1, len(families), figsize=(3.0 * len(families), 2.5), squeeze=False)
    for ax, family in zip(axes[0], families, strict=True):
        rows = pd.concat([parent[parent["family"] == family], frame[frame["family"] == family]])
        rows = rows.drop_duplicates("dose", keep="last").sort_values("dose")
        ax.plot(
            rows["dose"],
            rows["motor_disruption"],
            "^-",
            color="#CC79A7",
            ms=4,
            label="motor disruption",
        )
        ax.plot(
            rows["dose"],
            rows["abs_delta_mfr"],
            "D-",
            color="#D55E00",
            ms=3.5,
            label="chip |log2 change in firing rate|",
        )
        ax.axhline(THRESHOLD, color="black", ls="--", lw=0.8)
        ax.set_title(LABELS[family], fontsize=7.5)
        ax.set_xlabel("knob change from untreated", fontsize=6.5)
        ax.tick_params(labelsize=6)
    axes[0][0].legend(fontsize=5.5, frameon=False)
    fig.suptitle("SIMULATED male CNS: finer doses inside the two tied steps", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: run the fine levels, score TB0-TB1, write outputs and metrics."""
    from neuromps.fly.replication import _context
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    conds = fine_conditions(spec["design"]["fine_levels"], spec["design"]["reproducibility_levels"])
    context = _context(design.registration())
    frame = table(run(context, conds), conds)
    predictions, resolutions = score(frame, spec)
    result = {
        "label": "SIMULATED male CNS: motor-level ties resolved on a finer grid",
        "predictions": predictions,
        "resolutions": resolutions,
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    frame.to_csv(out / "malecns_motor_tie_break_SIMULATED.csv", index=False)
    (out / "malecns_motor_tie_break_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("fly_motor_tie_break", scoring.clean(result))
    plot(frame, figures_dir() / "fly_motor_tie_break_SIMULATED.png")
    for name, entry in predictions.items():
        print(f"{name:38s} {entry['outcome']:>5}  {entry['value']}")
    print(json.dumps(resolutions, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
