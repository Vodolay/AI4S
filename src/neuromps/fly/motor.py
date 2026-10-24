"""Motor-neuron readouts on the male CNS: behaviour one step closer (registered study).

The published readouts are named interneurons (MN9 excepted). MaleCNS includes
the nerve cord, so the same protocols can be read at the motor pools that move
the body: the proboscis pool for feeding and its bitter gate, the front-leg pool
for antennal grooming, and how many motor neurons any drive recruits. The
behaviour trials repeat the replication's seeds, so its interneuron readouts
must come back unchanged. See `preregistration/fly_motor_readout.json`.
Everything here is **simulated**; the readouts are firing rates of motor
neurons, not movements.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neuromps.fly import analysis, behaviour, design, jobs, model, scoring
from neuromps.fly.study import PROTOCOLS, memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "fly_motor_readout.json"
MOTOR = ("vnc_motor", "cb_motor")
READOUTS = (
    "proboscis_sugar",
    "proboscis_sugar_bitter",
    "front_leg_jon_ce",
    "motor_recruited_sugar",
)
SEEDS = range(600, 608)
WORKERS_GB = 12.0
_STATE: dict = {}


def pools(superclass: np.ndarray, subclass: np.ndarray) -> dict[str, np.ndarray]:
    """Model indices of the proboscis, front-leg and all motor neurons."""
    is_motor = np.isin(superclass.astype(str), MOTOR)
    return {
        "proboscis": np.flatnonzero(is_motor & (subclass == "pm")),
        "front_leg": np.flatnonzero(is_motor & (subclass == "fl")),
        "all_motor": np.flatnonzero(is_motor),
    }


def disruption(treated: dict, vehicle: dict) -> float:
    """Mean absolute log change of the four motor readouts."""
    return float(np.mean([abs(analysis.log_change(treated[k], vehicle[k])) for k in READOUTS]))


def first_doses(table: pd.DataFrame, column: str, threshold: float) -> dict[str, float | None]:
    """Per family: the lowest dose at which `column` reaches the threshold."""
    out = {}
    for family, rows in table[table["family"] != "vehicle"].groupby("family"):
        hit = rows[rows[column] >= threshold]
        out[family] = None if hit.empty else float(hit["dose"].min())
    return out


def compare_onsets(
    table, first: str, second: str, t_first: float, t_second: float, strict: bool
) -> dict:
    """Families (impaired at both) where `first`'s onset is earlier (strict) or no later."""
    a, b = first_doses(table, first, t_first), first_doses(table, second, t_second)
    scored = [f for f in a if a[f] is not None and b.get(f) is not None]
    meeting = sum((a[f] < b[f]) if strict else (a[f] <= b[f]) for f in scored)
    return {"meeting": int(meeting), "scored": len(scored)}


def _init(context: jobs.Context) -> None:
    from neuromps.fly import malecns

    jobs.init(context, "malecns")
    con = jobs._STATE["con"]
    full = pd.read_feather(malecns.RAW / malecns.source.FILES[1])
    full = full[full["status"] == "Traced"].set_index("bodyId").reindex(con.root_ids)
    _STATE["pools"] = pools(
        full["superclass"].to_numpy(dtype=object), full["subclass"].to_numpy(dtype=object)
    )


def _trial(job) -> dict:
    """One trial, exactly as the replication ran it, read at interneurons and motor pools."""
    overrides, protocol, seed = job
    ctx, con = jobs._STATE["context"], jobs._STATE["con"]
    sets = ctx.sets
    rate = ctx.spec["design"]["behaviour"]["stim_rate_hz"]
    drive = {
        "sugar": ("sugar_grn_right", None),
        "sugar_bitter": ("sugar_grn_right", "bitter_grn"),
        "jon_ce": ("jon_ce", None),
    }[protocol]
    params = jobs._base(
        stim_rate_hz=rate, stim2_rate_hz=rate if drive[1] else 0.0, seed=seed, **overrides
    )
    spikes = model.simulate(
        con,
        params,
        excite=con.index_of(sets[drive[0]]),
        excite2=con.index_of(sets[drive[1]]) if drive[1] else [],
    )
    out = {
        name: behaviour.rate_hz([spikes], con.index_of(sets[name]))
        for name in ("mn9", "abn1", "adn1")
    }
    out["dn_recruited"] = behaviour.recruited([spikes], ctx.descending)
    p = _STATE["pools"]
    out["proboscis"] = behaviour.rate_hz([spikes], p["proboscis"])
    out["front_leg"] = behaviour.rate_hz([spikes], p["front_leg"])
    out["motor_recruited"] = behaviour.recruited([spikes], p["all_motor"])
    return out


def chip_late(table: pd.DataFrame, impairment: float, detection: float) -> dict:
    """Families with any motor impairment in which the chip detects later, or never."""
    motor = first_doses(table, "motor_disruption", impairment)
    chip = first_doses(table, "abs_delta_mfr", detection)
    scored = [f for f, dose in motor.items() if dose is not None]
    meeting = sum(chip.get(f) is None or motor[f] < chip[f] for f in scored)
    return {"meeting": int(meeting), "scored": len(scored)}


def table(rows: pd.DataFrame, chip: pd.DataFrame) -> pd.DataFrame:
    """Per condition: interneuron and motor readouts, motor disruption, and the chip's columns."""
    m = rows.groupby(["condition", "protocol"]).mean(numeric_only=True)

    def at(c, protocol, column):
        return float(m.loc[(c, protocol), column])

    out = []
    for c in chip["condition"]:
        out.append(
            {
                "condition": c,
                "mn9_sugar": at(c, "sugar", "mn9"),
                "mn9_sugar_bitter": at(c, "sugar_bitter", "mn9"),
                "abn1_jon_ce": at(c, "jon_ce", "abn1"),
                "dn_recruited": at(c, "sugar", "dn_recruited"),
                "proboscis_sugar": at(c, "sugar", "proboscis"),
                "proboscis_sugar_bitter": at(c, "sugar_bitter", "proboscis"),
                "front_leg_jon_ce": at(c, "jon_ce", "front_leg"),
                "front_leg_sugar": at(c, "sugar", "front_leg"),
                "motor_recruited_sugar": at(c, "sugar", "motor_recruited"),
            }
        )
    frame = pd.DataFrame(out)
    vehicle = frame[frame["condition"] == "vehicle"].iloc[0]
    frame["motor_disruption"] = [disruption(r, vehicle) for _, r in frame.iterrows()]
    keep = ["condition", "family", "level", "dose", "delta.mfr_hz", "abs_delta_mfr", "disruption"]
    return chip[keep].merge(frame, on="condition")


def score(frame: pd.DataFrame, chip: pd.DataFrame, spec: dict) -> dict:
    """MR0-MR6 with values and outcomes; MR2-MR6 uninterpretable if MR0 or MR1 fails."""
    d = spec["design"]
    cols = ["mn9_sugar", "mn9_sugar_bitter", "abn1_jon_ce", "dn_recruited"]
    merged = frame.merge(chip[["condition", *cols]], on="condition", suffixes=("", "_rep"))
    mr0 = float(max((merged[c] - merged[f"{c}_rep"]).abs().max() for c in cols))
    v = frame[frame["family"] == "vehicle"].iloc[0]
    mr1 = bool(
        v["proboscis_sugar"] >= 10
        and v["proboscis_sugar_bitter"] <= 0.5 * v["proboscis_sugar"]
        and v["front_leg_jon_ce"] >= 1.5 * v["front_leg_sugar"]
    )

    def trend(family, column):
        rows = frame[frame["family"].isin(["vehicle", family])]
        rho = spearmanr(rows["dose"], rows[column])[0]
        return None if not np.isfinite(rho) else round(float(rho), 4)

    mr2 = trend("gaba_block", "proboscis_sugar_bitter")
    mr3 = {
        "proboscis_sugar": trend("ach_block", "proboscis_sugar"),
        "front_leg_jon_ce": trend("ach_block", "front_leg_jon_ce"),
    }
    drugs = frame[frame["family"] != "vehicle"]
    mr4 = round(float(spearmanr(drugs["abs_delta_mfr"], drugs["motor_disruption"])[0]), 4)
    late = chip_late(frame, d["impairment_threshold"], d["chip_detection_threshold"])
    order = compare_onsets(
        frame,
        "motor_disruption",
        "disruption",
        d["impairment_threshold"],
        d["impairment_threshold"],
        strict=False,
    )
    found = {
        "MR0_interneuron_readouts_reproduce": (mr0, mr0 <= 1e-9),
        "MR1_the_motor_pools_carry_the_behaviours": (
            {
                k: round(float(v[k]), 3)
                for k in (
                    "proboscis_sugar",
                    "proboscis_sugar_bitter",
                    "front_leg_jon_ce",
                    "front_leg_sugar",
                )
            },
            mr1,
        ),
        "MR2_disinhibition_releases_the_gate_at_the_motor_level": (
            mr2,
            mr2 is not None and mr2 >= 0.8,
        ),
        "MR3_excitation_block_silences_motor_output": (
            mr3,
            all(x is not None and x <= -0.8 for x in mr3.values()),
        ),
        "MR4_chip_tracks_motor_disruption": (mr4, mr4 >= 0.6),
        "MR5_chip_still_late_at_the_motor_level": (late, late["meeting"] >= 4),
        "MR6_motor_output_fails_no_later_than_interneurons": (order, order["meeting"] >= 4),
    }
    gate = found["MR0_interneuron_readouts_reproduce"][1] and mr1
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        if not gate and name[:3] not in ("MR0", "MR1"):
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    return out


def summary(frame: pd.DataFrame, spec: dict) -> dict:
    """Citable per-family onsets at each level, and how the chip compares with motor output."""
    d = spec["design"]
    motor_onset = first_doses(frame, "motor_disruption", d["impairment_threshold"])
    interneuron = first_doses(frame, "disruption", d["impairment_threshold"])
    chip = first_doses(frame, "abs_delta_mfr", d["chip_detection_threshold"])
    scored = [f for f, v in motor_onset.items() if v is not None]
    return {
        "first_dose": {"motor": motor_onset, "interneuron": interneuron, "chip": chip},
        "families_chip_later_than_motor": int(
            sum(chip[f] is None or chip[f] > motor_onset[f] for f in scored)
        ),
        "families_chip_tied_with_motor": int(sum(chip[f] == motor_onset[f] for f in scored)),
        "families_chip_earlier_than_motor": int(
            sum(chip[f] is not None and chip[f] < motor_onset[f] for f in scored)
        ),
        "families_motor_earlier_than_interneuron": int(
            sum(interneuron.get(f) is not None and motor_onset[f] < interneuron[f] for f in scored)
        ),
        "families_scored": len(scored),
    }


def rescore() -> int:
    """Rewrite the summary and figure from the saved table, without simulating."""
    from neuromps.paths import figures_dir

    spec = json.loads(REGISTRATION.read_text())
    out = outputs_dir() / "fly"
    frame = pd.read_csv(out / "malecns_motor_SIMULATED.csv")
    result = json.loads((out / "malecns_motor_SIMULATED.json").read_text())
    result["summary"] = summary(frame, spec)
    (out / "malecns_motor_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("fly_motor_readout", scoring.clean(result))
    plot(frame, spec, figures_dir() / "fly_motor_onsets_SIMULATED.png")
    print(json.dumps(result["summary"], indent=1))
    return 0


def plot(frame: pd.DataFrame, spec: dict, path) -> None:
    """Per family: first dose impairing interneuron readouts, motor readouts, and chip detection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neuromps.fly.figures import LABELS

    d = spec["design"]
    marks = (
        ("disruption", d["impairment_threshold"], "#000000", "s", "interneuron readouts impaired"),
        ("motor_disruption", d["impairment_threshold"], "#CC79A7", "^", "motor readouts impaired"),
        ("abs_delta_mfr", d["chip_detection_threshold"], "#D55E00", "D", "chip: firing rate"),
    )
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    families = [f for f in LABELS if f in set(frame["family"])]
    for row, family in enumerate(families):
        for k, (column, threshold, colour, marker, label) in enumerate(marks):
            dose = first_doses(frame, column, threshold).get(family)
            if dose is not None:
                ax.scatter(
                    dose,
                    row + (k - 1) * 0.18,
                    color=colour,
                    marker=marker,
                    s=24,
                    label=label if row == 0 else None,
                    zorder=3,
                )
    ax.set_yticks(range(len(families)), [LABELS[f] for f in families], fontsize=6.5)
    ax.invert_yaxis()
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.set_xlabel("knob change from untreated (first dose reaching threshold)", fontsize=6.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    ax.set_title("SIMULATED male CNS: motor output against interneurons and the chip", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pool_sizes() -> dict[str, int]:
    from neuromps.fly import malecns

    con, _ = malecns.cached()
    full = pd.read_feather(malecns.RAW / malecns.source.FILES[1])
    full = full[full["status"] == "Traced"].set_index("bodyId").reindex(con.root_ids)
    found = pools(
        full["superclass"].to_numpy(dtype=object), full["subclass"].to_numpy(dtype=object)
    )
    return {k: int(v.size) for k, v in found.items()}


def main(argv: list[str] | None = None) -> int:
    """CLI: run the trials, build the table, score MR0-MR6, write outputs and metrics."""
    from neuromps.fly.replication import _context
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    context = _context(design.registration())
    conditions = design.conditions(context.spec)
    work = [(c.overrides, p, s) for c in conditions for p in PROTOCOLS for s in SEEDS]
    names = [c.name for c in conditions for _ in PROTOCOLS for _ in SEEDS]
    workers = memory_bounded_workers(per_worker_gb=WORKERS_GB)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(context,)) as pool:
        got = list(pool.map(_trial, work))
    rows = pd.DataFrame(
        [{"condition": n, "protocol": w[1], **r} for n, w, r in zip(names, work, got, strict=True)]
    )
    chip = pd.read_csv(outputs_dir() / "fly" / "malecns_conditions_SIMULATED.csv")
    frame = table(rows, chip)
    result = {
        "label": "SIMULATED male CNS: motor-neuron readouts",
        "predictions": score(frame, chip, spec),
        "pool_sizes": _pool_sizes(),
        "summary": None,
        "registration": design.provenance(REGISTRATION),
    }
    result["summary"] = summary(frame, spec)
    out = outputs_dir() / "fly"
    frame.to_csv(out / "malecns_motor_SIMULATED.csv", index=False)
    (out / "malecns_motor_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("fly_motor_readout", scoring.clean(result))
    plot(frame, spec, figures_dir() / "fly_motor_onsets_SIMULATED.png")
    for name, entry in result["predictions"].items():
        print(f"{name:56s} {entry['outcome']:>15}  {entry['value']}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(rescore() if "--rescore" in sys.argv else main())
