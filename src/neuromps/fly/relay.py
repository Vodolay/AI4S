"""Embodied fly, stage 1e: the FlyWire brain drives the male nerve cord through matched DNs.

``python -m neuromps.fly.relay`` runs the registered design
(`preregistration/embodied_stage1e_flywire_brain_to_male_nerve_cord.json`): the
published brain model on FlyWire under antennal, sugar or no drive; every
descending neuron's spikes recorded; each replayed into the male-CNS descending
neuron of the same cell type and side; the stage-1c nerve cord moving the
FlyMimic leg; and stage 1d's readouts. A lesion removes the grooming descending
neurons' spikes. Open loop. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import (
    behaviour,
    body,
    connectome,
    design,
    embodied_stage1,
    embodied_stage1c,
    kinematics,
    malecns,
    model,
    scoring,
)
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = (
    REPO_ROOT / "preregistration" / "embodied_stage1e_flywire_brain_to_male_nerve_cord.json"
)
SIDE = {"left": "L", "right": "R"}
GROOMING_TYPES = ("DNg62", "DNge078")
DRIVES = {"jon_ce": "jon_ce", "sugar": "sugar_grn_right", "none": None}
SEEDS = range(750, 758)
NEAR_HEAD_MM = body.REST_DISTANCE_MM - 0.2
_STATE: dict = {}


def match(flywire: pd.DataFrame, male: pd.DataFrame) -> dict[int, int]:
    """Male-CNS body ID -> FlyWire root ID, by cell type and side, paired in ID order."""
    out: dict[int, int] = {}
    fw = flywire.assign(code=flywire["side"].map(SIDE))
    for (kind, side), rows in fw.groupby(["cell_type", "code"]):
        sources = sorted(int(r) for r in rows["root_id"])
        targets = sorted(
            int(b)
            for b, names, s in zip(male["bodyId"], male["names"], male["somaSide"], strict=True)
            if kind in names and s == side
        )
        for k, target in enumerate(targets):
            out[target] = sources[k % len(sources)]
    return out


def coverage(pairs: dict[int, int], flywire_ids, male_ids) -> dict[str, int]:
    """How many neurons on each side of the match take part."""
    used = set(pairs.values())
    return {
        "flywire_matched": sum(int(i) in used for i in flywire_ids),
        "flywire_total": len(list(flywire_ids)),
        "male_receiving": sum(int(i) in pairs for i in male_ids),
        "male_total": len(list(male_ids)),
    }


def _brain_init() -> None:
    con = connectome.cached()
    ann = connectome.read_annotations().drop_duplicates("root_id")
    dn = ann[ann["super_class"] == "descending"]
    ids = [r for r in dn["root_id"] if r in set(con.root_ids.tolist())]
    _STATE.update(
        brain=con, dn_index=con.index_of(ids), dn_ids=np.asarray(ids), sets=behaviour.neuron_sets()
    )


def _brain(job) -> dict:
    """One FlyWire trial: spike times of every descending neuron, by root ID."""
    drive, seed = job
    con = _STATE["brain"]
    group = DRIVES[drive]
    excite = con.index_of(_STATE["sets"][group]) if group else []
    spikes = model.simulate(con, model.FlyParams(seed=seed, stim_rate_hz=150.0), excite=excite)
    out = {}
    for root, index in zip(_STATE["dn_ids"], _STATE["dn_index"], strict=True):
        times = spikes.times[spikes.neurons == index]
        if times.size:
            out[int(root)] = times
    return {"drive": drive, "seed": seed, "dn_spikes": out}


def _cord_init(pairs: dict[int, int]) -> None:
    embodied_stage1c._init()
    sub = embodied_stage1c._STATE["con"]
    position = {int(b): k for k, b in enumerate(sub.root_ids)}
    _STATE.update(pairs={position[b]: f for b, f in pairs.items() if b in position})


def _cord(job) -> dict:
    """The nerve cord driven by one brain trial's descending spikes (optionally lesioned)."""
    brain, lesion, removed = job
    state = embodied_stage1c._STATE
    replay = {}
    for target, source in _STATE["pairs"].items():
        if lesion and source in removed:
            continue
        times = brain["dn_spikes"].get(source)
        if times is not None:
            replay[target] = times
    con = state["con"]
    params = model.FlyParams(duration_s=embodied_stage1.DURATION_S, seed=brain["seed"])
    spikes = model.simulate(con, params, replay=replay)
    counts = np.bincount(spikes.neurons, minlength=con.n_neurons) / spikes.duration_s
    neurons = sorted({n for v in state["assignment"].neurons_by_muscle.values() for n in v})
    label = brain["drive"] + ("_lesion" if lesion else "")
    return {
        "drive": label,
        "seed": brain["seed"],
        "spikes": {n: spikes.times[spikes.neurons == n] for n in neurons},
        "assignment": state["assignment"],
        "pool_rates": {leg: float(counts[idx].mean()) for leg, idx in state["pools"].items()},
        "replayed_neurons": len(replay),
    }


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    fw = connectome.read_annotations().drop_duplicates("root_id")
    fw = fw[fw["super_class"] == "descending"][["root_id", "cell_type", "side"]]
    _, ann = malecns.cached()
    full = pd.read_feather(malecns.RAW / malecns.source.FILES[1])
    male = full[(full["status"] == "Traced") & (full["superclass"] == "descending_neuron")].copy()
    male["names"] = male.apply(malecns._names, axis=1)
    return fw, male[["bodyId", "names", "somaSide"]]


def score(table: pd.DataFrame, pools: pd.DataFrame, grooming_rate: float, spec: dict) -> dict:
    """E0-E4 with values and outcomes; E1-E4 uninterpretable if E0 fails."""

    def e1(scale):
        med = table[table["r_full"] == scale].groupby("drive")["time_near_head"].median()
        ok = med["jon_ce"] >= 0.5 and all(med["jon_ce"] >= med[d] + 0.25 for d in ("sugar", "none"))
        return {k: round(float(v), 4) for k, v in med.items()}, bool(ok)

    primary, ok1 = e1(20.0)
    rates = pools.groupby("drive")[list(embodied_stage1c.LEGS)].mean()
    j = rates.loc["jon_ce"]
    others = {f"r_full_{s:g}": e1(s) for s in (10.0, 40.0)}
    found = {
        "E0_the_grooming_neurons_are_reached": (round(grooming_rate, 3), grooming_rate >= 10),
        "E1_antennal_input_holds_the_leg_at_the_head": (primary, ok1),
        "E2_the_front_leg_is_the_one_that_moves": (
            {
                "jon_ce": {leg: round(float(j[leg]), 3) for leg in embodied_stage1c.LEGS},
                "front_under_sugar": round(float(rates.loc["sugar", "front"]), 3),
            },
            bool(
                j["front"] > j["middle"]
                and j["front"] > j["hind"]
                and j["front"] > rates.loc["sugar", "front"]
            ),
        ),
        "E3_the_activation_scale_does_not_decide_it": (
            {k: v[0] for k, v in others.items()},
            all(v[1] for v in others.values()),
        ),
        "E4_the_reach_runs_through_the_grooming_neurons": (
            {"intact": primary["jon_ce"], "lesioned": primary["jon_ce_lesion"]},
            bool(primary["jon_ce_lesion"] <= primary["jon_ce"] - 0.25),
        ),
    }
    gate = found["E0_the_grooming_neurons_are_reached"][1]
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        if not gate and not name.startswith("E0"):
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    return out


def plot(table: pd.DataFrame, pools: pd.DataFrame, path) -> None:
    """Time near the head per condition, and the three left legs' motor pools."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["jon_ce", "jon_ce_lesion", "sugar", "none"]
    labels = ["antennal input", "antennal input,\naDN1+aDN2 removed", "sugar input", "no input"]
    colours = ["#D55E00", "#E69F00", "#0072B2", "#999999"]
    rows = table[table["r_full"] == 20.0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for k, cond in enumerate(order):
        values = rows[rows["drive"] == cond]["time_near_head"].to_numpy()
        left.bar(k, np.median(values), color=colours[k], alpha=0.6)
        left.scatter(
            np.full(values.size, k) + np.linspace(-0.15, 0.15, values.size),
            values,
            s=10,
            color=colours[k],
            zorder=3,
        )
    left.set_xticks(range(len(order)), labels, fontsize=6)
    left.set_ylabel("fraction of the second, claw near the head", fontsize=7)
    left.set_ylim(0, 1.05)
    left.tick_params(labelsize=6)
    means = pools.groupby("drive")[list(embodied_stage1c.LEGS)].mean().reindex(order)
    width = 0.2
    for k, cond in enumerate(order):
        right.bar(np.arange(3) + (k - 1.5) * width, means.loc[cond], width, color=colours[k])
    right.set_xticks(range(3), [f"left {leg} leg" for leg in embodied_stage1c.LEGS], fontsize=6.5)
    right.set_ylabel("motor pool rate (Hz)", fontsize=7)
    right.tick_params(labelsize=6)
    fig.suptitle(
        "SIMULATED FlyWire brain driving the male nerve cord and the FlyMimic leg", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(seeds) -> dict:
    """Brain trials, matched replay into the nerve cord (intact, and lesioned under JO-CE), the leg.

    Returns the per-trial readout table, the motor-pool table, the matching coverage and the
    grooming descending neurons' mean rate under each drive.
    """
    fw, male = _tables()
    pairs = match(fw, male)
    grooming_ids = set(fw.loc[fw["cell_type"].isin(GROOMING_TYPES), "root_id"].astype(int))
    with ProcessPoolExecutor(memory_bounded_workers(), initializer=_brain_init) as pool:
        brains = list(pool.map(_brain, [(d, s) for d in DRIVES for s in seeds]))
    grooming_rate = {
        drive: float(
            np.mean(
                [
                    sum(b["dn_spikes"].get(i, np.empty(0)).size for i in grooming_ids)
                    / len(grooming_ids)
                    for b in brains
                    if b["drive"] == drive
                ]
            )
        )
        for drive in DRIVES
    }
    jon = [b for b in brains if b["drive"] == "jon_ce"]
    work = [(b, False, grooming_ids) for b in brains] + [(b, True, grooming_ids) for b in jon]
    with ProcessPoolExecutor(
        memory_bounded_workers(per_worker_gb=8.0), initializer=_cord_init, initargs=(pairs,)
    ) as pool:
        cords = list(pool.map(_cord, work))
    with ProcessPoolExecutor(12) as pool:
        moves = list(
            pool.map(embodied_stage1._move, [(c, r) for c in cords for r in embodied_stage1.R_FULL])
        )
    table = pd.DataFrame(
        [
            {
                "drive": m["drive"],
                "seed": m["seed"],
                "r_full": m["r_full"],
                "min_distance_mm": m["min_distance_mm"],
                "time_near_head": kinematics.time_near_head(m["distance_mm"], NEAR_HEAD_MM),
                "rhythmicity": kinematics.rhythmicity(m["distance_mm"]),
            }
            for m in moves
        ]
    )
    pools = pd.DataFrame(
        [
            {
                "drive": c["drive"],
                "seed": c["seed"],
                "replayed_neurons": c["replayed_neurons"],
                **c["pool_rates"],
            }
            for c in cords
        ]
    )
    return {
        "table": table,
        "pools": pools,
        "grooming_rate": grooming_rate,
        "matching": coverage(pairs, fw["root_id"], male["bodyId"]),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: brain trials, matched replay into the nerve cord (intact and lesioned), the leg."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    ran = run(SEEDS)
    table, pools, cover = ran["table"], ran["pools"], ran["matching"]
    grooming_rate = ran["grooming_rate"]["jon_ce"]
    result = {
        "label": "SIMULATED FlyWire brain -> male nerve cord -> FlyMimic leg (open loop)",
        "predictions": score(table, pools, grooming_rate, spec),
        "matching": cover,
        "replayed_neurons_median": pools.groupby("drive")["replayed_neurons"].median().to_dict(),
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage1e_SIMULATED.csv", index=False)
    pools.to_csv(out / "embodied_stage1e_pools_SIMULATED.csv", index=False)
    (out / "embodied_stage1e_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage1e", scoring.clean(result))
    plot(table, pools, figures_dir() / "embodied_stage1e_relay_SIMULATED.png")
    for name, entry in result["predictions"].items():
        print(f"{name:46s} {entry['outcome']:>15}  {entry['value']}")
    print("matching", cover, "replayed", result["replayed_neurons_median"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
