"""Embodied fly, stage 2g: Pugliese et al.'s front-leg network on the male CNS.

``python -m neuromps.fly.embodied_stage2g`` runs the registered design
(`preregistration/embodied_stage2g_front_leg_network.json`): the left front leg's
motor neurons, the premotor neurons onto them and the descending neurons onto
those, as a firing-rate network (`fly.rate_model`); per draw, the input is halved
or doubled until 5-500 neurons are recruited (Pugliese et al.'s rule); within-leg
rhythm and phases with the audit's corrected measure. Everything is **simulated**.
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
    kinematics,
    rate_model,
    scoring,
)
from neuromps.fly import embodied_stage2c as s2c
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage2g_front_leg_network.json"
SEEDS = list(range(940, 972))
SIZE_RULES = ("synapse_count", "none")
DRIVES = {
    "forward_walking": ("DNg100",),
    "grooming": ("DNg62", "DNge078"),
    "dnb08": ("DNb08",),
    "none": (),
}
CPG = ("IN17A001", "INXXX466", "IN16B036")
POOLS = {
    "promotor": "Tergopleural/Pleural promotor MN",
    "remotor": "Pleural remotor/abductor MN",
    "flexor": "Ti flexor MN",
    "extensor": "Ti extensor MN",
}
START_INPUT, MAX_RUNS, FEWEST, MOST = 250.0, 10, 5, 500
DURATION_S, DT_S, START_S = 2.0, 1e-4, 0.2
_STATE: dict = {}


def front_leg_network(con: connectome.Connectome, motor: np.ndarray, descending: np.ndarray):
    """Keep-mask: the motor neurons, non-descending premotor neurons with a >= 5-synapse
    connection onto them, and descending neurons with such a connection onto a premotor one."""
    strong = con.synapses >= rate_model.SYNAPSE_FLOOR
    is_dn = np.zeros(con.n_neurons, bool)
    is_dn[descending] = True
    is_mn = np.zeros(con.n_neurons, bool)
    is_mn[motor] = True
    onto_mn = strong & is_mn[con.post]
    pre = np.unique(con.pre[onto_mn])
    pre = pre[~is_dn[pre] & ~is_mn[pre]]
    is_pre = np.zeros(con.n_neurons, bool)
    is_pre[pre] = True
    dns = np.unique(con.pre[strong & is_pre[con.post] & is_dn[con.pre]])
    return is_mn | is_pre | np.isin(np.arange(con.n_neurons), dns)


def next_inputs(inputs: np.ndarray, recruited: np.ndarray) -> np.ndarray:
    """Pugliese et al.'s rule: halve above 500 recruited neurons, double below 5."""
    out = np.asarray(inputs, float).copy()
    out[recruited > MOST] /= 2
    out[recruited < FEWEST] *= 2
    return out


def _init(with_volumes: bool = False) -> None:
    """Build the front-leg network; `with_volumes` adds the volume size rule (stage 2i)."""
    from neuromps.fly import malecns

    embodied_stage1c._init()
    full = embodied_stage1c._STATE["con"]
    table = embodied_stage1c._annotations(full.root_ids)
    motor = (table["superclass"] == "vnc_motor") & (table["subclass"] == "fl")
    motor = np.flatnonzero((motor & (table["somaSide"] == "L")).to_numpy())
    descending = np.flatnonzero((table["superclass"] == "descending_neuron").to_numpy())
    keep = front_leg_network(full, motor, descending)
    con = connectome.subnetwork(full, keep)
    sub = table[keep].reset_index(drop=True)
    types = sub["type"].astype(str)
    mn = np.flatnonzero(
        ((sub["superclass"] == "vnc_motor") & (sub["subclass"] == "fl")).to_numpy()
        & (sub["somaSide"] == "L").to_numpy()
    )
    sizes = rate_model.sizes(con)
    rules = {"synapse_count": sizes, "none": np.ones_like(sizes)}
    if with_volumes:
        known = malecns.neuron_sizes()
        volumes = known.reindex(con.root_ids).to_numpy()
        rules["volume"] = rate_model.volume_sizes(volumes)
        cord = known.reindex(full.root_ids).to_numpy()
        rules["volume_cord"] = rate_model.volume_sizes(volumes, reference=cord)
        _STATE["volumes_missing"] = int(np.sum(~np.isfinite(volumes) | (volumes <= 0)))
    _STATE.update(
        con=con,
        w=rate_model.weights(con),
        sizes=rules,
        drive={k: np.flatnonzero(types.isin(v).to_numpy()) for k, v in DRIVES.items()},
        cpg=np.flatnonzero(types.isin(CPG).to_numpy()),
        mn=mn,
        pools={
            k: np.searchsorted(mn, np.intersect1d(np.flatnonzero((types == v).to_numpy()), mn))
            for k, v in POOLS.items()
        },
        counts={"neurons": int(con.n_neurons), "motor": int(mn.size)},
        types=types.to_numpy(),
    )


def _simulate(
    rule: str, drive: str, inputs: np.ndarray, silenced=None, seeds: list[int] = SEEDS
) -> rate_model.Run:
    idx = _STATE["drive"][drive]
    p = rate_model.draw(_STATE["sizes"][rule], seeds)
    amplitude = np.broadcast_to(inputs, (len(idx), len(seeds))) if len(idx) else 0.0
    return rate_model.simulate(
        _STATE["w"],
        p,
        idx,
        amplitude,
        _STATE["mn"],
        duration_s=DURATION_S,
        dt_s=DT_S,
        silenced=silenced,
    )


def _recruited(run: rate_model.Run, drive: str) -> np.ndarray:
    active = run.mean_rate >= fictive.MIN_RATE_HZ
    active[_STATE["drive"][drive]] = False
    return active.sum(axis=0)


def find_inputs(
    rule: str, drive: str, seeds: list[int] = SEEDS
) -> tuple[rate_model.Run, np.ndarray, np.ndarray, int]:
    """The recruitment rule, per draw: final run, inputs, feasibility, runs used."""
    inputs = np.full(len(seeds), START_INPUT)
    for runs in range(1, MAX_RUNS + 1):
        run = _simulate(rule, drive, inputs, seeds=seeds)
        n = _recruited(run, drive)
        ok = (n >= FEWEST) & (n <= MOST)
        if ok.all() or runs == MAX_RUNS:
            return run, inputs, ok, runs
        inputs = np.where(ok, inputs, next_inputs(inputs, n))
    raise AssertionError("unreachable")


def draw_measures(run: rate_model.Run, rep: int) -> dict:
    """One draw: motor rhythm score, median period, pool rates and the two antagonist phases."""
    start = int(round(START_S / 1e-3))
    traces = run.rates[start:, :, rep]
    active = np.flatnonzero(traces.mean(axis=0) >= fictive.MIN_RATE_HZ)
    found = [kinematics.rhythm(traces[:, j], start_s=0.0) for j in active]
    periods = [per for _, per in found if per is not None]
    pools = {k: traces[:, v].mean(axis=1) for k, v in _STATE["pools"].items()}
    rates = {k: float(v.mean()) for k, v in pools.items()}

    def ph(a: str, b: str):
        if rates[a] < fictive.MIN_RATE_HZ or rates[b] < fictive.MIN_RATE_HZ:
            return None
        return fictive.phase(pools[a], pools[b], method="audited")

    return {
        "score": float(np.mean([sc for sc, _ in found])) if found else 0.0,
        "period_ms": float(np.median(periods) * 1000) if periods else None,
        "active_motor": int(active.size),
        "phase_promotor_remotor": ph("promotor", "remotor"),
        "phase_flexor_extensor": ph("flexor", "extensor"),
        **{f"rate_{k}": v for k, v in rates.items()},
    }


def run_job(job, seeds: list[int] = SEEDS, search=None) -> dict:
    """One (size rule, drive): the input search (stage 2g's rule unless `search` is given),
    then measures per draw; the lesion for forward walking at the final inputs. Example pool
    traces of the first feasible draw. `search(rule, drive, seeds)` returns
    (run, inputs, feasible, runs) and may add a fifth item of per-draw extras."""
    rule, drive = job
    extras: dict = {}
    if drive == "none":
        run = _simulate(rule, drive, np.zeros(len(seeds)), seeds=seeds)
        inputs, ok, runs = np.zeros(len(seeds)), np.ones(len(seeds), bool), 1
    else:
        found = (search or find_inputs)(rule, drive, seeds)
        run, inputs, ok, runs = found[:4]
        extras = found[4] if len(found) > 4 else {}
    rows = []
    n = _recruited(run, drive)
    for rep, seed in enumerate(seeds):
        rows.append(
            {
                "size_rule": rule,
                "condition": drive,
                "seed": seed,
                "input": float(inputs[rep]),
                "feasible": bool(ok[rep]),
                "recruited": int(n[rep]),
                **{k: v[rep] for k, v in extras.items()},
                **draw_measures(run, rep),
            }
        )
    out = {"rows": rows, "runs": runs}
    if ok.any():
        first = int(np.flatnonzero(ok)[0])
        start = int(round(START_S / 1e-3))
        out["example"] = {
            "seed": seeds[first],
            **{k: run.rates[start:, v, first].mean(axis=1) for k, v in _STATE["pools"].items()},
        }
    if drive == "forward_walking":
        lesion = _simulate(rule, drive, inputs, silenced=_STATE["cpg"], seeds=seeds)
        for rep, seed in enumerate(seeds):
            rows.append(
                {
                    "size_rule": rule,
                    "condition": "lesion",
                    "seed": seed,
                    "input": float(inputs[rep]),
                    "feasible": bool(ok[rep]),
                    "recruited": int(_recruited(lesion, drive)[rep]),
                    **draw_measures(lesion, rep),
                }
            )
    return out


def resultant(phases: pd.Series) -> float | None:
    """Mean resultant length of phases (cycles); None when there are none."""
    v = phases.dropna().to_numpy(dtype=float)
    return None if v.size == 0 else float(np.abs(np.mean(np.exp(2j * np.pi * v))))


def score(table: pd.DataFrame, spec: dict) -> dict:
    """P0-P6 for one size rule, over feasible draws."""
    feas = table[table["feasible"]]
    fw = feas[feas["condition"] == "forward_walking"]
    gr = feas[feas["condition"] == "grooming"]
    lesion = feas[feas["condition"] == "lesion"]
    n_ok = int(len(fw))
    s_fw = float(fw["score"].median()) if n_ok else 0.0
    s_gr = float(gr["score"].median()) if len(gr) else 0.0
    s_le = float(lesion["score"].median()) if len(lesion) else 0.0
    defined = fw["phase_promotor_remotor"].dropna()
    r = resultant(fw["phase_promotor_remotor"])
    p3 = fictive._close(fw["phase_flexor_extensor"], 0.5) if n_ok else None
    periods = fw["period_ms"].dropna()
    period = float(periods.median()) if len(periods) else None
    p1 = s_fw >= 0.3
    found = {
        "P0_the_rule_finds_a_working_input": (n_ok, n_ok >= 16),
        "P1_the_walking_command_makes_a_leg_rhythm": (round(s_fw, 4), p1),
        "P2_promotor_and_remotor_keep_a_reliable_offset": (
            {
                "defined": int(len(defined)),
                "feasible": n_ok,
                "R": None if r is None else round(r, 4),
            },
            n_ok > 0 and len(defined) >= n_ok / 2 and r is not None and r >= 0.7,
        ),
        "P3_tibia_flexor_and_extensor_alternate": (p3, p3 is not None and p3 <= 0.15),
        "P4_the_walking_command_drives_the_rhythm_best": (
            {"forward_walking": round(s_fw, 4), "grooming": round(s_gr, 4)},
            s_fw > s_gr,
        ),
        "P5_the_rhythm_needs_the_three_neuron_circuit": (
            {"forward_walking": round(s_fw, 4), "lesion": round(s_le, 4)},
            p1 and s_le < 0.3 and s_le <= 0.5 * s_fw,
        ),
        "P6_a_fly_like_step_period": (
            period,
            period is not None and s2c.PERIOD_MS[0] <= period <= s2c.PERIOD_MS[1],
        ),
    }
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        no_input = name != "P0_the_rule_finds_a_working_input" and n_ok < 16
        nothing_to_lesion = name == "P5_the_rhythm_needs_the_three_neuron_circuit" and not p1
        if no_input or nothing_to_lesion:
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    return out


def reported(table: pd.DataFrame) -> dict:
    """Per condition: feasible draws, median input, recruited, active motor neurons, score."""
    out = {}
    for cond, g in table.groupby("condition", sort=False):
        f = g[g["feasible"]]
        out[cond] = {
            "feasible": int(len(f)),
            "median_input": float(f["input"].median()) if len(f) else None,
            "median_recruited": float(f["recruited"].median()) if len(f) else None,
            "median_active_motor": float(f["active_motor"].median()) if len(f) else None,
            "median_score": round(float(f["score"].median()), 4) if len(f) else None,
            "median_period_ms": float(f["period_ms"].median())
            if f["period_ms"].notna().any()
            else None,
            "infeasible_too_few": int(((~g["feasible"]) & (g["recruited"] < FEWEST)).sum()),
            "infeasible_too_many": int(((~g["feasible"]) & (g["recruited"] > MOST)).sum()),
            "infeasible_median_recruited_when_too_many": (
                float(g.loc[(~g["feasible"]) & (g["recruited"] > MOST), "recruited"].median())
                if ((~g["feasible"]) & (g["recruited"] > MOST)).any()
                else None
            ),
        }
    return out


def plot(examples: dict, table: pd.DataFrame, path, rules=SIZE_RULES) -> None:
    """Per size rule: one feasible forward-walking draw's pools, and every draw's score."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "promotor": "#D55E00",
        "remotor": "#0072B2",
        "flexor": "#009E73",
        "extensor": "#E69F00",
    }
    order = ["forward_walking", "grooming", "dnb08", "none", "lesion"]
    fig, axes = plt.subplots(
        len(rules),
        2,
        figsize=(7.4, 2.7 * len(rules)),
        gridspec_kw={"width_ratios": [1.5, 1]},
        squeeze=False,
    )
    for r, rule in enumerate(rules):
        left, right = axes[r]
        ex = examples.get(rule)
        if ex is not None:
            t = START_S + np.arange(ex["promotor"].size) * 1e-3
            for k, colour in colours.items():
                left.plot(t, ex[k], color=colour, lw=0.9, label=k)
            left.set_xlim(START_S, 1.2)
            left.set_title(f"size rule '{rule}': DNg100, feasible draw {ex['seed']}", fontsize=7)
        else:
            left.set_title(f"size rule '{rule}': no feasible DNg100 draw", fontsize=7)
        left.set_ylabel("pool rate (Hz)", fontsize=7)
        left.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=2)
        left.tick_params(labelsize=6)
        sub = table[(table["size_rule"] == rule) & table["feasible"]]
        for k, cond in enumerate(order):
            v = sub.loc[sub["condition"] == cond, "score"].to_numpy()
            right.scatter(k + np.linspace(-0.2, 0.2, v.size), v, s=6, color="#0072B2")
        right.axhline(0.3, color="black", lw=0.6, ls="--")
        right.set_ylim(-0.03, 1.0)
        right.set_xticks(
            range(len(order)), [o.replace("_", " ") for o in order], fontsize=5.5, rotation=30
        )
        right.set_ylabel("motor rhythm score (corrected)", fontsize=7)
        right.set_title("feasible draws; dashed: registered 0.3", fontsize=7)
        right.tick_params(labelsize=6)
    axes[-1, 0].set_xlabel("time (s)", fontsize=7)
    fig.suptitle(
        "SIMULATED left front-leg network of the male CNS (Pugliese et al. construction)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-2g design."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    jobs = [(rule, drive) for rule in SIZE_RULES for drive in DRIVES]
    with ProcessPoolExecutor(len(jobs), initializer=_init) as pool:
        got = list(pool.map(run_job, jobs))
    _init()
    table = pd.DataFrame([row for g in got for row in g["rows"]])
    examples = {
        rule: g.get("example")
        for (rule, drive), g in zip(jobs, got, strict=True)
        if drive == "forward_walking"
    }
    result = {
        "label": "SIMULATED left front-leg network of the male CNS (Pugliese et al. construction)",
        "network": _STATE["counts"]
        | {"pool_sizes": {k: int(len(v)) for k, v in _STATE["pools"].items()}},
        "runs_used": {
            f"{rule}/{drive}": g["runs"] for (rule, drive), g in zip(jobs, got, strict=True)
        },
    }
    for rule in SIZE_RULES:
        sub = table[table["size_rule"] == rule]
        result[rule] = {"predictions": score(sub, spec), "reported": reported(sub)}
    result["registration"] = design.provenance(REGISTRATION)
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage2g_SIMULATED.csv", index=False)
    (out / "embodied_stage2g_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage2g", scoring.clean(result))
    plot(examples, table, figures_dir() / "embodied_stage2g_front_leg_SIMULATED.png")
    print("network", result["network"], "runs", result["runs_used"])
    for rule in SIZE_RULES:
        print("size rule:", rule)
        for name, entry in result[rule]["predictions"].items():
            print(f"   {name:48s} {entry['outcome']:>15}  {entry['value']}")
        print("   reported", json.dumps(scoring.clean(result[rule]["reported"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
