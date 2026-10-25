"""Embodied fly, stage 3c: the perturbation test on the right hind leg (servoed flybody tibia).

``python -m neuromps.fly.embodied_stage3c`` runs the registered design
(`preregistration/embodied_stage3c_hind_leg_reflex.json`): the stage-1c male nerve cord
under the walking command, its right hind-leg tibia pools commanding flybody's servoed
tibia every millisecond, the leg's 25 claw neurons driven by the tibia angle (resistance or
assistance assignment, or a sham), and a 20 ms flexing push. Everything here is
**simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from neuromps.fly import design, embodied_stage1c, model, proboscis, scoring
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage3c_hind_leg_reflex.json"
SEEDS = list(range(1300, 1316))
JOINT = "tibia_T3_right"
EXTENDED, FLEXED = 1.3, -1.35
PERTURB = (0.500, 0.520, -0.3)
CLAW = {"SNpp51": "extension", "SNpp50": "flexion"}  # the resistance assignment
OPPOSITE = {"extension": "flexion", "flexion": "extension"}
RATE_MAX_HZ, R_FULL, TAU_S, DT_S, SETTLE_S = 150.0, 20.0, 0.02, 0.001, 0.2
PRE_S, WINDOW_END_S, RECOVERY_RAD = (0.40, 0.50), 0.60, 0.05
_STATE: dict = {}


class HindLeg:
    """flybody with only the right hind tibia commanded; one control step at a time."""

    def __init__(self):
        import mujoco

        self._mj = mujoco
        self.m = proboscis.model()
        self.d = mujoco.MjData(self.m)
        joint = self.m.joint(JOINT).id
        self.actuator = self.m.actuator(JOINT).id
        self.qpos = self.m.jnt_qposadr[joint]
        self.dof = self.m.jnt_dofadr[joint]
        self.substeps = int(round(DT_S / self.m.opt.timestep))
        for _ in range(int(round(SETTLE_S / self.m.opt.timestep))):
            mujoco.mj_step(self.m, self.d)

    def angle(self) -> float:
        return float(self.d.qpos[self.qpos])

    def step(self, command: float, torque: float = 0.0) -> float:
        self.d.ctrl[self.actuator] = command
        self.d.qfrc_applied[:] = 0.0
        self.d.qfrc_applied[self.dof] = torque
        for _ in range(self.substeps):
            self._mj.mj_step(self.m, self.d)
        return self.angle()


def command(net: float) -> float:
    """Servo command from net drive in [-1, 1]: positive extends, negative flexes."""
    net = float(np.clip(net, -1.0, 1.0))
    return net * EXTENDED if net > 0 else -net * FLEXED


def claw_rate(theta: float, theta_0: float, tuning: str) -> float:
    """Linear position tuning from rest to the joint's limit (extension is a larger angle)."""
    if tuning == "extension":
        x = (theta - theta_0) / (EXTENDED - theta_0)
    elif tuning == "flexion":
        x = (theta_0 - theta) / (theta_0 - FLEXED)
    else:
        raise ValueError(tuning)
    return RATE_MAX_HZ * float(np.clip(x, 0.0, 1.0))


def _init() -> None:
    embodied_stage1c._init()
    con = embodied_stage1c._STATE["con"]
    t = embodied_stage1c._annotations(con.root_ids)
    leg = (t["superclass"] == "vnc_motor") & (t["subclass"] == "hl") & (t["somaSide"] == "R")
    sensor = (
        (t["superclass"] == "vnc_sensory")
        & (t["entryNerve"].astype(str) == "MetaLN")
        & (t["rootSide"].astype(str) == "R")
        & t["type"].isin(list(CLAW))
    )
    idx = np.flatnonzero(sensor.to_numpy())
    types = t["type"].astype(str).to_numpy()
    _STATE.update(
        extensor=np.flatnonzero((leg & (t["type"] == "Ti extensor MN")).to_numpy()),
        flexor=np.flatnonzero(
            (leg & t["type"].isin(["Ti flexor MN", "Acc. ti flexor MN"])).to_numpy()
        ),
        sensors=idx,
        resistance=[CLAW[types[i]] for i in idx],
    )


def _extender(tunings, sham, perturb, record: dict):
    def extend(neurons, b2):
        pools = [_STATE["extensor"], _STATE["flexor"]]
        members = np.concatenate(pools)
        counter = b2.NeuronGroup(len(members), "count : 1", name="fly_loop_counter")
        tally = b2.Synapses(neurons, counter, on_pre="count_post += 1", name="fly_loop_tally")
        tally.connect(i=members, j=np.arange(len(members)))
        sensors = _STATE["sensors"]
        source = b2.PoissonGroup(len(sensors), rates=0 * b2.Hz, name="fly_loop_sensors")
        w = model.FlyParams().w_syn_mv * model.FlyParams().f_poi
        feed = b2.Synapses(source, neurons, on_pre=f"v_post += {w} * mV", name="fly_loop_feed")
        feed.connect(i=np.arange(len(sensors)), j=sensors)
        leg = HindLeg()
        theta_0 = leg.angle()
        split = len(pools[0])
        state = {"last": np.zeros(len(members)), "level": np.zeros(len(members)), "k": 0}
        decay = np.exp(-DT_S / TAU_S)
        record.update(theta_0=theta_0, tibia=[], rates=[])

        def step():
            counts = np.array(counter.count[:], dtype=float)  # a copy: Brian2 returns a live view
            state["level"] = state["level"] * decay + (counts - state["last"]) / TAU_S
            state["last"] = counts
            ext = np.clip(state["level"][:split].mean() / R_FULL, 0, 1)
            flx = np.clip(state["level"][split:].mean() / R_FULL, 0, 1)
            t = state["k"] * DT_S
            torque = perturb[2] if perturb and perturb[0] <= t < perturb[1] else 0.0
            theta = leg.step(command(ext - flx), torque)
            if sham is not None:
                rates = sham[min(state["k"], len(sham) - 1)]
            elif tunings is not None:
                rates = np.array([claw_rate(theta, theta_0, u) for u in tunings])
            else:
                rates = np.zeros(len(sensors))
            source.rates = rates * b2.Hz
            record["tibia"].append(theta)
            record["rates"].append(np.asarray(rates, float))
            state["k"] += 1

        op = b2.NetworkOperation(step, dt=DT_S * b2.second, when="start", name="fly_loop_step")
        return [counter, tally, source, feed, op]

    return extend


def readouts(tibia: np.ndarray) -> dict:
    """Pre-pulse angle, excursion, recovery time (ms after the push ends)."""
    ms = lambda s: int(round(s / DT_S))  # noqa: E731
    pre = float(np.median(tibia[ms(PRE_S[0]) : ms(PRE_S[1])]))
    excursion = float(pre - tibia[ms(PERTURB[0]) : ms(WINDOW_END_S)].min())
    after = tibia[ms(PERTURB[1]) :]
    back = np.flatnonzero(after >= pre - RECOVERY_RAD)
    return {
        "pre_rad": pre,
        "excursion_rad": excursion,
        "recovery_ms": float(back[0] if back.size else after.size),
    }


def _trial(job) -> dict:
    arm, seed, sham = job
    if arm in ("resistance", "resistance_unperturbed"):
        tunings = list(_STATE["resistance"])
    elif arm == "assistance":
        tunings = [OPPOSITE[u] for u in _STATE["resistance"]]
    else:
        tunings = None
    perturb = None if arm == "resistance_unperturbed" else PERTURB
    record: dict = {}
    params = model.FlyParams(duration_s=1.0, stim_rate_hz=150.0, seed=seed)
    spikes = model.simulate(
        embodied_stage1c._STATE["con"],
        params,
        excite=embodied_stage1c._STATE["drive"]["forward_walking"],
        extend=_extender(tunings, sham, perturb, record),
    )

    def hz(idx):
        return float(np.isin(spikes.neurons, idx).sum() / max(len(idx), 1))

    tibia = np.asarray(record["tibia"])
    return {
        "arm": arm,
        "seed": seed,
        **readouts(tibia),
        "extensor_hz": hz(_STATE["extensor"]),
        "flexor_hz": hz(_STATE["flexor"]),
        "claw_mean_hz": float(np.mean(record["rates"])),
        "rates": np.asarray(record["rates"]),
        "tibia": tibia,
    }


def _median(table: pd.DataFrame, arm: str, column: str) -> float:
    return float(table.loc[table["arm"] == arm, column].median())


def score(table: pd.DataFrame, spec: dict) -> dict:
    """K0-K3; K1-K3 uninterpretable without a hold and a push, K2 also without K1."""
    pre, exc = _median(table, "open", "pre_rad"), _median(table, "open", "excursion_rad")
    rec = {
        a: _median(table, a, "recovery_ms") for a in ("open", "resistance", "assistance", "sham")
    }
    k0 = pre >= 0.3 and exc >= 0.2
    k1 = rec["resistance"] <= 0.8 * rec["open"]
    found = {
        "K0_the_leg_is_held_and_the_push_moves_it": (
            {"pre_rad": round(pre, 4), "excursion_rad": round(exc, 4)},
            k0,
        ),
        "K1_feedback_speeds_the_return": ({k: rec[k] for k in ("open", "resistance")}, k1),
        "K2_the_speed_needs_the_loop": (
            {k: rec[k] for k in ("resistance", "sham")},
            rec["resistance"] < rec["sham"],
        ),
        "K3_the_opposite_assignment_does_not_speed_it": (
            {k: rec[k] for k in ("open", "assistance")},
            rec["assistance"] >= rec["open"],
        ),
    }
    out = {}
    for name, (value, ok) in found.items():
        outcome = "pass" if ok else "fail"
        no_hold = name != "K0_the_leg_is_held_and_the_push_moves_it" and not k0
        nothing_to_explain = name.startswith("K2") and not k1
        if no_hold or nothing_to_explain:
            outcome = "uninterpretable"
        out[name] = {"test": spec["predictions"][name]["test"], "value": value, "outcome": outcome}
    return out


def plot(trials: list[dict], table: pd.DataFrame, path) -> None:
    """The tibia around the push (one seed per arm), and every trial's recovery time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {
        "open": "#999999",
        "resistance": "#0072B2",
        "assistance": "#D55E00",
        "sham": "#009E73",
    }
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    window = slice(450, 650)
    for arm, colour in colours.items():
        t = next(x for x in trials if x["arm"] == arm)
        left.plot(
            np.arange(1000)[window] * DT_S, t["tibia"][window], color=colour, lw=1.0, label=arm
        )
    left.axvspan(PERTURB[0], PERTURB[1], color="black", alpha=0.08)
    left.set_xlabel("time (s); shaded: imposed flexion", fontsize=7)
    left.set_ylabel("right hind tibia angle (rad; higher = extended)", fontsize=7)
    left.set_title(f"walking drive, seed {SEEDS[0]}", fontsize=7)
    left.legend(fontsize=6, frameon=False)
    left.tick_params(labelsize=6)
    for k, (arm, colour) in enumerate(colours.items()):
        v = table.loc[table["arm"] == arm, "recovery_ms"].to_numpy()
        right.scatter(k + np.linspace(-0.2, 0.2, v.size), v, s=8, color=colour)
    right.set_xticks(range(len(colours)), list(colours), fontsize=6.5)
    right.set_ylabel("recovery time (ms after the push)", fontsize=7)
    right.set_title("every trial", fontsize=7)
    right.tick_params(labelsize=6)
    fig.suptitle(
        "SIMULATED closed loop: male nerve cord, flybody hind leg, claw-neuron feedback", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered stage-3c design."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    arms = ("open", "resistance", "assistance", "resistance_unperturbed")
    with ProcessPoolExecutor(memory_bounded_workers(per_worker_gb=8.0), initializer=_init) as pool:
        trials = list(pool.map(_trial, [(a, s, None) for a in arms for s in SEEDS]))
        tonic = {t["seed"]: t["rates"] for t in trials if t["arm"] == "resistance_unperturbed"}
        trials += list(pool.map(_trial, [("sham", s, tonic[s]) for s in SEEDS]))
    table = pd.DataFrame(
        [{k: v for k, v in t.items() if k not in ("rates", "tibia")} for t in trials]
    )
    reported = {
        a: {
            c: round(float(g[c].median()), 4)
            for c in (
                "pre_rad",
                "excursion_rad",
                "recovery_ms",
                "extensor_hz",
                "flexor_hz",
                "claw_mean_hz",
            )
        }
        for a, g in table.groupby("arm")
    }
    result = {
        "label": "SIMULATED hind-leg perturbation test through the claw neurons",
        "predictions": score(table, spec),
        "reported": reported,
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage3c_SIMULATED.csv", index=False)
    (out / "embodied_stage3c_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage3c", scoring.clean(result))
    plot(trials, table, figures_dir() / "embodied_stage3c_hind_leg_SIMULATED.png")
    for name, entry in result["predictions"].items():
        print(f"{name:46s} {entry['outcome']:>15}  {entry['value']}")
    print("reported", json.dumps(reported))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
