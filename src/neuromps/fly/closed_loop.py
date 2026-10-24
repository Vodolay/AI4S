"""A closed loop between the LIF nerve cord and the FlyMimic leg, 1 ms at a time.

Every millisecond a Brian2 network operation (1) reads the leg motor neurons' new spikes
from a counter, filters them into rates (tau 20 ms) and turns each muscle's pool rate
into an activation (rate / r_full, as stage 1); (2) advances the leg by one control step
(`body.Session`); (3) sets the firing rate of chosen sensory neurons from the leg's
femur-tibia angle, or from a given time course (a sham), delivered as Poisson spikes.
Everything it produces is **simulated**.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neuromps.fly import body, model

TIBIA = "joint_LFTibia_pitch"
EXTENDED, FLEXED = 0.479, 2.502  # the femur-tibia joint's limits (rad)
RATE_MAX_HZ = 150.0
TAU_S = 0.02


def claw_rate(theta: float, theta_0: float, tuning: str) -> float:
    """Linear position tuning from rest (0 Hz) to the joint limit (RATE_MAX_HZ)."""
    if tuning == "extension":
        x = (theta_0 - theta) / (theta_0 - EXTENDED)
    elif tuning == "flexion":
        x = (theta - theta_0) / (FLEXED - theta_0)
    else:
        raise ValueError(tuning)
    return RATE_MAX_HZ * float(np.clip(x, 0.0, 1.0))


@dataclass
class Loop:
    """What the loop connects; `tunings` (one per sensor) or `sham` (steps x sensors)."""

    neurons_by_muscle: dict[str, list[int]]
    sensors: np.ndarray
    tunings: list[str] | None = None
    sham: np.ndarray | None = None
    r_full: float = 20.0
    perturb: tuple[float, float, float] | None = None  # (start s, end s, torque on the tibia)
    record: dict = field(default_factory=dict)


def _activation(level: np.ndarray, columns: list[list[int]], r_full: float) -> np.ndarray:
    return np.array([np.clip(level[c].mean() / r_full, 1e-4, 1.0) if c else 1e-4 for c in columns])


def extender(loop: Loop, params: model.FlyParams):
    """The `extend` hook for `model.simulate`: counter, sensor drive and the loop step."""

    def extend(neurons, b2):
        names = body.muscle_names()
        motor = sorted({n for v in loop.neurons_by_muscle.values() for n in v})
        column = {n: k for k, n in enumerate(motor)}
        columns = [[column[n] for n in loop.neurons_by_muscle.get(m, [])] for m in names]
        counter = b2.NeuronGroup(len(motor), "count : 1", name="fly_loop_counter")
        tally = b2.Synapses(neurons, counter, on_pre="count_post += 1", name="fly_loop_tally")
        tally.connect(i=np.asarray(motor, int), j=np.arange(len(motor)))
        source = b2.PoissonGroup(len(loop.sensors), rates=0 * b2.Hz, name="fly_loop_sensors")
        feed = b2.Synapses(
            source,
            neurons,
            on_pre=f"v_post += {params.w_syn_mv * params.f_poi} * mV",
            name="fly_loop_feed",
        )
        feed.connect(i=np.arange(len(loop.sensors)), j=np.asarray(loop.sensors, int))
        session = body.Session(names)
        tibia = session.joint_names.index(TIBIA)
        theta_0 = float(session.angles()[tibia])
        state = {"last": np.zeros(len(motor)), "level": np.zeros(len(motor)), "k": 0}
        decay = np.exp(-body.CONTROL_DT_S / TAU_S)
        rec = loop.record
        rec.update(theta_0=theta_0, distance=[], tibia=[], sensor_rates=[])

        def step():
            counts = np.array(counter.count[:], dtype=float)  # a copy: Brian2 returns a live view
            state["level"] = state["level"] * decay + (counts - state["last"]) / TAU_S
            state["last"] = counts
            torques = None
            if loop.perturb is not None:
                start, end, torque = loop.perturb
                if start <= state["k"] * body.CONTROL_DT_S < end:
                    torques = {TIBIA: torque}
            distance, angles = session.step(
                _activation(state["level"], columns, loop.r_full), torques
            )
            theta = float(angles[tibia])
            if loop.sham is not None:
                rates = loop.sham[min(state["k"], len(loop.sham) - 1)]
            elif loop.tunings is not None:
                rates = np.array([claw_rate(theta, theta_0, t) for t in loop.tunings])
            else:
                rates = np.zeros(len(loop.sensors))
            source.rates = rates * b2.Hz
            rec["distance"].append(distance)
            rec["tibia"].append(theta)
            rec["sensor_rates"].append(np.asarray(rates, float))
            state["k"] += 1

        op = b2.NetworkOperation(
            step, dt=body.CONTROL_DT_S * b2.second, when="start", name="fly_loop_step"
        )
        return [counter, tally, source, feed, op]

    return extend
