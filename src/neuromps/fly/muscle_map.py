"""Motor neurons of the left front leg to FlyMimic muscles, and spikes to muscle activation.

The map is read from the stage-1 registration, so what was registered is what
runs: each MaleCNS front-leg motor-neuron type drives the FlyMimic muscle(s) of
the same anatomical name; types whose muscle the model lacks are left out and
counted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from neuromps.paths import REPO_ROOT

REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage1_open_loop_leg.json"
UNMAPPED_KEY = "unmapped (no muscle in the model)"
MOTOR = ("vnc_motor", "cb_motor")


def registered_map() -> tuple[dict[str, list[str]], list[str]]:
    """Motor-neuron type -> muscles, and the types the model has no muscle for."""
    spec = json.loads(REGISTRATION.read_text())["design"]["map"]
    unmapped = spec[UNMAPPED_KEY]
    return {k: v for k, v in spec.items() if k != UNMAPPED_KEY}, unmapped


@dataclass(frozen=True)
class Assignment:
    """Row positions (in the annotation table given) of each muscle's motor neurons."""

    neurons_by_muscle: dict[str, list[int]]
    coverage: dict[str, int]


def assign(annotations: pd.DataFrame) -> Assignment:
    """Left front-leg motor neurons, grouped by the muscle their type drives."""
    mapping, _ = registered_map()
    leg = annotations[
        annotations["superclass"].isin(MOTOR)
        & (annotations["subclass"] == "fl")
        & (annotations["somaSide"] == "L")
    ]
    by_muscle: dict[str, list[int]] = {}
    mapped = 0
    for row, kind in zip(leg.index, leg["type"], strict=True):
        muscles = mapping.get(kind)
        if not muscles:
            continue
        mapped += 1
        for muscle in muscles:
            by_muscle.setdefault(muscle, []).append(int(annotations.index.get_loc(row)))
    return Assignment(by_muscle, {"mapped": mapped, "left_front_leg_motor_neurons": int(len(leg))})


def filtered_rate(spike_times: np.ndarray, duration_s: float, dt_s: float, tau_s: float):
    """A spike train as a rate (Hz): exponential kernel with time constant `tau_s`."""
    n = int(round(duration_s / dt_s))
    counts = np.bincount(
        np.clip((np.asarray(spike_times) / dt_s).astype(int), 0, n - 1), minlength=n
    ).astype(float)
    decay = np.exp(-dt_s / tau_s)
    rate = np.empty(n)
    level = 0.0
    for k in range(n):
        level = level * decay + counts[k] / tau_s
        rate[k] = level
    return rate


def activation(rates: np.ndarray, neurons: list[int], r_full: float) -> np.ndarray:
    """A muscle's activation over time: its neurons' mean rate over `r_full`, in [1e-4, 1]."""
    if not neurons:
        return np.full(rates.shape[0], 1e-4)
    return np.clip(rates[:, neurons].mean(axis=1) / r_full, 1e-4, 1.0)
