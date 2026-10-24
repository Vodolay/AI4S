"""A firing-rate model of a connectome, after Pugliese et al. (2025), batched over replicates.

Reimplemented from the paper (bioRxiv 10.1101/2025.09.12.675944) and the
configuration published with it; their code is not used. Each neuron's rate
follows

    tau dr/dt = -r + max(0, r_max tanh((a / r_max)(I + W r - theta)))

with W = 0.03 x signed synapse count (connections of at least 5 synapses), and
tau, a, theta, r_max drawn per neuron and replicate; a is divided and theta
multiplied by the neuron's median-normalized size. All replicates of one
condition are integrated together: rates are a (neurons x replicates) array.
Everything it produces is **simulated**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from neuromps.fly.connectome import Connectome

SYNAPSE_FLOOR = 5
SCALE = 0.03
# (mean, standard deviation) per parameter, as published
DISTRIBUTIONS = {
    "tau_s": (0.02, 0.002),
    "a": (1.0, 0.1),
    "theta": (7.5, 0.6),
    "r_max": (200.0, 10.0),
}


def weights(con: Connectome, floor: int = SYNAPSE_FLOOR, scale: float = SCALE) -> sparse.csr_matrix:
    """W[post, pre] = scale x sign x synapses, for connections with at least `floor` synapses."""
    keep = con.synapses >= floor
    values = scale * con.sign[keep] * con.synapses[keep]
    shape = (con.n_neurons, con.n_neurons)
    return sparse.csr_matrix((values, (con.post[keep], con.pre[keep])), shape=shape)


def sizes(con: Connectome) -> np.ndarray:
    """Stand-in for neuron size: total synapses in and out, median-normalized; 0 -> median."""
    total = np.bincount(con.pre, con.synapses, con.n_neurons) + np.bincount(
        con.post, con.synapses, con.n_neurons
    )
    median = np.median(total[total > 0])
    total[total == 0] = median
    return total / median


def _positive_normal(rng, mean: float, sd: float, shape) -> np.ndarray:
    out = rng.normal(mean, sd, shape)
    while np.any(bad := out <= 0):
        out[bad] = rng.normal(mean, sd, int(bad.sum()))
    return out


@dataclass(frozen=True)
class Draws:
    """Per-neuron, per-replicate parameters, each (neurons x replicates)."""

    tau_s: np.ndarray
    a: np.ndarray
    theta: np.ndarray
    r_max: np.ndarray


def draw(size: np.ndarray, seeds: list[int]) -> Draws:
    """One parameter draw per seed (a column each), scaled by size as published."""
    cols = {k: [] for k in DISTRIBUTIONS}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for k, (mean, sd) in DISTRIBUTIONS.items():
            cols[k].append(_positive_normal(rng, mean, sd, size.size))
    got = {k: np.stack(v, axis=1) for k, v in cols.items()}
    got["a"] = got["a"] / size[:, None]
    got["theta"] = got["theta"] * size[:, None]
    return Draws(**got)


def activation(x: np.ndarray, p: Draws) -> np.ndarray:
    """The rectified, saturating response to total input `x`."""
    return np.maximum(p.r_max * np.tanh((p.a / p.r_max) * (x - p.theta)), 0.0)


@dataclass(frozen=True)
class Run:
    """Recorded rates (Hz) at each 1 ms sample, (samples x recorded x replicates); mean rates."""

    rates: np.ndarray
    mean_rate: np.ndarray


def simulate(
    w: sparse.csr_matrix,
    p: Draws,
    drive: np.ndarray,
    amplitude: float,
    record: np.ndarray,
    duration_s: float = 2.0,
    dt_s: float = 1e-4,
    onset_s: float = 0.02,
    silenced: np.ndarray | None = None,
    sample_s: float = 1e-3,
) -> Run:
    """Forward-Euler integration from rest; step input to `drive` from `onset_s`."""
    n, reps = p.tau_s.shape
    r = np.zeros((n, reps))
    inputs = np.zeros((n, 1))
    inputs[np.asarray(drive, dtype=int)] = amplitude
    every = int(round(sample_s / dt_s))
    steps = int(round(duration_s / dt_s))
    out = np.empty((steps // every, len(record), reps))
    total = np.zeros((n, reps))
    gain = dt_s / p.tau_s
    for k in range(steps):
        on = inputs if k * dt_s >= onset_s else 0.0
        r += gain * (activation(on + w @ r, p) - r)
        if silenced is not None:
            r[silenced] = 0.0
        total += r
        if (k + 1) % every == 0:
            out[(k + 1) // every - 1] = r[record]
    return Run(rates=out, mean_rate=total / steps)
