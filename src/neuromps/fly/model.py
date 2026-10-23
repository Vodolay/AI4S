"""Leaky integrate-and-fire whole-brain model of the adult fly, with drug knobs.

The neuron and synapse equations and every constant are those of Shiu et al.
2024 (MIT; github.com/philshiu/Drosophila_brain_model, `model.py`): each neuron
integrates a current that each presynaptic spike raises by its synapse count
times `w_syn`, signed by transmitter. Two things are added:

- **Drug knobs**, mirroring the chip twin's (`twin/network.py`): a multiplier on
  the weights of each transmitter class, and a threshold multiplier. 1.0 is
  untreated. In the fly, GABA (Rdl receptors) and glutamate (GluCl channels) are
  the fast inhibitory transmitters and acetylcholine the main fast excitatory
  one, so `gaba_scale` plays the part `gabaa_scale` plays on the chip and
  `ach_scale` the part of `ampa_scale`.
- **An optional Poisson background** to every neuron, which the published model
  does not have. It holds the brain in ongoing activity, as a culture on a chip
  is, so a virtual array has spontaneous spikes to record. Its size is set as
  the mean depolarisation it produces; 0 leaves the published model unchanged.

Everything this module produces is **simulated**.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from neuromps.fly.connectome import Connectome

log = logging.getLogger("neuromps.fly")
KNOB_OF_CLASS = {"ach": "ach_scale", "gaba": "gaba_scale", "glut": "glut_scale"}


@dataclass(frozen=True)
class FlyParams:
    """Shiu et al. constants, stimulation, background and drug knobs."""

    duration_s: float = 1.0
    seed: int = 0
    v_0_mv: float = -52.0
    v_rst_mv: float = -52.0
    v_th_mv: float = -45.0
    t_mbr_ms: float = 20.0
    tau_ms: float = 5.0
    t_rfc_ms: float = 2.2
    t_dly_ms: float = 1.8
    w_syn_mv: float = 0.275
    # Poisson stimulation of chosen neurons, as in the published experiments
    stim_rate_hz: float = 150.0
    stim2_rate_hz: float = 0.0
    f_poi: float = 250.0
    # Pulsed stimulation, as an electrode delivers it: drive only for the first
    # `stim_width_ms` of every `stim_period_ms`. A period of 0 drives throughout.
    stim_period_ms: float = 0.0
    stim_width_ms: float = 10.0
    # background: mean depolarisation (mV) from `background_inputs` Poisson inputs
    background_mean_mv: float = 0.0
    background_inputs: int = 100
    background_rate_hz: float = 10.0
    # drug knobs
    ach_scale: float = 1.0
    gaba_scale: float = 1.0
    glut_scale: float = 1.0
    threshold_scale: float = 1.0

    def as_record(self) -> dict:
        """Parameter dictionary for provenance."""
        return asdict(self)


@dataclass(frozen=True)
class FlySpikes:
    """Spikes of the recorded neurons (model indices), times in seconds."""

    times: np.ndarray
    neurons: np.ndarray
    duration_s: float


def synaptic_weights(con: Connectome, params: FlyParams) -> np.ndarray:
    """Signed weight of each connection in mV, after the drug knobs."""
    weights = con.sign * con.synapses * params.w_syn_mv
    scale = np.ones(weights.size)
    for cls, knob in KNOB_OF_CLASS.items():
        scale[con.pre_class == cls] = getattr(params, knob)
    return weights * scale


def threshold_mv(params: FlyParams) -> float:
    """Spike threshold: the knob scales its distance above rest."""
    return params.v_0_mv + params.threshold_scale * (params.v_th_mv - params.v_0_mv)


def background_weight_mv(params: FlyParams) -> float:
    """Per-event weight giving the requested mean depolarisation."""
    events_per_tau = params.background_inputs * params.background_rate_hz * params.t_mbr_ms / 1000
    return params.background_mean_mv / events_per_tau


EQUATIONS = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau : volt (unless refractory)
rfc : second
"""


def _network(con: Connectome, params: FlyParams, excite, excite2, b2):
    mv, ms = b2.mV, b2.ms
    namespace = {
        "v_0": params.v_0_mv * mv,
        "v_rst": params.v_rst_mv * mv,
        "v_th": threshold_mv(params) * mv,
        "t_mbr": params.t_mbr_ms * ms,
        "tau": params.tau_ms * ms,
    }
    neurons = b2.NeuronGroup(
        con.n_neurons,
        EQUATIONS,
        method="linear",
        threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV",
        refractory="rfc",
        namespace=namespace,
        name="fly_neurons",
    )
    neurons.v = params.v_0_mv * mv
    neurons.rfc = params.t_rfc_ms * ms
    synapses = b2.Synapses(
        neurons,
        neurons,
        "w : volt",
        on_pre="g += w",
        delay=params.t_dly_ms * ms,
        name="fly_synapses",
    )
    synapses.connect(i=con.pre, j=con.post)
    synapses.w = synaptic_weights(con, params) * mv
    objects = [neurons, synapses]
    for label, indices, rate in (
        ("stim", excite, params.stim_rate_hz),
        ("stim2", excite2, params.stim2_rate_hz),
    ):
        objects += _stimulus(neurons, np.asarray(indices, dtype=int), rate, label, params, b2)
    if params.background_mean_mv > 0:
        objects.append(_background(neurons, params, b2))
    monitor = b2.SpikeMonitor(neurons, name="fly_monitor")
    objects.append(monitor)
    # Fixed names: Brian2 writes object names into the code it compiles, so the
    # auto-numbered defaults made every trial in a process compile afresh.
    return b2.Network(*objects, name="fly_network"), monitor


def _stimulus(neurons, indices: np.ndarray, rate_hz: float, label: str, params: FlyParams, b2):
    """Independent Poisson drive straight onto `v` of each driven neuron, as published.

    One named Poisson source per driven neuron, connected one to one. The
    published model uses a `PoissonInput` per neuron, which delivers the same
    events but cannot be given a fixed name (see `_network`).
    """
    if indices.size == 0 or rate_hz <= 0:
        return []
    neurons.rfc[indices] = 0 * b2.ms  # as published: no refractory period for driven neurons
    rates = rate_hz * b2.Hz
    if params.stim_period_ms > 0:
        rates = (
            f"{rate_hz} * Hz * int((t % ({params.stim_period_ms} * ms)) < "
            f"{params.stim_width_ms} * ms)"
        )
    source = b2.PoissonGroup(indices.size, rates, name=f"fly_{label}_source")
    link = b2.Synapses(
        source,
        neurons,
        on_pre=f"v_post += {params.w_syn_mv * params.f_poi} * mV",
        name=f"fly_{label}_link",
    )
    link.connect(i=np.arange(indices.size), j=indices)
    return [source, link]


def _background(neurons, params: FlyParams, b2):
    """`background_inputs` Poisson inputs per neuron, summed as one binomial draw per step.

    This is what `PoissonInput` compiles to; written out so the object has a
    fixed name.
    """
    p_event = params.background_rate_hz * float(b2.defaultclock.dt / b2.second)
    events = b2.BinomialFunction(params.background_inputs, p_event, name="fly_background_events")
    neurons.namespace["fly_background_events"] = events
    return neurons.run_regularly(
        f"v += {background_weight_mv(params)} * mV * fly_background_events()",
        when="synapses",
        name="fly_background",
    )


def simulate(
    con: Connectome,
    params: FlyParams,
    excite: Sequence[int] = (),
    excite2: Sequence[int] = (),
    record: Sequence[int] | None = None,
) -> FlySpikes:
    """One trial: drive `excite` at `stim_rate_hz` (and `excite2` at `stim2_rate_hz`).

    `record` limits the spike monitor to those model indices; None records all.
    """
    import brian2 as b2

    with _quiet_brian():
        b2.start_scope()
        b2.seed(params.seed)
        network, monitor = _network(con, params, excite, excite2, b2)
        network.run(params.duration_s * b2.second)
        times = np.asarray(monitor.t / b2.second, dtype=float)
        neurons = np.asarray(monitor.i, dtype=int)
    if record is not None:  # this Brian2 records all or nothing, so filter after
        keep = np.isin(neurons, np.asarray(record, dtype=int))
        times, neurons = times[keep], neurons[keep]
    return FlySpikes(times=times, neurons=neurons, duration_s=params.duration_s)


@contextlib.contextmanager
def _quiet_brian():
    """Silence Brian2's per-run logging, which is noisy inside sweeps."""
    brian_logger = logging.getLogger("brian2")
    level = brian_logger.level
    brian_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        brian_logger.setLevel(level)
