"""Two-compartment spiking network: the digital twin of an asymmetric neural chip.

Two populations of conductance-based LIF neurons with spike-frequency
adaptation, 80/20 excitatory/inhibitory, recurrently connected within each
compartment and projecting to each other through a microchannel-like pathway
whose forward and backward connection probabilities differ by ``forward_ratio``
- the "axon diode" of the real device.

Adaptation plus recurrent excitation is what makes the network burst rather than
fire tonically, which is the behaviour the real cultures show.

Everything this module produces is **simulated**; outputs belong under
``outputs/twin/`` and must be labelled as simulated wherever they appear.
"""

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

log = logging.getLogger("neuromps.twin")
FORWARD, BACKWARD = "forward", "backward"


@dataclass(frozen=True)
class NetworkParams:
    """Network size, connectivity, drive and drug knobs.

    Drug knobs multiply a conductance or threshold, so 1.0 is untreated:
    ``gabaa_scale`` < 1 is disinhibition (a GABA-A antagonist), > 1 is a
    positive modulator; ``ampa_scale`` scales excitatory transmission;
    ``threshold_scale`` raises the spike threshold, as a sodium-channel blocker
    effectively does.
    """

    n_per_compartment: int = 250
    excitatory_fraction: float = 0.8
    duration_s: float = 60.0
    seed: int = 0
    # Connectivity is set by in-degree, not by probability: with a fixed
    # probability the number of inputs per neuron grows with the network, so a
    # calibration done at one size does not hold at another.
    intra_in_degree: float = 20.0
    inter_in_degree: float = 10.0
    forward_ratio: float = 1.0
    intra_delay_ms: float = 1.5
    inter_delay_ms: float = 4.0
    # synapses (dimensionless conductance steps)
    excitatory_weight: float = 0.25
    inhibitory_weight: float = 2.6
    # background drive
    background_synapses: int = 100
    background_rate_hz: float = 10.0
    background_weight: float = 0.042
    # Background drive to the receiving compartment, relative to the source one.
    # A directed pathway also *adds* excitation to its target, which makes the
    # target more excitable and can make it burst first; lowering this isolates
    # the connectivity asymmetry from that excitability asymmetry.
    background_scale_target: float = 1.0
    # adaptation
    adaptation_increment: float = 1.5
    adaptation_tau_ms: float = 380.0
    # drug knobs
    gabaa_scale: float = 1.0
    ampa_scale: float = 1.0
    threshold_scale: float = 1.0
    extra: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        """Parameter dictionary for provenance and for sbi priors."""
        return {k: v for k, v in asdict(self).items() if k != "extra"}


@dataclass
class SimulationResult:
    """Spikes from one simulated recording, with the neuron -> compartment map."""

    spike_times: np.ndarray
    spike_neurons: np.ndarray
    compartment_of_neuron: np.ndarray
    is_excitatory: np.ndarray
    duration_s: float
    params: dict

    @property
    def n_neurons(self) -> int:
        return int(self.compartment_of_neuron.size)

    @property
    def n_excitatory(self) -> int:
        """Excitatory neurons per compartment."""
        return int(self.is_excitatory[self.compartment_of_neuron == 0].sum())

    @property
    def n_inhibitory(self) -> int:
        """Inhibitory neurons per compartment."""
        return int((~self.is_excitatory[self.compartment_of_neuron == 0]).sum())

    def spikes_by_compartment(self) -> dict[int, np.ndarray]:
        """Spike times split by compartment."""
        compartment = self.compartment_of_neuron[self.spike_neurons]
        return {c: self.spike_times[compartment == c] for c in (0, 1)}


def _neuron_layout(params: NetworkParams) -> tuple[np.ndarray, np.ndarray]:
    """Compartment id and excitatory flag per neuron, compartments stacked in order."""
    n = params.n_per_compartment
    n_exc = int(round(n * params.excitatory_fraction))
    compartment = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    excitatory = np.tile(np.arange(n) < n_exc, 2)
    return compartment, excitatory


def synapse_counts(params: NetworkParams) -> dict[str, int]:
    """Inter-compartment synapse counts, without running the simulation."""
    rng = np.random.default_rng(params.seed)
    compartment, excitatory = _neuron_layout(params)
    forward_p, backward_p = _inter_probabilities(params)
    counts = {}
    for name, (source, target, probability) in {
        FORWARD: (0, 1, forward_p),
        BACKWARD: (1, 0, backward_p),
    }.items():
        sources = np.flatnonzero((compartment == source) & excitatory)
        targets = np.flatnonzero(compartment == target)
        counts[name] = int(rng.binomial(sources.size * targets.size, probability))
    return counts


def intra_probability(params: NetworkParams) -> float:
    """Connection probability giving the requested within-compartment in-degree."""
    return min(params.intra_in_degree / max(params.n_per_compartment, 1), 1.0)


def _inter_probabilities(params: NetworkParams) -> tuple[float, float]:
    """Forward/backward probabilities for the requested mean inter-compartment in-degree.

    The ratio tilts the pathway without changing how many inter-compartment
    synapses exist in total, so asymmetry and overall coupling stay separable.
    """
    mean_p = min(params.inter_in_degree / max(params.n_per_compartment, 1), 1.0)
    ratio = max(params.forward_ratio, 1e-6)
    backward = 2 * mean_p / (1 + ratio)
    return min(ratio * backward, 1.0), min(backward, 1.0)


EQUATIONS = """
dv/dt = ((v_rest - v) + g_e * (e_exc - v) + g_i * (e_inh - v) - w) / tau_m : 1 (unless refractory)
dg_e/dt = -g_e / tau_e : 1
dg_i/dt = -g_i / tau_i : 1
dw/dt = -w / tau_w : 1
v_threshold : 1 (constant)
"""


def _build(params: NetworkParams):
    """Assemble the Brian2 network. Returns (network, spike monitor, neuron group)."""
    import brian2 as b2

    b2.prefs.codegen.target = "numpy"  # portable, and fastest here for short runs
    b2.seed(params.seed)
    compartment, excitatory = _neuron_layout(params)
    namespace = {
        "v_rest": 0.0,
        "e_exc": 4.0,
        "e_inh": -1.0,
        "tau_m": 20 * b2.ms,
        "tau_e": 5 * b2.ms,
        "tau_i": 10 * b2.ms,
        "tau_w": params.adaptation_tau_ms * b2.ms,
    }
    neurons = b2.NeuronGroup(
        compartment.size,
        EQUATIONS,
        threshold="v > v_threshold",
        reset=f"v = 0; w += {params.adaptation_increment}",
        refractory=3 * b2.ms,
        method="euler",
        namespace=namespace,
    )
    neurons.v = "rand() * 0.5"
    neurons.v_threshold = params.threshold_scale
    network = b2.Network(neurons)
    network.add(_synapses(params, neurons, compartment, excitatory, b2))
    network.add(_background(params, neurons, compartment, b2))
    monitor = b2.SpikeMonitor(neurons)
    network.add(monitor)
    return network, monitor, compartment, excitatory


def _ranges(params: NetworkParams) -> dict:
    """Index ranges of each neuron block; compartments are laid out contiguously."""
    n = params.n_per_compartment
    n_exc = int(round(n * params.excitatory_fraction))
    return {
        c: {
            "all": (c * n, (c + 1) * n),
            "exc": (c * n, c * n + n_exc),
            "inh": (c * n + n_exc, (c + 1) * n),
        }
        for c in (0, 1)
    }


def _condition(source: tuple[int, int], target: tuple[int, int]) -> str:
    """Brian2 connection condition for one index block onto another, no self-synapses."""
    return (
        f"(i >= {source[0]}) and (i < {source[1]}) and "
        f"(j >= {target[0]}) and (j < {target[1]}) and (i != j)"
    )


def _background(params: NetworkParams, neurons, compartment, b2) -> list:
    """Poisson drive, optionally weaker into the pathway's target compartment."""
    if params.background_rate_hz <= 0 or params.background_weight <= 0:
        return []
    inputs = []
    for c, scale in ((0, 1.0), (1, params.background_scale_target)):
        if scale <= 0:
            continue
        subset = neurons[
            int(np.flatnonzero(compartment == c)[0]) : int(np.flatnonzero(compartment == c)[-1] + 1)
        ]
        inputs.append(
            b2.PoissonInput(
                subset,
                "g_e",
                params.background_synapses,
                params.background_rate_hz * b2.Hz,
                params.background_weight * scale,
            )
        )
    return inputs


def _synapses(params: NetworkParams, neurons, compartment, excitatory, b2) -> list:
    """Recurrent connections inside each compartment plus the directed pathway between them."""
    exc_weight = params.excitatory_weight * params.ampa_scale
    inh_weight = params.inhibitory_weight * params.gabaa_scale
    blocks = _ranges(params)
    made = []
    for kind, weight, target_var in (("exc", exc_weight, "g_e"), ("inh", inh_weight, "g_i")):
        synapses = b2.Synapses(
            neurons,
            neurons,
            on_pre=f"{target_var}_post += {weight}",
            delay=params.intra_delay_ms * b2.ms,
        )
        for c in (0, 1):  # recurrent, within each compartment only
            synapses.connect(
                _condition(blocks[c][kind], blocks[c]["all"]), p=intra_probability(params)
            )
        made.append(synapses)
    forward_p, backward_p = _inter_probabilities(params)
    for source, target, probability in ((0, 1, forward_p), (1, 0, backward_p)):
        if probability <= 0:
            continue
        pathway = b2.Synapses(
            neurons,
            neurons,
            on_pre=f"g_e_post += {exc_weight}",
            delay=params.inter_delay_ms * b2.ms,
        )
        pathway.connect(_condition(blocks[source]["exc"], blocks[target]["all"]), p=probability)
        made.append(pathway)
    return made


def simulate(params: NetworkParams) -> SimulationResult:
    """Run the two-compartment network and return its simulated spikes."""
    import brian2 as b2

    with _quiet_brian():
        network, monitor, compartment, excitatory = _build(params)
        network.run(params.duration_s * b2.second)
        times = np.asarray(monitor.t / b2.second, dtype=float)
        neurons = np.asarray(monitor.i, dtype=int)
    log.debug("simulated %d spikes in %.1f s", times.size, params.duration_s)
    return SimulationResult(
        spike_times=times,
        spike_neurons=neurons,
        compartment_of_neuron=compartment,
        is_excitatory=excitatory,
        duration_s=params.duration_s,
        params=params.as_record(),
    )


def _quiet_brian():
    """Silence Brian2's per-run logging, which is noisy inside sweeps."""
    import contextlib
    import logging as std_logging

    @contextlib.contextmanager
    def quiet():
        brian_logger = std_logging.getLogger("brian2")
        level = brian_logger.level
        brian_logger.setLevel(std_logging.ERROR)
        try:
            yield
        finally:
            brian_logger.setLevel(level)

    return quiet()
