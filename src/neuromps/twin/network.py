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
    # Connectivity and threshold heterogeneity are drawn at build time from
    # `seed`; the background Poisson drive is drawn at run time. Setting
    # `noise_seed` reseeds between the two, which gives the *same chip* under a
    # different noise realisation - the relation between two recordings of one
    # real device. None keeps the single-seed behaviour every earlier result
    # was produced under.
    noise_seed: int | None = None
    # Connectivity is set by in-degree, not by probability: with a fixed
    # probability the number of inputs per neuron grows with the network, so a
    # calibration done at one size does not hold at another.
    intra_in_degree: float = 20.0
    inter_in_degree: float = 10.0
    forward_ratio: float = 1.0
    intra_delay_ms: float = 1.5
    inter_delay_ms: float = 4.0
    # Spread of the recurrent conduction delay (Gaussian SD in ms around
    # `intra_delay_ms`). One delay shared by every synapse makes recurrent
    # activity arrive in lockstep; real axonal delays are spread over several
    # ms, and that spread is a standard desynchroniser. 0.0 keeps the single
    # fixed delay every result before this parameter was produced under.
    intra_delay_sd_ms: float = 0.0
    # Range of the within-compartment connection rule, in micrometres: the
    # Gaussian SD of a neuron's reach. `None` connects uniformly at random, so
    # every pair of neurons - and therefore every pair of electrodes - is
    # statistically identical however far apart they sit, and mean pairwise
    # STTC is pushed to one extreme or the other. A finite range gives near
    # neighbours more shared input than distant ones, which is what lets the
    # measured synchrony sit between the two. `None` is what every result
    # before this parameter was produced under. Density is renormalised so the
    # requested `intra_in_degree` is preserved either way.
    connection_sigma_um: float | None = None
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
    # Short-term synaptic depression on recurrent excitatory synapses
    # (Tsodyks-Markram). `depression_u` is the fraction of available resources
    # each spike consumes and `depression_tau_rec_ms` how fast they come back;
    # u = 0 disables it, which is what every result before HD18 was produced
    # under. Parameterised so the *first* spike after rest delivers the same
    # conductance either way, which makes the two comparable without
    # recalibrating the weights - see `twin/depression.py`.
    depression_u: float = 0.0
    depression_tau_rec_ms: float = 600.0
    # Per-neuron threshold heterogeneity (fractional SD). Identical neurons
    # synchronise almost perfectly, which no real culture does; excitability
    # varies from cell to cell.
    threshold_spread: float = 0.08
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
x_um : 1 (constant)
y_um : 1 (constant)
"""


def _distance_scale(params: NetworkParams, positions: np.ndarray, compartment: np.ndarray) -> float:
    """Density scale making a Gaussian connection rule give the requested in-degree.

    In-degree, not probability, is the project's parameterisation, so narrowing
    the range must concentrate the same number of inputs rather than remove
    them. Both chambers share a geometry, so one is enough to estimate it.
    """
    sigma = float(params.connection_sigma_um or 0.0)
    block = positions[compartment == 0]
    squared = ((block[:, None, :] - block[None, :, :]) ** 2).sum(-1)
    kernel = np.exp(-squared / (2.0 * sigma**2))
    np.fill_diagonal(kernel, 0.0)
    expected = float(kernel.sum(axis=0).mean())
    return float(params.intra_in_degree / max(expected, 1e-9))


def _intra_connection_p(params: NetworkParams, positions: np.ndarray, compartment: np.ndarray):
    """Connection probability for within-compartment synapses: a number, or an expression."""
    if params.connection_sigma_um is None:
        return intra_probability(params)
    sigma = float(params.connection_sigma_um)
    scale = _distance_scale(params, positions, compartment)
    return (
        f"clip({scale} * exp(-((x_um_pre - x_um_post)**2 + (y_um_pre - y_um_post)**2)"
        f" / {2.0 * sigma**2}), 0, 1)"
    )


def _build(params: NetworkParams, layout=None):
    """Assemble the Brian2 network. Returns (network, spike monitor, neuron group).

    `layout` is the electrode layout the recording will use. The network needs
    it because neuron positions must be the *same* ones the electrodes later
    read from, or a distance-dependent wiring rule and the measurement would
    describe different cultures.
    """
    import brian2 as b2

    from neuromps.twin.mea import ElectrodeLayout, place_neurons

    layout = ElectrodeLayout() if layout is None else layout

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
    spread = max(params.threshold_spread, 0.0)
    neurons.v_threshold = f"{params.threshold_scale} * (1 + {spread} * randn())"
    positions, _ = place_neurons(params, layout, seed=params.seed)
    neurons.x_um, neurons.y_um = positions[:, 0], positions[:, 1]
    network = b2.Network(neurons)
    network.add(_synapses(params, neurons, compartment, excitatory, b2, positions))
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


# Brian2's default clock step. A delay below it rounds down to zero.
MIN_DELAY_MS = 0.1


def _spread_delays(synapses, params: NetworkParams, rng: np.random.Generator, b2) -> None:
    """Give each recurrent synapse its own conduction delay, if a spread is asked for.

    Drawn from the build-time generator, like connectivity and thresholds, so
    one seed fixes the whole chip. Clipped at the clock step, below which a
    delay would round to zero.
    """
    if params.intra_delay_sd_ms <= 0 or not len(synapses):
        return
    drawn = rng.normal(params.intra_delay_ms, params.intra_delay_sd_ms, len(synapses))
    synapses.delay = np.clip(drawn, MIN_DELAY_MS, None) * b2.ms


def _synapses(params: NetworkParams, neurons, compartment, excitatory, b2, positions) -> list:
    """Recurrent connections inside each compartment plus the directed pathway between them."""
    exc_weight = params.excitatory_weight * params.ampa_scale
    inh_weight = params.inhibitory_weight * params.gabaa_scale
    blocks = _ranges(params)
    made = []
    delay_rng = np.random.default_rng(params.seed + 1_000)
    # A scalar `delay=` makes Brian2 treat the delay as one shared value, which
    # cannot then be set per synapse, so it is omitted when a spread is wanted.
    fixed_delay = {} if params.intra_delay_sd_ms > 0 else {"delay": params.intra_delay_ms * b2.ms}
    depressing = params.depression_u > 0
    for kind, weight, target_var in (("exc", exc_weight, "g_e"), ("inh", inh_weight, "g_i")):
        if depressing and kind == "exc":
            # Vesicle depletion: each spike consumes a fraction u of what is
            # available and it recovers with tau_rec. x starts at 1, so the
            # first spike out of rest is unchanged and only sustained firing
            # is attenuated - which is how a burst terminates itself.
            synapses = b2.Synapses(
                neurons,
                neurons,
                model="dx/dt = (1 - x) / tau_rec : 1 (event-driven)",
                on_pre=f"{target_var}_post += {weight} * x; x = x * (1 - u_se)",
                **fixed_delay,
                namespace={
                    "tau_rec": params.depression_tau_rec_ms * b2.ms,
                    "u_se": params.depression_u,
                },
            )
        else:
            synapses = b2.Synapses(
                neurons,
                neurons,
                on_pre=f"{target_var}_post += {weight}",
                **fixed_delay,
            )
        connection_p = _intra_connection_p(params, positions, compartment)
        for c in (0, 1):  # recurrent, within each compartment only
            synapses.connect(_condition(blocks[c][kind], blocks[c]["all"]), p=connection_p)
        if depressing and kind == "exc":
            synapses.x = 1.0
        _spread_delays(synapses, params, delay_rng, b2)
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


def simulate(params: NetworkParams, layout=None) -> SimulationResult:
    """Run the two-compartment network and return its simulated spikes.

    `layout` is passed through so the wiring and the electrodes agree on where
    the neurons are; the default layout is the one `twin/mea.py` uses.
    """
    import brian2 as b2

    with _quiet_brian():
        network, monitor, compartment, excitatory = _build(params, layout)
        if params.noise_seed is not None:
            b2.seed(params.noise_seed)
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
