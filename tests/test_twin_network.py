"""Tests for the two-compartment spiking network.

Kept small and short so the suite stays fast; the scientific validation (burst
rates in the range the real cultures show) lives in the calibration script.
"""

import numpy as np
import pytest

from neuromps.twin.network import NetworkParams, simulate, synapse_counts


def _params(**kw) -> NetworkParams:
    base = {
        "n_per_compartment": 80,
        "duration_s": 1.0,
        "seed": 0,
    }
    base.update(kw)
    return NetworkParams(**base)


def test_excitatory_inhibitory_split_follows_the_fraction() -> None:
    result = simulate(_params(n_per_compartment=100, excitatory_fraction=0.8))
    assert result.n_excitatory == 80
    assert result.n_inhibitory == 20
    assert result.n_neurons == 200  # two compartments


def test_compartment_labels_cover_every_neuron() -> None:
    result = simulate(_params())
    assert set(result.compartment_of_neuron) == {0, 1}
    assert len(result.compartment_of_neuron) == result.n_neurons


def test_simulation_is_reproducible_with_a_seed() -> None:
    a, b = simulate(_params(seed=3)), simulate(_params(seed=3))
    assert np.array_equal(a.spike_times, b.spike_times)
    assert np.array_equal(a.spike_neurons, b.spike_neurons)


def test_different_seeds_give_different_spike_trains() -> None:
    a, b = simulate(_params(seed=1)), simulate(_params(seed=2))
    assert not np.array_equal(a.spike_times, b.spike_times)


def test_no_background_drive_leaves_the_network_silent() -> None:
    result = simulate(_params(background_rate_hz=0.0, background_weight=0.0))
    assert result.spike_times.size == 0


def test_forward_ratio_shapes_the_inter_compartment_projection() -> None:
    """The 'axon diode': forward and backward connection probabilities differ by the ratio."""
    symmetric = synapse_counts(_params(inter_in_degree=8.0, forward_ratio=1.0))
    asymmetric = synapse_counts(_params(inter_in_degree=8.0, forward_ratio=8.0))
    assert symmetric["forward"] == pytest.approx(symmetric["backward"], rel=0.35)
    assert asymmetric["forward"] > 3 * asymmetric["backward"]


def test_zero_inter_probability_disconnects_the_compartments() -> None:
    counts = synapse_counts(_params(inter_in_degree=0.0))
    assert counts["forward"] == 0
    assert counts["backward"] == 0


def test_spikes_are_within_the_simulated_window() -> None:
    result = simulate(_params(duration_s=0.5))
    assert result.duration_s == pytest.approx(0.5)
    if result.spike_times.size:
        assert result.spike_times.min() >= 0.0
        assert result.spike_times.max() <= 0.5


def test_spikes_by_compartment_splits_the_train() -> None:
    result = simulate(_params())
    per_compartment = result.spikes_by_compartment()
    assert set(per_compartment) == {0, 1}
    total = sum(len(v) for v in per_compartment.values())
    assert total == result.spike_times.size


def test_drug_knobs_scale_conductances_not_silently_ignored() -> None:
    """A strong GABA-A block must raise activity; the knob has to be wired in.

    Needs enough neurons and time for inhibitory cells to fire at all: at 80
    neurons for 1 s the network is too quiet for inhibition to matter.
    """
    baseline = simulate(_params(n_per_compartment=150, duration_s=4.0, seed=0))
    disinhibited = simulate(_params(n_per_compartment=150, duration_s=4.0, seed=0, gabaa_scale=0.0))
    assert disinhibited.spike_times.size > 1.5 * baseline.spike_times.size


def test_asymmetry_follows_drive_balance_not_only_connectivity() -> None:
    """Regression guard for the HD11 finding.

    A directed pathway also excites its target, so the target can burst *first*.
    Coupling asymmetry therefore tracks the compartments' relative drive, and a
    strongly directed pathway can read as reversed when drive is equal. Anything
    that infers direction from this measure must control for firing rate.
    """
    from neuromps.features.coupling import directed_coupling

    def asymmetry(scale: float) -> float:
        result = simulate(
            _params(
                n_per_compartment=150,
                duration_s=20.0,
                seed=0,
                inter_in_degree=9.0,
                forward_ratio=16.0,
                background_scale_target=scale,
            )
        )
        spikes = result.spikes_by_compartment()
        out = directed_coupling(
            {"p": np.sort(spikes[0])}, {"p": np.sort(spikes[1])}, result.duration_s
        )
        return out["asymmetry"]

    equal_drive, weaker_target = asymmetry(1.0), asymmetry(0.7)
    assert equal_drive < weaker_target  # same wiring, opposite apparent direction


def test_parameters_are_recorded_for_provenance() -> None:
    params = _params(forward_ratio=4.0)
    record = params.as_record()
    assert record["forward_ratio"] == 4.0
    assert record["n_per_compartment"] == 80
    assert "seed" in record


def test_noise_seed_changes_the_recording_but_not_the_chip() -> None:
    """A pair of recordings from one device: same wiring, different noise."""
    base = NetworkParams(n_per_compartment=60, duration_s=1.0, seed=3)
    same = NetworkParams(n_per_compartment=60, duration_s=1.0, seed=3, noise_seed=99)
    counts_base = synapse_counts(base)
    counts_same = synapse_counts(same)
    assert counts_base == counts_same, "noise_seed must not move the connectivity"
    a = simulate(base)
    b = simulate(same)
    assert a.spike_times.size and b.spike_times.size
    assert not np.array_equal(a.spike_times, b.spike_times), "the noise should differ"


def test_both_seeding_paths_stay_deterministic() -> None:
    """Reproducibility, with and without the extra reseed.

    `noise_seed=seed` is deliberately *not* the same run as `noise_seed=None`:
    the build consumes draws before the reseed, so restarting the stream at the
    same number lands elsewhere in it. What matters is that the default path is
    untouched - every earlier result reproduces - and that a stated noise seed
    reproduces itself.
    """
    for noise_seed in (None, 5):
        first = NetworkParams(n_per_compartment=60, duration_s=1.0, seed=5, noise_seed=noise_seed)
        second = NetworkParams(n_per_compartment=60, duration_s=1.0, seed=5, noise_seed=noise_seed)
        assert np.array_equal(simulate(first).spike_times, simulate(second).spike_times)


def test_two_noise_seeds_on_one_chip_give_different_recordings() -> None:
    chip = dict(n_per_compartment=60, duration_s=1.0, seed=5)
    a = simulate(NetworkParams(**chip, noise_seed=1))
    b = simulate(NetworkParams(**chip, noise_seed=2))
    assert not np.array_equal(a.spike_times, b.spike_times)


def _recurrent_delays_ms(params: NetworkParams) -> np.ndarray:
    """Delays of the within-compartment synapses of one built network, in ms.

    The inter-compartment pathway carries its own fixed `inter_delay_ms` and is
    excluded, so it is identified the way the model defines it: a synapse is
    recurrent when its source and target sit in the same compartment.
    """
    import brian2 as b2

    from neuromps.twin.network import _build

    network, _, _, _ = _build(params)
    n = params.n_per_compartment
    delays = []
    # `network.objects` has no stable order, so two builds would otherwise
    # concatenate the same synapses in different orders.
    for o in sorted(network.objects, key=lambda o: o.name):
        if not isinstance(o, b2.Synapses) or not len(o):
            continue
        source, target = np.asarray(o.i[:]) // n, np.asarray(o.j[:]) // n
        if not np.array_equal(source, target):
            continue  # the pathway between compartments
        # With no spread the delay is a shared scalar, so broadcast it to one
        # value per synapse and the two cases stay comparable.
        value = np.atleast_1d(np.asarray(o.delay / b2.ms, dtype=float))
        delays.append(np.full(len(o), value[0]) if value.size == 1 else value)
    return np.concatenate(delays)


def test_delay_spread_defaults_to_zero_so_earlier_results_are_unchanged() -> None:
    """Every result before this parameter existed was produced with one fixed delay."""
    assert NetworkParams().intra_delay_sd_ms == 0.0
    delays = _recurrent_delays_ms(NetworkParams(n_per_compartment=40, duration_s=0.1))
    assert np.allclose(delays, delays[0]), "with sd 0 every synapse must share one delay"


def test_delay_spread_jitters_the_recurrent_delays_around_the_mean() -> None:
    """The mechanism: identical delays lock recurrent activity into volleys."""
    params = NetworkParams(
        n_per_compartment=120, duration_s=0.1, intra_delay_ms=1.5, intra_delay_sd_ms=0.6
    )
    delays = _recurrent_delays_ms(params)
    assert delays.size > 200
    assert delays.std() > 0.2, "requested spread did not reach the synapses"
    assert 1.0 < delays.mean() < 2.0
    assert (delays > 0).all(), "a non-positive delay is not physical"


def test_delay_spread_is_reproducible_under_a_seed() -> None:
    """Heterogeneity is drawn at build time, like connectivity and thresholds."""
    kw = dict(n_per_compartment=60, duration_s=0.1, intra_delay_sd_ms=0.6)
    first = _recurrent_delays_ms(NetworkParams(seed=3, **kw))
    again = _recurrent_delays_ms(NetworkParams(seed=3, **kw))
    other = _recurrent_delays_ms(NetworkParams(seed=4, **kw))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other[: first.size]) or first.size != other.size


def _recurrent_pairs_and_positions(params: NetworkParams):
    """Connected within-compartment (source, target) pairs, and neuron positions."""
    import brian2 as b2

    from neuromps.twin.mea import ElectrodeLayout, place_neurons
    from neuromps.twin.network import _build

    layout = ElectrodeLayout()
    network, _, _, _ = _build(params, layout)
    n = params.n_per_compartment
    pairs = []
    for o in sorted(network.objects, key=lambda o: o.name):
        if not isinstance(o, b2.Synapses) or not len(o):
            continue
        i, j = np.asarray(o.i[:]), np.asarray(o.j[:])
        if not np.array_equal(i // n, j // n):
            continue  # the pathway between compartments
        pairs.append(np.column_stack([i, j]))
    positions, _ = place_neurons(params, layout, seed=params.seed)
    return np.vstack(pairs), positions


def _distances(pairs: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.hypot(*(positions[pairs[:, 0]] - positions[pairs[:, 1]]).T)


def test_connectivity_is_distance_independent_by_default() -> None:
    """Every result so far was produced with uniform random wiring."""
    assert NetworkParams().connection_sigma_um is None
    params = NetworkParams(n_per_compartment=200, duration_s=0.1, intra_in_degree=20.0)
    pairs, positions = _recurrent_pairs_and_positions(params)
    connected = _distances(pairs, positions).mean()
    same_chamber = positions[: params.n_per_compartment]
    everything = np.hypot(
        *(same_chamber[:, None, :] - same_chamber[None, :, :]).reshape(-1, 2).T
    ).mean()
    assert abs(connected - everything) / everything < 0.10


def test_a_connection_range_wires_neighbours_preferentially() -> None:
    """Uniform wiring makes every electrode pair statistically identical."""
    params = NetworkParams(
        n_per_compartment=200, duration_s=0.1, intra_in_degree=20.0, connection_sigma_um=200.0
    )
    pairs, positions = _recurrent_pairs_and_positions(params)
    uniform = NetworkParams(n_per_compartment=200, duration_s=0.1, intra_in_degree=20.0)
    wide_pairs, wide_positions = _recurrent_pairs_and_positions(uniform)
    assert _distances(pairs, positions).mean() < 0.6 * _distances(wide_pairs, wide_positions).mean()


def test_a_connection_range_preserves_the_requested_in_degree() -> None:
    """In-degree is the parameterisation; a range must retune density, not change it."""
    for sigma in (None, 400.0):
        params = NetworkParams(
            n_per_compartment=200,
            duration_s=0.1,
            intra_in_degree=20.0,
            connection_sigma_um=sigma,
        )
        pairs, _ = _recurrent_pairs_and_positions(params)
        in_degree = len(pairs) / (2 * params.n_per_compartment)
        assert 14.0 < in_degree < 27.0, (sigma, in_degree)


def test_distance_dependent_wiring_is_reproducible_from_the_seed() -> None:
    kw = dict(n_per_compartment=120, duration_s=0.1, connection_sigma_um=250.0)
    first, _ = _recurrent_pairs_and_positions(NetworkParams(seed=5, **kw))
    again, _ = _recurrent_pairs_and_positions(NetworkParams(seed=5, **kw))
    assert np.array_equal(first, again)


def test_the_wiring_and_the_electrodes_place_neurons_identically() -> None:
    """A distance rule is meaningless if the electrodes read a different culture."""
    import brian2 as b2

    from neuromps.twin.mea import ElectrodeLayout, place_neurons
    from neuromps.twin.network import _build

    layout = ElectrodeLayout(detection_radius_um=75.0)
    params = NetworkParams(n_per_compartment=60, duration_s=0.1, connection_sigma_um=250.0)
    network, _, _, _ = _build(params, layout)
    neurons = next(o for o in network.objects if isinstance(o, b2.NeuronGroup))
    expected, _ = place_neurons(params, layout, seed=params.seed)
    assert np.allclose(np.asarray(neurons.x_um[:]), expected[:, 0])
    assert np.allclose(np.asarray(neurons.y_um[:]), expected[:, 1])
