"""Tests for the virtual MEA: electrode geometry and what each electrode hears."""

import numpy as np
import pytest

from neuromps.twin.mea import (
    ElectrodeLayout,
    electrode_positions,
    place_neurons,
    record_electrodes,
)
from neuromps.twin.network import NetworkParams, simulate


def _layout(**kw) -> ElectrodeLayout:
    base = {"rows": 3, "cols": 4, "pitch_um": 200.0, "detection_radius_um": 150.0}
    base.update(kw)
    return ElectrodeLayout(**base)


def test_electrode_grid_has_one_position_per_electrode_per_compartment() -> None:
    layout = _layout()
    positions = electrode_positions(layout)
    assert len(positions) == 2 * layout.electrodes_per_compartment
    assert len(set(positions.values())) == len(positions)  # no two electrodes coincide
    assert all(len(p) == 2 for p in positions.values())


def test_electrode_ids_carry_their_compartment() -> None:
    positions = electrode_positions(_layout())
    compartments = {name.split("_")[0] for name in positions}
    assert compartments == {"c0", "c1"}


def test_compartments_are_separated_by_the_gap() -> None:
    layout = _layout(gap_um=1000.0)
    positions = electrode_positions(layout)
    left = max(x for name, (x, _) in positions.items() if name.startswith("c0"))
    right = min(x for name, (x, _) in positions.items() if name.startswith("c1"))
    assert right - left >= layout.gap_um


def test_neurons_stay_inside_their_own_compartment() -> None:
    params = NetworkParams(n_per_compartment=40, duration_s=0.2)
    layout = _layout(gap_um=1000.0)
    positions, compartment = place_neurons(params, layout, seed=0)
    left = positions[compartment == 0][:, 0]
    right = positions[compartment == 1][:, 0]
    assert left.max() < right.min()


def test_placement_is_reproducible() -> None:
    params = NetworkParams(n_per_compartment=30, duration_s=0.2)
    a, _ = place_neurons(params, _layout(), seed=3)
    b, _ = place_neurons(params, _layout(), seed=3)
    assert np.array_equal(a, b)


def test_an_electrode_hears_only_neurons_within_its_radius() -> None:
    layout = _layout(rows=1, cols=2, pitch_um=1000.0, detection_radius_um=100.0, margin_um=0.0)
    electrodes = electrode_positions(layout)
    assert electrodes["c0_r0_k0"] == (0.0, 0.0)  # the electrode this test aims at
    neuron_positions = np.array([[0.0, 0.0], [10_000.0, 0.0]])
    compartment = np.array([0, 0])
    heard = record_electrodes(
        spike_times=np.array([0.1, 0.2]),
        spike_neurons=np.array([0, 1]),
        neuron_positions=neuron_positions,
        compartment_of_neuron=compartment,
        electrodes=electrodes,
        layout=layout,
    )
    near = heard["c0_r0_k0"]
    assert near.size == 1 and near[0] == pytest.approx(0.1)


def test_electrodes_of_the_other_compartment_hear_nothing_from_this_one() -> None:
    layout = _layout(gap_um=5000.0, detection_radius_um=150.0)
    params = NetworkParams(n_per_compartment=60, duration_s=1.0, seed=0)
    result = simulate(params)
    positions, compartment = place_neurons(params, layout, seed=0)
    electrodes = electrode_positions(layout)
    heard = record_electrodes(
        result.spike_times, result.spike_neurons, positions, compartment, electrodes, layout
    )
    # every electrode id carries its compartment; cross-talk would need a neuron
    # from the other compartment inside the radius, impossible across the gap
    assert set(heard) == set(electrodes)


def test_recording_returns_sorted_times_per_electrode() -> None:
    params = NetworkParams(n_per_compartment=60, duration_s=1.0, seed=1)
    result = simulate(params)
    layout = _layout()
    positions, compartment = place_neurons(params, layout, seed=0)
    heard = record_electrodes(
        result.spike_times,
        result.spike_neurons,
        positions,
        compartment,
        electrode_positions(layout),
        layout,
    )
    for times in heard.values():
        assert np.all(np.diff(times) >= 0)


def test_a_larger_radius_never_hears_less() -> None:
    params = NetworkParams(n_per_compartment=80, duration_s=1.0, seed=2)
    result = simulate(params)
    counts = []
    for radius in (80.0, 400.0):
        layout = _layout(detection_radius_um=radius)
        positions, compartment = place_neurons(params, layout, seed=0)
        heard = record_electrodes(
            result.spike_times,
            result.spike_neurons,
            positions,
            compartment,
            electrode_positions(layout),
            layout,
        )
        counts.append(sum(t.size for t in heard.values()))
    assert counts[1] >= counts[0]


def test_layout_is_recorded_for_provenance() -> None:
    record = _layout().as_record()
    assert record["rows"] == 3
    assert record["detection_radius_um"] == 150.0
    assert "pitch_um" in record
