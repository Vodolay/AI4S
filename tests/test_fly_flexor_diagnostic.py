"""The exploratory flexor diagnostic's two helpers, on a hand-built network."""

from __future__ import annotations

import numpy as np

from neuromps.fly import connectome, flexor_diagnostic


def _net() -> connectome.Connectome:
    return connectome.Connectome(
        root_ids=np.arange(4),
        pre=np.array([0, 1, 2, 0]),
        post=np.array([3, 3, 3, 2]),
        synapses=np.array([5, 7, 11, 3]),
        sign=np.array([1, -1, 1, 1]),
        pre_class=np.array(["ach", "gaba", "ach", "ach"], dtype=object),
        neuropil=np.array(["A"] * 4, dtype=object),
    )


def test_removing_inhibition_zeroes_only_inhibitory_connections() -> None:
    got = flexor_diagnostic.without_inhibition(_net())
    np.testing.assert_array_equal(got.synapses, [5, 0, 11, 3])


def test_inputs_count_signed_synapses_from_the_sources_per_target() -> None:
    got = flexor_diagnostic.inputs_from(_net(), np.array([0, 1]), np.array([3]))
    assert got == {
        "excitatory_per_neuron": 5.0,
        "inhibitory_per_neuron": 7.0,
        "inhibitory_by_transmitter_per_neuron": {"gaba": 7.0},
    }
