"""The firing-rate model: weights, size scaling, the response function, integration."""

from __future__ import annotations

import numpy as np

from neuromps.fly import connectome, rate_model


def _net() -> connectome.Connectome:
    """0 -> 1 (10 synapses, excitatory); 2 -| 1 (8, inhibitory); 0 -> 2 (3, below the floor)."""
    return connectome.Connectome(
        root_ids=np.arange(3),
        pre=np.array([0, 2, 0]),
        post=np.array([1, 1, 2]),
        synapses=np.array([10, 8, 3]),
        sign=np.array([1, -1, 1]),
        pre_class=np.array(["ach", "gaba", "ach"], dtype=object),
        neuropil=np.array(["A"] * 3, dtype=object),
    )


def test_weights_are_signed_scaled_and_floored_postsynaptic_by_presynaptic() -> None:
    w = rate_model.weights(_net()).toarray()
    expected = np.zeros((3, 3))
    expected[1, 0] = 0.3
    expected[1, 2] = -0.24
    np.testing.assert_allclose(w, expected)


def test_size_is_total_synapses_over_the_median() -> None:
    np.testing.assert_allclose(rate_model.sizes(_net()), [13 / 13, 18 / 13, 11 / 13])


def test_draws_are_positive_and_scaled_by_size() -> None:
    size = np.array([1.0, 2.0])
    p = rate_model.draw(size, [1, 2, 3])
    assert p.a.shape == (2, 3) and np.all(p.tau_s > 0)
    raw = rate_model.draw(np.ones(2), [1, 2, 3])
    np.testing.assert_allclose(p.theta[1], raw.theta[1] * 2)
    np.testing.assert_allclose(p.a[1], raw.a[1] / 2)


def test_no_input_stays_at_rest_and_a_step_settles_at_the_response() -> None:
    con = _net()
    w = rate_model.weights(con)
    p = rate_model.draw(np.ones(3), [7])
    rest = rate_model.simulate(w, p, [], 0.0, record=np.arange(3), duration_s=0.2)
    assert rest.rates.max() == 0.0
    run = rate_model.simulate(w, p, [0], 50.0, record=np.array([0]), duration_s=0.5)
    expected = rate_model.activation(np.full((3, 1), 50.0), p)[0, 0]
    assert abs(run.rates[-1, 0, 0] - expected) < 1e-3 * expected


def test_silenced_neurons_stay_at_zero() -> None:
    w = rate_model.weights(_net())
    p = rate_model.draw(np.ones(3), [7])
    run = rate_model.simulate(
        w, p, [0], 50.0, record=np.array([0, 1]), duration_s=0.2, silenced=np.array([0])
    )
    assert run.rates[:, 0].max() == 0.0 and run.rates[:, 1].max() == 0.0
