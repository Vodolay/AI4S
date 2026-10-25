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


def test_the_drive_input_brings_an_isolated_neuron_to_the_target_rate() -> None:
    con = _net()
    w = rate_model.weights(con)
    p = rate_model.draw(np.array([1.0, 3.0, 0.5]), [7, 8])
    amp = rate_model.drive_input(p, np.array([0, 2]), 50.0)
    assert amp.shape == (2, 2)
    run = rate_model.simulate(w, p, [0, 2], amp, record=np.array([0, 2]), duration_s=0.5)
    np.testing.assert_allclose(run.rates[-1], 50.0, rtol=1e-3)


def test_volume_sizes_are_median_normalized_and_fill_missing_with_the_median() -> None:
    got = rate_model.volume_sizes(np.array([100.0, 200.0, 300.0, np.nan, 0.0]))
    assert np.allclose(got, [0.5, 1.0, 1.5, 1.0, 1.0])


def test_volume_sizes_can_be_normalized_by_a_reference_population() -> None:
    got = rate_model.volume_sizes(np.array([100.0, 200.0]), reference=np.array([50.0, 50.0, 400.0]))
    assert np.allclose(got, [2.0, 4.0])


def test_draw_accepts_a_size_per_neuron_and_replicate() -> None:
    size = np.array([[1.0, 2.0], [4.0, 1.0]])
    p = rate_model.draw(size, [1, 2])
    q = rate_model.draw(np.ones(2), [1, 2])
    assert np.allclose(p.a * size, q.a) and np.allclose(p.theta / size, q.theta)


def test_class_scales_act_on_their_presynaptic_class_only() -> None:
    w = rate_model.weights(_net(), class_scale={"gaba": 0.5, "glut": 0.0}).toarray()
    assert np.isclose(w[1, 0], 0.3) and np.isclose(w[1, 2], -0.12)
