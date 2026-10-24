"""The fly-brain model: transmitter-class drug knobs, regions, electrodes, readouts.

Everything here runs on a hand-built six-neuron connectome, so each rule can be
checked against an answer worked out by hand. The whole-brain run is exercised
by the fly stage itself, not by the unit tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuromps.features.mea import FEATURE_NAMES
from neuromps.fly import behaviour, connectome, electrodes, model


def _toy() -> connectome.Connectome:
    """0 (ACh) -> 1 -> 2; 3 (GABA) -| 2; 4 (Glut, inhibitory) -| 2; 5 (dopamine) -> 2."""
    pre = np.array([0, 1, 3, 4, 5])
    post = np.array([1, 2, 2, 2, 2])
    synapses = np.array([10, 10, 6, 4, 2])
    sign = np.array([1, 1, -1, -1, 1])
    nt = ["acetylcholine", "acetylcholine", "acetylcholine", "gaba", "glutamate", "dopamine"]
    return connectome.Connectome(
        root_ids=np.arange(100, 106),
        pre=pre,
        post=post,
        synapses=synapses,
        sign=sign,
        pre_class=connectome.transmitter_class(sign, np.asarray(nt, dtype=object)[pre]),
        neuropil=np.array(["AL_R", "LH_R", "LH_R", "LH_R", "", "LH_R"], dtype=object),
    )


def test_inhibitory_connections_split_into_gaba_and_glutamate() -> None:
    assert _toy().pre_class.tolist() == ["ach", "ach", "gaba", "glut", "other_exc"]


def test_a_label_that_disagrees_with_the_sign_is_left_to_no_knob() -> None:
    """A neuron the model treats as inhibitory but annotated cholinergic is not 'ach'."""
    classes = connectome.transmitter_class(
        np.array([-1, 1, 1]), np.array(["acetylcholine", "gaba", None], dtype=object)
    )
    assert classes.tolist() == ["other_inh", "other_exc", "other_exc"]


def test_each_knob_scales_only_its_own_transmitter() -> None:
    con = _toy()
    base = model.synaptic_weights(con, model.FlyParams())
    assert base.tolist() == pytest.approx([2.75, 2.75, -1.65, -1.1, 0.55])
    gaba = model.synaptic_weights(con, model.FlyParams(gaba_scale=0.0))
    assert gaba.tolist() == pytest.approx([2.75, 2.75, 0.0, -1.1, 0.55])
    glut = model.synaptic_weights(con, model.FlyParams(glut_scale=2.0))
    assert glut.tolist() == pytest.approx([2.75, 2.75, -1.65, -2.2, 0.55])
    ach = model.synaptic_weights(con, model.FlyParams(ach_scale=0.5))
    assert ach.tolist() == pytest.approx([1.375, 1.375, -1.65, -1.1, 0.55])


def test_the_threshold_knob_scales_the_distance_from_rest() -> None:
    p = model.FlyParams(threshold_scale=2.0)
    assert model.threshold_mv(p) == pytest.approx(-52.0 + 2.0 * 7.0)
    assert model.threshold_mv(model.FlyParams()) == pytest.approx(-45.0)


def test_background_weight_gives_the_stated_mean_depolarisation() -> None:
    p = model.FlyParams(background_mean_mv=5.0, background_inputs=100, background_rate_hz=10.0)
    weight = model.background_weight_mv(p)
    assert weight * p.background_inputs * p.background_rate_hz * p.t_mbr_ms / 1000 == 5.0


def test_a_neuron_sits_in_the_region_where_it_receives_most_synapses() -> None:
    counts = pd.DataFrame(
        {
            "post_pt_root_id": [100, 100, 101, 101, 102],
            "neuropil": ["AL_R", "LH_R", "LH_R", "None", "SMP_L"],
            "count": [5, 3, 2, 9, 1],
        }
    )
    got = connectome.primary_neuropil(counts, np.array([100, 101, 102, 103]))
    assert got.tolist() == ["AL_R", "LH_R", "SMP_L", ""]


def test_region_to_region_synapses_follow_the_neurons_regions() -> None:
    flow = connectome.neuropil_flow(_toy())
    assert flow.loc[("AL_R", "LH_R")] == 10
    assert flow.loc[("LH_R", "LH_R")] == 10 + 6 + 2  # the unplaced neuron 4 is dropped
    assert ("LH_R", "AL_R") not in flow.index


def test_electrodes_draw_disjoint_units_from_their_own_region() -> None:
    region = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 3, dtype=object)
    array = electrodes.build_array(region, electrodes_per_region=2, units_per_electrode=4, seed=1)
    assert sorted({array.region[e] for e in array.units}) == ["A", "B"]  # C is too small
    units = np.concatenate(list(array.units.values()))
    assert units.size == len(set(units.tolist())) == 16
    assert all((region[array.units[e]] == array.region[e]).all() for e in array.units)
    again = electrodes.build_array(region, electrodes_per_region=2, units_per_electrode=4, seed=1)
    assert {e: u.tolist() for e, u in again.units.items()} == {
        e: u.tolist() for e, u in array.units.items()
    }


def test_an_electrode_hears_the_merged_spikes_of_its_units() -> None:
    array = electrodes.VirtualArray(
        units={"A_0": np.array([0, 2]), "B_0": np.array([1])}, region={"A_0": "A", "B_0": "B"}
    )
    spikes = model.FlySpikes(
        times=np.array([0.3, 0.1, 0.2, 0.4]), neurons=np.array([2, 0, 1, 5]), duration_s=1.0
    )
    trains = electrodes.electrode_trains(spikes, array)
    assert trains["A_0"].tolist() == [0.1, 0.3]
    assert trains["B_0"].tolist() == [0.2]


def test_virtual_recordings_are_scored_by_the_one_mea_extractor() -> None:
    rng = np.random.default_rng(0)
    array = electrodes.VirtualArray(
        units={f"A_{k}": np.array([k]) for k in range(4)}, region={f"A_{k}": "A" for k in range(4)}
    )
    times = np.sort(rng.uniform(0, 10, 400))
    spikes = model.FlySpikes(times=times, neurons=rng.integers(0, 4, 400), duration_s=10.0)
    features = electrodes.mea_features(spikes, array)
    assert set(features) == set(FEATURE_NAMES)
    assert features["active_electrodes"] == 4


def test_readout_rates_are_averaged_over_trials() -> None:
    spikes = [
        model.FlySpikes(np.array([0.1, 0.2]), np.array([7, 7]), 1.0),
        model.FlySpikes(np.array([0.1]), np.array([7]), 1.0),
    ]
    assert behaviour.rate_hz(spikes, [7]) == pytest.approx(1.5)
    assert behaviour.rate_hz(spikes, [8]) == 0.0


def test_gating_is_the_fraction_of_feeding_drive_the_bitter_input_removes() -> None:
    assert behaviour.gating_index(sugar_hz=40.0, sugar_bitter_hz=10.0) == pytest.approx(0.75)
    assert behaviour.gating_index(sugar_hz=0.0, sugar_bitter_hz=0.0) is None


@pytest.mark.slow
def test_blocking_gaba_releases_a_neuron_the_gaba_input_was_holding_down() -> None:
    """A real three-neuron Brian2 run: drive 0 -> 1, with 3 inhibiting 1 by GABA."""
    con = connectome.Connectome(
        root_ids=np.arange(4),
        pre=np.array([0, 3]),
        post=np.array([1, 1]),
        synapses=np.array([40, 60]),
        sign=np.array([1, -1]),
        pre_class=np.array(["ach", "gaba"], dtype=object),
        neuropil=np.array(["A"] * 4, dtype=object),
    )
    rates = {}
    for scale in (1.0, 0.0):
        p = model.FlyParams(duration_s=1.0, gaba_scale=scale, stim_rate_hz=100.0, seed=3)
        got = model.simulate(con, p, excite=[0, 3])
        rates[scale] = int(np.sum(got.neurons == 1))
    assert rates[0.0] > rates[1.0]


def test_tables_from_another_flywire_release_are_refused() -> None:
    """Root IDs change when a neuron is edited, so a 630 table matches few 783 neurons."""
    model_ids = np.arange(1000)
    connectome.check_release("annotations", model_ids, np.arange(995))  # 99.5%: fine
    with pytest.raises(connectome.ReleaseMismatch, match="annotations"):
        connectome.check_release("annotations", model_ids, np.arange(760))


@pytest.mark.slow
def test_the_background_alone_makes_unconnected_neurons_fire() -> None:
    """With no synapses the rate comes from the background drive only, and grows with it."""
    con = connectome.Connectome(
        root_ids=np.arange(200),
        pre=np.array([0]),
        post=np.array([1]),
        synapses=np.array([0]),
        sign=np.array([1]),
        pre_class=np.array(["ach"], dtype=object),
        neuropil=np.array(["A"] * 200, dtype=object),
    )
    counts = [
        model.simulate(
            con, model.FlyParams(duration_s=2.0, background_mean_mv=m, seed=1)
        ).times.size
        for m in (0.0, 5.0, 7.0)
    ]
    assert counts[0] == 0 < counts[1] < counts[2]


@pytest.mark.slow
def test_a_restricted_monitor_reports_model_indices() -> None:
    con = connectome.Connectome(
        root_ids=np.arange(50),
        pre=np.array([0]),
        post=np.array([1]),
        synapses=np.array([0]),
        sign=np.array([1]),
        pre_class=np.array(["ach"], dtype=object),
        neuropil=np.array(["A"] * 50, dtype=object),
    )
    got = model.simulate(con, model.FlyParams(duration_s=1.0), excite=[40], record=[40, 41])
    assert set(got.neurons.tolist()) == {40}


def test_the_cached_connectome_is_rebuilt_when_an_input_changes(monkeypatch, tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in connectome.INPUTS:
        (raw / name).write_text("v1")
    builds = []
    monkeypatch.setattr(connectome, "load", lambda r: builds.append(1) or _toy())
    first = connectome.cached(raw, tmp_path / "cache")
    again = connectome.cached(raw, tmp_path / "cache")
    assert len(builds) == 1
    assert again.pre_class.tolist() == first.pre_class.tolist()
    assert again.neuropil.tolist() == first.neuropil.tolist()
    (raw / connectome.INPUTS[1]).write_text("v2")
    connectome.cached(raw, tmp_path / "cache")
    assert len(builds) == 2


@pytest.mark.slow
def test_pulsed_stimulation_drives_only_inside_the_pulses() -> None:
    con = connectome.Connectome(
        root_ids=np.arange(10),
        pre=np.array([0]),
        post=np.array([1]),
        synapses=np.array([0]),
        sign=np.array([1]),
        pre_class=np.array(["ach"], dtype=object),
        neuropil=np.array(["A"] * 10, dtype=object),
    )
    p = model.FlyParams(
        duration_s=2.0, stim_rate_hz=400.0, stim_period_ms=200.0, stim_width_ms=20.0, seed=2
    )
    got = model.simulate(con, p, excite=[3])
    phase_ms = (got.times * 1000) % 200.0
    assert got.times.size > 20
    assert np.all(phase_ms < 20.0 + 1.0)  # a spike can land one step after the pulse ends


def test_a_subnetwork_keeps_only_connections_among_the_kept_neurons() -> None:
    con = _toy()
    keep = np.array([False, True, True, True, False, False])
    sub = connectome.subnetwork(con, keep)
    assert sub.root_ids.tolist() == [101, 102, 103]
    # 1 -> 2 and 3 -| 2 survive, renumbered 0 -> 1 and 2 -| 1; 0 -> 1, 4 -| 2, 5 -> 2 do not
    assert list(zip(sub.pre.tolist(), sub.post.tolist(), strict=True)) == [(0, 1), (2, 1)]
    assert sub.pre_class.tolist() == ["ach", "gaba"]
    assert sub.neuropil.tolist() == ["LH_R", "LH_R", "LH_R"]


@pytest.mark.slow
def test_replayed_spikes_make_their_target_fire_at_those_times() -> None:
    con = connectome.Connectome(
        root_ids=np.arange(5),
        pre=np.array([0]),
        post=np.array([1]),
        synapses=np.array([0]),
        sign=np.array([1]),
        pre_class=np.array(["ach"], dtype=object),
        neuropil=np.array(["A"] * 5, dtype=object),
    )
    times = np.array([0.1, 0.3, 0.5])
    got = model.simulate(con, model.FlyParams(duration_s=0.7), replay={3: times})
    fired = got.times[got.neurons == 3]
    assert fired.size == 3
    assert np.allclose(fired, times, atol=0.002)
    assert set(got.neurons.tolist()) == {3}


def _driven_pair() -> connectome.Connectome:
    return connectome.Connectome(
        root_ids=np.arange(2),
        pre=np.array([0]),
        post=np.array([1]),
        synapses=np.array([100]),
        sign=np.array([1]),
        pre_class=np.array(["ach"], dtype=object),
        neuropil=np.array(["A"] * 2, dtype=object),
    )


@pytest.mark.slow
def test_without_adaptation_the_model_is_the_published_one() -> None:
    con = _driven_pair()
    plain = model.simulate(con, model.FlyParams(seed=4), excite=[0])
    zero = model.simulate(con, model.FlyParams(seed=4, adapt_mv=0.0, adapt_tau_ms=37.0), excite=[0])
    np.testing.assert_array_equal(plain.times, zero.times)
    np.testing.assert_array_equal(plain.neurons, zero.neurons)


@pytest.mark.slow
def test_stronger_adaptation_means_fewer_spikes_under_the_same_drive() -> None:
    con = _driven_pair()
    counts = [
        int(
            np.sum(
                model.simulate(
                    con, model.FlyParams(seed=4, adapt_mv=b, adapt_tau_ms=200.0), excite=[0]
                ).neurons
                == 1
            )
        )
        for b in (0.0, 2.0, 8.0)
    ]
    assert counts[0] > counts[1] > counts[2] > 0


def test_rewiring_keeps_degrees_and_weights_with_their_presynaptic_neuron() -> None:
    con = _toy()
    got = connectome.rewire(con, seed=0)
    np.testing.assert_array_equal(got.pre, con.pre)
    np.testing.assert_array_equal(np.sort(got.post), np.sort(con.post))
    np.testing.assert_array_equal(got.synapses, con.synapses)
    np.testing.assert_array_equal(got.sign, con.sign)
    big = connectome.Connectome(
        root_ids=np.arange(50),
        pre=np.repeat(np.arange(50), 4),
        post=np.tile(np.arange(4), 50),
        synapses=np.ones(200, int),
        sign=np.ones(200, int),
        pre_class=np.array(["ach"] * 200, dtype=object),
        neuropil=np.array(["A"] * 50, dtype=object),
    )
    assert not np.array_equal(connectome.rewire(big, seed=0).post, big.post)


def test_making_connections_excitatory_flips_only_those_and_keeps_weights() -> None:
    con = _toy()
    got = connectome.make_excitatory(con, con.pre_class == "glut")
    np.testing.assert_array_equal(got.sign, [1, 1, -1, 1, 1])
    np.testing.assert_array_equal(got.synapses, con.synapses)
    np.testing.assert_array_equal(con.sign, [1, 1, -1, -1, 1])
