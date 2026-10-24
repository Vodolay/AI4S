"""Embodied stage 1: spikes to rates to muscle activation to a moving leg."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuromps.fly import body, muscle_map

BODY_READY = body.model_ready()


def test_the_rate_filter_turns_a_steady_train_into_its_rate() -> None:
    spikes = np.arange(0.0, 2.0, 0.02)  # 50 Hz for 2 s
    rate = muscle_map.filtered_rate(spikes, duration_s=2.0, dt_s=0.001, tau_s=0.02)
    assert rate[-500:].mean() == pytest.approx(50.0, rel=0.05)
    assert rate[0] < 60


def test_activation_is_the_mean_rate_of_a_muscles_neurons_over_the_full_scale() -> None:
    rates = np.array([[10.0, 30.0], [0.0, 100.0]])  # two neurons, two time steps
    got = muscle_map.activation(rates, [0, 1], r_full=20.0)
    assert got.tolist() == pytest.approx([1.0, 1.0])
    assert muscle_map.activation(rates[:, :1], [0], r_full=20.0).tolist() == pytest.approx(
        [0.5, 1e-4]
    )


def test_the_map_comes_from_the_registration() -> None:
    mapping, unmapped = muscle_map.registered_map()
    assert mapping["Tr flexor MN"] == ["LFF_trochanter_flexor_a", "LFF_trochanter_flexor_b"]
    assert "Fe reductor MN" in unmapped
    assert len({m for muscles in mapping.values() for m in muscles}) == 15


def test_motor_neurons_are_assigned_to_muscles_by_type_and_side() -> None:
    ann = pd.DataFrame(
        {
            "superclass": ["vnc_motor", "vnc_motor", "vnc_motor", "vnc_sensory"],
            "subclass": ["fl", "fl", "fl", "fl"],
            "somaSide": ["L", "R", "L", "L"],
            "type": ["Tr flexor MN", "Tr flexor MN", "Fe reductor MN", "Tr flexor MN"],
        }
    )
    got = muscle_map.assign(ann)
    assert got.neurons_by_muscle["LFF_trochanter_flexor_a"] == [0]
    assert got.coverage == {"mapped": 1, "left_front_leg_motor_neurons": 2}


@pytest.mark.skipif(not BODY_READY, reason="needs the fetched FlyMimic model")
def test_no_activation_leaves_the_leg_at_rest_and_a_flexor_brings_it_to_the_head() -> None:
    names = body.muscle_names()
    rest = body.run(np.full((300, len(names)), 1e-4), names)
    flex = np.full((300, len(names)), 1e-4)
    flex[:, names.index("LFF_trochanter_flexor_b")] = 1.0
    moved = body.run(flex, names)
    assert abs(rest.distance_mm.min() - body.REST_DISTANCE_MM) < 0.05
    assert moved.distance_mm.min() < rest.distance_mm.min() - 0.3


def test_stage1_scoring_reads_the_registered_rules() -> None:
    import json

    from neuromps.fly import embodied_stage1 as s1

    spec = json.loads(s1.REGISTRATION.read_text())
    rows = []
    for scale in s1.R_FULL:
        for seed in range(8):
            rows += [
                {"drive": "jon_ce", "seed": seed, "r_full": scale, "min_distance_mm": 1.0},
                {"drive": "sugar", "seed": seed, "r_full": scale, "min_distance_mm": 1.4},
                {"drive": "none", "seed": seed, "r_full": scale, "min_distance_mm": 1.53},
            ]
    got = s1.score(pd.DataFrame(rows), spec)
    assert all(v["outcome"] == "pass" for v in got.values())
