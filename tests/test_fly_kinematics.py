"""Reach and swing readouts, on signals whose answers are known."""

from __future__ import annotations

import numpy as np
import pytest

from neuromps.fly import kinematics


def test_time_near_head_is_the_fraction_below_the_threshold() -> None:
    d = np.r_[np.full(300, 1.5), np.full(700, 1.2)]
    assert kinematics.time_near_head(d, 1.328) == pytest.approx(0.7)


def test_a_held_posture_is_not_rhythmic_and_a_swing_is() -> None:
    t = np.arange(1000) * 0.001
    held = 1.21 + 0.002 * np.random.default_rng(0).standard_normal(1000)
    swing = 1.5 + 0.3 * np.sin(2 * np.pi * 4 * t)  # 4 Hz
    assert kinematics.rhythmicity(held) < 0.2
    assert kinematics.rhythmicity(swing) > 0.6


def test_a_flat_trace_has_zero_rhythmicity() -> None:
    assert kinematics.rhythmicity(np.full(1000, 1.528)) == 0.0


def test_stage1d_scoring_reads_the_registered_rules() -> None:
    import json

    import pandas as pd

    from neuromps.fly import embodied_stage1d as s1d

    spec = json.loads(s1d.REGISTRATION.read_text())
    rows = []
    for scale in (20.0, 10.0, 40.0):
        for drive in s1d.DRIVES:
            for seed in range(8):
                near = 0.9 if drive == "grooming" else 0.2
                rhythm = 0.6 if drive == "forward_walking" else 0.1
                rows.append(
                    {
                        "drive": drive,
                        "seed": seed,
                        "r_full": scale,
                        "time_near_head": near,
                        "rhythmicity": rhythm,
                    }
                )
    pools = pd.DataFrame(
        [
            {"drive": d, "front": 10.0 if d == "grooming" else 1.0, "middle": 0.5, "hind": 0.5}
            for d in s1d.DRIVES
        ]
    )
    got = s1d.score(pd.DataFrame(rows), pools, spec)
    assert all(v["outcome"] == "pass" for v in got.values())
