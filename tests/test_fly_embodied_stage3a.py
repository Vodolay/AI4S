"""Stage 3a scoring: the registered rules on synthetic tables."""

from __future__ import annotations

import json

import pandas as pd

from neuromps.fly import embodied_stage3a as s3a

SPEC = json.loads(s3a.REGISTRATION.read_text())


def _table(shift: dict[str, float], near: float = 0.9) -> pd.DataFrame:
    rows = []
    for arm in ("open", "resistance", "assistance", "sham"):
        for drive in ("grooming", "none"):
            for seed in range(8):
                rows.append(
                    {
                        "arm": arm,
                        "drive": drive,
                        "seed": seed,
                        "median_tibia_rad": 0.6 + shift.get(arm, 0.0),
                        "time_near_head": near if drive == "grooming" else 0.0,
                        "leg_motor_spikes": 0 if drive == "none" else 100,
                    }
                )
    return pd.DataFrame(rows)


def test_a_resistance_reflex_passes_every_rule() -> None:
    got = s3a.score(_table({"resistance": 0.3, "sham": 0.05, "assistance": -0.02}), SPEC)
    assert all(v["outcome"] == "pass" for v in got.values()), got


def test_no_resistance_makes_the_contingency_test_uninterpretable() -> None:
    got = s3a.score(_table({"resistance": 0.02, "sham": 0.0}), SPEC)
    assert got["L1_feedback_resists_the_extension"]["outcome"] == "fail"
    assert got["L2_the_change_needs_the_loop"]["outcome"] == "uninterpretable"


def test_a_sham_as_strong_as_the_loop_fails_the_contingency_test() -> None:
    got = s3a.score(_table({"resistance": 0.3, "sham": 0.35}), SPEC)
    assert got["L2_the_change_needs_the_loop"]["outcome"] == "fail"
