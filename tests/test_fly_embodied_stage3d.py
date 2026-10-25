"""Stage 3d scoring: the paired sign count and the median margin."""

from __future__ import annotations

import json

import pandas as pd

from neuromps.fly import embodied_stage3d as s3d

SPEC = json.loads(s3d.REGISTRATION.read_text())


def _table(favour: int, gap: float) -> pd.DataFrame:
    rows = []
    for seed in range(32):
        res = 20.0
        sham = res + gap if seed < favour else res - 1.0
        rows += [
            {"arm": "resistance", "seed": seed, "recovery_ms": res},
            {"arm": "sham", "seed": seed, "recovery_ms": sham},
        ]
    return pd.DataFrame(rows)


def test_enough_seeds_by_enough_margin_pass() -> None:
    got = s3d.score(_table(24, 10.0), SPEC)
    assert [v["outcome"] for v in got.values()] == ["pass", "pass"]


def test_too_few_seeds_or_a_small_margin_fail() -> None:
    assert (
        s3d.score(_table(21, 10.0), SPEC)["Q1_feedback_that_follows_the_push_speeds_the_return"][
            "outcome"
        ]
        == "fail"
    )
    assert s3d.score(_table(24, 2.0), SPEC)["Q2_by_a_meaningful_margin"]["outcome"] == "fail"
