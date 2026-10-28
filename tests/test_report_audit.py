"""The registration audit counts outcomes at the top level and one level down."""

from __future__ import annotations

from neuromps.report import audit


def test_counts_top_level_and_sub_designs_and_skips_the_rest() -> None:
    metrics = {
        "a": {"predictions": {"x": {"outcome": "pass"}, "y": {"outcome": "fail"}}},
        "b": {"rule1": {"predictions": {"x": {"outcome": "uninterpretable"}}}, "n": 3},
        "c": {"value": 1.0},
        "d": {"predictions": {"x": {"value": 1}}},
        "registration_audit": {"stale": True},
    }
    got = audit.audit(metrics)
    assert set(got["studies"]) == {"a", "b__rule1"}
    assert got["studies"]["a"] == {
        "pass": 1,
        "fail": 1,
        "uninterpretable": 0,
        "other": 0,
        "total": 2,
    }
    assert got["total"]["total"] == 3 and got["n_studies"] == 2
