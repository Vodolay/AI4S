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


def test_names_the_studies_whose_predictions_it_cannot_read() -> None:
    metrics = {
        "a": {"predictions": {"x": {"outcome": "pass"}}},
        "d": {"predictions": {"x": {"pass": True}}},
        "e": {"rule": {"predictions": {"x": {"pass": False}}}},
    }
    got = audit.audit(metrics)
    assert set(got["studies"]) == {"a"}
    assert got["unrecognised"] == ["d", "e__rule"]


def test_leaves_out_the_rhythm_audit_which_rereads_counted_studies() -> None:
    metrics = {
        "a": {"predictions": {"x": {"outcome": "pass"}}},
        "rhythm_audit": {"stage1d": {"predictions": {"x": {"audited": {"outcome": "fail"}}}}},
    }
    got = audit.audit(metrics)
    assert set(got["studies"]) == {"a"} and got["unrecognised"] == []
    assert set(got["excluded"]) == {"rhythm_audit"}


def test_main_skips_a_run_that_wrote_no_metrics(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(audit, "outputs_dir", lambda: tmp_path)
    assert audit.main([]) == 0
    assert not (tmp_path / "metrics.json").exists()
