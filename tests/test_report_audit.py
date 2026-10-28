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


def test_the_submission_leaves_out_the_manuscript_only_registrations() -> None:
    metrics = {
        "a": {"registration": {"file": "preregistration/a.json"}, "predictions": PASS_FAIL},
        "b": {"registration": {"file": "preregistration/b.json"}, "predictions": PASS_FAIL},
        "c": {"x": {"registration": {"file": "c.json"}, "predictions": PASS_FAIL}},
        "d": {"predictions": PASS_FAIL},  # names no registration: counted in the submission
    }
    got = audit.audit(metrics, manuscript_only={"b.json", "c.json"})
    assert got["n_studies"] == 4 and got["total"]["total"] == 8
    assert got["submission"]["n_studies"] == 2
    assert got["submission"]["total"] == {
        "pass": 2,
        "fail": 2,
        "uninterpretable": 0,
        "other": 0,
        "total": 4,
    }
    assert audit.audit(metrics)["submission"]["n_studies"] == 4  # nothing listed: all of it


def test_reads_the_manuscript_only_list_and_its_absence(tmp_path) -> None:
    assert audit.manuscript_only(tmp_path) == set()
    (tmp_path / audit.MANUSCRIPT_ONLY).write_text("# private\nb.json\n\nc.json\n")
    assert audit.manuscript_only(tmp_path) == {"b.json", "c.json"}


PASS_FAIL = {"x": {"outcome": "pass"}, "y": {"outcome": "fail"}}


def test_reads_predictions_nested_under_by_prediction() -> None:
    nested = {"by_prediction": PASS_FAIL, "n_passed": 1, "n_registered": 2}
    got = audit.audit({"a": {"predictions": nested}})
    assert got["studies"]["a"]["total"] == 2 and got["unrecognised"] == []
