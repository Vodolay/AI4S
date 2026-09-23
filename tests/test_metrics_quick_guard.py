"""The guard against a `--quick` stage overwriting the numbers the report cites."""

import json

import pytest

from neuromps.metrics import update_metrics, warn_if_quick_run_writes_to_the_full_root


@pytest.fixture
def _no_run_mode(monkeypatch):
    monkeypatch.delenv("NEUROMPS_RUN_MODE", raising=False)


def test_it_warns_when_quick_is_passed_but_the_root_is_the_full_one(_no_run_mode, monkeypatch):
    monkeypatch.setattr("sys.argv", ["neuromps.models.x", "--quick"])
    assert warn_if_quick_run_writes_to_the_full_root() is True


def test_it_stays_quiet_when_the_run_mode_matches_the_flag(monkeypatch):
    monkeypatch.setenv("NEUROMPS_RUN_MODE", "quick")
    monkeypatch.setattr("sys.argv", ["neuromps.models.x", "--quick"])
    assert warn_if_quick_run_writes_to_the_full_root() is False


def test_it_stays_quiet_for_an_ordinary_full_run(_no_run_mode, monkeypatch):
    monkeypatch.setattr("sys.argv", ["neuromps.models.x"])
    assert warn_if_quick_run_writes_to_the_full_root() is False


def test_the_warning_names_the_way_out(_no_run_mode, monkeypatch, caplog):
    monkeypatch.setattr("sys.argv", ["neuromps.models.x", "--quick"])
    with caplog.at_level("WARNING"):
        warn_if_quick_run_writes_to_the_full_root()
    assert "NEUROMPS_RUN_MODE=quick" in caplog.text


def test_it_warns_but_still_writes(_no_run_mode, monkeypatch, tmp_path, caplog):
    """The guard reports; it must never silently discard a stage's output."""
    monkeypatch.setattr("sys.argv", ["neuromps.models.x", "--quick"])
    monkeypatch.setattr("neuromps.metrics.metrics_path", lambda: tmp_path / "metrics.json")
    with caplog.at_level("WARNING"):
        update_metrics("section", {"value": 1})
    assert json.loads((tmp_path / "metrics.json").read_text())["section"]["value"] == 1
    assert "FULL metrics root" in caplog.text


def test_an_explicit_path_is_the_callers_business(_no_run_mode, monkeypatch, tmp_path, caplog):
    monkeypatch.setattr("sys.argv", ["neuromps.models.x", "--quick"])
    with caplog.at_level("WARNING"):
        update_metrics("section", {"value": 2}, path=tmp_path / "elsewhere.json")
    assert "FULL metrics root" not in caplog.text
