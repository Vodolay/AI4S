"""The harness record is the one thing a quick run writes to the full metrics root.

`run_all.py --quick` is isolated from the full run's results on purpose, so that
running it cannot rewrite the numbers the report cites. The reproducibility
appendix needs one number that isolation puts out of reach: how long `--quick`
itself takes. The exception is narrow, and these tests are what keeps it narrow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuromps import harness, paths

STAGES = ["fetch", "data", "models", "twin", "demo"]


@pytest.fixture
def canonical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A full-run metrics.json holding a result, with the run set to quick."""
    outputs = tmp_path / "outputs"
    (outputs / "quick").mkdir(parents=True)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", outputs)
    monkeypatch.setattr(paths, "PROCESSED_ROOT", tmp_path / "processed")
    paths.set_run_mode(paths.QUICK)
    metrics = outputs / "metrics.json"
    metrics.write_text(json.dumps({"mechanism_baselines": {"balanced_accuracy": 0.537}}))
    yield metrics
    paths.set_run_mode(paths.FULL)


def test_a_quick_run_records_its_wall_clock_where_the_report_can_cite_it(
    canonical: Path,
) -> None:
    """The timing lands in the full root, under a key that names its tier."""
    harness.record({"models": 64.9}, 412.2, quick=True, stages=STAGES, all_stages=STAGES)
    recorded = json.loads(canonical.read_text())["harness_quick"]
    assert recorded["total_s"] == 412.2
    assert recorded["total_is_complete_pass"] is True
    assert recorded["cpu_count"] >= 1
    assert recorded["numpy"] and recorded["python"]


def test_a_quick_run_touches_no_other_key_in_the_full_metrics(canonical: Path) -> None:
    """The exception is one key. Any result it reached would be a reduced-tier number."""
    harness.record({"models": 64.9}, 412.2, quick=True, stages=STAGES, all_stages=STAGES)
    recorded = json.loads(canonical.read_text())
    assert recorded["mechanism_baselines"] == {"balanced_accuracy": 0.537}
    assert set(recorded) == {"mechanism_baselines", "harness_quick"}


def test_a_partial_run_does_not_shrink_the_end_to_end_timing(canonical: Path) -> None:
    """`--stage models` must not leave the appendix claiming the pipeline takes a minute."""
    harness.record({"models": 64.9}, 412.2, quick=True, stages=STAGES, all_stages=STAGES)
    harness.record({"models": 61.0}, 61.0, quick=True, stages=["models"], all_stages=STAGES)
    recorded = json.loads(canonical.read_text())["harness_quick"]
    assert recorded["total_s"] == 412.2, "a one-stage run replaced the end-to-end figure"
    assert recorded["by_stage_s"]["models"] == 61.0, "the stage's own time should update"
    assert recorded["total_is_complete_pass"] is True


def test_the_pinned_versions_are_read_from_what_actually_imported() -> None:
    """Reading pyproject would let the appendix describe an environment that never ran."""
    import numpy

    assert harness.pinned_versions()["numpy"] == numpy.__version__


def test_the_tier_is_named_in_the_key() -> None:
    """A full run and a quick run must not overwrite each other's wall clock."""
    assert harness.key_for(quick=True) != harness.key_for(quick=False)
    quick_key, _ = harness.harness_record({}, 1.0, True, STAGES, STAGES)
    full_key, _ = harness.harness_record({}, 1.0, False, STAGES, STAGES)
    assert quick_key == "harness_quick" and full_key == "harness_full"


def test_run_all_delegates_rather_than_keeping_its_own_copy() -> None:
    """These functions moved into the package so they could be imported by a test.

    `import run_all` only worked because `python -m pytest` puts the working
    directory on the path; CI calls `pytest` directly, where it does not, and the
    module failed to import there.
    """
    source = (Path(__file__).resolve().parents[1] / "run_all.py").read_text()
    assert "harness.record(" in source
    assert "def _pinned_versions" not in source, "the copy in run_all.py came back"
