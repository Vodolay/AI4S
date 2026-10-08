"""A run records its own cost; publishing it to the cited root is a separate step.

`run_all.py --quick` writes only under `outputs/quick/`, so that running it
cannot alter the numbers the report cites. CI enforces that with a `git diff
--exit-code` after the smoke run. An earlier version of this recorded straight
into the full root and was caught by exactly that check, so these tests hold the
boundary from this side too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuromps import harness, paths

STAGES = ["fetch", "data", "models", "twin", "demo"]


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A full metrics.json holding a result, and a quick root beside it."""
    outputs = tmp_path / "outputs"
    (outputs / "quick").mkdir(parents=True)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", outputs)
    monkeypatch.setattr(paths, "PROCESSED_ROOT", tmp_path / "processed")
    paths.set_run_mode(paths.QUICK)
    full = outputs / "metrics.json"
    full.write_text(json.dumps({"mechanism_baselines": {"balanced_accuracy": 0.537}}))
    yield full, outputs / "quick" / "metrics.json"
    paths.set_run_mode(paths.FULL)


def test_a_quick_run_writes_only_into_the_quick_root(roots) -> None:
    """The invariant CI checks with git diff, asserted here as well."""
    full, quick = roots
    before = full.read_text()
    harness.record({"models": 64.9}, 412.2, quick=True, stages=STAGES, all_stages=STAGES)
    assert full.read_text() == before, "a quick run reached the metrics the report cites"
    assert json.loads(quick.read_text())["harness_quick"]["total_s"] == 412.2


def test_the_record_carries_the_machine_and_the_versions(roots) -> None:
    """A wall clock without a core count is not a reproducibility claim."""
    _, quick = roots
    harness.record({"models": 64.9}, 412.2, quick=True, stages=STAGES, all_stages=STAGES)
    recorded = json.loads(quick.read_text())["harness_quick"]
    assert recorded["cpu_count"] >= 1
    assert recorded["numpy"] and recorded["python"]
    assert recorded["total_is_complete_pass"] is True


def test_a_partial_run_does_not_shrink_the_end_to_end_timing(roots) -> None:
    """`--stage models` must not leave the appendix claiming the pipeline takes a minute."""
    _, quick = roots
    harness.record({"models": 64.9}, 412.2, quick=True, stages=STAGES, all_stages=STAGES)
    harness.record({"models": 61.0}, 61.0, quick=True, stages=["models"], all_stages=STAGES)
    recorded = json.loads(quick.read_text())["harness_quick"]
    assert recorded["total_s"] == 412.2, "a one-stage run replaced the end-to-end figure"
    assert recorded["by_stage_s"]["models"] == 61.0, "the stage's own time should update"


def test_publishing_copies_the_record_and_touches_nothing_else(roots) -> None:
    """The one write that crosses the boundary, and it is asked for explicitly."""
    full, quick = roots
    harness.record({"models": 64.9}, 412.2, quick=True, stages=STAGES, all_stages=STAGES)
    harness.publish(quick=True)
    published = json.loads(full.read_text())
    assert published["harness_quick"]["total_s"] == 412.2
    assert published["mechanism_baselines"] == {"balanced_accuracy": 0.537}
    assert set(published) == {"mechanism_baselines", "harness_quick"}


def test_publishing_a_partial_run_is_refused(roots) -> None:
    """The report quotes this as an end-to-end time, so a part of one must not pass."""
    _, _ = roots
    harness.record({"models": 61.0}, 61.0, quick=True, stages=["models"], all_stages=STAGES)
    with pytest.raises(ValueError, match="partial run"):
        harness.publish(quick=True)


def test_publishing_a_tier_that_never_ran_says_so(roots) -> None:
    """Better than publishing nothing and leaving the old number in place."""
    with pytest.raises(KeyError, match="harness_full"):
        harness.publish(quick=False)


def test_the_tier_is_named_in_the_key() -> None:
    """A full run and a quick run must not overwrite each other's wall clock."""
    assert harness.key_for(quick=True) == "harness_quick"
    assert harness.key_for(quick=False) == "harness_full"


def test_the_pinned_versions_are_read_from_what_actually_imported() -> None:
    """Reading pyproject would let the appendix describe an environment that never ran."""
    import numpy

    assert harness.pinned_versions()["numpy"] == numpy.__version__


def test_run_all_delegates_rather_than_keeping_its_own_copy() -> None:
    """These functions moved into the package so a test could import them.

    `import run_all` only worked because `python -m pytest` puts the working
    directory on the path; CI calls `pytest` directly, where it does not.
    """
    source = (Path(__file__).resolve().parents[1] / "run_all.py").read_text()
    assert "harness.record(" in source
    assert "def _pinned_versions" not in source, "the copy in run_all.py came back"
