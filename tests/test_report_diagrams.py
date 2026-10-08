"""The model diagrams must be drawn from the metrics, not captioned by hand.

They appear in the app, the video and the report, so a number typed into one of
them would be the one figure in the project that could disagree with the result
beside it and have nothing notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuromps.report import diagrams

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((REPO_ROOT / "outputs" / "metrics.json").read_text())


@pytest.mark.parametrize("name,entry", sorted(diagrams.FIGURES.items()))
def test_each_diagram_is_drawn(tmp_path: Path, name: str, entry: tuple) -> None:
    """Every entry in the registry produces a file with something in it."""
    written = getattr(diagrams, entry[0])(METRICS, tmp_path / name)
    assert written.exists()
    assert written.stat().st_size > 10_000, "a near-empty PNG means the draw failed quietly"


def test_a_missing_metric_stops_the_build(tmp_path: Path) -> None:
    """A diagram built against metrics that lack a key must fail, not draw a default."""
    stripped = {key: value for key, value in METRICS.items() if key != "twin_hd14_inference"}
    with pytest.raises(KeyError):
        diagrams.draw_inference(stripped, tmp_path / "inference.png")


def test_the_numbers_on_the_diagram_follow_the_metrics(tmp_path: Path) -> None:
    """Change a cited value and the picture must change, or it was a literal.

    This is the check that a caption is actually wired to its key. Comparing the
    rendered bytes is blunt, but it cannot be satisfied by a diagram that quotes
    a hard-coded number.
    """
    before = diagrams.draw_twin(METRICS, tmp_path / "before.png").read_bytes()
    edited = json.loads(json.dumps(METRICS))
    edited["twin_hd12_calibration"]["best"]["draw"]["n_per_compartment"] = 1234
    after = diagrams.draw_twin(edited, tmp_path / "after.png").read_bytes()
    assert before != after, "the neuron count is drawn from a literal, not from metrics.json"


def test_the_inference_diagram_names_the_parameters_that_fail(tmp_path: Path) -> None:
    """The failing parameters are the point of the panel; they must not be hard-coded."""
    ks_p = METRICS["twin_hd14_inference"]["rank_uniformity_ks_p"]
    failing = {name for name, value in ks_p.items() if value["ks_p"] <= 0.05}
    assert failing, "this test is meaningless if every parameter passes"
    edited = json.loads(json.dumps(METRICS))
    for value in edited["twin_hd14_inference"]["rank_uniformity_ks_p"].values():
        value["ks_p"] = 0.9  # everything passes
    before = diagrams.draw_inference(METRICS, tmp_path / "a.png").read_bytes()
    after = diagrams.draw_inference(edited, tmp_path / "b.png").read_bytes()
    assert before != after, "the pass/fail verdicts are not read from the KS p-values"


def test_the_registry_and_the_drawing_functions_agree() -> None:
    """A figure named in the registry with no function would fail only at build time."""
    for name, (function, required) in diagrams.FIGURES.items():
        assert name.endswith(".png")
        assert callable(getattr(diagrams, function, None)), function
        assert required, f"{name} declares no metrics requirement"
        assert all(key in METRICS for key in required), name


def test_a_tier_without_the_numbers_skips_rather_than_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The quick tier does not calibrate, so it has no `twin_hd12_calibration`.

    The twin stage already falls back to the full run's calibration file for
    everything else; the diagrams must not be the one step that takes the stage
    down. This failed on CI and not locally, because a local quick metrics.json
    still carried the key from an older run that had the full data.
    """
    from neuromps import paths
    from neuromps.report import diagrams as module

    reduced = {key: value for key, value in METRICS.items() if not key.startswith("twin_")}
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps(reduced))
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path)
    with caplog.at_level("WARNING"):
        assert module.main([]) == 0
    skipped = [
        record.getMessage() for record in caplog.records if "skipping" in record.getMessage()
    ]
    assert len(skipped) == 2, skipped
    assert any("twin_hd12_calibration" in message for message in skipped)
    assert (tmp_path / "figures" / "mechanism_classifier_real.png").exists(), (
        "the classifier diagram needs no twin metrics and should still be drawn"
    )


def test_a_malformed_metric_still_raises(tmp_path: Path) -> None:
    """Skipping an absent tier must not also swallow a key that is present but wrong."""
    broken = json.loads(json.dumps(METRICS))
    del broken["twin_hd14_inference"]["rank_uniformity_ks_p"]
    with pytest.raises(KeyError):
        diagrams.draw_inference(broken, tmp_path / "inference.png")


def test_every_diagram_declares_whether_it_shows_real_or_simulated_data() -> None:
    """The project's figure rule, applied to the filenames the app and video load."""
    for name in diagrams.FIGURES:
        assert "_real" in name or "_simulated" in name, name
