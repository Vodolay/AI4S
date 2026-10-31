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
RECALIBRATION = METRICS["twin_hd14b_recalibration"]


def _inference_text(metrics: dict, path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every string drawn on the inversion diagram, captured as the figure is saved."""
    drawn: list[str] = []
    save = diagrams._save

    def capture(fig, target):
        drawn.extend(text.get_text() for ax in fig.axes for text in ax.texts)
        return save(fig, target)

    monkeypatch.setattr(diagrams, "_save", capture)
    diagrams.draw_inference(metrics, path)
    return drawn


@pytest.mark.parametrize("name,entry", sorted(diagrams.FIGURES.items()))
def test_each_diagram_is_drawn(tmp_path: Path, name: str, entry: tuple) -> None:
    """Every entry in the registry produces a file with something in it."""
    written = getattr(diagrams, entry[0])(METRICS, tmp_path / name)
    assert written.exists()
    assert written.stat().st_size > 10_000, "a near-empty PNG means the draw failed quietly"


@pytest.mark.parametrize("key", ["twin_hd14_inference", "twin_hd14b_recalibration"])
def test_a_missing_metric_stops_the_build(tmp_path: Path, key: str) -> None:
    """A diagram built against metrics that lack a key must fail, not draw a default."""
    stripped = {name: value for name, value in METRICS.items() if name != key}
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


def test_the_inference_panel_marks_only_the_adopted_parameters_as_calibrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D31's adoption rule decides which interval is quoted, not round 1's 300-run check.

    The panel used to label three parameters calibrated from the round-1 holdout;
    on 1,000 fresh simulations that model fails all five, and the registered rule
    quotes only what it adopts from the recalibrated ensemble.
    """
    adopted = RECALIBRATION["adopted_parameters"]
    assert 0 < len(adopted) < 5, "this test is meaningless if every parameter, or none, is adopted"
    drawn = _inference_text(METRICS, tmp_path / "a.png", monkeypatch)
    assert drawn.count(diagrams.QUOTED) == len(adopted)
    assert drawn.count(diagrams.NOT_QUOTED) == 5 - len(adopted)
    assert "over-confident" not in drawn and "calibrated" not in drawn

    edited = json.loads(json.dumps(METRICS))
    edited["twin_hd14b_recalibration"]["adopted_parameters"] = sorted(
        edited["twin_hd14b_recalibration"]["results"]["M2"]["H1000"]["ks"]
    )
    drawn = _inference_text(edited, tmp_path / "b.png", monkeypatch)
    assert drawn.count(diagrams.QUOTED) == 5, "the verdicts are not read from adopted_parameters"


def test_the_inference_caption_reports_the_recalibration_not_round_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The footer quotes D31's fresh-simulation result and the adopted parameter only."""
    fresh = RECALIBRATION["results"]["M0"]["H1000"]
    failing = [
        name for name, value in fresh["ks"].items() if value["ks_p"] < RECALIBRATION["alpha"]
    ]
    caption = diagrams.inference_caption(METRICS)
    assert f"On {fresh['n']:,} fresh simulations" in caption
    assert f"{len(failing)} of {len(fresh['ks'])} parameters" in caption
    assert {name for name in fresh["ks"] if name in caption} == set(
        RECALIBRATION["adopted_parameters"]
    )
    assert "pass rank uniformity" not in caption and "The two that fail" not in caption
    assert caption in _inference_text(METRICS, tmp_path / "inference.png", monkeypatch)


def test_the_inference_caption_follows_the_metrics() -> None:
    """Change the cited values and the sentence must change, or it was typed by hand."""
    edited = json.loads(json.dumps(METRICS))
    recalibration = edited["twin_hd14b_recalibration"]
    recalibration["results"]["M0"]["H1000"]["n"] = 1234
    recalibration["adopted_parameters"] = ["ampa_scale", "gabaa_scale"]
    caption = diagrams.inference_caption(edited)
    assert "On 1,234 fresh simulations" in caption
    assert "ampa_scale" in caption and "gabaa_scale" in caption
    assert "threshold_scale" not in caption
    recalibration["adopted_parameters"] = []
    assert "no interval is quoted" in diagrams.inference_caption(edited)


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
    del broken["twin_hd14b_recalibration"]["adopted_parameters"]
    with pytest.raises(KeyError):
        diagrams.draw_inference(broken, tmp_path / "inference.png")


def test_every_diagram_declares_whether_it_shows_real_or_simulated_data() -> None:
    """The project's figure rule, applied to the filenames the app and video load."""
    for name in diagrams.FIGURES:
        assert "_real" in name or "_simulated" in name, name
