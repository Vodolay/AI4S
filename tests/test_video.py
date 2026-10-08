"""Tests for the demo video build.

The video quotes results, so it is held to the same rule as the report: its
numbers come from `outputs/metrics.json` at build time, and it cannot be built
against metrics that lack them.
"""

import json
import shutil
from pathlib import Path

import pytest

from neuromps.video.shots import Shot, metric, shot_list

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((REPO_ROOT / "outputs" / "metrics.json").read_text())
SHOTS = shot_list(METRICS)
LIMIT_SECONDS = 300


def test_the_video_fits_the_five_minute_limit() -> None:
    total = sum(shot.seconds for shot in SHOTS)
    assert 200 <= total <= LIMIT_SECONDS, f"{total} s"


def test_every_still_exists_and_crops_inside_its_image() -> None:
    from PIL import Image

    for shot in SHOTS:
        if not shot.image:
            continue
        path = REPO_ROOT / shot.image
        assert path.exists(), shot.image
        if shot.crop:
            width, height = Image.open(path).size
            left, top, right, bottom = shot.crop
            assert 0 <= left < right <= width, shot.image
            assert 0 <= top < bottom <= height, shot.image


def test_each_shot_says_something() -> None:
    for shot in SHOTS:
        assert shot.caption or shot.body, shot
        assert shot.seconds >= 5


def test_the_running_order_covers_every_act() -> None:
    """The sections, and the order they appear in.

    PLAN.md budgeted this as problem, data, transfer, twin, reproducibility. The
    running order was deliberately rebalanced: the models lead, the data asset
    follows rather than opening, and the time cut from the asset went into the
    three model diagrams. Changing it again should be a decision, not a drift.
    """
    order = list(dict.fromkeys(shot.section for shot in SHOTS if shot.section))
    assert order == [
        "Problem",
        "Mechanism",
        "Twin",
        "Inversion",
        "Data asset",
        "Reproducibility",
    ]


def test_the_models_get_more_time_than_the_data_asset() -> None:
    """The rebalance this running order exists for, held in place by a number."""
    spend = {}
    for shot in SHOTS:
        spend[shot.section] = spend.get(shot.section, 0) + shot.seconds
    models = spend["Mechanism"] + spend["Twin"] + spend["Inversion"]
    assert models >= 3 * spend["Data asset"], (models, spend["Data asset"])


def test_each_model_diagram_gets_a_shot_of_its_own() -> None:
    """The three diagrams are the point of this cut; a dropped one should fail here."""
    from neuromps.video import shots as shot_module

    wanted = {
        shot_module.CLASSIFIER_DIAGRAM,
        shot_module.TWIN_DIAGRAM,
        shot_module.INVERSION_DIAGRAM,
    }
    shown = {shot.image for shot in SHOTS if shot.image in wanted}
    assert shown == wanted, sorted(wanted - shown)


def test_the_diagrams_are_used_as_files_not_cropped_out_of_a_screenshot() -> None:
    """Screenshotting a diagram the app rendered meant racing the app's slowest panel.

    The video kept catching a half-painted page. The diagrams are generated PNGs
    already, so the video uses the original and the race disappears.
    """
    from neuromps.video import shots as shot_module

    for name in ("CLASSIFIER_DIAGRAM", "TWIN_DIAGRAM", "INVERSION_DIAGRAM"):
        path = getattr(shot_module, name)
        assert path.startswith("outputs/figures/"), path
        matching = [shot for shot in SHOTS if shot.image == path]
        assert matching and all(shot.crop is None for shot in matching), path


def test_the_numbers_come_from_metrics_not_from_the_script() -> None:
    """Building against metrics without a cited key must fail, not invent a value."""
    with pytest.raises((KeyError, TypeError)):
        shot_list({"data_asset": {"n_records": 1, "counts": {"by_data_level": {}}}})


def test_quoted_values_match_the_live_metrics() -> None:
    """Spot-check that captions carry the metrics' values, not stale copies."""
    baseline = metric(METRICS, "twin_hd15_real_inference.gabaa_scale_at_baseline.median")
    records = f"{metric(METRICS, 'data_asset.n_records'):,}"
    text = " ".join(shot.caption + " ".join(shot.body) for shot in SHOTS)
    assert str(baseline) in text
    assert records in text


def test_the_research_use_framing_is_on_screen() -> None:
    text = " ".join(" ".join(shot.body) for shot in SHOTS)
    assert "Research decision-support only." in text


def test_a_card_renders_without_a_display(tmp_path) -> None:
    from neuromps.video.build import render_card

    image = render_card(Shot(5, "", title="T", body=["a", "b"], section="Problem"))
    assert image.size == (1920, 1080)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_the_encoder_is_wired_to_ffmpeg() -> None:
    source = (REPO_ROOT / "src" / "neuromps" / "video" / "build.py").read_text()
    assert '"ffmpeg"' in source
    assert "libx264" in source


def test_the_live_simulation_gets_a_frame() -> None:
    """The claim that the twin is a simulator needs the frame that shows it running.

    This still is the one a screenshot tool cannot take by clicking. It is
    reachable because `?live=1` runs the simulation on load and the result is
    cached on its parameter tuple, so the capture script pays for Brian2 on a
    warm-up load and photographs the cached render.
    """
    live = [shot for shot in SHOTS if shot.image == "outputs/video/stills/twin_live.png"]
    assert len(live) == 1, "the live Brian2 frame is missing"
    assert live[0].section == "Twin"
    assert "wall clock" in live[0].caption
