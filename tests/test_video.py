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


def test_the_plan_sections_are_all_covered() -> None:
    sections = {shot.section for shot in SHOTS}
    assert {
        "Problem",
        "Data asset & standard",
        "Transfer",
        "Twin & inference",
        "Reproducibility",
    } <= sections


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
