"""Smoke tests for the Streamlit demo.

These run the app the way Streamlit does, assert that no tab raises, that the
widgets each tab is built around exist, and that an interaction stays inside the
responsiveness budget the app promises. They need the precomputed demo inputs,
so they skip when `outputs/demo/` is absent rather than failing a fresh clone.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "app" / "app.py"
DEMO_READY = (REPO_ROOT / "outputs" / "demo" / "meta.json").exists()
# Either run's asset: the app falls back to the quick one, so after a fresh
# clone's `run_all.py --quick` the table renders and the "missing" note does not.
ASSET_BUILT = any(
    (REPO_ROOT / "data" / "processed" / sub / "neuromps.parquet").exists() for sub in ("", "quick")
)
INTERACTION_BUDGET_S = 1.0
# What this budget is for: catching a callback that has started doing real work.
# A live Brian2 run in the app takes about four seconds, so anything that slow
# is unambiguous. The one-second figure is the promise the app makes on a
# developer's machine, and a shared CI runner cannot be held to it - it has no
# timing guarantee and measured 1.26 s on the very code that took well under a
# second here. Allow it room on CI rather than leaving a coin-flip in the suite,
# and keep the strict number where it can actually be measured.
CI_BUDGET_MULTIPLIER = 3.0
_on_ci = bool(os.environ.get("CI"))
interaction_budget_s = INTERACTION_BUDGET_S * (CI_BUDGET_MULTIPLIER if _on_ci else 1.0)

needs_demo = pytest.mark.skipif(not DEMO_READY, reason="run `python run_all.py --stage demo` first")


@pytest.fixture(scope="module")
def app() -> AppTest:
    """The app, run once."""
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def test_the_app_runs_without_raising(app: AppTest) -> None:
    assert app.exception == [], [str(e.value) for e in app.exception]


def test_the_tabs_are_in_the_order_the_app_declares(app: AppTest) -> None:
    """The models lead and the asset follows; the order is the app's, not a guess."""
    from app.app import TABS

    labels = [tab.label for tab in app.tabs]
    assert labels == [label for _, label in TABS]
    assert "Mechanism" in labels[0]
    assert "twin" in labels[1].lower()
    assert "Data asset" in labels[2]
    assert "chip to animal" in labels[3]


@pytest.mark.parametrize("slug,expected", [("models", 0), ("twin", 1), ("data", 2), ("fly", 3)])
def test_a_tab_can_be_deep_linked_by_slug(slug: str, expected: int) -> None:
    """The video screenshots one tab at a time, and the numbers moved when the order did."""
    from app.app import TABS

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.query_params["tab"] = slug
    at.run()
    assert at.exception == [], [str(e.value) for e in at.exception]
    assert at.tabs == [], "a deep link should render one tab, not the tab strip"
    assert TABS[expected][1] in [s.value for s in at.subheader]


def test_an_unknown_tab_value_falls_back_to_the_full_strip() -> None:
    """A stale link must not render a blank page."""
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.query_params["tab"] = "provenance"
    at.run()
    assert at.exception == []
    from app.app import TABS

    assert len(at.tabs) == len(TABS)


def test_the_research_use_banner_is_shown(app: AppTest) -> None:
    """The project's framing rule, checked rather than trusted."""
    warnings = " ".join(w.value for w in app.warning)
    assert "Research decision-support only" in warnings
    assert "clinical claim" in warnings


def test_tab_one_shows_the_dataset_card_without_the_built_asset(app: AppTest) -> None:
    """The card reads metrics.json, which is committed; only the table needs the asset.

    A clone that has not run the pipeline - which is what CI checks out - must
    still show the dataset card, and say why the table is missing rather than
    rendering an empty tab.
    """
    labels = [m.label for m in app.metric]
    assert "Records" in labels and "Sources" in labels


def test_the_wells_needed_panel_reads_metrics_and_follows_the_selection() -> None:
    """The panel shows the three designs from metrics.json and updates on a new change size."""
    import json

    from neuromps.paths import OUTPUTS_ROOT

    metrics = json.loads((OUTPUTS_ROOT / "metrics.json").read_text())
    floors = metrics.get("toxcast_noise_floors")
    if not floors:
        pytest.skip("toxcast_noise_floors not in metrics.json")
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.query_params["tab"] = "data"
    at.run()
    labels = [m.label for m in at.metric]
    assert "Wells across days" in labels and "One day, same-day vehicles" in labels
    needed = floors["components"][floors["headline_component"]]["wells_needed"]
    across = next(m for m in at.metric if m.label == "Wells across days")
    assert across.value == str(needed["independent"]["down_20pct"])
    next(s for s in at.segmented_control if s.label == "Change").set_value("50% fall").run()
    across = next(m for m in at.metric if m.label == "Wells across days")
    assert across.value == str(needed["independent"]["down_50pct"])


@pytest.mark.skipif(ASSET_BUILT, reason="the asset is present, so the table renders")
def test_tab_one_explains_a_missing_asset(app: AppTest) -> None:
    messages = " ".join(i.value for i in app.info)
    assert "record table is unavailable" in messages


@pytest.mark.skipif(not ASSET_BUILT, reason="needs the built asset")
def test_tab_one_shows_the_record_filters_when_the_asset_is_built(app: AppTest) -> None:
    assert any(s.label == "Source" for s in app.selectbox)
    assert any(s.label == "Data level" for s in app.selectbox)


@needs_demo
def test_tab_two_offers_a_prediction_and_a_conformal_set(app: AppTest) -> None:
    assert any(m.label == "Predicted mechanism" for m in app.metric)
    text = " ".join(m.value for m in app.markdown)
    assert "Conformal prediction set" in text


@needs_demo
def test_tab_three_has_the_three_twin_sliders_snapped_to_the_grid(app: AppTest) -> None:
    labels = [s.label for s in app.select_slider]
    assert "forward / backward ratio" in labels
    assert "GABA-A scale" in labels
    assert "AMPA scale" in labels


@needs_demo
def test_tab_three_offers_a_live_simulation_button(app: AppTest) -> None:
    assert any("live simulation" in b.label for b in app.button)


def _metrics() -> dict:
    import json

    from neuromps.paths import OUTPUTS_ROOT

    return json.loads((OUTPUTS_ROOT / "metrics.json").read_text())


def _calibration_claims(text: str, names) -> set[str]:
    """Parameters a caption names in a sentence about calibration or quoting."""
    sentences = [s for s in text.split(". ") if "calibrat" in s or "quoted" in s]
    return {name for name in names if any(f"`{name}`" in s for s in sentences)}


@needs_demo
def test_the_inversion_caption_reports_the_recalibration(app: AppTest) -> None:
    """D31 superseded the round-1 300-run check; the caption must say what it found."""
    recalibration = _metrics()["twin_hd14b_recalibration"]
    fresh = recalibration["results"]["M0"]["H1000"]
    budget = recalibration["budget"]
    failing = [n for n, v in fresh["ks"].items() if v["ks_p"] < recalibration["alpha"]]
    caption = next(c.value for c in app.caption if "run backwards" in c.value)
    assert f"On {fresh['n']:,} fresh simulations" in caption
    assert f"{len(failing)} of {len(fresh['ks'])} parameters" in caption
    assert f"{budget['pool_simulations']:,}" in caption and f"{budget['hd14_train']:,}" in caption
    assert _calibration_claims(caption, fresh["ks"]) == set(recalibration["adopted_parameters"])
    assert "Two of the five" not in caption


def test_the_posterior_note_calls_only_the_adopted_parameter_calibrated() -> None:
    """The panel samples the round-1 model; only D31's adopted interval is quoted."""
    from app.tab_twin import _identifiability_note

    metrics = _metrics()
    recalibration = metrics["twin_hd14b_recalibration"]
    names = recalibration["results"]["M0"]["H1000"]["ks"]
    note = _identifiability_note(metrics)
    assert "pass the rank-uniformity check" not in note
    assert "round-1" in note
    assert _calibration_claims(note, names) == set(recalibration["adopted_parameters"])
    recalibration["adopted_parameters"] = ["ampa_scale"]
    assert _calibration_claims(_identifiability_note(metrics), names) == {"ampa_scale"}


@needs_demo
def test_the_posterior_panel_shows_the_note(app: AppTest) -> None:
    from app.tab_twin import _identifiability_note

    assert _identifiability_note(_metrics()) in [c.value for c in app.caption]


@needs_demo
def test_changing_a_twin_slider_responds_within_the_budget() -> None:
    """The app's promise: an interaction never waits on a simulator."""
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    # What a rerun with no input costs on this machine. A slider rerenders the
    # page, so it can never beat this; on a busy or slower machine a fixed one
    # second failed with the slider costing exactly a plain rerun. A simulator
    # call costs seconds on top, so twice the plain rerun still catches it.
    started = time.time()
    at.run()
    plain = time.time() - started
    budget = max(interaction_budget_s, 2 * plain)
    slider = next(s for s in at.select_slider if s.label == "GABA-A scale")
    options = list(slider.options)
    started = time.time()
    slider.set_value(options[-1]).run()
    elapsed = time.time() - started
    assert at.exception == [], [str(e.value) for e in at.exception]
    assert elapsed < budget, (
        f"slider took {elapsed:.2f}s, over the {budget:.1f}s budget "
        f"(a plain rerun took {plain:.2f}s)"
    )


@needs_demo
def test_switching_to_manual_features_exposes_one_slider_per_feature() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    radio = next(r for r in at.radio if r.label == "Input")
    radio.set_value("Dial a feature vector").run()
    assert at.exception == []
    assert len(at.slider) == 9, [s.label for s in at.slider]


def test_the_live_simulation_is_not_wired_into_page_load() -> None:
    """Brian2 must never run as a side effect of opening the app.

    `?live=1` exists so the demo video's capture script can photograph the panel,
    and it is inert unless the server was started with the opt-in set. Opening
    the app, or following that link to a deployed one, must still cost nothing.
    """
    source = (REPO_ROOT / "app" / "tab_twin.py").read_text()
    call = source.index("loaders.live_simulation(")
    guard = source.index('clicked = st.button("Run one live simulation')
    assert guard < call, "the live run must sit behind the button"
    assert 'os.environ.get(LIVE_ON_LOAD_ENV, "") != "1"' in source, (
        "the URL alone must not be able to start a simulation"
    )


def test_the_url_alone_does_not_start_a_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the server-side opt-in, `?live=1` is ignored."""
    from app.tab_twin import LIVE_ON_LOAD_ENV

    monkeypatch.delenv(LIVE_ON_LOAD_ENV, raising=False)
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.query_params["tab"] = "twin"
    at.query_params["live"] = "1"
    at.run()
    assert at.exception == []
    reported = " ".join(element.value for element in at.success)
    assert "wall clock" not in reported, "a link started a Brian2 run"


@needs_demo
def test_the_live_simulation_actually_runs_brian2_and_returns_a_raster() -> None:
    """The button is the claim that the twin is a simulator, not a lookup table."""
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    button = next(b for b in at.button if "live simulation" in b.label)
    started = time.time()
    button.click().run()
    elapsed = time.time() - started
    assert at.exception == [], [str(e.value) for e in at.exception]
    messages = " ".join(s.value for s in at.success)
    assert "of wall clock" in messages, messages
    assert 1.0 < elapsed < 120.0, f"a live Brian2 run took {elapsed:.1f}s"


def test_brian2_runs_in_its_own_interpreter() -> None:
    """Importing Brian2 in Streamlit's thread raises; it must stay in a subprocess."""
    source = (REPO_ROOT / "app" / "loaders.py").read_text()
    assert "subprocess.run" in source
    assert "neuromps.demo.simulate_one" in source


@needs_demo
def test_the_live_simulation_can_be_asked_for_by_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the opt-in set, `?live=1` runs it without a click.

    That is how the video gets its frame. Without it the still has to be captured
    by hand, and a hand-captured still is one that stays on disk through the next
    correction - which is exactly what happened to the first set.
    """
    from app.tab_twin import LIVE_ON_LOAD_ENV

    monkeypatch.setenv(LIVE_ON_LOAD_ENV, "1")
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.query_params["tab"] = "twin"
    at.query_params["live"] = "1"
    at.run()
    assert at.exception == [], [str(e.value) for e in at.exception]
    text = " ".join(element.value for element in at.success) + " ".join(
        element.value for element in at.error
    )
    assert "wall clock" in text or "Brian2" in text, "the live run did not report back"


FLY_READY = (REPO_ROOT / "outputs" / "fly" / "conditions_SIMULATED.csv").exists()


@pytest.mark.skipif(not FLY_READY, reason="needs the fly stage's outputs")
def test_the_fly_tab_moves_both_readouts_with_the_dose_slider() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.query_params["tab"] = "fly"
    at.run()
    assert at.exception == [], [str(e.value) for e in at.exception]
    firing = [m.value for m in at.metric if m.label == "Firing rate"]
    at.select_slider[0].set_value("untreated").run()
    assert at.exception == []
    assert [m.value for m in at.metric if m.label == "Firing rate"] == ["+0.00"]
    assert firing != ["+0.00"]
