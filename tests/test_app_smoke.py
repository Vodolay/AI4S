"""Smoke tests for the Streamlit demo.

These run the app the way Streamlit does, assert that no tab raises, that the
widgets each tab is built around exist, and that an interaction stays inside the
responsiveness budget the app promises. They need the precomputed demo inputs,
so they skip when `outputs/demo/` is absent rather than failing a fresh clone.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "app" / "app.py"
DEMO_READY = (REPO_ROOT / "outputs" / "demo" / "meta.json").exists()
INTERACTION_BUDGET_S = 1.0

needs_demo = pytest.mark.skipif(not DEMO_READY, reason="run `python run_all.py --stage demo` first")


@pytest.fixture(scope="module")
def app() -> AppTest:
    """The app, run once."""
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def test_the_app_runs_without_raising(app: AppTest) -> None:
    assert app.exception == [], [str(e.value) for e in app.exception]


def test_the_three_tabs_are_present(app: AppTest) -> None:
    labels = [tab.label for tab in app.tabs]
    assert len(labels) == 3
    assert "Data asset" in labels[0]
    assert "Mechanism" in labels[1]
    assert "twin" in labels[2].lower()


def test_the_research_use_banner_is_shown(app: AppTest) -> None:
    """The project's framing rule, checked rather than trusted."""
    warnings = " ".join(w.value for w in app.warning)
    assert "Research decision-support only" in warnings
    assert "clinical claim" in warnings


@needs_demo
def test_tab_one_shows_the_dataset_card_and_record_filters(app: AppTest) -> None:
    labels = [m.label for m in app.metric]
    assert "Records" in labels and "Sources" in labels
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


@needs_demo
def test_changing_a_twin_slider_responds_within_the_budget() -> None:
    """The app's promise: an interaction never waits on a simulator."""
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    slider = next(s for s in at.select_slider if s.label == "GABA-A scale")
    options = list(slider.options)
    started = time.time()
    slider.set_value(options[-1]).run()
    elapsed = time.time() - started
    assert at.exception == [], [str(e.value) for e in at.exception]
    assert elapsed < INTERACTION_BUDGET_S, f"slider took {elapsed:.2f}s"


@needs_demo
def test_switching_to_manual_features_exposes_one_slider_per_feature() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    radio = next(r for r in at.radio if r.label == "Input")
    radio.set_value("Dial a feature vector").run()
    assert at.exception == []
    assert len(at.slider) == 9, [s.label for s in at.slider]


def test_the_live_simulation_is_not_wired_into_page_load() -> None:
    """Brian2 must never run as a side effect of opening the app."""
    source = (REPO_ROOT / "app" / "tab_twin.py").read_text()
    call = source.index("loaders.live_simulation")
    guard = source.index('if not st.button("Run one live simulation')
    assert guard < call, "the live run must sit behind the button"


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
