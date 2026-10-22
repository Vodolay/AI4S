"""The demo must also run as a hosted page, from the repository alone.

Streamlit Community Cloud installs `app/requirements.txt` and runs `app/app.py`;
it does not `pip install -e .`, and the light requirements leave out the
simulator (Brian2), torch and sbi. So the app has to find `neuromps` in `src/`
by itself, and the one control that needs Brian2 has to say why it is absent
rather than fail.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "app" / "app.py"
HOSTED_REQUIREMENTS = REPO_ROOT / "app" / "requirements.txt"
DEMO_READY = (REPO_ROOT / "outputs" / "demo" / "meta.json").exists()


def _pins(text: str) -> dict[str, str]:
    return dict(re.findall(r'^\s*"?([A-Za-z0-9_.-]+)==([0-9][^",\s#]*)', text, re.M))


def test_hosted_requirements_pin_the_same_versions_as_the_project() -> None:
    """A hosted page on different versions would show a different app."""
    hosted = _pins(HOSTED_REQUIREMENTS.read_text())
    project = _pins((REPO_ROOT / "pyproject.toml").read_text())
    assert hosted, "app/requirements.txt pins nothing"
    for name, version in hosted.items():
        assert project.get(name) == version, f"{name}: {version} hosted, {project.get(name)} pinned"


def test_hosted_requirements_leave_out_the_heavy_simulation_stack() -> None:
    hosted = _pins(HOSTED_REQUIREMENTS.read_text())
    assert not {"torch", "brian2", "sbi"} & set(hosted)
    assert {"streamlit", "scikit-learn", "pandas", "pyarrow"} <= set(hosted)


def test_the_entry_point_puts_src_on_the_path_when_neuromps_is_not_installed() -> None:
    source = APP.read_text()
    assert 'REPO_ROOT / "src"' in source


def test_the_live_run_reports_itself_unavailable_without_brian2(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))  # the app package sits at the repo root
    from app import loaders

    monkeypatch.setattr(loaders.importlib.util, "find_spec", lambda name: None)
    assert loaders.live_simulation_available() is False


@pytest.mark.skipif(not DEMO_READY, reason="needs the precomputed demo inputs")
def test_the_twin_tab_explains_a_missing_simulator_instead_of_offering_the_button(
    monkeypatch,
) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "brian2" else real(name, *a)
    )
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.query_params["tab"] = "twin"
    at.run()
    assert at.exception == []
    assert not any("live simulation" in b.label for b in at.button)
    assert any("full local install" in i.value for i in at.info)
