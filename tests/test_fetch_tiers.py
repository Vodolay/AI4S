"""A fetch records into the manifest of the tier it fetched.

The committed `data/raw/MANIFEST.json` describes the full tier. A small-tier
fetch written into it replaces each source's entry with fewer files: on a fresh
clone, the README's own `neuromps.fetch.runner --only ...` and a bare
`run_all.py` both did that, and the provenance checks then failed against the
shrunken record.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuromps import paths
from neuromps.fetch import runner

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fetch_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Stand in for the network: record each fetch's tier and active run mode."""
    calls: list[dict] = []

    def fake_run_fetchers(raw_root=None, names=None, full=False):
        calls.append({"full": full, "quick_mode": paths.is_quick()})
        return SimpleNamespace(sources={})

    monkeypatch.setattr(runner, "run_fetchers", fake_run_fetchers)
    yield calls
    paths.set_run_mode(paths.FULL)


def test_a_standalone_small_tier_fetch_keeps_out_of_the_committed_manifest(fetch_calls) -> None:
    runner.main(["--only", "kapucu2022"])
    assert fetch_calls == [{"full": False, "quick_mode": True}]


def test_a_standalone_full_fetch_records_into_the_committed_manifest(fetch_calls) -> None:
    runner.main(["--full", "--only", "kapucu2022"])
    assert fetch_calls == [{"full": True, "quick_mode": False}]


def _run_all():
    spec = importlib.util.spec_from_file_location("run_all", REPO_ROOT / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], {"full": True, "quick_mode": False}),  # the README: a bare run is the full run
        (["--full"], {"full": True, "quick_mode": False}),
        (["--quick"], {"full": False, "quick_mode": True}),
    ],
)
def test_run_all_fetches_the_tier_of_the_run_it_is(
    fetch_calls, argv, expected, monkeypatch, tmp_path
) -> None:
    # Every write a run makes goes under tmp_path: the first version of this test
    # recorded its stage timing into the real outputs/metrics.json.
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(paths, "PROCESSED_ROOT", tmp_path / "processed")
    run_all = _run_all()
    monkeypatch.setattr(run_all.harness, "record", lambda *a, **k: None)
    monkeypatch.setattr(run_all, "_record_timings", lambda *a, **k: None)
    run_all.main([*argv, "--stage", "fetch"])
    assert fetch_calls == [expected]


def test_a_quick_run_leaves_the_fly_stage_out_unless_asked() -> None:
    module = _run_all()
    assert "fly" not in module.select_stages("all", quick=True, with_fly=False)
    assert module.select_stages("all", quick=True, with_fly=True)[-1] == "fly"
    assert module.select_stages("all", quick=False, with_fly=False)[-1] == "fly"
    assert module.select_stages("fly", quick=True, with_fly=False) == ["fly"]
