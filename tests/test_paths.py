"""Tests for run-root resolution: a quick run must not write over a full one."""

import pytest

from neuromps import paths


@pytest.fixture(autouse=True)
def _full_run(monkeypatch) -> None:
    """Each test states its own mode; none inherits the caller's environment."""
    monkeypatch.delenv(paths.MODE_ENV, raising=False)


def test_a_full_run_writes_the_reported_locations() -> None:
    assert paths.metrics_path() == paths.REPO_ROOT / "outputs" / "metrics.json"
    assert paths.asset_path() == paths.REPO_ROOT / "data" / "processed" / "neuromps.parquet"
    assert paths.figures_dir() == paths.REPO_ROOT / "outputs" / "figures"


def test_a_quick_run_writes_under_its_own_root(monkeypatch) -> None:
    monkeypatch.setenv(paths.MODE_ENV, "quick")
    assert paths.is_quick()
    assert paths.metrics_path() == paths.REPO_ROOT / "outputs" / "quick" / "metrics.json"
    assert paths.twin_dir() == paths.REPO_ROOT / "outputs" / "quick" / "twin"
    assert paths.demo_dir() == paths.REPO_ROOT / "outputs" / "quick" / "demo"
    assert (
        paths.asset_path() == paths.REPO_ROOT / "data" / "processed" / "quick" / "neuromps.parquet"
    )


def test_no_quick_path_lands_inside_the_full_run(monkeypatch) -> None:
    """The point of the split: nothing a quick run writes can be a full-run file."""
    monkeypatch.setenv(paths.MODE_ENV, "quick")
    quick = [paths.metrics_path(), paths.twin_dir(), paths.demo_dir(), paths.asset_path()]
    monkeypatch.delenv(paths.MODE_ENV)
    full = {paths.metrics_path(), paths.twin_dir(), paths.demo_dir(), paths.asset_path()}
    assert not full.intersection(quick)


def test_set_run_mode_rejects_anything_else() -> None:
    with pytest.raises(ValueError, match="run mode"):
        paths.set_run_mode("fast")


def test_reading_prefers_what_this_run_produced(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setenv(paths.MODE_ENV, "quick")
    mine = paths.twin_dir() / "posterior.pt"
    mine.parent.mkdir(parents=True)
    mine.write_text("quick")
    paths.full_run_equivalent(mine).parent.mkdir(parents=True)
    paths.full_run_equivalent(mine).write_text("full")
    assert paths.for_reading(mine) == mine


def test_reading_falls_back_to_the_full_run_when_this_one_has_no_copy(
    monkeypatch, tmp_path, caplog
) -> None:
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setenv(paths.MODE_ENV, "quick")
    wanted = paths.twin_dir() / "posterior.pt"
    full = paths.full_run_equivalent(wanted)
    full.parent.mkdir(parents=True)
    full.write_text("full")
    with caplog.at_level("WARNING"):
        assert paths.for_reading(wanted) == full
    assert "full run" in caplog.text  # a silent fallback is the thing to avoid


def test_reading_returns_the_asked_for_path_when_nothing_exists(monkeypatch, tmp_path) -> None:
    """The caller's own missing-file branch has to keep running."""
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setenv(paths.MODE_ENV, "quick")
    wanted = paths.twin_dir() / "posterior.pt"
    assert paths.for_reading(wanted) == wanted


def test_a_full_run_never_falls_back(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path / "outputs")
    wanted = paths.twin_dir() / "posterior.pt"
    assert paths.for_reading(wanted) == wanted


def test_the_simulation_cache_is_never_shared_between_modes(monkeypatch) -> None:
    """150 quick draws must not become the next full run's training set."""
    full = paths.simulation_cache()
    monkeypatch.setenv(paths.MODE_ENV, "quick")
    assert paths.simulation_cache() != full


def test_a_quick_fetch_writes_its_own_manifest(monkeypatch) -> None:
    """The committed manifest records the full run's downloads; quick keeps its own."""
    monkeypatch.setenv(paths.MODE_ENV, "quick")
    assert paths.manifest_path() == paths.RAW_ROOT / "MANIFEST.quick.json"
    monkeypatch.delenv(paths.MODE_ENV)
    assert paths.manifest_path() == paths.RAW_ROOT / "MANIFEST.json"


def test_the_manifest_follows_the_raw_root_it_is_given(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(paths.MODE_ENV, "quick")
    assert paths.manifest_path(tmp_path) == tmp_path / "MANIFEST.quick.json"


def test_interim_dir_is_split_by_run_mode(monkeypatch) -> None:
    """A quick run's partial cache must not be read by a full one."""
    from neuromps import paths

    monkeypatch.delenv(paths.MODE_ENV, raising=False)
    full = paths.interim_dir()
    monkeypatch.setenv(paths.MODE_ENV, paths.QUICK)
    assert paths.interim_dir() != full
    assert paths.interim_dir().name == paths.QUICK
