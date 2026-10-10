"""A flaky host must cost a retry, and an incomplete download must stop the run.

The first end-to-end `run_all.py --full` from a fresh clone lost three sources to
the network - a read timeout on gin.g-node.org, a connection that broke 670 MB
into a 930 MB file, a Zenodo 504 - recorded them as partial, carried on, and
crashed an hour later in the models stage with `min() arg is an empty sequence`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from neuromps import paths
from neuromps.fetch import download, runner

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Response:
    """A streamed response whose body may break part-way, as a real one did."""

    def __init__(self, chunks: list[bytes], breaks: bool) -> None:
        self.chunks, self.breaks, self.status_code = chunks, breaks, 200

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, size: int):
        yield self.chunks[0]
        if self.breaks:
            raise requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")
        yield from self.chunks[1:]


@pytest.fixture
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waited: list[float] = []
    monkeypatch.setattr(download.time, "sleep", waited.append)
    return waited


def test_a_connection_broken_mid_file_is_retried(monkeypatch, tmp_path, no_waiting) -> None:
    responses = iter([_Response([b"ab", b"cd"], breaks=True), _Response([b"ab", b"cd"], False)])
    monkeypatch.setattr(download, "get_with_retry", lambda url, timeout: next(responses))
    record = download.download_file("https://example.org/f", tmp_path, "src/f.bin")
    assert (tmp_path / "src/f.bin").read_bytes() == b"abcd"
    assert record.bytes == 4
    assert len(no_waiting) == 1


def test_a_host_that_keeps_failing_still_fails_and_leaves_no_fragment(
    monkeypatch, tmp_path, no_waiting
) -> None:
    def always_times_out(url, timeout):
        raise requests.exceptions.ReadTimeout("Read timed out.")

    monkeypatch.setattr(download, "get_with_retry", always_times_out)
    with pytest.raises(requests.exceptions.ReadTimeout):
        download.download_file("https://example.org/f", tmp_path, "src/f.bin")
    assert len(no_waiting) == download.RETRY_ATTEMPTS - 1
    assert not list(tmp_path.rglob("*.part")) and not (tmp_path / "src/f.bin").exists()


def _manifest(**statuses: str) -> SimpleNamespace:
    return SimpleNamespace(
        sources={
            name: SimpleNamespace(name=name, status=status, note=f"{name} note")
            for name, status in statuses.items()
        }
    )


def test_only_partial_and_blocked_sources_count_as_incomplete() -> None:
    manifest = _manifest(a="ok", b="skipped", c="partial", d="blocked")
    assert [s.name for s in runner.incomplete_sources(manifest)] == ["c", "d"]


def test_run_all_stops_after_an_incomplete_fetch_and_says_how_to_resume(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(paths, "PROCESSED_ROOT", tmp_path / "processed")
    monkeypatch.setattr(
        runner, "run_fetchers", lambda **kwargs: _manifest(kapucu2022="partial", fofie2025="ok")
    )
    spec = importlib.util.spec_from_file_location("run_all", REPO_ROOT / "run_all.py")
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)
    monkeypatch.setattr(run_all.harness, "record", lambda *a, **k: None)
    monkeypatch.setattr(run_all, "_record_timings", lambda *a, **k: None)
    with pytest.raises(SystemExit) as stopped:
        run_all.main(["--full", "--stage", "fetch"])
    message = str(stopped.value.code)
    assert "kapucu2022" in message and "fofie2025" not in message
    assert "rerun" in message.lower()
    paths.set_run_mode(paths.FULL)


def test_the_shared_session_retries_busy_statuses_and_dropped_connections() -> None:
    """Range reads go through a session, not `download_file`, so it needs its own retries."""
    retry = download.retrying_session().get_adapter("https://zenodo.org/").max_retries
    assert retry.total == download.RETRY_ATTEMPTS - 1
    assert 504 in retry.status_forcelist and 403 in retry.status_forcelist
    assert retry.connect and retry.read  # a refused or stalled connection is retried too


def test_the_range_reader_uses_the_retrying_session_by_default(monkeypatch) -> None:
    from neuromps.fetch import http_random

    used = []

    class _Session:
        def get(self, url, **kwargs):
            used.append(url)
            return SimpleNamespace(
                headers={"Content-Range": "bytes 0-0/1000"},
                raise_for_status=lambda: None,
                close=lambda: None,
            )

    monkeypatch.setattr(download, "retrying_session", _Session)
    reader = http_random.CachedRangeReader("https://zenodo.org/f")
    assert used == ["https://zenodo.org/f"] and reader.size == 1000
