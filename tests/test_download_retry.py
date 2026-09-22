"""Backing off when a host refuses to serve us.

Written after a CI run died on a 403 from gin.g-node.org for a file that
downloaded fine from a laptop the same minute: the host rate-limits by
address, and says so with the status code for "forbidden".
"""

from __future__ import annotations

import pytest

from neuromps.fetch import download


class FakeResponse:
    """Just enough of requests.Response for the retry loop."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _responses(monkeypatch, statuses: list[int]) -> list[str]:
    """Serve `statuses` in order; return the list that records sleeps."""
    queue = list(statuses)
    slept: list[float] = []
    monkeypatch.setattr(download.requests, "get", lambda *a, **k: FakeResponse(queue.pop(0)))
    return slept


def test_a_rate_limited_request_is_retried_until_it_succeeds(monkeypatch) -> None:
    slept = _responses(monkeypatch, [403, 403, 200])
    response = download.get_with_retry("https://example.invalid/f", timeout=1, sleep=slept.append)
    assert response.status_code == 200
    assert slept == [2.0, 6.0], "delays should grow between attempts"


def test_the_last_response_is_returned_rather_than_retried_forever(monkeypatch) -> None:
    slept = _responses(monkeypatch, [429, 429, 429, 429])
    response = download.get_with_retry("https://example.invalid/f", timeout=1, sleep=slept.append)
    assert response.status_code == 429
    assert len(slept) == download.RETRY_ATTEMPTS - 1


def test_an_ordinary_failure_is_not_retried(monkeypatch) -> None:
    """A 404 is an answer, not a request to come back later."""
    slept = _responses(monkeypatch, [404])
    response = download.get_with_retry("https://example.invalid/f", timeout=1, sleep=slept.append)
    assert response.status_code == 404
    assert slept == []


@pytest.mark.parametrize("status", [403, 429, 503])
def test_the_refusing_response_is_closed_before_the_next_attempt(monkeypatch, status: int) -> None:
    """A streaming response left open holds a connection to a host already under load."""
    seen: list[FakeResponse] = []

    def fake_get(*_args, **_kwargs) -> FakeResponse:
        seen.append(FakeResponse(status if len(seen) == 0 else 200))
        return seen[-1]

    monkeypatch.setattr(download.requests, "get", fake_get)
    download.get_with_retry("https://example.invalid/f", timeout=1, sleep=lambda _s: None)
    assert seen[0].closed is True
