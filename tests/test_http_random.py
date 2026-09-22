"""The block-caching remote reader, checked against a served local file."""

from __future__ import annotations

import io
import os

import pytest

from neuromps.fetch.http_random import CachedRangeReader

PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes of known content


class FakeResponse:
    def __init__(self, status_code: int, content: bytes, headers: dict) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise OSError(f"status {self.status_code}")

    def close(self) -> None:
        pass


class FakeSession:
    """Serves PAYLOAD over byte ranges and counts what was asked for."""

    def __init__(self, data: bytes = PAYLOAD) -> None:
        self.data = data
        self.ranges: list[tuple[int, int]] = []

    def get(self, url, headers=None, timeout=None, stream=False):
        start, _, end = (headers or {})["Range"].removeprefix("bytes=").partition("-")
        start = int(start)
        end = int(end) if end else len(self.data) - 1
        self.ranges.append((start, end))
        chunk = self.data[start : end + 1]
        return FakeResponse(
            206, chunk, {"Content-Range": f"bytes {start}-{end}/{len(self.data)}"}
        )

    def close(self) -> None:
        pass


@pytest.fixture
def reader() -> CachedRangeReader:
    return CachedRangeReader("https://example.invalid/f", block_size=4096, session=FakeSession())


def test_it_reports_the_total_size_without_downloading_the_file(reader) -> None:
    assert reader.size == len(PAYLOAD)
    assert reader.bytes_fetched <= 1


def test_a_read_returns_exactly_the_requested_bytes(reader) -> None:
    reader.seek(5000)
    assert reader.read(300) == PAYLOAD[5000:5300]


def test_reading_across_a_block_boundary_is_contiguous(reader) -> None:
    reader.seek(4090)
    assert reader.read(20) == PAYLOAD[4090:4110]


def test_a_second_read_of_the_same_region_costs_nothing(reader) -> None:
    reader.seek(2048)
    reader.read(100)
    after_first = reader.bytes_fetched
    reader.seek(2048)
    reader.read(100)
    assert reader.bytes_fetched == after_first, "the block cache should have served this"


def test_seeking_backwards_does_not_refetch(reader) -> None:
    """The failure this class exists to fix: HDF5 seeks backwards constantly."""
    reader.seek(40000)
    reader.read(50)
    reader.seek(0)
    reader.read(50)
    before = reader.requests_made
    for position in (40000, 0, 40000, 0):
        reader.seek(position)
        reader.read(50)
    assert reader.requests_made == before


def test_adjacent_missing_blocks_are_fetched_in_one_request(reader) -> None:
    reader.seek(0)
    reader.read(4096 * 5)
    assert reader.requests_made == 1, reader._session.ranges


def test_reading_past_the_end_stops_at_the_end(reader) -> None:
    reader.seek(len(PAYLOAD) - 10)
    assert reader.read(500) == PAYLOAD[-10:]
    assert reader.read(10) == b""


def test_seek_from_the_end_and_from_the_current_position(reader) -> None:
    reader.seek(-20, io.SEEK_END)
    assert reader.tell() == len(PAYLOAD) - 20
    reader.seek(0)
    reader.seek(10, io.SEEK_CUR)
    assert reader.read(5) == PAYLOAD[10:15]


def test_the_cache_is_bounded(reader) -> None:
    small = CachedRangeReader(
        "https://example.invalid/f", block_size=4096, max_blocks=3, session=FakeSession()
    )
    for position in range(0, 4096 * 8, 4096):
        small.seek(position)
        small.read(10)
    assert len(small._blocks) <= 3


@pytest.mark.skipif(not os.environ.get("NEUROMPS_NETWORK_TESTS"), reason="needs the network")
def test_it_reads_a_real_remote_hdf5_header() -> None:
    url = "https://zenodo.org/records/4751759/files/LFP_Sp_160810.mat?download=1"
    remote = CachedRangeReader(url)
    remote.seek(512)
    assert remote.read(4) == b"\x89HDF"
