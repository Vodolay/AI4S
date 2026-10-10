"""A seekable, block-caching file over HTTP, for reading inside a remote file.

``remote_zip.HttpRangeFile`` streams forwards: it is built for extracting a ZIP
member, where one sequential pass does the job. HDF5 is the opposite. Its
object headers, B-tree nodes and data blocks are scattered, so h5py seeks
backwards constantly, and a reader that opens a new request per backward seek
spends all its time in round trips - opening one 1 GB MATLAB v7.3 file that way
did not finish in ten minutes.

This reader caches fixed-size blocks and keeps one connection open, which turns
those thousands of small scattered reads into a few hundred block fetches. That
is what makes it possible to take the spike times out of a file whose bulk is
the LFP traces we do not use: bytes actually transferred are reported by
:attr:`bytes_fetched`, so a fetcher can record how much of the remote file it
really read.
"""

from __future__ import annotations

import io
from collections import OrderedDict

import requests

from neuromps.fetch import download
from neuromps.fetch.download import USER_AGENT

BLOCK_SIZE = 1 << 20
MAX_BLOCKS = 768


class CachedRangeReader(io.RawIOBase):
    """Read-only random access to a remote file, one cached block at a time."""

    def __init__(
        self,
        url: str,
        block_size: int = BLOCK_SIZE,
        max_blocks: int = MAX_BLOCKS,
        timeout: float = 120,
        session: requests.Session | None = None,
    ) -> None:
        self.url = url
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.timeout = timeout
        self._session = session or download.retrying_session()
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self.bytes_fetched = 0
        self.requests_made = 0
        self.pos = 0
        self.size = self._probe_size()

    def _probe_size(self) -> int:
        """Total length, from an explicit one-byte range rather than a HEAD.

        Some repositories answer HEAD without Content-Length, and suffix ranges
        with an error; a 0-0 range is the form every host here honours.
        """
        r = self._session.get(
            self.url,
            headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0", "Accept-Encoding": "identity"},
            timeout=self.timeout,
            stream=True,
        )
        try:
            r.raise_for_status()
            total = r.headers.get("Content-Range", "").rpartition("/")[2]
            if not total.isdigit():
                raise OSError(f"server does not report a total size for {self.url}")
            return int(total)
        finally:
            r.close()

    def _fetch(self, first: int, last: int) -> None:
        """Fetch blocks `first`..`last` inclusive in one request."""
        start = first * self.block_size
        end = min((last + 1) * self.block_size, self.size) - 1
        r = self._session.get(
            self.url,
            headers={
                "User-Agent": USER_AGENT,
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        if r.status_code != 206:
            raise OSError(f"{self.url} ignored a range request (status {r.status_code})")
        payload = r.content
        self.bytes_fetched += len(payload)
        self.requests_made += 1
        for index in range(first, last + 1):
            offset = (index - first) * self.block_size
            chunk = payload[offset : offset + self.block_size]
            if not chunk:
                break
            self._blocks[index] = chunk
            self._blocks.move_to_end(index)
        while len(self._blocks) > self.max_blocks:
            self._blocks.popitem(last=False)

    def _ensure(self, first: int, last: int) -> None:
        """Fetch any missing blocks in the range, coalescing adjacent misses."""
        missing = [i for i in range(first, last + 1) if i not in self._blocks]
        if not missing:
            return
        run_start = previous = missing[0]
        for index in missing[1:]:
            if index == previous + 1:
                previous = index
                continue
            self._fetch(run_start, previous)
            run_start = previous = index
        self._fetch(run_start, previous)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readall(self) -> bytes:
        return self.read(self.size - self.pos)

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.pos
        size = max(0, min(size, self.size - self.pos))
        if size == 0:
            return b""
        first = self.pos // self.block_size
        last = (self.pos + size - 1) // self.block_size
        self._ensure(first, last)
        out = bytearray()
        remaining, position = size, self.pos
        while remaining > 0:
            index = position // self.block_size
            inside = position % self.block_size
            block = self._blocks.get(index, b"")
            if index in self._blocks:
                self._blocks.move_to_end(index)
            piece = block[inside : inside + remaining]
            if not piece:
                break
            out += piece
            remaining -= len(piece)
            position += len(piece)
        self.pos += len(out)
        return bytes(out)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self._session.close()
        super().close()
