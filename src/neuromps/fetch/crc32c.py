"""CRC32C (Castagnoli), the checksum Google Cloud Storage publishes for composite objects.

The bucket gives no MD5 for objects uploaded in parts, only a CRC32C. Python's standard
library has no CRC32C, and a byte-at-a-time loop over gigabytes takes hours, so the file
checksum runs the CRC over many fixed-size chunks at once (one vectorized table lookup per
byte position) and then joins the chunk CRCs with the GF(2) operator that appends zero
bytes. The result is the standard CRC32C; ``crc32c`` is the slow reference it is tested
against.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import numpy as np

POLY = 0x82F63B78  # reflected Castagnoli polynomial
CHUNK = 4096  # bytes per lane in the vectorized pass
BLOCK = CHUNK * 65536  # bytes read per step (256 MiB)


@lru_cache(maxsize=1)
def _table() -> np.ndarray:
    out = np.zeros(256, dtype=np.uint32)
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ POLY if c & 1 else c >> 1
        out[i] = c
    return out


def crc32c(data: bytes, crc: int = 0) -> int:
    """Reference CRC32C of ``data`` (byte at a time); ``crc`` continues an earlier value."""
    table = _table()
    c = crc ^ 0xFFFFFFFF
    for b in data:
        c = int(table[(c ^ b) & 0xFF]) ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def _apply(columns: np.ndarray, v: np.ndarray | int) -> np.ndarray:
    """A 32 x 32 GF(2) matrix (as 32 column words) applied to each 32-bit value."""
    v = np.asarray(v, dtype=np.uint32)
    out = np.zeros_like(v)
    for j in range(32):
        out ^= np.where((v >> np.uint32(j)) & np.uint32(1), columns[j], np.uint32(0))
    return out


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The matrix a . b (apply b, then a)."""
    return _apply(a, b)


@lru_cache(maxsize=64)
def _zeros(n: int) -> np.ndarray:
    """The operator that advances a raw (unfinalized) CRC over ``n`` zero bytes."""
    table = _table()
    one = np.array([table[(1 << j) & 0xFF] ^ ((1 << j) >> 8) for j in range(32)], dtype=np.uint32)
    result = np.array([1 << j for j in range(32)], dtype=np.uint32)
    power = one
    while n:
        if n & 1:
            result = _compose(power, result)
        power = _compose(power, power)
        n >>= 1
    return result


def _raw_lanes(data: np.ndarray) -> np.ndarray:
    """Raw CRC (no init, no final xor) of each row of a 2-D uint8 array."""
    table = _table()
    s = np.zeros(data.shape[0], dtype=np.uint32)
    for k in range(data.shape[1]):
        s = table[(s ^ data[:, k]) & np.uint32(0xFF)] ^ (s >> np.uint32(8))
    return s


def _raw_block(buf: bytes) -> int:
    """Raw CRC of one buffer: full chunks in parallel, joined by a pairwise tree, then the tail.

    A raw CRC starts at zero and stays zero over leading zero bytes, so the lanes are padded
    in front with zero lanes up to a power of two and every tree level joins equal halves.
    """
    n_full = len(buf) // CHUNK
    raw = 0
    if n_full:
        lanes = _raw_lanes(np.frombuffer(buf, np.uint8, n_full * CHUNK).reshape(n_full, CHUNK))
        width = 1 << (n_full - 1).bit_length()
        lanes = np.concatenate([np.zeros(width - n_full, dtype=np.uint32), lanes])
        size = CHUNK
        while lanes.size > 1:
            lanes = _apply(_zeros(size), lanes[0::2]) ^ lanes[1::2]
            size *= 2
        raw = int(lanes[0])
    table = _table()
    for b in buf[n_full * CHUNK :]:
        raw = int(table[(raw ^ b) & 0xFF]) ^ (raw >> 8)
    return raw


def crc32c_file(path: Path) -> int:
    """CRC32C of a file, read in 256 MiB blocks."""
    raw, total = 0, 0
    with Path(path).open("rb") as fh:
        while buf := fh.read(BLOCK):
            raw = int(_apply(_zeros(len(buf)), raw)) ^ _raw_block(buf)
            total += len(buf)
    init = int(_apply(_zeros(total), 0xFFFFFFFF)) if total else 0xFFFFFFFF
    return raw ^ init ^ 0xFFFFFFFF


def from_gcs(value: str) -> int:
    """The bucket's base64, big-endian CRC32C as an integer."""
    return int.from_bytes(base64.b64decode(value), "big")
