"""CRC32C: the standard check value, the file checksum against the reference, the bucket form."""

from __future__ import annotations

import base64

import numpy as np

from neuromps.fetch import crc32c as c


def test_check_value_and_continuation() -> None:
    assert c.crc32c(b"123456789") == 0xE3069283
    assert c.crc32c(b"") == 0
    assert c.crc32c(b"6789", c.crc32c(b"12345")) == 0xE3069283


def test_file_checksum_matches_the_reference_for_awkward_lengths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(c, "CHUNK", 16)
    monkeypatch.setattr(c, "BLOCK", 16 * 8)
    c._zeros.cache_clear()
    rng = np.random.default_rng(0)
    for n in (0, 1, 15, 16, 17, 100, 128, 129, 300, 1000):
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        path = tmp_path / f"f{n}"
        path.write_bytes(data)
        assert c.crc32c_file(path) == c.crc32c(data), n
    c._zeros.cache_clear()


def test_default_chunking_on_a_few_chunks(tmp_path) -> None:
    data = np.random.default_rng(1).integers(0, 256, 3 * c.CHUNK + 5, dtype=np.uint8).tobytes()
    path = tmp_path / "f"
    path.write_bytes(data)
    assert c.crc32c_file(path) == c.crc32c(data)


def test_bucket_value_is_big_endian_base64() -> None:
    assert c.from_gcs(base64.b64encode((0xE3069283).to_bytes(4, "big")).decode()) == 0xE3069283
