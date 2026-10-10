"""The Kapucu quick tier falls back to a checksummed mirror when GIN is down.

Every run - quick and full - needs Kapucu, and its only host, gin.g-node.org,
stopped accepting connections in the middle of the first fresh-clone full run.
The mirror is a GitHub release asset holding the quick tier's files exactly as
GIN serves them (CC BY 4.0, attribution inside the archive). It is used only for
the quick tier, only when GIN fails, and only after its pinned SHA-256 matches.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import pytest
import requests

from neuromps.fetch import kapucu2022, mirror
from neuromps.fetch.manifest import FileRecord, Manifest, utc_now


def _zip(tmp_path: Path, members: dict[str, bytes]) -> tuple[Path, str]:
    for rel, body in members.items():
        (tmp_path / "src" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "src" / rel).write_bytes(body)
    archive = tmp_path / "mirror.zip"
    sha = mirror.build_zip(tmp_path / "src", sorted(members), archive, readme="credit")
    return archive, sha


def _local_download(archive: Path):
    """A downloader that serves the archive from disk instead of the network."""

    def download(url, raw_root, rel_path, **kwargs):
        dest = Path(raw_root) / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, dest)
        body = dest.read_bytes()
        return FileRecord(rel_path, url, hashlib.sha256(body).hexdigest(), len(body), utc_now())

    return download


def test_the_archive_is_byte_for_byte_reproducible(tmp_path) -> None:
    """The pinned hash is only checkable if the same files always zip the same way."""
    _, first = _zip(tmp_path / "a", {"k/x.csv": b"1,2\n", "k/y.csv": b"3\n"})
    _, second = _zip(tmp_path / "b", {"k/y.csv": b"3\n", "k/x.csv": b"1,2\n"})
    assert first == second


def test_members_are_extracted_and_recorded_with_their_own_hashes(tmp_path) -> None:
    archive, sha = _zip(tmp_path, {"kapucu2022/a.csv": b"a\n", "kapucu2022/b.csv": b"bb\n"})
    raw = tmp_path / "raw"
    records = mirror.fetch_members(
        "https://example.org/m.zip",
        sha,
        raw,
        ["kapucu2022/a.csv", "kapucu2022/b.csv"],
        {},
        download=_local_download(archive),
    )
    assert [r.path for r in records] == ["kapucu2022/a.csv", "kapucu2022/b.csv"]
    assert (raw / "kapucu2022/b.csv").read_bytes() == b"bb\n"
    assert records[1].sha256 == hashlib.sha256(b"bb\n").hexdigest()
    assert records[0].url == "https://example.org/m.zip#kapucu2022/a.csv"
    assert not list(raw.rglob("*.zip")), "the archive is removed once unpacked"


def test_an_archive_with_the_wrong_hash_is_refused_and_nothing_is_written(tmp_path) -> None:
    archive, _ = _zip(tmp_path, {"kapucu2022/a.csv": b"a\n"})
    raw = tmp_path / "raw"
    with pytest.raises(mirror.MirrorMismatch):
        mirror.fetch_members(
            "https://example.org/m.zip",
            "0" * 64,
            raw,
            ["kapucu2022/a.csv"],
            {},
            download=_local_download(archive),
        )
    assert not (raw / "kapucu2022/a.csv").exists()
    assert not list(raw.rglob("*.zip"))


def test_a_member_that_was_not_asked_for_is_never_written(tmp_path) -> None:
    """Only listed paths come out, so a stray or hostile name cannot land anywhere."""
    archive, sha = _zip(tmp_path, {"kapucu2022/a.csv": b"a\n", "kapucu2022/extra.csv": b"x\n"})
    raw = tmp_path / "raw"
    records = mirror.fetch_members(
        "https://example.org/m.zip",
        sha,
        raw,
        ["kapucu2022/a.csv"],
        {},
        download=_local_download(archive),
    )
    assert [r.path for r in records] == ["kapucu2022/a.csv"]
    assert not (raw / "kapucu2022/extra.csv").exists()


def test_a_requested_file_missing_from_the_archive_is_an_error(tmp_path) -> None:
    archive, sha = _zip(tmp_path, {"kapucu2022/a.csv": b"a\n"})
    with pytest.raises(mirror.MirrorMismatch, match="missing"):
        mirror.fetch_members(
            "https://example.org/m.zip",
            sha,
            tmp_path / "raw",
            ["kapucu2022/a.csv", "kapucu2022/b.csv"],
            {},
            download=_local_download(archive),
        )


@pytest.fixture
def offline_kapucu(monkeypatch, tmp_path):
    """Kapucu's fetcher with GIN, the mirror and the article all stood in for."""
    used = {"mirror": 0, "gin": 0}

    def fake_mirror(url, sha256, raw_root, wanted, cached, **kwargs):
        used["mirror"] += 1
        assert url == kapucu2022.MIRROR_URL and sha256 == kapucu2022.MIRROR_SHA256
        return [FileRecord(rel, url + "#" + rel, "f" * 64, 1, utc_now()) for rel in wanted]

    def fake_gin(items, raw_root, cached, **kwargs):
        used["gin"] += 1
        raise requests.exceptions.ConnectTimeout("gin.g-node.org timed out")

    monkeypatch.setattr(kapucu2022, "gin_reachable", lambda: True)  # fails mid-way instead
    monkeypatch.setattr(kapucu2022.mirror, "fetch_members", fake_mirror)
    monkeypatch.setattr(kapucu2022, "fetch_many", fake_gin)
    monkeypatch.setattr(kapucu2022, "fetch_article", lambda *a, **k: ([], None))
    monkeypatch.setattr(kapucu2022, "retain_existing", lambda s, m, r, files: files)
    return used


def test_the_quick_tier_falls_back_to_the_mirror_when_gin_fails(offline_kapucu, tmp_path) -> None:
    record = kapucu2022.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=False)
    assert offline_kapucu == {"mirror": 1, "gin": 1}
    assert record.status == "ok"
    assert len(record.files) == len(kapucu2022.file_list(False))
    assert "mirror" in record.note and kapucu2022.COMMIT in record.note


def test_the_full_tier_never_uses_the_mirror(offline_kapucu, tmp_path) -> None:
    """The mirror holds the quick tier only; a full run must say GIN failed."""
    with pytest.raises(requests.exceptions.ConnectTimeout):
        kapucu2022.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=True)
    assert offline_kapucu["mirror"] == 0


def test_the_mirror_is_pinned_to_this_repository_and_a_full_hash() -> None:
    assert kapucu2022.MIRROR_URL.startswith("https://github.com/Vodolay/AI4S/releases/download/")
    assert len(kapucu2022.MIRROR_SHA256) == 64
    int(kapucu2022.MIRROR_SHA256, 16)


@pytest.mark.network
def test_the_published_mirror_matches_its_pin_and_the_quick_tier(tmp_path) -> None:
    response = requests.get(kapucu2022.MIRROR_URL, timeout=300)
    response.raise_for_status()
    assert hashlib.sha256(response.content).hexdigest() == kapucu2022.MIRROR_SHA256
    archive = tmp_path / "m.zip"
    archive.write_bytes(response.content)
    with zipfile.ZipFile(archive) as z:
        names = {n for n in z.namelist() if not n.endswith(mirror.README_NAME)}
    assert names == {f"{kapucu2022.SOURCE}/{p}" for p in kapucu2022.file_list(False)}


def test_a_host_that_is_down_goes_straight_to_the_mirror(
    offline_kapucu, monkeypatch, tmp_path
) -> None:
    """In the real outage each connection hung for minutes; 169 of them must not be tried."""
    monkeypatch.setattr(kapucu2022, "gin_reachable", lambda: False)
    record = kapucu2022.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=False)
    assert offline_kapucu == {"mirror": 1, "gin": 0}
    assert record.status == "ok"


def test_the_reachability_probe_gives_up_quickly(monkeypatch) -> None:
    asked = []

    def unreachable(url, timeout, **kwargs):
        asked.append(timeout)
        raise requests.exceptions.ConnectTimeout("no route")

    monkeypatch.setattr(kapucu2022.requests, "get", unreachable)
    assert kapucu2022.gin_reachable() is False
    assert len(asked) <= 2 and sum(asked) <= 60


def test_a_failing_fetch_stops_queued_downloads_instead_of_trying_them_all(monkeypatch, tmp_path):
    """The pool used to drain every queued file before reporting the first failure."""
    from neuromps.fetch import common

    started = []

    def failing_download(url, raw_root, rel, **kwargs):
        started.append(rel)
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(common, "download_file", failing_download)
    items = [(f"https://example.org/{i}", f"s/{i}") for i in range(100)]
    with pytest.raises(requests.exceptions.ConnectionError):
        common.fetch_many(items, tmp_path, {}, workers=4)
    assert len(started) < 20
