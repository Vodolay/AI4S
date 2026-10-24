"""The FlyMimic leg is fetched from pinned, checksummed copies, and only when asked for."""

from __future__ import annotations

import pytest

from neuromps.fetch import flymimic, runner
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.manifest import FileRecord, Manifest


@pytest.fixture
def offline(monkeypatch, tmp_path):
    calls = []

    def fake_fetch_many(items, raw_root, cached, **kwargs):
        calls.extend(items)
        return [FileRecord(rel, url, "a" * 64, 1, "t") for url, rel in items]

    monkeypatch.setattr(flymimic, "fetch_many", fake_fetch_many)
    monkeypatch.setattr(flymimic, "retain_existing", lambda s, m, r, files: files)
    monkeypatch.setattr(flymimic, "published_sha256", lambda: "a" * 64)
    return calls


def test_github_files_are_read_at_the_pinned_commit(offline, tmp_path) -> None:
    flymimic.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=False)
    github = [u for u, _ in offline if "github" in u]
    assert github and all(flymimic.FLYGYM_COMMIT in u for u in github)
    names = {rel.rsplit("/", 1)[-1] for _, rel in offline}
    assert {flymimic.MODEL_NAME, "LICENSE", flymimic.MESH_ARCHIVE + ".tar"} <= names


def test_an_archive_that_does_not_match_its_published_checksum_is_refused(
    offline, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(flymimic, "published_sha256", lambda: "b" * 64)
    with pytest.raises(SourceBlocked, match="checksum"):
        flymimic.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=False)


def test_the_leg_is_optional() -> None:
    assert flymimic.SOURCE in runner.OPTIONAL


def test_extraction_keeps_only_mesh_files_inside_the_target(tmp_path) -> None:
    import io
    import tarfile

    archive = tmp_path / "m.tar"
    with tarfile.open(archive, "w") as tar:
        for name, body in (
            ("meshes/stl/Thorax.stl", b"solid"),
            ("../evil.stl", b"x"),
            ("meshes/run.sh", b"x"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    out = flymimic.extract_meshes(archive, tmp_path / "out")
    assert sorted(p.relative_to(tmp_path / "out").as_posix() for p in out) == [
        "meshes/stl/Thorax.stl"
    ]
