"""The flybody body model is fetched at one commit, every file checked against its git hash."""

from __future__ import annotations

import hashlib

import pytest

from neuromps.fetch import flybody, runner
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.manifest import FileRecord, Manifest


def test_git_blob_hash_matches_gits_own_rule() -> None:
    body = b"hello\n"
    expected = hashlib.sha1(b"blob 6\x00hello\n").hexdigest()
    assert flybody.git_blob_sha1(body) == expected == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_only_model_files_are_wanted() -> None:
    tree = [
        {"path": "flybody/fruitfly/assets/fruitfly.xml", "type": "blob", "sha": "a"},
        {"path": "flybody/fruitfly/assets/thorax.obj", "type": "blob", "sha": "b"},
        {"path": "flybody/fruitfly/assets/drosophila.blend", "type": "blob", "sha": "c"},
        {"path": "flybody/tasks/walk.py", "type": "blob", "sha": "d"},
    ]
    assert [e["path"].rsplit("/", 1)[-1] for e in flybody.wanted(tree)] == [
        "fruitfly.xml",
        "thorax.obj",
    ]


def test_a_file_that_does_not_match_its_git_hash_blocks_the_source(monkeypatch, tmp_path) -> None:
    tree = [{"path": "flybody/fruitfly/assets/fruitfly.xml", "type": "blob", "sha": "0" * 40}]
    monkeypatch.setattr(flybody, "listing", lambda: tree)

    def fake_fetch_many(items, raw_root, cached, **kwargs):
        out = []
        for url, rel in items:
            path = raw_root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"<mujoco/>")
            out.append(FileRecord(rel, url, "a" * 64, 9, "t"))
        return out

    monkeypatch.setattr(flybody, "fetch_many", fake_fetch_many)
    with pytest.raises(SourceBlocked, match="git hash"):
        flybody.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=False)


def test_the_body_is_optional() -> None:
    assert flybody.SOURCE in runner.OPTIONAL
