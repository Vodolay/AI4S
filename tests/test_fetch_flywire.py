"""The fly connectome is fetched only when asked for, from licensed, pinned copies.

The whole-brain model needs about 350 MB that no chip analysis uses, so a
default fetch must leave it alone. Every file comes from a source whose reuse
license is checked at fetch time (the two Zenodo records) or pinned with the
code (the MIT-licensed model repository, read at one commit).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuromps.fetch import flywire783, runner
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.manifest import FileRecord, Manifest


def _zenodo_record(license_id: str, names: tuple[str, ...]) -> dict:
    return {
        "metadata": {"license": {"id": license_id}},
        "files": [
            {
                "key": n,
                "links": {"self": f"https://zenodo.org/x/{n}"},
                "size": 1,
                "checksum": "md5:0",
            }
            for n in names
        ],
    }


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Answer the two Zenodo records and record every download instead of making it."""
    records = {
        flywire783.CONNECTIVITY_RECORD: _zenodo_record("cc-by-4.0", flywire783.CONNECTIVITY_FILES),
        flywire783.ANNOTATION_RECORD: _zenodo_record("cc-by-4.0", (flywire783.ANNOTATION_ZIP,)),
    }
    monkeypatch.setattr(flywire783.zenodo, "record", lambda rid: records[rid])
    calls: list[tuple[str, str]] = []

    def fake_fetch_many(items, raw_root, cached, **kwargs):
        calls.extend(items)
        return [FileRecord(rel, url, "0" * 64, 1, "t") for url, rel in items]

    monkeypatch.setattr(flywire783, "fetch_many", fake_fetch_many)
    monkeypatch.setattr(flywire783, "retain_existing", lambda s, m, r, files: files)
    return calls


def test_the_model_files_are_read_at_one_pinned_commit(offline, tmp_path: Path) -> None:
    flywire783.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=False)
    github = [url for url, _ in offline if "github" in url]
    assert github and all(flywire783.SHIU_COMMIT in url for url in github)
    assert {rel.rsplit("/", 1)[-1] for _, rel in offline} >= {
        "Connectivity_783.parquet",
        "Completeness_783.csv",
        "LICENSE",
        flywire783.ANNOTATION_ZIP.rsplit("/", 1)[-1],
        "per_neuron_neuropil_count_post_783.feather",
        "2023_03_23_connectivity_630_final.parquet",
    }


def test_a_changed_zenodo_license_blocks_the_source(offline, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(flywire783.zenodo, "record", lambda rid: _zenodo_record("cc-by-nc-4.0", ()))
    with pytest.raises(SourceBlocked, match="license"):
        flywire783.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=False)


def test_the_record_names_every_license_it_relies_on(offline, tmp_path: Path) -> None:
    record = flywire783.fetch(tmp_path, Manifest(tmp_path / "MANIFEST.json"), full=False)
    assert "CC-BY-4.0" in record.license and "MIT" in record.license
    assert "Shiu" in record.citation and "Dorkenwald" in record.citation


def test_a_default_fetch_leaves_the_fly_connectome_alone(monkeypatch, tmp_path) -> None:
    fetched: list[str] = []
    fake = {
        name: ((lambda n: lambda *a: fetched.append(n) or _ok(n))(name), meta)
        for name, (_, meta) in runner.registry().items()
    }
    monkeypatch.setattr(runner, "registry", lambda: fake)
    monkeypatch.setattr(runner, "excluded_records", lambda: [])
    runner.run_fetchers(tmp_path)
    assert flywire783.SOURCE not in fetched and fetched
    runner.run_fetchers(tmp_path, names=[flywire783.SOURCE])
    assert fetched[-1] == flywire783.SOURCE


def _ok(name: str):
    from neuromps.fetch.manifest import SourceRecord

    return SourceRecord(name, "skipped", "u", "", "", "", files=[])
