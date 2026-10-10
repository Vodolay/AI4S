"""A checksummed fallback copy of a source's files, for when its host is down.

The archive holds the files byte for byte as the original host serves them, so
the provenance hashes in MANIFEST.json describe either copy. It is trusted only
after its own SHA-256 matches the value pinned in the fetcher, and only the paths
the fetcher asks for are ever unpacked.

Build one with ``python -m neuromps.fetch.mirror``, which writes the Kapucu quick
tier's archive from files already verified against the committed manifest.
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from collections.abc import Callable
from pathlib import Path

from neuromps.fetch.download import download_file
from neuromps.fetch.manifest import FileRecord, utc_now
from neuromps.qc.provenance import sha256_file

log = logging.getLogger("neuromps.fetch")
README_NAME = "MIRROR_README.txt"
# A fixed timestamp and mode, so the same files always produce the same archive
# and the pinned hash can be re-derived by anyone holding them.
_EPOCH = (1980, 1, 1, 0, 0, 0)


class MirrorMismatch(RuntimeError):
    """The archive is not the one pinned, or lacks a file it should hold."""


def build_zip(root: Path, rel_paths: list[str], dest: Path, readme: str) -> str:
    """Write ``rel_paths`` (relative to ``root``) and a readme into ``dest``; return its SHA-256."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for rel in sorted(rel_paths):
            _add(archive, rel, (Path(root) / rel).read_bytes())
        _add(archive, README_NAME, readme.encode())
    return sha256_file(dest)


def _add(archive: zipfile.ZipFile, name: str, body: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, body)


def fetch_members(
    url: str,
    sha256: str,
    raw_root: Path,
    wanted: list[str],
    cached: dict[str, FileRecord],
    download: Callable[..., FileRecord] = download_file,
) -> list[FileRecord]:
    """Download the archive, check it against ``sha256``, and unpack ``wanted``.

    Files already on disk with the hash the manifest records are kept as they
    are. Each unpacked file is recorded with its own hash and with the archive's
    URL plus the member path, so the manifest says where the bytes came from.
    """
    raw_root = Path(raw_root)
    rel_archive = f"{wanted[0].split('/')[0]}/.mirror/{url.rsplit('/', 1)[-1]}"
    archive_path = raw_root / rel_archive
    try:
        fetched = download(url, raw_root, rel_archive, max_bytes=2 << 30)
        if fetched.sha256 != sha256:
            raise MirrorMismatch(f"{url} has SHA-256 {fetched.sha256}, pinned {sha256}")
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            missing = [rel for rel in wanted if rel not in names]
            if missing:
                raise MirrorMismatch(f"{url} is missing {len(missing)} file(s): {missing[:3]}")
            return [_unpack(archive, rel, raw_root, url, cached.get(rel)) for rel in wanted]
    finally:
        archive_path.unlink(missing_ok=True)


def _unpack(
    archive: zipfile.ZipFile, rel: str, raw_root: Path, url: str, cached: FileRecord | None
) -> FileRecord:
    """One member, written whole or not at all."""
    dest = raw_root / rel
    if cached and dest.exists() and sha256_file(dest) == cached.sha256:
        return cached
    body = archive.read(rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(body)
    part.replace(dest)
    digest = hashlib.sha256(body).hexdigest()
    return FileRecord(rel, f"{url}#{rel}", digest, len(body), utc_now())


def main(argv: list[str] | None = None) -> int:
    """CLI: build the Kapucu quick-tier archive from verified local files."""
    import argparse
    import json

    from neuromps.fetch import kapucu2022
    from neuromps.fetch.runner import RAW_ROOT
    from neuromps.paths import REPO_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    committed = json.loads((REPO_ROOT / "data/raw/MANIFEST.json").read_text())
    expected = {f["path"]: f["sha256"] for f in committed["sources"][kapucu2022.SOURCE]["files"]}
    wanted = [f"{kapucu2022.SOURCE}/{p}" for p in kapucu2022.file_list(False)]
    wrong = [rel for rel in wanted if sha256_file(RAW_ROOT / rel) != expected.get(rel)]
    if wrong:
        raise SystemExit(f"{len(wrong)} file(s) differ from the committed manifest: {wrong[:3]}")
    digest = build_zip(RAW_ROOT, wanted, args.out, kapucu2022.MIRROR_README)
    print(f"{args.out}  {len(wanted)} files  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
