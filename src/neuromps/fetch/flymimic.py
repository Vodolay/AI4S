"""The FlyMimic musculoskeletal fly leg, for the embodied fly (report/ROADMAP_EMBODIED_FLY.md).

FlyMimic (Özdil et al., 2026) is a MuJoCo model of the *Drosophila* left front
leg driven by 15 Hill-type muscles named after the fly's leg muscles, shipped
with FlyGym 2 (NeLy-EPFL; Apache-2.0; the model file's own header states
"This model is licensed under Apache 2.0"). Fetched, all from pinned copies:

- from FlyGym's GitHub at one commit: the model file, the license, and the one
  motion-capture clip of a real fly's front leg that FlyGym ships (500 Hz);
- from the project's public bucket: the mesh archive, checked against the
  SHA-256 the bucket publishes beside it.

Optional: only the embodied-fly work reads it.
"""

from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath

from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import SourceBlocked, get_with_retry
from neuromps.fetch.manifest import Manifest, SourceRecord
from neuromps.qc.provenance import sha256_file

SOURCE = "flymimic"
FLYGYM_COMMIT = "a63dc91a7ce432452e6521c2f3dcc1a7c2c7c6bc"
RAW = f"https://raw.githubusercontent.com/NeLy-EPFL/flygym/{FLYGYM_COMMIT}"
MODEL_PATH = "src/flygym/assets/model/musculoskeletal"
MODEL_NAME = "best_combined_arm_damping_stiff_cvt3.xml"
MOCAP_PATH = "src/flygym_demo/muscle_imitation/assets/mocap"
MOCAP_ARRAYS = ("qpos", "qvel", "xipos", "xivel")
MOCAP_CLIP = "0002"
BUCKET = "https://datasets.epfl.ch/nely-public-share/flygym_assets"
MESH_ARCHIVE = "neuromechfly_musculoskeletal_meshes_20260623a"
META = {
    "landing_url": "https://github.com/NeLy-EPFL/flygym",
    "license": "Apache-2.0 (FlyGym; the FlyMimic model file states Apache 2.0)",
    "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
    "citation": (
        "Özdil PG, et al. FlyMimic: musculoskeletal model of Drosophila limb movement (2026), as "
        "shipped in FlyGym 2 (NeuroMechFly v2; Wang-Chen S, et al. Nature Methods 21, 2353-2362 "
        "(2024)); github.com/NeLy-EPFL/flygym at commit " + FLYGYM_COMMIT[:12]
    ),
}


def published_sha256() -> str:
    """The SHA-256 the bucket publishes for the mesh archive."""
    response = get_with_retry(f"{BUCKET}/{MESH_ARCHIVE}.checksum", timeout=60)
    return response.text.split()[0].strip()


def _items() -> list[tuple[str, str]]:
    items = [
        (f"{RAW}/{MODEL_PATH}/{MODEL_NAME}", f"{SOURCE}/{MODEL_NAME}"),
        (f"{RAW}/LICENSE", f"{SOURCE}/LICENSE"),
    ]
    items += [
        (f"{RAW}/{MOCAP_PATH}/{a}/{MOCAP_CLIP}.npy", f"{SOURCE}/mocap/{a}_{MOCAP_CLIP}.npy")
        for a in MOCAP_ARRAYS
    ]
    items.append((f"{BUCKET}/{MESH_ARCHIVE}.tar", f"{SOURCE}/{MESH_ARCHIVE}.tar"))
    return items


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the model, license, motion-capture clip and checksummed mesh archive."""
    expected = published_sha256()
    files = fetch_many(_items(), raw_root, manifest.file_index(), max_bytes=1 << 30)
    archive = next(f for f in files if f.path.endswith(".tar"))
    if archive.sha256 != expected:
        (Path(raw_root) / archive.path).unlink(missing_ok=True)
        raise SourceBlocked(
            f"mesh archive checksum {archive.sha256} does not match the published {expected}"
        )
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note=f"optional (embodied fly only); FlyGym commit {FLYGYM_COMMIT[:12]}; "
        f"mesh archive {MESH_ARCHIVE} checked against its published SHA-256",
        **META,
    )


def extract_meshes(archive: Path, dest: Path) -> list[Path]:
    """Unpack only the .stl/.obj files of the archive, refusing any path that leaves `dest`."""
    dest = Path(dest)
    out = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            if not member.isfile() or name.suffix.lower() not in (".stl", ".obj"):
                continue
            if name.is_absolute() or ".." in name.parts:
                continue
            target = dest / Path(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source:
                target.write_bytes(source.read())
            out.append(target)
    return out


def check_archive(path: Path, expected: str) -> bool:
    """Whether a local archive still matches the published checksum."""
    return sha256_file(path) == expected
