"""The flybody whole-fly body model, for stage 2 of the embodied fly.

flybody (Vaxenburg et al., Nature 2025; TuragaLab and Google DeepMind; Apache-2.0)
is a MuJoCo model of the whole adult fly: a free body, six legs with tarsal force
and touch sensors, wings, and an actuated proboscis. Only its model file and
meshes are fetched (not the Blender source), from GitHub at one pinned commit;
GitHub's tree listing gives every file's git blob hash, and each download is
checked against it. Loaded with plain MuJoCo, so the package (which pins an old
numpy) is not installed. Optional: only the embodied-fly work reads it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import SourceBlocked, get_json
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "flybody"
COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"
REPO = "TuragaLab/flybody"
ASSETS = "flybody/fruitfly/assets/"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"
SUFFIXES = (".xml", ".obj")
META = {
    "landing_url": "https://github.com/TuragaLab/flybody",
    "license": "Apache-2.0",
    "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
    "citation": (
        "Vaxenburg R, Siwanowicz I, Merel J, et al. Whole-body physics simulation of fruit fly "
        "locomotion. Nature (2025); flybody model, github.com/TuragaLab/flybody at commit "
        + COMMIT[:12]
    ),
}


def git_blob_sha1(body: bytes) -> str:
    """The hash git stores for a file's content."""
    return hashlib.sha1(b"blob %d\x00" % len(body) + body).hexdigest()


def listing() -> list[dict]:
    """The repository tree at the pinned commit."""
    return get_json(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1")["tree"]


def wanted(tree: list[dict]) -> list[dict]:
    """The model file and meshes; not the Blender source or anything outside the assets."""
    return [
        e
        for e in tree
        if e["type"] == "blob" and e["path"].startswith(ASSETS) and e["path"].endswith(SUFFIXES)
    ]


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the model and meshes and the license, each checked against its git hash."""
    entries = wanted(listing())
    if not entries:
        raise SourceBlocked(f"no model files under {ASSETS} at {COMMIT[:12]}")
    items = [(f"{RAW}/{e['path']}", f"{SOURCE}/assets/{e['path'][len(ASSETS) :]}") for e in entries]
    items.append((f"{RAW}/LICENSE", f"{SOURCE}/LICENSE"))
    files = fetch_many(items, raw_root, manifest.file_index(), max_bytes=1 << 30)
    expected = {f"{SOURCE}/assets/{e['path'][len(ASSETS) :]}": e["sha"] for e in entries}
    for record in files:
        want = expected.get(record.path)
        if want and git_blob_sha1((Path(raw_root) / record.path).read_bytes()) != want:
            (Path(raw_root) / record.path).unlink(missing_ok=True)
            raise SourceBlocked(f"{record.path} does not match its git hash {want}")
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note=f"optional (embodied fly only); commit {COMMIT[:12]}; model file and meshes, "
        "each checked against its git blob hash; Blender source not fetched",
        **META,
    )
