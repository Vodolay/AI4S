"""The Janelia/Cambridge/Google male CNS connectome v1.0, to replicate the fly study.

A second, independently reconstructed fly connectome (a male; FlyWire is a
female) with brain and nerve cord together. Three flat files are fetched from
the project's public bucket, about 1.1 GB: neuron-to-neuron synapse weights,
body annotations (type, class, region) and per-body transmitter predictions.
The project states "The Male CNS dataset is licensed under CC-BY" on its home
page; the bucket publishes an MD5 for each file, checked on download.

Optional, like `flywire783`: only the fly replication reads it.
"""

import base64
from pathlib import Path

from neuromps.fetch.common import fetch_many, retain_existing
from neuromps.fetch.download import SourceBlocked, get_json
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "malecns"
BUCKET = "flyem-male-cns"
PREFIX = "v1.0/connectome-data/flat-connectome/"
FILES = (
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
)
META = {
    "landing_url": "https://male-cns.janelia.org/",
    "license": "CC-BY-4.0 (stated on the project home page as 'licensed under CC-BY')",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "citation": (
        "Berg S, Beckett IR, Costa M, et al. Sexual dimorphism in the complete connectome of "
        "the Drosophila male central nervous system. Nature (2026); MaleCNS v1.0, "
        "https://male-cns.janelia.org/"
    ),
}


def _object(name: str) -> dict:
    quoted = (PREFIX + name).replace("/", "%2F")
    return get_json(
        f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{quoted}?fields=name,size,md5Hash"
    )


def md5_hex(md5_base64: str) -> str:
    """The bucket reports MD5 in base64; the downloader checks hex."""
    return base64.b64decode(md5_base64).hex()


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the three flat files, each checked against the bucket's MD5."""
    items, md5s = [], {}
    for name in FILES:
        meta = _object(name)
        if "md5Hash" not in meta:
            raise SourceBlocked(f"no MD5 published for {name}")
        rel = f"{SOURCE}/{name}"
        items.append((f"https://storage.googleapis.com/{BUCKET}/{PREFIX}{name}", rel))
        md5s[rel] = md5_hex(meta["md5Hash"])
    files = fetch_many(items, raw_root, manifest.file_index(), max_bytes=2 << 30, md5s=md5s)
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=retain_existing(SOURCE, manifest, raw_root, files),
        note="optional (fly replication only); v1.0 flat connectome, minconf 0.5",
        **META,
    )
