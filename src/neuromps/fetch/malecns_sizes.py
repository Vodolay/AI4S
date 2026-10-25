"""Neuron sizes (voxel counts) of the male CNS v1.0, for the rate model's size rule.

The flat connectome files carry no neuron volumes. The project's neuPrint input table
(`Neuprint_Neurons.feather`, same bucket, same CC-BY release) has a ``size`` column per
body. The bucket stores it as a composite object and publishes a CRC32C, not an MD5, so
the file is checked against that CRC32C after download. About 4.65 GB.

Optional: only the embodied stages that use the volume size rule read it.
"""

from pathlib import Path

from neuromps.fetch.crc32c import crc32c_file, from_gcs
from neuromps.fetch.download import SourceBlocked, download_file, get_json
from neuromps.fetch.malecns import BUCKET
from neuromps.fetch.malecns import META as MALECNS_META
from neuromps.fetch.manifest import Manifest, SourceRecord

SOURCE = "malecns_sizes"
OBJECT = "v1.0/database/neuprint-inputs/Neuprint_Neurons.feather"
REL = f"{SOURCE}/Neuprint_Neurons.feather"
MAX_BYTES = 6 << 30
META = dict(MALECNS_META)


def published() -> dict:
    """The bucket's size, generation and CRC32C for the object."""
    quoted = OBJECT.replace("/", "%2F")
    return get_json(
        f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{quoted}"
        "?fields=size,generation,crc32c"
    )


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Download the neuron table and check it against the bucket's CRC32C."""
    meta = published()
    if "crc32c" not in meta:
        raise SourceBlocked(f"no CRC32C published for {OBJECT}")
    url = f"https://storage.googleapis.com/{BUCKET}/{OBJECT}"
    cached = manifest.file_index().get(REL)
    record = download_file(url, raw_root, REL, cached=cached, max_bytes=MAX_BYTES)
    if record is not cached:
        got, want = crc32c_file(raw_root / REL), from_gcs(meta["crc32c"])
        if got != want:
            (raw_root / REL).unlink(missing_ok=True)
            raise SourceBlocked(f"CRC32C mismatch for {OBJECT}: {got:08x} != {want:08x}")
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=[record],
        note=(
            f"optional (embodied volume size rule); {OBJECT}, generation {meta['generation']}, "
            f"checked against the bucket's CRC32C {meta['crc32c']}"
        ),
        **META,
    )
