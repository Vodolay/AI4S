"""Trujillo et al. 2019: human cortical organoids on Axion Maestro MEAs.

The largest correction available to the asset's human/rat imbalance. Twelve
organoids per session, 64 electrodes each, 240 s at 12.5 kHz, recorded weekly
over ten months.

The deposit stores each session as a MATLAB v7.3 file of 1.0-1.5 GB whose bulk
is LFP traces this project does not use; the spike times inside are a few MB.
Zenodo serves this record at about 2.3 MB/s, so downloading all 42 files would
take some eight hours. Instead the HDF5 is read in place over HTTP range
requests and only the spike references are pulled: about 5 MB and 8 s per
session. What is written locally is the extracted spike table, and that is what
is hashed - the same arrangement `remote_zip.py` uses for the ToxCast member,
with the URL carrying a fragment saying which part of the remote file was read.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.fetch import zenodo
from neuromps.fetch.download import SourceBlocked
from neuromps.fetch.http_random import CachedRangeReader
from neuromps.fetch.manifest import FileRecord, Manifest, SourceRecord, utc_now

log = logging.getLogger("neuromps.fetch")

SOURCE = "trujillo2019"
RECORD_ID = 4751759
EXPECTED_LICENSE = "cc-by-4.0"
# Two sessions for the smoke path: one early, one late in the ten months.
QUICK_SESSIONS = ("LFP_Sp_160810.mat", "LFP_Sp_170207.mat")
META = {
    "landing_url": "https://doi.org/10.5281/zenodo.4751759",
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "citation": (
        "Trujillo CA, Gao R, Negraes PD, et al. Complex Oscillatory Waves Emerging from "
        "Cortical Organoids Model Early Human Brain Network Development. Cell Stem Cell "
        "25(4):558-569.e7 (2019). Dataset: doi:10.5281/zenodo.4751759"
    ),
}


def mat_version(url: str) -> str:
    """"7.3", "5.0" or "?" - read from the file's 128-byte text header.

    This deposit is mixed. Nineteen of the 42 sessions are MATLAB 5, which is
    not HDF5 and cannot be read in place at all: scipy has to have the whole
    file. One range request of 140 bytes tells us which kind we are looking at,
    before committing to either a fast read or a gigabyte download.
    """
    import requests

    from neuromps.fetch.download import USER_AGENT

    response = requests.get(
        url,
        headers={"Range": "bytes=0-140", "Accept-Encoding": "identity", "User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    header = response.content[:128].decode("latin-1", "replace")
    if "MATLAB 7.3" in header:
        return "7.3"
    if "MATLAB 5.0" in header:
        return "5.0"
    return "?"


def _is_empty(dataset) -> bool:
    """Whether a referenced cell holds a MATLAB empty array.

    MATLAB writes an empty array as a two-element dataset of zeros carrying a
    `MATLAB_empty` attribute. Both are checked: the attribute is the contract,
    the shape is what older writers actually produce.
    """
    if dataset.attrs.get("MATLAB_empty"):
        return True
    values = np.asarray(dataset).ravel()
    return values.size <= 2 and not values.any()


def session_spikes(handle) -> tuple[pd.DataFrame, float, float]:
    """Spike table, sample rate and duration for one open session file.

    Spike times are stored as sample indices into the recording's timebase, so
    the sample rate is recovered from that timebase rather than assumed.
    """
    timebase = handle["t_s"]
    n_samples = int(timebase.shape[1])
    duration_s = float(timebase[0, n_samples - 1])
    sample_rate_hz = (n_samples - 1) / duration_s
    references = handle["spikes"]
    n_electrodes, n_wells = references.shape
    wells, electrodes, times = [], [], []
    for well in range(n_wells):
        for electrode in range(n_electrodes):
            cell = handle[references[electrode, well]]
            if _is_empty(cell):
                continue
            indices = np.asarray(cell).ravel()
            wells.append(np.full(indices.size, well, dtype=np.int16))
            electrodes.append(np.full(indices.size, electrode, dtype=np.int16))
            times.append(indices.astype(np.float64) / sample_rate_hz)
    if not times:
        frame = pd.DataFrame({"well": [], "electrode": [], "time_s": []})
    else:
        frame = pd.DataFrame(
            {
                "well": np.concatenate(wells),
                "electrode": np.concatenate(electrodes),
                "time_s": np.concatenate(times).astype(np.float32),
            }
        )
    return frame, sample_rate_hz, duration_s


def extract_session(url: str, raw_root: Path, rel_path: str) -> tuple[FileRecord, int]:
    """Read one remote session's spikes and write them locally.

    Returns the provenance record for what was written and the number of bytes
    actually transferred, which is the point of doing it this way.
    """
    import h5py

    dest = Path(raw_root) / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    import pyarrow as pa
    import pyarrow.parquet as pq

    reader = CachedRangeReader(url)
    try:
        with h5py.File(reader, "r") as handle:
            frame, sample_rate_hz, duration_s = session_spikes(handle)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        # Duration belongs with the spikes: a rate computed from the last spike
        # time instead would be biased upwards, and by a different amount in
        # every recording.
        table = table.replace_schema_metadata(
            {
                **(table.schema.metadata or {}),
                b"duration_s": f"{duration_s}".encode(),
                b"sample_rate_hz": f"{sample_rate_hz}".encode(),
            }
        )
        pq.write_table(table, part)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        transferred = reader.bytes_fetched
        reader.close()
    payload = part.read_bytes()
    part.replace(dest)
    log.info(
        "  %s: %d spikes, %.1f s at %.0f Hz (%.1f MB read of %.0f MB remote)",
        rel_path,
        len(frame),
        duration_s,
        sample_rate_hz,
        transferred / 1e6,
        reader.size / 1e6,
    )
    return (
        FileRecord(
            path=rel_path,
            url=f"{url}#spikes",
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
            downloaded_at=utc_now(),
        ),
        transferred,
    )


def fetch(raw_root: Path, manifest: Manifest, full: bool) -> SourceRecord:
    """Extract spike times from every session (full) or two of them (quick)."""
    rec = zenodo.record(RECORD_ID)
    got = zenodo.license_id(rec)
    if got != EXPECTED_LICENSE:
        raise SourceBlocked(f"license changed upstream: expected {EXPECTED_LICENSE!r}, got {got!r}")
    available = zenodo.files(rec)
    sessions = sorted(name for name in available if name.endswith(".mat"))
    if not full:
        sessions = [name for name in sessions if name in QUICK_SESSIONS]
    missing = [name for name in sessions if name not in available]
    if missing:
        raise SourceBlocked(f"files missing from record {RECORD_ID}: {missing}")
    cached = manifest.file_index()
    files, transferred, remote_bytes = [], 0, 0
    skipped: list[str] = []
    for index, name in enumerate(sessions):
        rel = f"{SOURCE}/{Path(name).stem}.spikes.parquet"
        known = cached.get(rel)
        if known and (Path(raw_root) / rel).exists():
            files.append(known)
            continue
        version = mat_version(available[name].url)
        if version != "7.3":
            skipped.append(name)
            log.info(
                "  [%d/%d] %s: MATLAB %s, not readable in place - skipped (%.1f GB)",
                index + 1,
                len(sessions),
                name,
                version,
                available[name].size / 1e9,
            )
            continue
        log.info("  [%d/%d] %s", index + 1, len(sessions), name)
        record, read_bytes = extract_session(available[name].url, raw_root, rel)
        files.append(record)
        transferred += read_bytes
        remote_bytes += available[name].size
    if not files:
        raise SourceBlocked("no session in this tier could be read in place")
    saved = 1 - (transferred / remote_bytes) if remote_bytes else 0.0
    return SourceRecord(
        name=SOURCE,
        status="ok",
        files=files,
        note=(
            f"tier={'full' if full else 'quick'}; Zenodo record {RECORD_ID}; "
            f"{len(files)} sessions, spike times extracted in place over HTTP ranges "
            f"({transferred / 1e6:.0f} MB read of {remote_bytes / 1e6:.0f} MB remote, "
            f"{saved:.1%} not transferred); LFP traces not read; "
            f"{len(skipped)} sessions skipped as MATLAB 5 (not readable in place, "
            "29 GB of early timepoints); "
            "'_Drugs' sessions carry no well-to-compound mapping in the deposit"
        ),
        **META,
    )
