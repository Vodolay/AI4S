"""The spike trains behind each spike-time record of the asset, source by source.

The ETL modules read a deposit, extract features and keep only the features. A
consumer that wants the same recordings under another definition set
(`features/definitions.py`) needs the trains again, keyed by the asset's
``record_id``. This module re-reads them with the ETL modules' own loaders, so no
deposit is parsed twice in two ways; it only adds the record id each builder gives
and, for Kapucu 2022, the deposit's noisy-electrode list for that well.

Work comes in *units*, the smallest piece one loader reads at once (a plate, a
culture, an archive member, a session), so a caller can spread units over
processes. :func:`units` finds the units a tier has fetched, with the same checks
`etl.build` applies; :func:`load` turns one unit into its records.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.etl import brewer, kapucu, screenneuropharm, sharf, trujillo, vassallo


@dataclass(frozen=True)
class SpikeRecord:
    """One asset record's spike trains: electrode id -> spike times in seconds."""

    record_id: str
    source: str
    channels: Mapping[str, np.ndarray]
    duration_s: float
    noisy: tuple[str, ...] = ()


@dataclass(frozen=True)
class Unit:
    """One piece a loader reads at once; ``key`` is what the loader needs, picklable."""

    source: str
    loader: str
    key: tuple = field(default_factory=tuple)


# --- Kapucu 2022 pharmacology plates ------------------------------------------


def _kapucu_units(raw_root: Path, hashes: dict, full: bool) -> list[Unit]:
    """One unit per pharmacology plate recording on disk, as `etl.build` reads them."""
    data = raw_root / kapucu.SOURCE / "Data"
    return [
        Unit(kapucu.SOURCE, "kapucu", (i,))
        for i, plate in enumerate(kapucu.PHARMACOLOGY_PLATES)
        if (data / plate.directory).exists()
    ]


def _kapucu(unit: Unit, raw_root: Path) -> list[SpikeRecord]:
    """Every well of one plate recording, wells in the log without spikes included."""
    plate = kapucu.PHARMACOLOGY_PLATES[unit.key[0]]
    files = kapucu._files(raw_root / kapucu.SOURCE / "Data" / plate.directory)
    logged = set(kapucu.parse_explog(files["explog"])["well"])
    noisy = kapucu.parse_noisy_electrodes(files["noisy"].read_text()) if "noisy" in files else {}
    wells, duration = kapucu._load_spikes(files["spikes"])
    stem = Path(plate.directory).name.lower()
    return [
        SpikeRecord(
            f"{kapucu.SOURCE}:{stem}:{well.lower()}",
            kapucu.SOURCE,
            wells.get(well, {}),
            duration,
            tuple(noisy.get(well, [])),
        )
        for well in sorted(set(wells) | logged)
    ]


# --- archive-backed sources ---------------------------------------------------


def _fetched(raw_root: Path, source: str, name: str, hashes: dict) -> Path | None:
    """The archive, if this tier's manifest vouches for it (the build's own check)."""
    from neuromps.etl.build import fetched_in_this_tier

    archive = raw_root / source / name
    return archive if fetched_in_this_tier(archive, hashes, source) else None


def _brewer_units(raw_root: Path, hashes: dict, full: bool) -> list[Unit]:
    """One unit per culture; the quick tier builds two, as `etl.build` does."""
    archive = _fetched(raw_root, brewer.SOURCE, brewer.ARCHIVE_NAME, hashes)
    if archive is None:
        return []
    with zipfile.ZipFile(archive) as z:
        chosen = brewer.cultures(z)[: None if full else 2]
    return [Unit(brewer.SOURCE, "brewer", (c,)) for c in chosen]


def _brewer(unit: Unit, raw_root: Path) -> list[SpikeRecord]:
    """The four compartments of one culture.

    `brewer.build_records` gives every compartment the duration of the last one it
    read; that is kept here, so the reference features match the asset's exactly.
    """
    culture = unit.key[0]
    with zipfile.ZipFile(raw_root / brewer.SOURCE / brewer.ARCHIVE_NAME) as z:
        chambers = {}
        duration = 0.0
        for region in brewer.SUBREGIONS:
            chambers[region], duration = brewer.read_compartment(z, culture, region)
    device = culture.split()[0].lower()
    return [
        SpikeRecord(f"{brewer.SOURCE}:{device}:{region.lower()}", brewer.SOURCE, spikes, duration)
        for region, spikes in chambers.items()
    ]


def _sharf_units(raw_root: Path, hashes: dict, full: bool) -> list[Unit]:
    """One unit per organoid recording."""
    archive = _fetched(raw_root, sharf.SOURCE, sharf.ARCHIVE_NAME, hashes)
    if archive is None:
        return []
    with zipfile.ZipFile(archive) as z:
        return [Unit(sharf.SOURCE, "sharf", (m,)) for m in sharf.wanted_members(z)]


def _sharf(unit: Unit, raw_root: Path) -> list[SpikeRecord]:
    """One organoid recording, sorted units merged onto their electrodes."""
    import scipy.io as sio

    member = unit.key[0]
    with zipfile.ZipFile(raw_root / sharf.SOURCE / sharf.ARCHIVE_NAME) as z:
        raw = z.read(member)
    mat = sio.loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
    spikes, duration = sharf.spikes_by_electrode(mat)
    stem = Path(member).stem
    organoid, _, _ = sharf.parse_condition(stem, member.split("/")[1])
    return [
        SpikeRecord(f"{sharf.SOURCE}:{organoid}:{stem.lower()}", sharf.SOURCE, spikes, duration)
    ]


SNP_BATCH = 20


def _snp_units(raw_root: Path, hashes: dict, full: bool) -> list[Unit]:
    """Single-chamber recordings in batches, then one unit per multi-compartment device."""
    archive = _fetched(raw_root, screenneuropharm.SOURCE, screenneuropharm.ARCHIVE_NAME, hashes)
    if archive is None:
        return []
    with zipfile.ZipFile(archive) as z:
        single = screenneuropharm.discover_recordings(z)
        joint = screenneuropharm.discover_joint_devices(z)
    source = screenneuropharm.SOURCE
    batches = [tuple(single[i : i + SNP_BATCH]) for i in range(0, len(single), SNP_BATCH)]
    return [Unit(source, "snp_single", b) for b in batches] + [
        Unit(source, "snp_joint", (d,)) for d in joint
    ]


def _snp_single(unit: Unit, raw_root: Path) -> list[SpikeRecord]:
    """A batch of single-chamber recordings (array x condition)."""
    source = screenneuropharm.SOURCE
    out = []
    with zipfile.ZipFile(raw_root / source / screenneuropharm.ARCHIVE_NAME) as z:
        for rec in unit.key:
            spikes, duration = screenneuropharm._compartment_spikes(z, rec.members)
            record_id = f"{source}:{rec.unit_id.lower()}:{rec.condition.lower()}"
            out.append(SpikeRecord(record_id, source, spikes, duration))
    return out


def _snp_joint(unit: Unit, raw_root: Path) -> list[SpikeRecord]:
    """Every compartment of one multi-compartment device in one condition."""
    source, device = screenneuropharm.SOURCE, unit.key[0]
    out = []
    with zipfile.ZipFile(raw_root / source / screenneuropharm.ARCHIVE_NAME) as z:
        for colour, members in device.compartments.items():
            spikes, duration = screenneuropharm._compartment_spikes(z, members)
            record_id = f"{source}:{device.device_id.lower()}-{colour}:{device.condition.lower()}"
            out.append(SpikeRecord(record_id, source, spikes, duration))
    return out


# --- Mateus 2024: two chambers per recording ----------------------------------

MATEUS_ARCHIVE = "1_MEA_Recordings.zip"


def _mateus_units(raw_root: Path, hashes: dict, full: bool) -> list[Unit]:
    """One unit per recording file."""
    archive = _fetched(raw_root, "mateus2024", MATEUS_ARCHIVE, hashes)
    if archive is None:
        return []
    with zipfile.ZipFile(archive) as z:
        members = sorted(n for n in z.namelist() if n.endswith(".h5"))
    return [Unit("mateus2024", "mateus", (m,)) for m in members]


def _mateus(unit: Unit, raw_root: Path) -> list[SpikeRecord]:
    """The two chambers of one recording, split where `mateus2024_records` splits them."""
    from neuromps.etl import mateus2024_records as records
    from neuromps.etl.mateus2024 import chip_summary, read_recording
    from neuromps.models.microchannels import parse_filename

    member = unit.key[0]
    with (
        zipfile.ZipFile(raw_root / "mateus2024" / MATEUS_ARCHIVE) as z,
        tempfile.TemporaryDirectory() as tmp,
    ):
        target = Path(tmp) / "recording.h5"
        target.write_bytes(z.read(member))
        spikes, duration_s, _ = read_recording(target)
    band = chip_summary(spikes, duration_s)["channel_rows"]
    if not band:
        return []
    above, _, below = records.chambers(list(spikes), tuple(band))
    if not above or not below:
        return []
    parts = member.split("/")
    meta = {"experiment": parts[2], "condition": parts[3].lower(), **parse_filename(parts[-1])}
    return [
        SpikeRecord(
            records.record_id(
                str(meta.get("experiment", "exp")),
                str(meta.get("chip", "chip")),
                str(meta.get("condition", "unknown")),
                f"div{meta.get('div')}",
                name,
            ),
            records.SOURCE,
            {label: spikes[label] for label in labels},
            duration_s,
        )
        for name, labels in (("source", above), ("target", below))
    ]


# --- Trujillo 2019 and Vassallo 2017 ------------------------------------------


def _trujillo_units(raw_root: Path, hashes: dict, full: bool) -> list[Unit]:
    """One unit per session file this tier's manifest lists."""
    paths = sorted((raw_root / trujillo.SOURCE).glob("*.spikes.parquet"))
    return [Unit(trujillo.SOURCE, "trujillo", (p.name,)) for p in paths if p.name in hashes]


def _trujillo(unit: Unit, raw_root: Path) -> list[SpikeRecord]:
    """One organoid per well of one session."""
    path = raw_root / trujillo.SOURCE / unit.key[0]
    session, _ = trujillo.session_id(path)
    frame, duration_s = trujillo.read_session(path)
    return [
        SpikeRecord(
            f"{trujillo.SOURCE}:{session}:w{int(well):02d}",
            trujillo.SOURCE,
            trujillo.spikes_by_electrode(rows),
            duration_s,
        )
        for well, rows in frame.groupby("well", sort=True)
    ]


def _vassallo_units(raw_root: Path, hashes: dict, full: bool) -> list[Unit]:
    """One unit per logged chip with exactly one spike file, as `vassallo.build_records`."""
    archive = _fetched(raw_root, vassallo.SOURCE, vassallo.ARCHIVE_NAME, hashes)
    if archive is None:
        return []
    with zipfile.ZipFile(archive) as z:
        members = [n for n in z.namelist() if n.endswith(".txt") and "--" in n]
        entries = vassallo.read_logfile(z)
    by_id: dict[tuple[str, str], list[str]] = {}
    for member in members:
        by_id.setdefault(vassallo.session_and_chip(member), []).append(member)
    units = []
    for entry in entries:
        candidates = by_id.get(vassallo.session_and_chip(entry["experiment"]), [])
        if len(candidates) == 1:
            units.append(Unit(vassallo.SOURCE, "vassallo", (entry, candidates[0])))
    return units


def _vassallo(unit: Unit, raw_root: Path) -> list[SpikeRecord]:
    """Every logged window of one chip, times measured from the window's start."""
    entry, member = unit.key
    with zipfile.ZipFile(raw_root / vassallo.SOURCE / vassallo.ARCHIVE_NAME) as z:
        table: pd.DataFrame = vassallo.spike_table(z, member)
    session, chip = vassallo.session_and_chip(entry["experiment"])
    return [
        SpikeRecord(
            f"{vassallo.SOURCE}:{session}-{chip}:{window['label']}",
            vassallo.SOURCE,
            vassallo.window_spikes(table, window["start_s"], window["end_s"]),
            float(window["end_s"] - window["start_s"]),
        )
        for window in entry["windows"]
    ]


# --- the registry ---------------------------------------------------------------

UNIT_FINDERS: tuple[Callable[[Path, dict, bool], list[Unit]], ...] = (
    _kapucu_units,
    _snp_units,
    _sharf_units,
    _mateus_units,
    _trujillo_units,
    _brewer_units,
    _vassallo_units,
)
LOADERS: dict[str, Callable[[Unit, Path], list[SpikeRecord]]] = {
    "kapucu": _kapucu,
    "brewer": _brewer,
    "sharf": _sharf,
    "snp_single": _snp_single,
    "snp_joint": _snp_joint,
    "mateus": _mateus,
    "trujillo": _trujillo,
    "vassallo": _vassallo,
}


def units(raw_root: Path, hashes: dict[str, str], full: bool) -> list[Unit]:
    """Every unit this tier has fetched, sources in the build's order."""
    return [unit for find in UNIT_FINDERS for unit in find(Path(raw_root), hashes, full)]


def load(unit: Unit, raw_root: Path) -> list[SpikeRecord]:
    """The records one unit holds."""
    return LOADERS[unit.loader](unit, Path(raw_root))
