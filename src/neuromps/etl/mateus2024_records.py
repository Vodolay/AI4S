"""Schema records for the Mateus deposit: two chambers per recording.

Each chip carries straight or asymmetric microchannels joining two chambers,
with electrodes inside the channels. `etl/mateus2024.py` locates the channels
from the spike times; the rows they occupy split the array into the chamber
above and the chamber below, and everything outside the band belongs to one or
the other.

The chambers are named ``source`` and ``target`` to match the deposit's own
vocabulary, but **the names carry no direction**: the deposit does not record how
each device was seated on the array, so which physical chamber is which is not
knowable from the data. The names are labels for two sides, and the device block
says so.

Features come from `features/mea.py` and coupling from `features/coupling.py`,
the same functions the other sources and the twin use.
"""

from __future__ import annotations

import logging

import numpy as np

from neuromps.etl.mateus2024 import grid_position
from neuromps.features.coupling import CouplingParams, directed_coupling
from neuromps.features.mea import FeatureParams, extract_features
from neuromps.fetch.manifest import utc_now

log = logging.getLogger("neuromps.etl")

SOURCE = "mateus2024"
COMPARTMENTS = ("source", "target")
LICENSE = "CC-BY-4.0"
CITATION = (
    "Mateus J, Melo P, Aroso M, Charlot B, Aguiar P. Influence of asymmetric microchannels in "
    "the structure and function of engineered neuronal circuits. bioRxiv 2024, "
    "doi:10.1101/2024.07.09.602729. Dataset: doi:10.5281/zenodo.14525182 (CC BY 4.0)."
)

CONDITION_GEOMETRY = {
    "control": {"asymmetric": False, "reads_as": "straight microchannels"},
    "arrows": {"asymmetric": True, "reads_as": "arrow-shaped constrictions"},
    "rams": {"asymmetric": True, "reads_as": "ram-horn constrictions"},
    "tesla": {"asymmetric": True, "reads_as": "Tesla-valve channels"},
    "tesla_v2": {"asymmetric": True, "reads_as": "Tesla-valve channels, second design"},
}


def compartment_of(label: str, band: tuple[int, int]) -> str | None:
    """Which side of the microchannel band an electrode sits on."""
    position = grid_position(label)
    if position is None:
        return None
    row = position[1]
    if row < band[0]:
        return "source"
    if row > band[1]:
        return "target"
    return "microchannel"


def chambers(labels: list[str], band: tuple[int, int]) -> tuple[list[str], list[str], list[str]]:
    """Electrode labels above the band, inside it, and below it."""
    above, inside, below = [], [], []
    for label in sorted(labels, key=lambda name: grid_position(name) or (99, 99)):
        where = compartment_of(label, band)
        if where == "source":
            above.append(label)
        elif where == "target":
            below.append(label)
        elif where == "microchannel":
            inside.append(label)
    return above, inside, below


def record_id(experiment: str, chip: str, condition: str, div: str, compartment: str) -> str:
    """A stable identifier per recording and compartment."""
    return f"{SOURCE}:{experiment.lower()}-{chip}-{condition}-{str(div).lower()}:{compartment}"


def records_for_recording(
    spikes: dict[str, np.ndarray],
    duration_s: float,
    band: tuple[int, int] | None,
    meta: dict,
    sha: str,
) -> list[dict]:
    """Two records, one per chamber, or none when no microchannels were found.

    A recording whose channels could not be located has no defensible chamber
    split, and inventing one would put made-up geometry into the asset.
    """
    if band is None:
        return []
    above, inside, below = chambers(list(spikes), band)
    if not above or not below:
        return []
    sides = {"source": above, "target": below}
    trains = {name: {label: spikes[label] for label in labels} for name, labels in sides.items()}
    coupling_params = CouplingParams()
    records = []
    for name, labels in sides.items():
        other = "target" if name == "source" else "source"
        features = extract_features(trains[name], duration_s)
        measured = directed_coupling(trains[name], trains[other], duration_s, coupling_params)
        records.append(
            _record(
                name,
                other,
                labels,
                inside,
                features,
                measured,
                duration_s,
                meta,
                sha,
                coupling_params,
            )
        )
    return records


def _record(
    name: str,
    other: str,
    labels: list[str],
    inside: list[str],
    features: dict,
    measured: dict,
    duration_s: float,
    meta: dict,
    sha: str,
    coupling_params: CouplingParams,
) -> dict:
    """One compartment record in the project schema."""
    condition = str(meta.get("condition", "unknown"))
    geometry = CONDITION_GEOMETRY.get(condition, {"asymmetric": None, "reads_as": condition})
    active = features.get("active_electrodes") or 0
    return {
        "record_id": record_id(
            str(meta.get("experiment", "exp")),
            str(meta.get("chip", "chip")),
            condition,
            f"div{meta.get('div')}",
            name,
        ),
        "source": SOURCE,
        "license": LICENSE,
        "citation": CITATION,
        "data_level": "spike_times",
        "cell_source": {
            "species": "rat",
            "system": "primary_hippocampal",
            "line_or_strain": "rat hippocampal neurons, Mateus et al. 2024",
        },
        "platform": {
            "vendor": "Multi Channel Systems",
            "system": "MEA2100-256",
            "plate_format": "single-well",
            "electrodes_per_unit": len(labels),
            "sampling_rate_hz": 20000.0,
        },
        "plate_id": str(meta.get("experiment", "")),
        "unit_id": f"{meta.get('experiment', '')}-{meta.get('chip', '')}",
        "culture_age_days": meta.get("div"),
        "compound": {
            "name": "none",
            "role": "untreated_control",
            "mechanism_class": None,
            "dose_value": None,
            "dose_unit": None,
        },
        "recording": {"duration_s": duration_s, "phase": "baseline"},
        "features": features,
        "feature_method": {
            "computed_by": "neuromps.features.mea",
            **FeatureParams().as_method_record(),
            **coupling_params.as_method_record(),
            # The spikes are the depositors', not ours: they were detected before
            # publication and the deposit contains no raw traces to redetect from.
            "spike_detection": (
                "publisher: 200 Hz high-pass, threshold 5x SD of peak-to-peak electrode noise, "
                "3 ms dead time (Mateus et al. 2024 methods)"
            ),
        },
        "device": {
            "device_id": f"{meta.get('experiment', '')}-{meta.get('chip', '')}",
            "layout": "two_chamber_microchannel",
            "compartment_id": name,
            "compartment_role": "chamber",
            "n_compartments": 2,
            "microchannel_geometry": condition,
            "asymmetric_microchannels": geometry["asymmetric"],
            "n_in_channel_electrodes": len(inside),
        },
        "coupling": [
            {
                "to_compartment": other,
                "method": coupling_params.as_method_record()["coupling_method"],
                **measured,
            }
        ],
        "qc": {
            "usable": bool(active >= 10),
            "flags": [] if active >= 10 else ["too_few_active_electrodes"],
            "n_noisy_electrodes": None,
            "note": (
                "chambers assigned from the microchannel band recovered from spike times; "
                "compartment names are labels for two sides and carry no direction"
            ),
        },
        "provenance": {
            "source_files": [{"path": f"{SOURCE}/1_MEA_Recordings.zip", "sha256": sha}],
            "generated_by": "neuromps.etl.mateus2024_records",
            "generated_at": utc_now(),
            "transform": [
                f"read {meta.get('file', '')} from the archive",
                "locate microchannels by the deposit's 5-electrode, <=1 ms criterion",
                "split electrodes into the chambers either side of the channel band",
                "features/mea.py extract_features; features/coupling.py directed_coupling",
            ],
        },
    }


def build_records(archive_path, sha: str | None = None, cache_path=None, rebuild: bool = False):
    """Every recording in the deposit as two chamber records.

    Cached, because locating the microchannels means reading 57 recordings.
    """
    import json
    import tempfile
    import zipfile
    from pathlib import Path

    from neuromps.etl.mateus2024 import chip_summary, read_recording
    from neuromps.fetch.download import sha256_file
    from neuromps.models.microchannels import parse_filename

    archive_path = Path(archive_path)
    if cache_path and Path(cache_path).exists() and not rebuild:
        return json.loads(Path(cache_path).read_text())
    digest = sha or sha256_file(archive_path)
    records = []
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory() as tmp:
        members = sorted(n for n in archive.namelist() if n.endswith(".h5"))
        log.info("mateus2024: building records from %d recordings", len(members))
        for member in members:
            parts = member.split("/")
            target = Path(tmp) / "recording.h5"
            target.write_bytes(archive.read(member))
            spikes, duration_s, unreadable = read_recording(target)
            summary = chip_summary(spikes, duration_s)
            band = summary["channel_rows"]
            meta = {
                "experiment": parts[2],
                "condition": parts[3].lower(),
                "file": parts[-1],
                **parse_filename(parts[-1]),
            }
            for record in records_for_recording(
                spikes, duration_s, tuple(band) if band else None, meta, digest
            ):
                if unreadable:
                    record["qc"]["flags"] = [*record["qc"]["flags"], "unreadable_channels"]
                    record["qc"]["n_noisy_electrodes"] = len(unreadable)
                records.append(record)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(records, default=str))
    return records
