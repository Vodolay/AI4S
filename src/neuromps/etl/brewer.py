"""Brewer 2021 hippocampal subregion recordings -> compartment records.

Ten cultures, each grown in a four-chambered device over a 60-electrode Multi
Channel Systems array, recorded for 300 s of spontaneous activity at 25 kHz.
Each chamber holds one hippocampal subregion and is instrumented separately, so
a record here is one compartment and carries measured coupling to the other
three.

Two cautions, both from reading the files rather than the paper:

- **The anatomical labels are not trustworthy.** Only 4 of the 40 subregion
  files carry an internal `par.subregion` field at all, and 2 of those 4
  disagree with the file name they are stored under. The file name is the only
  labelling that exists for all ten cultures, so it is what `compartment_id`
  records - and every record carries a QC note saying the anatomy is the
  depositors' naming and was not verified. Nothing in this project's analyses
  depends on which subregion is which; what they need is that the compartments
  are distinct and instrumented, which the electrode coordinates confirm.
- Spikes are stored as sparse vectors over the full 7.5 M-sample recording,
  holding the spike amplitude at the sample of the spike. The non-zero indices
  are the spike times; the amplitudes are not used.
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

from neuromps.features.coupling import directed_coupling
from neuromps.features.mea import FeatureParams, extract_features
from neuromps.fetch.manifest import utc_now
from neuromps.qc.provenance import sha256_file

log = logging.getLogger("neuromps.etl")
SOURCE = "brewer2021"
ARCHIVE_NAME = "spike_trains_data.zip"
REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = REPO_ROOT / "data" / "raw" / SOURCE / ARCHIVE_NAME
SUBREGIONS = ("EC", "DG", "CA3", "CA1")
MIN_ACTIVE_ELECTRODES = 5
FEATURE_KEYS = (
    "mfr_hz",
    "burst_rate_per_min",
    "burst_duration_s",
    "spikes_per_burst",
    "pct_spikes_in_bursts",
    "network_burst_rate_per_min",
    "synchrony_index",
    "isi_cv",
    "active_electrodes",
)
CITATION = (
    "Vakilna YS, Tang WC, Wheeler BC, Brewer GJ. The flow of axonal information among "
    "hippocampal subregions: 1. Feedforward and feedback network spatial dynamics. "
    "Dataset doi:10.5281/zenodo.4679064 (CC BY 4.0)."
)
PLATFORM = {
    "vendor": "Multi Channel Systems",
    "system": "60-electrode planar MEA, four-chamber device with instrumented tunnels",
    "plate_format": "single-well",
    "electrodes_per_unit": 60,
    "sampling_rate_hz": 25000.0,
    "electrode_pitch_um": None,
}
CELL_SOURCE = {
    "species": "rat",
    "system": "primary_hippocampal",
    "line_or_strain": None,
    "differentiation_protocol": None,
    "note": "subregion cultures (EC, DG, CA3, CA1) in separate chambers of one device",
}


def cultures(archive: zipfile.ZipFile) -> list[str]:
    """Culture directory names inside the archive."""
    names = {n.split("/")[1] for n in archive.namelist() if n.count("/") >= 2 and n.split("/")[1]}
    return sorted(names)


def _cell(entry) -> object:
    """Unwrap however scipy handed back one channel's sparse vector."""
    if isinstance(entry, np.ndarray) and entry.dtype == object:
        return entry.reshape(-1)[0]
    return entry


def read_compartment(archive: zipfile.ZipFile, culture: str, region: str) -> tuple[dict, float]:
    """Spike times per electrode (seconds) and the recording duration."""
    member = f"spike_trains_data/{culture}/spikes/mea_{region}.mat"
    mea = sio.loadmat(
        io.BytesIO(archive.read(member)), squeeze_me=True, struct_as_record=False
    )["mea"]
    sample_rate_hz = float(mea.par.fs)
    names = [str(n) for n in np.atleast_1d(mea.channel_names)]
    spikes, n_samples = {}, 0
    for name, entry in zip(names, np.atleast_1d(mea.spike_data), strict=False):
        vector = _cell(entry)
        n_samples = max(n_samples, int(vector.shape[0]))
        indices = vector.nonzero()[0]
        if indices.size:
            spikes[name] = np.sort(indices.astype(float) / sample_rate_hz)
    return spikes, n_samples / sample_rate_hz


def build_records(
    archive_path: Path = ARCHIVE,
    hashes: dict[str, str] | None = None,
    params: FeatureParams | None = None,
    max_cultures: int | None = None,
) -> list[dict]:
    """One record per compartment, with coupling to the other three.

    `max_cultures` limits the quick tier: each culture costs four feature
    extractions and twelve cross-correlations, and the smoke path needs the code
    exercised rather than the dataset covered.
    """
    p = params or FeatureParams()
    archive_path = Path(archive_path)
    sha = (hashes or {}).get(f"{SOURCE}/{archive_path.name}") or sha256_file(archive_path)
    records: list[dict] = []
    with zipfile.ZipFile(archive_path) as archive:
        chosen = cultures(archive)[: max_cultures or None]
        for culture in chosen:
            device_id = culture.split()[0]
            chambers = {}
            duration_s = 0.0
            for region in SUBREGIONS:
                spikes, duration_s = read_compartment(archive, culture, region)
                chambers[region] = spikes
            for region, spikes in chambers.items():
                records.append(
                    _record(region, spikes, chambers, culture, device_id, duration_s, sha, p)
                )
            log.info(
                "  brewer %s: %d compartments, %d spikes",
                device_id,
                len(chambers),
                sum(len(t) for s in chambers.values() for t in s.values()),
            )
    return records


def _record(
    region: str,
    spikes: dict,
    chambers: dict,
    culture: str,
    device_id: str,
    duration_s: float,
    sha: str,
    p: FeatureParams,
) -> dict:
    """One compartment's record."""
    features = extract_features(spikes, duration_s, p)
    active = features.get("active_electrodes") or 0
    coupling = [
        {
            "to_compartment": other,
            "method": "cross-correlation of 10 ms population rates, +/- 200 ms",
            **directed_coupling(spikes, chambers[other], duration_s),
        }
        for other in SUBREGIONS
        if other != region
    ]
    return {
        "record_id": f"{SOURCE}:{device_id.lower()}:{region.lower()}",
        "source": SOURCE,
        "license": "CC-BY-4.0",
        "citation": CITATION,
        "data_level": "spike_times",
        "cell_source": dict(CELL_SOURCE),
        "platform": dict(PLATFORM),
        "plate_id": culture,
        "unit_id": device_id,
        "culture_age_days": _age_days(culture),
        "compound": {
            "name": "none",
            "role": "untreated_control",
            "mechanism_class": None,
            "dose_value": None,
            "dose_unit": None,
        },
        "recording": {"duration_s": duration_s, "phase": "baseline"},
        "features": {k: _value(features.get(k)) for k in FEATURE_KEYS},
        "feature_method": {
            **p.as_method_record(),
            "spike_detection": "publisher: spike trains as deposited (sparse, 25 kHz)",
        },
        "device": {
            "device_id": device_id,
            "layout": "four_chamber_microtunnel",
            "compartment_id": region,
            "compartment_role": "chamber",
            "n_compartments": len(SUBREGIONS),
            "n_in_channel_electrodes": None,
        },
        "coupling": coupling,
        "qc": {
            "usable": active >= MIN_ACTIVE_ELECTRODES,
            "flags": [] if active >= MIN_ACTIVE_ELECTRODES else ["too_few_active_electrodes"],
            "n_noisy_electrodes": None,
            "note": (
                "compartment named from the file name; the deposit's own subregion field is "
                "present in 4 of 40 files and disagrees with the file name in 2 of those"
            ),
        },
        "provenance": {
            "source_files": [{"path": f"{SOURCE}/{ARCHIVE_NAME}", "sha256": sha}],
            "transform": [
                f"read spike_trains_data/{culture}/spikes/mea_{region}.mat",
                "sparse non-zero indices / 25 kHz -> spike times in seconds",
                "features/mea.py extract_features",
                "features/coupling.py directed_coupling to each other compartment",
            ],
            "generated_by": "neuromps.etl.brewer",
            "generated_at": utc_now(),
        },
    }


def _age_days(culture: str) -> float | None:
    """Days in vitro, from the `d22` token the depositors put in the directory name."""
    for token in culture.split():
        if token.startswith("d") and token[1:].isdigit():
            return float(token[1:])
    return None


def _value(value: object) -> float | int | None:
    """Null-safe conversion for the features block."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value) if isinstance(value, (int, np.integer)) else float(value)


def main(argv: list[str] | None = None) -> int:
    """CLI: build the compartment records and report how many."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = build_records()
    log.info("built %d records from %s", len(records), ARCHIVE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
