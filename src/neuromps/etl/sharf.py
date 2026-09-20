"""Sharf et al. 2022 organoid HD-MEA recordings -> schema records.

Human iPSC-derived cortical organoid slices on a MaxWell MaxOne HD-CMOS array,
~180 s recordings at 20 kHz, spike-sorted with Kilosort2. Covers a diazepam
series (control, 3, 10, 30, 50 uM), six 7-month organoids, and one organoid
across 5-8 months.

Two choices worth knowing about:

- The archive publishes **sorted units**, but our features are defined per
  electrode, so units are grouped onto their peak electrode. `active_electrodes`
  then means what it means elsewhere in the asset. Spikes of different units on
  the same electrode that fall within 1 ms are collapsed into one: the electrode
  itself could not have resolved them, and sorters do split one neuron across
  units. This slightly lowers firing rates relative to unit-level counting.
- Recording duration is taken as the last spike time in the file (all files land
  within 0.2 s of 180 s). It is a lower bound, so rates are marginally high.
"""

import argparse
import io
import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from neuromps.features.mea import FeatureParams, extract_features
from neuromps.fetch.manifest import utc_now
from neuromps.qc.provenance import sha256_file

log = logging.getLogger("neuromps.etl")
SOURCE = "sharf2022"
FS = 20_000
ARCHIVE_NAME = "kilosort2.zip"
REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = REPO_ROOT / "data" / "raw" / SOURCE / ARCHIVE_NAME
CACHE = REPO_ROOT / "data" / "interim" / "sharf_features.parquet"
GROUPS = ("drug", "7month", "development")  # skip: interval (protocol), neuropixels (probe)
DAYS_PER_MONTH = 30.0
MERGE_REFRACTORY_S = 0.001  # coincident spikes on one electrode are one event
CITATION = (
    "Sharf T, van der Molen T, Glasauer SMK, et al. Extracellular recordings from human brain "
    "organoids using high-density CMOS arrays. Dataset doi:10.25349/D9031Z (CC0)."
)
PLATFORM = {
    "vendor": "MaxWell Biosystems",
    "system": "MaxOne HD-CMOS (26,400-electrode array, 1,020 recorded)",
    "plate_format": "single-well",
    "electrodes_per_unit": 1020,
    "electrode_pitch_um": 17.5,
    "sampling_rate_hz": float(FS),
}
_DRUG = re.compile(r"^(?P<drug>[a-z]+)(?P<dose>[0-9.]+)(?P<unit>uM|nM)_(?P<organoid>\d+)$", re.I)
_CONTROL = re.compile(r"^control_(?P<organoid>\d+)$", re.I)
_AGE = re.compile(r"^(?P<organoid>\d+)_(?P<age>\d+)month$", re.I)


def wanted_members(archive: zipfile.ZipFile) -> list[str]:
    """.mat members of the groups we convert."""
    return sorted(
        n
        for n in archive.namelist()
        if n.endswith(".mat") and n.split("/")[1] in GROUPS and "/kilosort2/" in n
    )


def spikes_by_electrode(mat: dict) -> tuple[dict[str, np.ndarray], float]:
    """Spike times in seconds per electrode, merging sorted units on the same electrode."""
    fs = float(mat.get("fs", FS))
    units = np.atleast_1d(mat["units"])
    spikes: dict[str, list[np.ndarray]] = {}
    for unit in units:
        train = np.ravel(_field(unit, "spike_train")).astype(float)
        electrode = str(int(_field(unit, "electrode")))
        spikes.setdefault(electrode, []).append(train / fs)
    merged = {k: _collapse(np.sort(np.concatenate(v))) for k, v in spikes.items()}
    duration = max((t.max() for t in merged.values() if t.size), default=0.0)
    return merged, float(duration)


def _collapse(times: np.ndarray, refractory: float = MERGE_REFRACTORY_S) -> np.ndarray:
    """Drop spikes closer than the refractory window; one electrode, one event."""
    if times.size < 2:
        return times
    keep = np.concatenate([[True], np.diff(times) > refractory])
    return times[keep]


def _field(unit: object, name: str):
    """Read a field from a MATLAB struct however scipy handed it to us."""
    if isinstance(unit, dict):
        return unit[name]
    return getattr(unit, name)


def parse_condition(stem: str, group: str) -> tuple[str, dict, float | None]:
    """Return (organoid id, compound block, culture age in days) for a file stem."""
    control = {
        "name": "none",
        "role": "untreated_control",
        "mechanism_class": None,
        "dose_value": None,
        "dose_unit": None,
    }
    if m := _DRUG.match(stem):
        compound = {
            "name": m.group("drug").lower(),
            "role": "test",
            "mechanism_class": "gabaa_agonist_or_pam",
            "dose_value": float(m.group("dose")),
            "dose_unit": "uM" if m.group("unit").lower() == "um" else "nM",
        }
        return m.group("organoid"), compound, None
    if m := _CONTROL.match(stem):
        return m.group("organoid"), control, None
    if m := _AGE.match(stem):
        return m.group("organoid"), control, float(m.group("age")) * DAYS_PER_MONTH
    return stem.split("_")[-1], control, None


def _feature_table(
    archive_path: Path, cache_path: Path | None, params: FeatureParams, rebuild: bool
) -> pd.DataFrame:
    """Features per recording, cached because each .mat is tens of megabytes."""
    if cache_path and cache_path.exists() and not rebuild:
        return pd.read_parquet(cache_path)
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        members = wanted_members(archive)
        for i, member in enumerate(members, 1):
            mat = sio.loadmat(
                io.BytesIO(archive.read(member)), squeeze_me=True, struct_as_record=False
            )
            spikes, duration = spikes_by_electrode(mat)
            rows.append(
                {
                    "member": member,
                    "group": member.split("/")[1],
                    "stem": Path(member).stem,
                    "duration_s": duration,
                    **extract_features(spikes, duration, params),
                }
            )
            log.info("  sharf: %d/%d %s", i, len(members), Path(member).name)
    table = pd.DataFrame(rows)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(cache_path, index=False)
    return table


def build_records(
    archive_path: Path = ARCHIVE,
    hashes: dict[str, str] | None = None,
    cache_path: Path | None = None,
    params: FeatureParams | None = None,
    rebuild: bool = False,
) -> list[dict]:
    """Build one record per organoid recording."""
    p = params or FeatureParams()
    table = _feature_table(Path(archive_path), cache_path, p, rebuild)
    sha = (hashes or {}).get(Path(archive_path).name) or sha256_file(Path(archive_path))
    feature_keys = [
        "mfr_hz",
        "burst_rate_per_min",
        "burst_duration_s",
        "spikes_per_burst",
        "pct_spikes_in_bursts",
        "network_burst_rate_per_min",
        "synchrony_index",
        "isi_cv",
        "active_electrodes",
    ]
    records = []
    for row in table.to_dict("records"):
        organoid, compound, age_days = parse_condition(row["stem"], row["group"])
        features = {k: _value(row.get(k)) for k in feature_keys}
        active = features["active_electrodes"] or 0
        records.append(
            {
                "record_id": f"{SOURCE}:{organoid}:{row['stem'].lower()}",
                "source": SOURCE,
                "license": "CC0-1.0",
                "citation": CITATION,
                "data_level": "spike_times",
                "cell_source": {
                    "species": "human",
                    "system": "brain_organoid_slice",
                    "line_or_strain": f"organoid array {organoid}",
                    "differentiation_protocol": "cortical organoid, 500 um slice",
                },
                "platform": dict(PLATFORM),
                "plate_id": row["group"],
                "unit_id": organoid,
                "culture_age_days": age_days,
                "compound": compound,
                "recording": {
                    "duration_s": float(row["duration_s"]),
                    "phase": "baseline" if compound["role"] == "untreated_control" else "exposure",
                },
                "features": features,
                "feature_method": {
                    **p.as_method_record(),
                    "spike_detection": (
                        "publisher: Kilosort2 sorted units, merged onto their peak electrode"
                    ),
                },
                "qc": {
                    "usable": active >= 10,
                    "flags": [] if active >= 10 else ["too_few_active_electrodes"],
                    "n_noisy_electrodes": None,
                    "note": "duration taken as the last spike time (~180 s recordings)",
                },
                "provenance": {
                    "source_files": [
                        {"path": f"{SOURCE}/{Path(archive_path).name}", "sha256": sha}
                    ],
                    "transform": [
                        f"extract {row['member']} from {ARCHIVE_NAME}",
                        "merge sorted units onto peak electrode; samples / 20 kHz",
                        "features/mea.py extract_features",
                    ]
                    + (["organoid age months -> days (x30)"] if age_days else []),
                    "generated_by": "neuromps.etl.sharf",
                    "generated_at": utc_now(),
                },
            }
        )
    return records


def _value(value: object) -> float | int | None:
    """Null-safe conversion out of the cached table."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value) if isinstance(value, (int, np.integer)) else float(value)


def main(argv: list[str] | None = None) -> int:
    """CLI: extract features for every organoid recording and refresh the cache."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild", action="store_true", help="ignore the feature cache")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = build_records(cache_path=CACHE, rebuild=args.rebuild)
    log.info("built %d records; cache at %s", len(records), CACHE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
