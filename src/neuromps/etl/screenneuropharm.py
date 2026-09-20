"""ScreenNeuroPharm bicuculline dose-response -> schema records.

Rat E18 cortical and hippocampal dissociated cultures on 60-electrode MCS MEAs,
8-minute recordings at 10 kHz, bicuculline from 100 nM to 300 uM (Poggio et al.
2023, Biomolecules, doi:10.3390/biom13010074; data MIT-licensed).

Two-compartment devices in the same archive (CxCx, CxHp, HpHp: "Joint"
recordings with Gray/Green/Red electrode groups) are **not** converted yet: the
schema has no representation for per-compartment units and directed connectivity
between them. They are the natural validation set for the digital twin, and the
schema extension is tracked in schema/STANDARD.md section 8.
"""

import argparse
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.features.mea import FeatureParams, extract_features
from neuromps.fetch.manifest import utc_now

log = logging.getLogger("neuromps.etl")
SOURCE = "screenneuropharm_bic"
SAMPLING_HZ = 10_000.0
ARCHIVE_NAME = "Bicuculline-BIC_v1.zip"
REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = REPO_ROOT / "data" / "raw" / SOURCE / ARCHIVE_NAME
CACHE = REPO_ROOT / "data" / "interim" / "screenneuropharm_features.parquet"
SINGLE_COMPARTMENT = {"Cx": "primary_cortical", "Hp": "primary_hippocampal"}
CITATION = (
    "Poggio F, Brofiga M, Tedesco M, Massobrio P, Adriano E, Balestrino M. Lack of epileptogenic "
    "effects of the creatine precursor guanidinoacetic acid on neuronal cultures in vitro. "
    "Biomolecules 13(1):74 (2023). doi:10.3390/biom13010074. "
    "Data: doi:10.5281/zenodo.7665283 (MIT)."
)
PLATFORM = {
    "vendor": "Multi Channel Systems",
    "system": "MEA60 planar array",
    "plate_format": "single-well",
    "electrodes_per_unit": 60,
    "sampling_rate_hz": SAMPLING_HZ,
}
_DOSE = re.compile(r"(?:BIC)?([0-9]*\.?[0-9]+)(nM|uM|mM)$", re.IGNORECASE)


@dataclass(frozen=True)
class Recording:
    """One condition of one array: the peak-train files of all its electrodes."""

    prep: str
    region: str
    array: str
    condition_dir: str
    members: tuple[str, ...]

    @property
    def unit_id(self) -> str:
        return f"{self.prep}-{self.region}-{self.array}"

    @property
    def condition(self) -> str:
        return self.condition_dir.rsplit("/", 1)[-1]


def parse_ptrain(text: str) -> tuple[np.ndarray, float]:
    """Read one peak train: a header of total samples, then `sample,amplitude` rows."""
    lines = text.splitlines()
    total = int(lines[0].split(",")[0])
    samples = [int(line.split(",")[0]) for line in lines[1:] if line.strip()]
    return np.asarray(samples, dtype=float) / SAMPLING_HZ, total / SAMPLING_HZ


def parse_condition(condition_dir: str) -> dict:
    """Compound block from a condition folder name (`..._nbasal`, `..._BIC100nM`, `..._1uM`)."""
    name = condition_dir.rsplit("/", 1)[-1]
    match = _DOSE.search(name)
    if not match:
        return {
            "name": "none",
            "role": "untreated_control",
            "mechanism_class": None,
            "dose_value": None,
            "dose_unit": None,
        }
    unit = {"nm": "nM", "um": "uM", "mm": "uM"}[match.group(2).lower()]
    return {
        "name": "bicuculline",
        "role": "test",
        "mechanism_class": "gabaa_antagonist",
        "dose_value": float(match.group(1)),
        "dose_unit": unit,
    }


def discover_recordings(archive: zipfile.ZipFile) -> list[Recording]:
    """Group single-compartment peak-train files into recordings."""
    groups: dict[tuple, list[str]] = {}
    skipped = set()
    for name in archive.namelist():
        if "/Data/" not in name or not name.endswith(".txt"):
            continue
        parts = name.split("/Data/")[1].split("/")
        if len(parts) < 5:
            continue
        prep, region, array = parts[0], parts[1], parts[2]
        if region not in SINGLE_COMPARTMENT:
            skipped.add((prep, region, array))
            continue
        groups.setdefault((prep, region, array, "/".join(parts[3:5])), []).append(name)
    if skipped:
        log.info("skipped %d two-compartment devices (no schema support yet)", len(skipped))
    return [
        Recording(prep, region, array, cond, tuple(sorted(members)))
        for (prep, region, array, cond), members in sorted(groups.items())
    ]


def _features_for(archive: zipfile.ZipFile, rec: Recording, params: FeatureParams) -> dict:
    """Extract features for one recording via the shared extractor."""
    spikes, duration = {}, 0.0
    for member in rec.members:
        times, duration = parse_ptrain(archive.read(member).decode())
        spikes[Path(member).stem.rsplit("_", 1)[-1]] = times
    return {"duration_s": duration, **extract_features(spikes, duration, params)}


def _feature_table(
    archive_path: Path, cache_path: Path | None, params: FeatureParams, rebuild: bool
) -> pd.DataFrame:
    """Features per recording, cached because extraction costs ~1 s per recording."""
    if cache_path and cache_path.exists() and not rebuild:
        return pd.read_parquet(cache_path)
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        recordings = discover_recordings(archive)
        for i, rec in enumerate(recordings, 1):
            rows.append(
                {
                    "prep": rec.prep,
                    "region": rec.region,
                    "array": rec.array,
                    "condition_dir": rec.condition_dir,
                    **_features_for(archive, rec, params),
                }
            )
            if i % 25 == 0:
                log.info("  screenneuropharm: %d/%d recordings", i, len(recordings))
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
    """Build one record per recording (array x condition)."""
    p = params or FeatureParams()
    table = _feature_table(Path(archive_path), cache_path, p, rebuild)
    sha = (hashes or {}).get(Path(archive_path).name, "")
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
        compound = parse_condition(row["condition_dir"])
        unit = f"{row['prep']}-{row['region']}-{row['array']}"
        condition = row["condition_dir"].rsplit("/", 1)[-1]
        features = {k: _value(row.get(k)) for k in feature_keys}
        active = features["active_electrodes"] or 0
        records.append(
            {
                "record_id": f"{SOURCE}:{unit.lower()}:{condition.lower()}",
                "source": SOURCE,
                "license": "MIT",
                "citation": CITATION,
                "data_level": "spike_times",
                "cell_source": {
                    "species": "rat",
                    "system": SINGLE_COMPARTMENT[row["region"]],
                    "line_or_strain": f"E18 rat embryo, preparation {row['prep']}",
                },
                "platform": dict(PLATFORM),
                "plate_id": row["array"],
                "unit_id": unit,
                "culture_age_days": None,
                "compound": compound,
                "recording": {
                    "duration_s": float(row["duration_s"]),
                    "phase": "baseline" if compound["role"] == "untreated_control" else "exposure",
                },
                "features": features,
                "feature_method": {
                    **p.as_method_record(),
                    "spike_detection": "publisher: PeakDetection PLP2ms RP1ms (MCS, 10 kHz)",
                },
                "qc": {
                    "usable": active >= 10,
                    "flags": [] if active >= 10 else ["too_few_active_electrodes"],
                    "n_noisy_electrodes": None,
                },
                "provenance": {
                    "source_files": [
                        {"path": f"{SOURCE}/{Path(archive_path).name}", "sha256": sha}
                    ],
                    "transform": [
                        f"extract {row['condition_dir']} from {ARCHIVE_NAME}",
                        "parse peak trains (samples / 10 kHz)",
                        "features/mea.py extract_features",
                    ],
                    "generated_by": "neuromps.etl.screenneuropharm",
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
    """CLI: extract features for every recording and refresh the cache."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild", action="store_true", help="ignore the feature cache")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = build_records(cache_path=CACHE, rebuild=args.rebuild)
    log.info("built %d records; cache at %s", len(records), CACHE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
