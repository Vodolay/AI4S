"""Kapucu 2022 pharmacology plates -> schema records.

One record per well per recording phase (baseline, exposure, TTX). Features come
from the shared extractor; upstream exclusions and noisy-electrode lists are
preserved as QC flags rather than dropped.
"""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.features.mea import FeatureParams, extract_features
from neuromps.fetch.manifest import utc_now
from neuromps.qc.provenance import sha256_file

log = logging.getLogger("neuromps.etl")
SOURCE = "kapucu2022"
LICENSE = "CC-BY-4.0"
CITATION = (
    "Kapucu FE, Vinogradov A, Hyvärinen T, Ylä-Outinen L, Narkilahti S. Comparative microelectrode "
    "array data of the functional development of hPSC-derived and rat neuronal networks. "
    "Sci Data 9, 120 (2022). doi:10.1038/s41597-022-01242-4"
)
PLATFORM = {
    "vendor": "Axion Biosystems",
    "system": "Maestro 768-channel",
    "plate_format": "48-well",
    "electrodes_per_unit": 16,
    "sampling_rate_hz": 12500.0,
}
MECHANISM = {
    "gabazine": "gabaa_antagonist",
    "gaba": "gabaa_agonist_or_pam",
    "kainicacid": "glutamatergic_agonist",
    "cnqx": "glutamatergic_antagonist",
    "d-ap5": "glutamatergic_antagonist",
    "ttx": "na_channel_blocker",
}
_DOSE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(uM|nM|mM)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Plate:
    """One recording of one plate: a species, a day in vitro and a treatment phase."""

    directory: str
    species: str
    system: str
    culture_age_days: int
    phase: str
    fixed_compound: str | None = None


PHARMACOLOGY_PLATES: tuple[Plate, ...] = (
    Plate("Rat_MEA2_Pharmacology/Rat_MEA2Pharma_DIV22", "rat", "primary_cortical", 22, "exposure"),
    Plate(
        "Rat_MEA2_Pharmacology/Rat_MEA2Baseline_DIV22", "rat", "primary_cortical", 22, "baseline"
    ),
    Plate(
        "Rat_MEA2_Pharmacology/Rat_MEA2TTX_DIV22", "rat", "primary_cortical", 22, "exposure", "ttx"
    ),
    Plate(
        "hPSC_MEA3_Pharmacology/hPSC_MEA3Pharma_DIV29",
        "human",
        "hpsc_derived_cortical",
        29,
        "exposure",
    ),
    Plate(
        "hPSC_MEA3_Pharmacology/hPSC_MEA3Baseline_DIV29",
        "human",
        "hpsc_derived_cortical",
        29,
        "baseline",
    ),
    Plate(
        "hPSC_MEA3_Pharmacology/hPSC_MEA3TTX_DIV29",
        "human",
        "hpsc_derived_cortical",
        29,
        "exposure",
        "ttx",
    ),
)
TTX_DOSE = (1.0, "uM")


def mechanism_class(compound: str) -> str | None:
    """Mechanism vocabulary for a normalized compound name; None for controls."""
    return MECHANISM.get(compound)


def normalize_compound(raw: str) -> str:
    """Lowercase and unify spellings (the logs contain both 'D-AP5' and 'DAP5')."""
    name = str(raw).strip().lower().replace(" ", "")
    return "d-ap5" if name in ("dap5", "d-ap5") else name


def parse_dose(raw: object) -> tuple[float | None, str | None]:
    """Split '30uM' into (30.0, 'uM'); the logs keep the unit inside the Dose column."""
    m = _DOSE.match(str(raw))
    if not m:
        return None, None
    return float(m.group(1)), {"um": "uM", "nm": "nM", "mm": "uM"}[m.group(2).lower()]


def parse_explog(lines: Iterable[str] | Path) -> pd.DataFrame:
    """Read an expLog into [well, compound, dose_value, dose_unit, excluded]."""
    df = pd.read_csv(Path(lines) if isinstance(lines, Path) else _as_buffer(lines))
    out = pd.DataFrame({"well": df["Well"].astype(str).str.strip()})
    compound = df["Treatment"].map(normalize_compound)
    out["excluded"] = compound.eq("excludedwell")
    out["compound"] = compound
    doses = df["Dose"].map(parse_dose)
    out["dose_value"] = [d[0] for d in doses]
    out["dose_unit"] = [d[1] for d in doses]
    return out


def _as_buffer(lines: Iterable[str]):
    import io

    return io.StringIO("".join(lines))


def parse_noisy_electrodes(text: str) -> dict[str, list[str]]:
    """Map well -> noisy electrode ids. The first line is a DIV header, not data."""
    out: dict[str, list[str]] = {}
    for line in text.splitlines()[1:]:
        name = line.strip()
        if "_" in name:
            out.setdefault(name.split("_")[0], []).append(name)
    return out


def _load_spikes(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    """Read a spikes CSV into {well: {channel: times}} and the recording duration."""
    df = pd.read_csv(path)
    df["well"] = df["Channel"].str.split("_").str[0]
    duration = float(df["Time"].max())
    wells: dict[str, dict[str, np.ndarray]] = {}
    for (well, channel), g in df.groupby(["well", "Channel"], sort=False):
        wells.setdefault(well, {})[channel] = g["Time"].to_numpy()
    return wells, duration


def _files(plate_dir: Path) -> dict[str, Path]:
    """Locate the three CSVs of a plate recording by suffix."""
    found = {}
    for path in sorted(plate_dir.rglob("*.csv")):
        if path.name.startswith("noisy_electrodes"):
            found["noisy"] = path
        elif path.name.endswith("_expLog.csv"):
            found["explog"] = path
        elif path.name.endswith("_spikes.csv"):
            found["spikes"] = path
    missing = {"explog", "spikes"} - set(found)
    if missing:
        raise FileNotFoundError(f"{plate_dir}: missing {sorted(missing)}")
    return found


def _source_file(path: Path, hashes: dict[str, str], raw_root: Path | None) -> dict:
    """Provenance entry for one input file, reusing the manifest hash when available."""
    try:
        rel = str(path.relative_to(raw_root)) if raw_root else f"{SOURCE}/{path.name}"
    except ValueError:
        rel = f"{SOURCE}/{path.name}"
    return {"path": rel, "sha256": hashes.get(path.name) or sha256_file(path)}


def _compound_block(row: pd.Series | None, plate: Plate) -> dict:
    """Compound fields for a well: the plate's fixed drug, the log's drug, or a control."""
    if plate.fixed_compound:
        dose, unit = TTX_DOSE
        return {
            "name": plate.fixed_compound,
            "role": "test",
            "mechanism_class": mechanism_class(plate.fixed_compound),
            "dose_value": dose,
            "dose_unit": unit,
        }
    if plate.phase == "baseline" or row is None or row["compound"] in ("control", "excludedwell"):
        return {"name": "none", "role": "untreated_control", "mechanism_class": None}
    return {
        "name": row["compound"],
        "role": "test",
        "mechanism_class": mechanism_class(row["compound"]),
        "dose_value": row["dose_value"],
        "dose_unit": row["dose_unit"],
    }


def _empty_features() -> dict:
    """Feature block for a well with no recorded spikes (e.g. an excluded well)."""
    keys = (
        "mfr_hz",
        "burst_rate_per_min",
        "burst_duration_s",
        "spikes_per_burst",
        "pct_spikes_in_bursts",
        "network_burst_rate_per_min",
        "synchrony_index",
        "isi_cv",
    )
    return {**dict.fromkeys(keys), "active_electrodes": 0}


def _qc_block(noisy: list[str], excluded: bool, features: dict, params: FeatureParams) -> dict:
    """QC decision and flags for one well."""
    flags = []
    if excluded:
        flags.append("excluded_well_upstream")
    if noisy:
        flags.append("noisy_electrodes_present")
    active = features["active_electrodes"] or 0
    if active < 10:
        flags.append("too_few_active_electrodes")
    return {
        "usable": not excluded and active >= 10,
        "flags": flags,
        "n_noisy_electrodes": len(noisy),
        "note": None,
    }


def build_plate_records(
    plate_dir: Path,
    plate: Plate,
    hashes: dict[str, str],
    params: FeatureParams | None = None,
    raw_root: Path | None = None,
) -> list[dict]:
    """Build one schema record per well of a plate recording.

    ``hashes`` maps file name -> SHA256 from data/raw/MANIFEST.json; any file not
    listed there is hashed here, so provenance is never empty or guessed.
    """
    p = params or FeatureParams()
    files = _files(Path(plate_dir))
    log_df = parse_explog(files["explog"]).set_index("well")
    noisy = parse_noisy_electrodes(files["noisy"].read_text()) if "noisy" in files else {}
    wells, duration = _load_spikes(files["spikes"])
    source_files = [_source_file(path, hashes, raw_root) for path in files.values()]
    records = []
    # Union of both sources: a well can appear in the log with no spikes at all
    # (upstream ExcludedWell). Those are recorded and flagged, never dropped.
    for well in sorted(set(wells) | set(log_df.index)):
        channels = wells.get(well, {})
        row = log_df.loc[well] if well in log_df.index else None
        excluded = bool(row["excluded"]) if row is not None else False
        features = extract_features(channels, duration, p) if channels else _empty_features()
        records.append(
            {
                "record_id": f"{SOURCE}:{Path(plate.directory).name.lower()}:{well.lower()}",
                "source": SOURCE,
                "license": LICENSE,
                "citation": CITATION,
                "data_level": "spike_times",
                "cell_source": {"species": plate.species, "system": plate.system},
                "platform": dict(PLATFORM),
                "plate_id": Path(plate.directory).name,
                "unit_id": well,
                "culture_age_days": plate.culture_age_days,
                "compound": _compound_block(row, plate),
                "recording": {"duration_s": duration, "phase": plate.phase},
                "features": features,
                "feature_method": {
                    **p.as_method_record(),
                    "spike_detection": "publisher: authors' MATLAB detection (GIN Codes/)",
                },
                "qc": _qc_block(noisy.get(well, []), excluded, features, p),
                "provenance": {
                    "source_files": source_files,
                    "transform": [
                        "read spikes csv",
                        "join expLog treatment",
                        "features/mea.py extract_features",
                    ],
                    "generated_by": "neuromps.etl.kapucu",
                    "generated_at": utc_now(),
                },
            }
        )
    return records
