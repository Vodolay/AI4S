"""Trujillo et al. 2019 organoid MEA sessions -> schema records.

One record per organoid: twelve organoids sit in one 12-well Axion Maestro
plate, each over 64 electrodes, recorded for 240 s at 12.5 kHz, and the same
plate is recorded again week after week. The well index is therefore an
organoid identity that persists across sessions, which is what the cluster
bootstrap needs to group by.

Three things a reader should know:

- **Culture age is not recorded.** The deposit gives session dates, not a
  plating date, so `culture_age_days` is null rather than a guess. The session
  date is kept in `plate_id`, so a developmental analysis can still order the
  sessions without anyone having to believe an invented age.
- **The drug sessions are flagged, not labelled.** The paper applied AP5+CNQX,
  baclofen, muscimol and bicuculline to four wells each, but the deposit
  carries no well-to-compound mapping. Those records get `role="test"` with
  `mechanism_class=None`, which keeps them out of the mechanism classifier
  while leaving them visible. Guessing the mapping would have been easy and
  wrong.
- **Nineteen early sessions are missing** because they are MATLAB 5 files that
  cannot be read in place; see the fetcher. The developmental trajectory here
  starts later than the deposit's does.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.features.mea import FeatureParams, extract_features
from neuromps.fetch.manifest import utc_now
from neuromps.qc.provenance import sha256_file

log = logging.getLogger("neuromps.etl")
SOURCE = "trujillo2019"
REPO_ROOT = Path(__file__).resolve().parents[3]
SPIKES_DIR = REPO_ROOT / "data" / "raw" / SOURCE
MIN_ACTIVE_ELECTRODES = 10
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
    "Trujillo CA, Gao R, Negraes PD, et al. Complex Oscillatory Waves Emerging from Cortical "
    "Organoids Model Early Human Brain Network Development. Cell Stem Cell 25(4):558-569.e7 "
    "(2019). Dataset doi:10.5281/zenodo.4751759 (CC BY 4.0)."
)
PLATFORM = {
    "vendor": "Axion Biosystems",
    "system": "Maestro (64 electrodes per well)",
    "plate_format": "12-well",
    "electrodes_per_unit": 64,
    "sampling_rate_hz": 12500.0,
    "electrode_pitch_um": None,
}
CELL_SOURCE = {
    "species": "human",
    "system": "brain_organoid",
    "line_or_strain": None,
    "differentiation_protocol": "cortical organoid, plated intact on the array",
    "note": "whole organoid on a planar MEA - not a slice, and not a monolayer",
}
_SESSION = re.compile(r"^LFP_Sp_(?P<date>\d{6})(?P<suffix>_\w+)?$")


def session_id(path: Path) -> tuple[str, bool]:
    """(session id, whether a compound was applied) from a spike-file name."""
    match = _SESSION.match(Path(path).name.removesuffix(".spikes.parquet"))
    if not match:
        raise ValueError(f"unexpected spike file name: {path.name}")
    suffix = (match.group("suffix") or "").lower()
    # Lowercase because record ids are, and a session id that differs in case
    # from the id built out of it is a trap for anyone joining the two.
    return f"{match.group('date')}{suffix}", suffix == "_drugs"


def read_session(path: Path) -> tuple[pd.DataFrame, float]:
    """Spike table and recording duration, the latter from the parquet metadata."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    if b"duration_s" not in metadata:
        raise ValueError(f"{path.name} predates the duration metadata; re-fetch the source")
    return table.to_pandas(), float(metadata[b"duration_s"])


def spikes_by_electrode(well_rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Electrode id -> sorted spike times in seconds, for one organoid."""
    return {
        str(int(electrode)): np.sort(group["time_s"].to_numpy(dtype=float))
        for electrode, group in well_rows.groupby("electrode", sort=True)
    }


def _compound(treated: bool) -> dict:
    """The compound block: flagged when a drug was applied, never invented."""
    if not treated:
        return {
            "name": "none",
            "role": "untreated_control",
            "mechanism_class": None,
            "dose_value": None,
            "dose_unit": None,
        }
    return {
        "name": "unspecified_in_deposit",
        "role": "test",
        "mechanism_class": None,
        "dose_value": None,
        "dose_unit": None,
    }


def build_records(
    spikes_dir: Path = SPIKES_DIR,
    hashes: dict[str, str] | None = None,
    params: FeatureParams | None = None,
) -> list[dict]:
    """One record per organoid per session, for every extracted session file."""
    p = params or FeatureParams()
    records: list[dict] = []
    for path in sorted(Path(spikes_dir).glob("*.spikes.parquet")):
        session, treated = session_id(path)
        frame, duration_s = read_session(path)
        rel = f"{SOURCE}/{path.name}"
        sha = (hashes or {}).get(rel) or sha256_file(path)
        for well, well_rows in frame.groupby("well", sort=True):
            spikes = spikes_by_electrode(well_rows)
            features = extract_features(spikes, duration_s, p)
            active = features.get("active_electrodes") or 0
            records.append(
                {
                    "record_id": f"{SOURCE}:{session}:w{int(well):02d}",
                    "source": SOURCE,
                    "license": "CC-BY-4.0",
                    "citation": CITATION,
                    "data_level": "spike_times",
                    "cell_source": dict(CELL_SOURCE),
                    "platform": dict(PLATFORM),
                    "plate_id": session,
                    "unit_id": f"well{int(well):02d}",
                    "culture_age_days": None,
                    "compound": _compound(treated),
                    "recording": {
                        "duration_s": duration_s,
                        "phase": "exposure" if treated else "baseline",
                    },
                    "features": {k: _value(features.get(k)) for k in FEATURE_KEYS},
                    "feature_method": {
                        **p.as_method_record(),
                        "spike_detection": (
                            "publisher: spike times as published in the deposit's "
                            "`spikes` array (sample indices at 12.5 kHz)"
                        ),
                    },
                    "qc": {
                        "usable": active >= MIN_ACTIVE_ELECTRODES,
                        "flags": [] if active >= MIN_ACTIVE_ELECTRODES else [
                            "too_few_active_electrodes"
                        ],
                        "n_noisy_electrodes": None,
                        "note": (
                            "culture age not recorded in the deposit; session date in plate_id"
                            + (
                                "; compound applied but not identified per well"
                                if treated
                                else ""
                            )
                        ),
                    },
                    "provenance": {
                        "source_files": [{"path": rel, "sha256": sha}],
                        "transform": [
                            "spike times read in place from the remote MATLAB v7.3 file",
                            "sample indices / 12.5 kHz -> seconds",
                            "group by well (one organoid per well)",
                            "features/mea.py extract_features",
                        ],
                        "generated_by": "neuromps.etl.trujillo",
                        "generated_at": utc_now(),
                    },
                }
            )
        log.info("  trujillo %s: %d organoids", session, frame["well"].nunique())
    return records


def _value(value: object) -> float | int | None:
    """Null-safe conversion for the features block."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value) if isinstance(value, (int, np.integer)) else float(value)


def main(argv: list[str] | None = None) -> int:
    """CLI: build the records and report how many."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = build_records()
    log.info("built %d records from %s", len(records), SPIKES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
