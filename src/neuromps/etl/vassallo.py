"""Vassallo 2017 EPA recordings -> one record per chip per concentration.

Each chip is recorded continuously for hours: a baseline, then a series of
rising concentrations of one compound, with no marker in the spike stream
saying where one ends and the next begins. `Logfile_EPAv1.xls`, inside the same
archive, is what makes the data usable - it gives the start and end second of
the baseline and of every concentration, on the same timebase as the spikes.

Two consequences worth stating:

- **The pairing is real.** The baseline and the exposures come from the same
  chip in the same session, so `models/dataset.py` pairs them without any
  matching heuristic. This is the only multi-compound source in the asset for
  which that is true.
- **Nothing is inferred about timing.** Spikes outside the logged windows -
  the minutes while a compound was being pipetted and mixed - are dropped
  rather than assigned to whichever window is nearer.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.features.mea import FeatureParams, extract_features
from neuromps.fetch.manifest import utc_now
from neuromps.qc.provenance import sha256_file

log = logging.getLogger("neuromps.etl")
SOURCE = "vassallo2017"
ARCHIVE_NAME = "Vassallo Data.zip"
LOGFILE_NAME = "Logfile_EPAv1.xls"
REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = REPO_ROOT / "data" / "raw" / SOURCE / ARCHIVE_NAME
SAMPLE_RATE_HZ = 25000.0
MIN_ACTIVE_ELECTRODES = 10
FIRST_DOSE_COLUMN = 6  # the logfile's (uM, tstart, tend) triples start here
MAX_DOSE_COLUMNS = 8
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
# The study's own design: three compounds chosen as neurotoxic, three as not.
MECHANISM = {
    "deltamethrin": "na_channel_modulator",
    "domoic acid": "glutamatergic_agonist",
    "chlorpyrifos-oxon": "ache_inhibitor",
    "acetaminophen": "inactive",
    "glyphosate": "inactive",
    "dimethyl phthalate": "inactive",
}
CITATION = (
    "Vassallo A, et al. A multi-laboratory evaluation of microelectrode array-based measurements "
    "of neural network activity for acute neurotoxicity testing. Neurotoxicology 60:280-292 "
    "(2017). Dataset doi:10.23719/1517583, U.S. EPA ORD (U.S. Public Domain)."
)
PLATFORM = {
    "vendor": "Multi Channel Systems",
    "system": "60-electrode single-well MEA",
    "plate_format": "single-well",
    "electrodes_per_unit": 60,
    "sampling_rate_hz": SAMPLE_RATE_HZ,
    "electrode_pitch_um": None,
}
CELL_SOURCE = {
    "species": "rat",
    "system": "primary_cortical",
    "line_or_strain": None,
    "differentiation_protocol": None,
    "note": "primary cortical cultures, EPA arm of the multi-laboratory study",
}


def session_and_chip(name: str) -> tuple[str, str]:
    """(session date, chip id) - the only parts the logfile and filenames agree on.

    A logfile row names every compound on the plate ("ACE DP GLY") while the
    spike file names the one that chip received, so the full strings do not
    match and these two fields are what join them.
    """
    stem = name.replace(".txt", "")
    return stem.split()[0], re.split(r"--", stem)[-1].strip()


def dose_windows(frame: pd.DataFrame) -> list[dict]:
    """Parse the logfile into one entry per chip, with its windows in seconds."""
    rows = []
    for _, row in frame.iloc[2:].iterrows():
        compound, experiment = row[0], row[1]
        if not isinstance(experiment, str) or not isinstance(compound, str):
            continue
        windows = [
            {"label": "baseline", "dose_um": None, "start_s": float(row[4]), "end_s": float(row[5])}
        ]
        for index in range(MAX_DOSE_COLUMNS):
            base = FIRST_DOSE_COLUMN + 3 * index
            if base + 2 >= len(row):
                break
            dose, start, end = row[base], row[base + 1], row[base + 2]
            if pd.isna(dose) or pd.isna(start) or pd.isna(end):
                continue
            windows.append(
                {
                    "label": f"conc{index + 1}",
                    "dose_um": float(dose),
                    "start_s": float(start),
                    "end_s": float(end),
                }
            )
        rows.append(
            {
                "compound": compound.strip().lower(),
                "experiment": experiment.strip(),
                "windows": windows,
            }
        )
    return rows


def read_logfile(archive: zipfile.ZipFile) -> list[dict]:
    """The dose windows, read from the .xls inside the archive."""
    with archive.open(LOGFILE_NAME) as handle:
        frame = pd.read_excel(io.BytesIO(handle.read()), header=None)
    return dose_windows(frame)


def spike_table(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """Spike ticks and channels for one chip."""
    with archive.open(member) as handle:
        return pd.read_csv(
            io.BytesIO(handle.read()),
            sep="\t",
            header=None,
            names=["tick", "channel"],
            dtype={"tick": np.int64, "channel": np.int32},
            on_bad_lines="skip",
        )


def window_spikes(table: pd.DataFrame, start_s: float, end_s: float) -> dict[str, np.ndarray]:
    """Spike times per electrode within a window, measured from the window start."""
    seconds = table["tick"].to_numpy(dtype=np.float64) / SAMPLE_RATE_HZ
    inside = (seconds >= start_s) & (seconds < end_s)
    if not inside.any():
        return {}
    times, channels = seconds[inside] - start_s, table["channel"].to_numpy()[inside]
    return {
        str(int(channel)): np.sort(times[channels == channel]) for channel in np.unique(channels)
    }


def _compound(compound: str, window: dict) -> dict:
    """The compound block for one window."""
    if window["dose_um"] is None:
        return {
            "name": "none",
            "role": "untreated_control",
            "mechanism_class": None,
            "dose_value": None,
            "dose_unit": None,
        }
    return {
        "name": compound,
        "role": "test",
        "mechanism_class": MECHANISM.get(compound),
        "dose_value": window["dose_um"],
        "dose_unit": "uM",
    }


def build_records(
    archive_path: Path = ARCHIVE,
    hashes: dict[str, str] | None = None,
    params: FeatureParams | None = None,
    max_chips: int | None = None,
) -> list[dict]:
    """One record per chip per logged window."""
    p = params or FeatureParams()
    archive_path = Path(archive_path)
    sha = (hashes or {}).get(archive_path.name) or sha256_file(archive_path)
    records: list[dict] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = [n for n in archive.namelist() if n.endswith(".txt") and "--" in n]
        by_id: dict[tuple[str, str], list[str]] = {}
        for member in members:
            by_id.setdefault(session_and_chip(member), []).append(member)
        entries = read_logfile(archive)[: max_chips or None]
        for entry in entries:
            key = session_and_chip(entry["experiment"])
            candidates = by_id.get(key, [])
            if len(candidates) != 1:
                log.warning("no unique spike file for %s; skipping", entry["experiment"])
                continue
            table = spike_table(archive, candidates[0])
            records += _chip_records(entry, candidates[0], table, sha, p)
            log.info(
                "  vassallo %s chip %s (%s): %d windows",
                key[0],
                key[1],
                entry["compound"],
                len(entry["windows"]),
            )
    return records


def _chip_records(
    entry: dict, member: str, table: pd.DataFrame, sha: str, p: FeatureParams
) -> list[dict]:
    """Every window of one chip."""
    session, chip = session_and_chip(entry["experiment"])
    out = []
    for window in entry["windows"]:
        spikes = window_spikes(table, window["start_s"], window["end_s"])
        duration_s = float(window["end_s"] - window["start_s"])
        features = extract_features(spikes, duration_s, p)
        active = features.get("active_electrodes") or 0
        out.append(
            {
                "record_id": f"{SOURCE}:{session}-{chip}:{window['label']}",
                "source": SOURCE,
                "license": "US-Public-Domain",
                "citation": CITATION,
                "data_level": "spike_times",
                "cell_source": dict(CELL_SOURCE),
                "platform": dict(PLATFORM),
                "plate_id": session,
                "unit_id": chip,
                "culture_age_days": None,
                "compound": _compound(entry["compound"], window),
                "recording": {
                    "duration_s": duration_s,
                    "phase": "baseline" if window["dose_um"] is None else "exposure",
                },
                "features": {k: _value(features.get(k)) for k in FEATURE_KEYS},
                "feature_method": {
                    **p.as_method_record(),
                    "spike_detection": (
                        "publisher: spike times as deposited (sample indices at 25 kHz)"
                    ),
                },
                "qc": {
                    "usable": active >= MIN_ACTIVE_ELECTRODES,
                    "flags": []
                    if active >= MIN_ACTIVE_ELECTRODES
                    else ["too_few_active_electrodes"],
                    "n_noisy_electrodes": None,
                    "note": (
                        f"window {window['start_s']:.0f}-{window['end_s']:.0f} s of a continuous "
                        "session, from Logfile_EPAv1.xls; spikes between windows are not used"
                    ),
                },
                "provenance": {
                    "source_files": [{"path": f"{SOURCE}/{ARCHIVE_NAME}", "sha256": sha}],
                    "transform": [
                        f"read {member} and {LOGFILE_NAME} from {ARCHIVE_NAME}",
                        f"slice {window['start_s']:.0f}-{window['end_s']:.0f} s",
                        "sample indices / 25 kHz -> seconds from the window start",
                        "features/mea.py extract_features",
                    ],
                    "generated_by": "neuromps.etl.vassallo",
                    "generated_at": utc_now(),
                },
            }
        )
    return out


def _value(value: object) -> float | int | None:
    """Null-safe conversion for the features block."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value) if isinstance(value, (int, np.integer)) else float(value)


def main(argv: list[str] | None = None) -> int:
    """CLI: build the records and report how many."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-chips", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = build_records(max_chips=args.max_chips)
    log.info("built %d records from %s", len(records), ARCHIVE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
