"""Fofie et al. 2025 nociceptor MEA workbook -> `per_well_features` records.

hiPSC-derived sensory neurons (RealDRG) on Axion Maestro plates, recorded at a
baseline and again at several times after a compound was applied. The deposit
publishes **total spike counts per well**, not spike times, which decides
everything about how this source can be used:

- Only rate-like features exist. `burst_rate_per_min`, `synchrony_index` and the
  rest of the nine-feature vector cannot be recomputed and are left absent, so
  the records carry `data_level = "per_well_features"` and never compare against
  a `spike_times` record.
- **Firing rate is recoverable as a ratio even though the recording duration is
  not in the workbook.** Every model in `models/` consumes
  ``log2(exposure / baseline)``; with both counts taken over equal-length
  recordings of the same well, the duration cancels out of that ratio. An
  absolute rate would need the duration and is not claimed.

Each well is therefore its own control, which is the strongest pairing in the
asset: no cross-well normalisation, no plate effect, no batch to match.

Two corrections to the note that parked this source in September:

- The dose units are **not** missing from the workbook. Four of its five data
  sheets label conditions with the unit (``GABA [10 uM]``, ``TTX (5 nM)``,
  ``DNQX (50 uM)``). One sheet, ``Curves TTX-HWTX-Lidocaine``, writes bare
  numbers (``TTX 10``); that sheet is skipped rather than guessed at.
- The sheets use two different layouts, both regular. `parse_row_per_well`
  handles the one whose rows are wells, and `parse_blocks` the one whose blocks
  are stacked down the sheet.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger("neuromps.etl")

SOURCE = "fofie2025"
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKBOOK = REPO_ROOT / "data" / "raw" / SOURCE / "Fofie_et_al_Cell_Reports_2025_Raw_data.xlsx"
# The sheet whose labels carry no unit. Skipped rather than read against the
# paper: a dose is a number and a unit, and half of one is not a dose.
UNITLESS_SHEET = "Curves TTX-HWTX-Lidocaine"

# Conditions that are a control rather than a test article.
VEHICLES = re.compile(r"^(no interv\w*|dpbs|dmso|control|ctrl|crtl)", re.I)
# "GABA [10 µM]", "TTX (5 nM)", "DMSO (0.25%)"
CONDITION = re.compile(
    r"^(?P<name>[A-Za-z0-9\- ]+?)\s*[\[(]\s*(?P<dose>[\d.]+)\s*(?P<unit>[µu]M|nM|mM|%)\s*[\])]"
)
TIMEPOINT = re.compile(r"^(base|[\d.]+\s*(hr|hrs|min)|24hrs)", re.I)
BASELINE = re.compile(r"^base", re.I)

# Mechanism classes, from the compounds' established primary targets. Sensory
# neurons, so these are the mechanism, not a claim about a cortical network.
MECHANISM = {
    "gaba": "gabaa_agonist_or_pam",
    "dnqx": "glutamatergic_antagonist",
    "ttx": "na_channel_blocker",
    "lidocaine": "na_channel_blocker",
    "gx201": "na_channel_blocker",
    "huwentoxin": "na_channel_blocker",
    "hwtx": "na_channel_blocker",
    "a887826": "na_channel_blocker",
    "conotoxin": "ca_channel_blocker",
    "nifedipine": "ca_channel_blocker",
    "mibefradil": "ca_channel_blocker",
    "diazoxide": "k_channel_opener",
    "retigabine": "k_channel_opener",
    "capsazepine": "trpv1_antagonist",
    "amitriptyline": "monoamine_reuptake_inhibitor",
    "duloxetine": "monoamine_reuptake_inhibitor",
    "dasatinib": "kinase_inhibitor",
    "baricitinib": "kinase_inhibitor",
}


@dataclass(frozen=True)
class WellSeries:
    """One well: its condition, and its spike count at each timepoint."""

    sheet: str
    block: str
    well: str
    condition: str
    counts: dict[str, float]

    @property
    def baseline(self) -> float | None:
        for label, value in self.counts.items():
            if BASELINE.match(label):
                return value
        return None


def normalise_condition(raw: str) -> tuple[str, float | None, str | None]:
    """Split a condition label into compound, dose and unit.

    A vehicle or no-intervention label returns no dose, which is what makes it
    usable as a control rather than as a zero-dose point of some compound.
    """
    text = " ".join(str(raw).split())
    vehicle = VEHICLES.match(text)
    if vehicle:
        # Keep the whole matched phrase, not its first word: "No Intervention"
        # reduced to "no" stops matching VEHICLES on the way back in, and every
        # no-intervention well is then silently dropped as an unknown compound.
        return vehicle.group(0).lower(), None, None
    match = CONDITION.match(text)
    if not match:
        return text.lower(), None, None
    unit = match.group("unit").replace("µ", "u")
    return match.group("name").strip().lower(), float(match.group("dose")), unit


def mechanism_of(compound: str) -> str | None:
    """Mechanism class for a compound name, or None when it is a control."""
    name = compound.lower()
    if VEHICLES.match(name):
        # The asset labels an untreated pair "control"; using a different word
        # here would silently hide this source's controls from every cross-source
        # comparison, which is the one thing they exist for.
        return "control"
    for key, mechanism in MECHANISM.items():
        if key in name:
            return mechanism
    return None


def _is_count(value) -> bool:
    """Whether a cell looks like a spike count rather than a percentage."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    # Counts are large positive integers; the derived %-change blocks beside
    # them are small and often negative, which is what separates the two.
    return number >= 0 and float(number).is_integer() and number > 0


def parse_row_per_well(frame: pd.DataFrame, sheet: str) -> list[WellSeries]:
    """Sheets whose rows are wells and whose columns are timepoints.

    The condition sits in a text column; the timepoint labels sit in row 0 above
    the count columns. A sheet holds several such groups side by side - raw
    counts, percentage change, and temperature-controlled repeats - and only the
    count groups are taken.
    """
    header = frame.iloc[0]
    out: list[WellSeries] = []
    for label_col in frame.columns:
        values = frame[label_col].iloc[1:]
        conditions = [v for v in values.dropna() if isinstance(v, str)]
        if len(conditions) < 4:
            continue
        # The timepoint columns are the run of headers immediately to the right.
        times: dict[int, str] = {}
        for col in frame.columns:
            if col <= label_col:
                continue
            name = header.get(col)
            if isinstance(name, str) and TIMEPOINT.match(name.strip()):
                times[col] = name.strip()
            elif times:
                break
        if not times or not any(BASELINE.match(t) for t in times.values()):
            continue
        for row in range(1, len(frame)):
            condition = frame.iloc[row][label_col]
            if not isinstance(condition, str) or not condition.strip():
                continue
            counts = {
                name: float(frame.iloc[row][col])
                for col, name in times.items()
                if _is_count(frame.iloc[row][col])
            }
            if len(counts) < 2 or not any(BASELINE.match(t) for t in counts):
                continue
            out.append(WellSeries(sheet, f"col{label_col}", f"r{row}", condition.strip(), counts))
    return out


def parse_blocks(frame: pd.DataFrame, sheet: str) -> list[WellSeries]:
    """Sheets whose condition blocks are stacked down the sheet.

    A block is a condition label at (row, col), well identifiers on the next row
    to its right, and one row per timepoint below that, the timepoint named in
    the column between the label and the wells.
    """
    out: list[WellSeries] = []
    for col in frame.columns:
        anchors = [
            row
            for row in range(len(frame) - 2)
            if isinstance(frame.iloc[row][col], str)
            and (
                CONDITION.match(frame.iloc[row][col].strip())
                or VEHICLES.match(frame.iloc[row][col].strip())
            )
        ]
        for row in anchors:
            condition = frame.iloc[row][col].strip()
            wells: dict[int, str] = {}
            for wcol in frame.columns:
                if wcol <= col + 1:
                    continue
                name = frame.iloc[row + 1][wcol]
                if isinstance(name, str) and re.fullmatch(r"[A-H]\d{1,2}", name.strip()):
                    wells[wcol] = name.strip()
                elif wells:
                    break
            if not wells:
                continue
            series: dict[str, dict[str, float]] = {w: {} for w in wells.values()}
            for below in range(row + 2, min(row + 14, len(frame))):
                label = frame.iloc[below][col + 1]
                if not isinstance(label, str) or not TIMEPOINT.match(label.strip()):
                    if series[next(iter(series))]:
                        break
                    continue
                for wcol, well in wells.items():
                    value = frame.iloc[below][wcol]
                    if _is_count(value):
                        series[well][label.strip()] = float(value)
            for well, counts in series.items():
                if len(counts) >= 2 and any(BASELINE.match(t) for t in counts):
                    out.append(WellSeries(sheet, f"r{row}c{col}", well, condition, counts))
    return out


def read_workbook(path: Path = WORKBOOK) -> list[WellSeries]:
    """Every usable well series in the workbook, from whichever layout it uses."""
    book = pd.ExcelFile(path)
    out: list[WellSeries] = []
    for sheet in book.sheet_names:
        if sheet == UNITLESS_SHEET or sheet.lower().startswith("read me"):
            continue
        frame = pd.read_excel(book, sheet_name=sheet, header=None, nrows=400)
        frame = frame.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
        if frame.empty:
            continue
        found = parse_row_per_well(frame, sheet) or parse_blocks(frame, sheet)
        log.info("%s: %d well series", sheet, len(found))
        out.extend(found)
    return out


def is_usable(series: WellSeries) -> bool:
    """Whether a parsed series is a real condition rather than a stray column.

    The sheets put plate-row letters and well identifiers in text columns beside
    the data, and a parser looking for "a text column with several distinct
    values" finds those too. A usable series names a vehicle or a compound with
    a dose, and has a baseline to divide by.
    """
    text = " ".join(str(series.condition).split())
    if not (VEHICLES.match(text) or CONDITION.match(text)):
        return False
    compound, _, _ = normalise_condition(text)
    if mechanism_of(compound) is None:
        return False
    baseline = series.baseline
    return baseline is not None and baseline > 0 and len(series.counts) >= 2


def usable_series(path: Path = WORKBOOK) -> list[WellSeries]:
    """Every series that names a real condition and carries a baseline."""
    return [s for s in read_workbook(path) if is_usable(s)]


# The exposure timepoint a pair is read at, in order of preference. Post-washout
# ("-PW") is excluded because it measures recovery rather than exposure, and the
# temperature-controlled repeats ("+T") because they re-measure wells that the
# primary blocks already contribute - keeping both would put the same culture in
# the table twice and narrow every interval computed from it.
EXPOSURE_PREFERENCE = ("2hrs", "1hr", "0.25hr", "0.5hr", "24hrs")


def exposure_timepoint(series: WellSeries) -> str | None:
    """The timepoint a pair is read at, or None when only a baseline exists."""
    for label in EXPOSURE_PREFERENCE:
        if label in series.counts:
            return label
    return None


def deduplicate(series: list[WellSeries]) -> list[WellSeries]:
    """Drop series whose sheet, condition and counts are all identical.

    The sheets repeat some blocks verbatim across column groups. A repeat with
    the same numbers is the same measurement read twice, not a replicate, and
    counting it twice would narrow every interval computed from it.
    """
    seen, out = set(), []
    for s in series:
        key = (s.sheet, s.condition, tuple(sorted(s.counts.items())))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def build_pairs(path: Path = WORKBOOK) -> pd.DataFrame:
    """One row per well series: its mechanism, dose and log2 rate change.

    ``delta_log2_rate`` is ``log2(exposure count / baseline count)`` for the same
    well. The recording duration and the electrode count both cancel in that
    ratio, which is why this source can contribute a rate change without
    publishing either.
    """
    import numpy as np

    rows = []
    for s in deduplicate(usable_series(path)):
        timepoint = exposure_timepoint(s)
        baseline = s.baseline
        if timepoint is None or not baseline:
            continue
        exposure = s.counts[timepoint]
        if exposure <= 0:
            # A silenced well is real and must not become -inf; floor it at the
            # same fraction `models/dataset.py` uses for the rest of the asset.
            exposure = 0.01 * baseline
        compound, dose, unit = normalise_condition(s.condition)
        rows.append(  # noqa: E501 - the group needs `compound`, bound just above
            {
                "sheet": s.sheet,
                "block": s.block,
                "well": s.well,
                # The six replicate wells of one condition share a plate and a
                # treatment, so the block is the unit a split must not straddle -
                # the same rule the rest of the asset applies to cultures. This is
                # the conservative choice: grouping by well instead would make
                # every pair independent and narrow every interval.
                "group": f"{s.sheet}:{s.block}:{compound}",
                "compound": compound,
                "dose_value": dose,
                "dose_unit": unit,
                "mechanism_class": mechanism_of(compound),
                "timepoint": timepoint,
                "baseline_count": baseline,
                "exposure_count": s.counts[timepoint],
                "delta_log2_rate": float(np.log2(exposure / baseline)),
            }
        )
    return pd.DataFrame(rows)
