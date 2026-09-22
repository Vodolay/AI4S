"""Microchannel propagation in the Mateus deposit, recovered from spike times.

The deposit publishes spike times and a description of the measurement, but not
the electrode map it was computed on (see `data/SOURCES.md`). Rather than
assuming a layout, this module reproduces the authors' own criterion and lets
the geometry fall out of it:

    a propagating event is detected on all five electrodes of a microchannel,
    with at most 1 ms between consecutive electrodes

Runs of five electrodes that carry such events *are* the microchannels; the rows
they occupy divide the array into the two chambers either side. Counting events
in each direction gives a per-chip forward/backward asymmetry, which is the
real-data counterpart of the twin's `forward_ratio`.

Direction is reported **per chip and unanchored**: which end is "source" depends
on how the device was seated on the array, and the deposit does not record that.
A chip's asymmetry says how one-sided its propagation is, not which way the
device points.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("neuromps.etl")

# Multi Channel Systems label columns: the letter I is skipped on the array.
COLUMNS = ("A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "O", "P", "R")
COLUMN_INDEX = {name: index for index, name in enumerate(COLUMNS)}
LABEL = re.compile(r"^([A-HJ-R])(\d{1,2})$")

CHANNEL_LENGTH = 5  # electrodes per microchannel, per the deposit's methods
MAX_STEP_S = 0.001  # <= 1 ms between consecutive electrodes, per the same
MIN_EVENTS = 20  # a run carrying fewer events than this is not called a channel


def grid_position(label: str) -> tuple[int, int] | None:
    """Column index and row for an electrode label, or None if it is not one."""
    match = LABEL.match(label.strip())
    if match is None or match.group(1) not in COLUMN_INDEX:
        return None
    return COLUMN_INDEX[match.group(1)], int(match.group(2))


def channel_runs(labels: list[str], orientation: str = "column") -> list[list[str]]:
    """Every run of five adjacent electrodes along one column, or along one row.

    The deposit does not record how each device was seated on the array, so
    which axis the microchannels run along is an assumption. `orientation`
    exists to let that assumption be tested rather than trusted: "column" walks
    consecutive rows within a column, "row" walks consecutive columns within a
    row.
    """
    if orientation not in ("column", "row"):
        raise ValueError(f"orientation must be 'column' or 'row', not {orientation!r}")
    lines: dict[int, dict[int, str]] = {}
    for label in labels:
        position = grid_position(label)
        if position is None:
            continue
        column, row = position
        fixed, moving = (column, row) if orientation == "column" else (row, column)
        lines.setdefault(fixed, {})[moving] = label
    runs = []
    for positions in lines.values():
        for start in sorted(positions):
            window = [positions.get(start + step) for step in range(CHANNEL_LENGTH)]
            if all(window):
                runs.append([label for label in window if label])
    return runs


def ordered_events(trains: list[np.ndarray]) -> int:
    """Events appearing on every train in order, each step within `MAX_STEP_S`.

    Each spike of the first train can begin at most one event: the chain follows
    the *next* spike on each subsequent electrode, so a dense downstream train
    cannot inflate the count.
    """
    current = np.asarray(trains[0], dtype=float)
    for following in trains[1:]:
        following = np.asarray(following, dtype=float)
        if current.size == 0 or following.size == 0:
            return 0
        index = np.searchsorted(following, current, side="right")
        valid = index < following.size
        if not valid.any():
            return 0
        candidate = following[np.clip(index, 0, following.size - 1)]
        delta = candidate - current
        keep = valid & (delta > 0) & (delta <= MAX_STEP_S)
        current = candidate[keep]
    return int(current.size)


def _best_chain(trains: list[np.ndarray], min_electrodes: int) -> int:
    """Events along the run, allowing `len(trains) - min_electrodes` electrodes to miss.

    A dead or low-SNR electrode in the middle of a channel breaks a strict chain
    and loses every event that crossed it. With one allowed omission this takes
    the best of the five four-electrode subsequences, which is optimistic - an
    event is counted if any of them fires - and is meant as a sensitivity
    bound, not as a better estimate.
    """
    if min_electrodes >= len(trains):
        return ordered_events(trains)
    from itertools import combinations

    best = 0
    for keep in combinations(range(len(trains)), min_electrodes):
        best = max(best, ordered_events([trains[i] for i in keep]))
    return best


def propagation(trains: list[np.ndarray], min_electrodes: int | None = None) -> tuple[int, int]:
    """Forward and backward event counts along one candidate microchannel."""
    required = min_electrodes or len(trains)
    return (
        _best_chain(trains, required),
        _best_chain(list(reversed(trains)), required),
    )


def chip_summary(
    spikes: dict[str, np.ndarray],
    duration_s: float,
    orientation: str = "column",
    min_electrodes: int | None = None,
) -> dict:
    """Detected microchannels and the chip's propagation asymmetry.

    ``asymmetry`` is ``(forward - backward) / (forward + backward)`` summed over
    the channels this chip actually carries, and is ``None`` when no run clears
    ``MIN_EVENTS`` - a chip with no detectable propagation has no direction, and
    reporting zero for it would confuse "balanced" with "not measured".
    """
    detected, forward_total, backward_total = [], 0, 0
    for run in channel_runs(list(spikes), orientation):
        forward, backward = propagation([spikes[label] for label in run], min_electrodes)
        if forward + backward >= MIN_EVENTS:
            detected.append({"electrodes": run, "forward": forward, "backward": backward})
            forward_total += forward
            backward_total += backward
    detected = _drop_overlaps(detected)
    forward_total = sum(d["forward"] for d in detected)
    backward_total = sum(d["backward"] for d in detected)
    total = forward_total + backward_total
    axis = 1 if orientation == "column" else 0
    rows = [int(grid_position(e)[axis]) for d in detected for e in d["electrodes"]]
    return {
        "n_channels_detected": len(detected),
        "forward_events": forward_total,
        "backward_events": backward_total,
        "asymmetry": round((forward_total - backward_total) / total, 4) if total else None,
        "events_per_minute": round(total / (duration_s / 60), 2) if duration_s else None,
        "channel_rows": [min(rows), max(rows)] if rows else None,
        "channels": detected,
    }


def _drop_overlaps(detected: list[dict]) -> list[dict]:
    """Greedily keep the strongest runs that share no electrode with a kept one.

    Consecutive five-electrode windows share four electrodes, so one physical
    microchannel would otherwise be counted several times over. Keeping the
    strongest and discarding anything overlapping it leaves one entry per
    channel, while still allowing two genuinely separate channels in one column.
    """
    kept: list[dict] = []
    for entry in sorted(detected, key=lambda d: -(d["forward"] + d["backward"])):
        electrodes = set(entry["electrodes"])
        if not any(electrodes & set(other["electrodes"]) for other in kept):
            kept.append(entry)
    return sorted(kept, key=lambda d: d["electrodes"][0])


def read_recording(path: Path) -> tuple[dict[str, np.ndarray], float, list[str]]:
    """Spike times in seconds per electrode, the duration, and any unreadable entities.

    One recording in the deposit (`2024-04-29T11-00-28chip_3_DIV17_control_baseline.h5`)
    has two channel datasets whose chunks fail to decompress. Dropping the
    recording would discard 250 good channels; failing on it would stop the
    source. The unreadable entities are skipped and returned, so the caller can
    record them as a QC flag rather than quietly losing channels.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        recording = handle["Data/Recording_0"]
        duration_s = float(recording.attrs.get("Duration", 0)) / 1e6
        stream = recording["TimeStampStream/Stream_0"]
        info = stream["InfoTimeStamp"][:]
        spikes, unreadable = {}, []
        for row in info:
            key = f"TimeStampEntity_{row['TimeStampEntityID']}"
            if key not in stream:
                continue
            label = row["Label"].decode() if isinstance(row["Label"], bytes) else str(row["Label"])
            try:
                raw = stream[key][0]
            except OSError:
                unreadable.append(label)
                continue
            spikes[label] = np.sort(raw.astype(np.float64)) * (10.0 ** int(row["Exponent"]))
    return spikes, duration_s, unreadable
