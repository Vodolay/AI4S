"""Brewer 2021 microtunnel electrodes: axonal crossings, measured not inferred.

The chamber-level analysis in ``models/tunnels.py`` asks whether coupling
asymmetry between two chambers tracks their firing rates. It cannot say what
the right answer would have been, because the chambers only see the ends of a
connection. This deposit also instruments the connection itself: 40 of the 44
electrodes in ``sorted-spikes/mea_tunnels.mat`` sit inside the microtunnels
between chambers, two per tunnel, so an axon potential travelling through a
tunnel is recorded twice and the sign of the delay is the direction it went.

That gives a physical direction to compare the chamber measure against.

Three things had to be read out of the files rather than assumed:

- **Channel names, not the ``coordinates`` field, give position.** In 8 of the
  50 spike files - the DG and CA1 chambers of four cultures - the stored
  coordinates are the array point-reflected through its centre and disagree
  with the channel name. The names follow the Multi Channel Systems convention
  in every file, so ``electrode_xy`` derives position from the name and
  ``coordinate_disagreements`` counts the files where the stored field differs.
- **The device was mounted in two different orientations.** Five cultures put
  the EC-DG tunnels on the left of the array and five put them on top. The
  orientation is recovered per culture from the tunnel file's own strip labels
  rather than assumed, and ``corner_subregions`` is checked against the chamber
  files' electrode positions in the nine cultures whose names are readable.
- **The anatomical labels are the depositors' naming**, as ``etl/brewer.py``
  already records. Nothing here depends on which subregion is which; it depends
  only on a tunnel connecting the two chambers it says it connects, which the
  electrode positions confirm.
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

log = logging.getLogger("neuromps.etl")

SOURCE = "brewer2021"
# Multi Channel Systems skips I so that no name reads as a digit.
COLUMN_LETTERS = "ABCDEFGHJKLM"
GRID = 12
# The four chambers sit in the array's corners and the tunnels in the strips
# between them; these are the coordinate bands, in electrode units.
LOW_BAND = (1, 5)
HIGH_BAND = (8, 12)
TUNNEL_BAND = (6, 7)
# The loop the depositors name their tunnels around. Used only to orient a
# tunnel's two electrodes consistently, never as an anatomical claim.
LOOP = ("EC", "DG", "CA3", "CA1")


@dataclass(frozen=True)
class Tunnel:
    """One microtunnel: two electrodes, and which chamber each end faces."""

    from_chamber: str
    to_chamber: str
    from_electrode: str
    to_electrode: str

    @property
    def pair(self) -> str:
        """The unordered chamber pair, named as ``models/tunnels.py`` names it."""
        first, second = sorted((self.from_chamber, self.to_chamber))
        return f"{first}->{second}"

    @property
    def forward_is_pair_order(self) -> bool:
        """Whether this tunnel's from-end is the first chamber of ``pair``."""
        return self.from_chamber == sorted((self.from_chamber, self.to_chamber))[0]


def electrode_xy(name: str) -> tuple[int, int] | None:
    """Grid position of an electrode from its name, or None if unparseable.

    Column letter gives x from the left; the number counts down from the top,
    so y increases upward and a larger y is nearer the top of the array.
    """
    if len(name) < 2 or name[0] not in COLUMN_LETTERS or not name[1:].isdigit():
        return None
    number = int(name[1:])
    if not 1 <= number <= GRID:
        return None
    return COLUMN_LETTERS.index(name[0]) + 1, GRID + 1 - number


def strip_of(position: tuple[int, int]) -> str | None:
    """Which band of tunnels an electrode lies in, or None for the centre."""
    x, y = position
    in_x, in_y = LOW_BAND[0] <= x <= LOW_BAND[1], LOW_BAND[0] <= y <= LOW_BAND[1]
    hi_x, hi_y = HIGH_BAND[0] <= x <= HIGH_BAND[1], HIGH_BAND[0] <= y <= HIGH_BAND[1]
    mid_x = TUNNEL_BAND[0] <= x <= TUNNEL_BAND[1]
    mid_y = TUNNEL_BAND[0] <= y <= TUNNEL_BAND[1]
    if mid_y and in_x:
        return "left"
    if mid_y and hi_x:
        return "right"
    if mid_x and hi_y:
        return "top"
    if mid_x and in_y:
        return "bottom"
    return None


# Which two strips meet at each corner of the array.
CORNER_STRIPS = {
    "top_left": ("left", "top"),
    "top_right": ("right", "top"),
    "bottom_left": ("left", "bottom"),
    "bottom_right": ("right", "bottom"),
}


def corner_subregions(strip_labels: dict[str, str]) -> dict[str, str]:
    """Which chamber sits in each corner, from the tunnel labels alone.

    A strip labelled ``EC-DG`` connects those two chambers, so the chamber in
    a corner is the one its two strips have in common. This recovers the
    culture's mounting orientation without reading the chamber files, which
    matters because one culture's channel names are unreadable.
    """
    corners = {}
    for corner, (first, second) in CORNER_STRIPS.items():
        if first not in strip_labels or second not in strip_labels:
            continue
        shared = set(strip_labels[first].split("-")) & set(strip_labels[second].split("-"))
        if len(shared) == 1:
            corners[corner] = shared.pop()
    return corners


def _corner_of(position: tuple[int, int]) -> str:
    """The corner an electrode's end of a tunnel faces."""
    x, y = position
    vertical = "top" if y > (TUNNEL_BAND[0] + TUNNEL_BAND[1]) / 2 else "bottom"
    horizontal = "left" if x <= (TUNNEL_BAND[0] + TUNNEL_BAND[1]) / 2 else "right"
    return f"{vertical}_{horizontal}"


def build_tunnels(names: list[str], subregions: list[str]) -> list[Tunnel]:
    """Pair up the tunnel electrodes and orient each pair between its chambers.

    Electrodes in the same strip that share the coordinate across the strip are
    the two ends of one tunnel. Each end is assigned to the corner it faces, and
    the corner's chamber comes from ``corner_subregions``.
    """
    placed = [
        (name, label, electrode_xy(name))
        for name, label in zip(names, subregions, strict=True)
        if electrode_xy(name) is not None
    ]
    strip_labels = {}
    for _, label, position in placed:
        strip = strip_of(position)
        if strip is not None and label != "NONE":
            strip_labels[strip] = label
    corners = corner_subregions(strip_labels)

    grouped: dict[tuple[str, int], list[tuple[str, tuple[int, int]]]] = {}
    for name, label, position in placed:
        strip = strip_of(position)
        if strip is None or label == "NONE":
            continue
        # The coordinate that runs across the strip identifies the tunnel; the
        # one that runs along it orders the two ends.
        across = position[0] if strip in ("left", "right") else position[1]
        grouped.setdefault((strip, across), []).append((name, position))

    tunnels = []
    for (_, _), members in sorted(grouped.items()):
        if len(members) != 2:
            continue
        ends = {}
        for name, position in members:
            chamber = corners.get(_corner_of(position))
            if chamber is not None:
                ends[chamber] = name
        if len(ends) != 2:
            continue
        # Orient along the depositors' loop so a tunnel's sign is stable across
        # cultures mounted differently.
        chambers = sorted(ends, key=lambda c: LOOP.index(c) if c in LOOP else len(LOOP))
        first, second = chambers
        if (LOOP.index(second) - LOOP.index(first)) % len(LOOP) != 1:
            first, second = second, first
        tunnels.append(Tunnel(first, second, ends[first], ends[second]))
    return tunnels


def read_tunnels(archive: zipfile.ZipFile, culture: str) -> dict:
    """Spike times per tunnel electrode, with names, labels and duration."""
    member = f"spike_trains_data/{culture}/sorted-spikes/mea_tunnels.mat"
    mea = sio.loadmat(io.BytesIO(archive.read(member)))["mea"][0, 0]
    sample_rate_hz = float(np.asarray(mea["par"][0, 0]["fs"]).ravel()[0])
    subregions = [str(x[0]) for x in mea["subregion"].ravel()]
    raw_names = [str(x[0]) for x in np.atleast_1d(mea["channel_names"]).ravel()]
    stored = [
        tuple(int(v) for v in x[0].ravel()) if np.size(x[0]) == 2 else None
        for x in mea["coordinates"].ravel()
    ]
    names = raw_names if len(raw_names) == len(subregions) else _names_from(stored)
    spikes, n_samples = {}, 0
    for index, name in enumerate(names):
        units = np.atleast_1d(mea["sorted_spike_data"][index, 0]).ravel()
        if not units.size:
            continue
        n_samples = max(n_samples, int(units[0].shape[0]))
        indices = np.concatenate([np.asarray(u.nonzero()[0]) for u in units])
        if indices.size:
            spikes[name] = np.sort(indices.astype(float) / sample_rate_hz)
    return {
        "culture": culture,
        "names": names,
        "subregions": subregions,
        "spikes": spikes,
        "duration_s": n_samples / sample_rate_hz if n_samples else 0.0,
        "names_recovered_from_coordinates": names is not raw_names,
    }


def _names_from(coordinates: list[tuple[int, int] | None]) -> list[str]:
    """Reconstruct channel names for the one culture that stores them as objects.

    Its ``channel_names`` is a MATLAB string object scipy cannot read. Its
    coordinates are recoverable because the tunnel files' stored coordinates
    agree with their names everywhere the names are readable, which
    ``coordinate_disagreements`` checks.
    """
    names = []
    for position in coordinates:
        if position is None:
            names.append("")
            continue
        x, y = position
        names.append(f"{COLUMN_LETTERS[x - 1]}{GRID + 1 - y}")
    return names


def coordinate_disagreements(archive: zipfile.ZipFile, culture: str) -> dict:
    """How often a file's stored coordinates contradict its own channel names.

    The reason this module derives position from names. Reported rather than
    silently worked around, because using the stored field would have flipped
    the direction of every tunnel in the affected cultures.
    """
    members = {"tunnels": f"spike_trains_data/{culture}/sorted-spikes/mea_tunnels.mat"}
    for region in LOOP:
        members[region] = f"spike_trains_data/{culture}/spikes/mea_{region}.mat"
    out = {}
    for label, member in members.items():
        mea = sio.loadmat(io.BytesIO(archive.read(member)))["mea"][0, 0]
        names = [str(x[0]) for x in np.atleast_1d(mea["channel_names"]).ravel()]
        stored = [
            tuple(int(v) for v in x[0].ravel()) if np.size(x[0]) == 2 else None
            for x in mea["coordinates"].ravel()
        ]
        if len(names) != len(stored):
            out[label] = None
            continue
        checked = [
            (electrode_xy(name), position)
            for name, position in zip(names, stored, strict=True)
            if electrode_xy(name) is not None and position is not None
        ]
        out[label] = {
            "n_checked": len(checked),
            "n_disagree": sum(1 for a, b in checked if a != b),
        }
    return out
