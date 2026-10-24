"""A virtual multi-electrode array laid over the fly brain's regions.

A real MEA electrode hears a few nearby units, not a whole population. Here
each brain region (neuropil) holds a small patch of electrodes, and each
electrode hears a handful of neurons drawn from that region without
replacement, so no neuron is heard twice. The recording is then scored by the
same extractor the chip recordings use (`features/mea.py`), so a fly feature and
a chip feature mean the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuromps.features.mea import extract_features
from neuromps.fly.connectome import UNPLACED
from neuromps.fly.model import FlySpikes


@dataclass(frozen=True)
class VirtualArray:
    """Electrode id -> model indices of the units it hears, and -> its region."""

    units: dict[str, np.ndarray]
    region: dict[str, str]

    def all_units(self) -> np.ndarray:
        """Every recorded neuron, for restricting the spike monitor."""
        return np.unique(np.concatenate(list(self.units.values())))

    def by_region(self) -> dict[str, list[str]]:
        """Region -> its electrodes."""
        out: dict[str, list[str]] = {}
        for electrode, region in self.region.items():
            out.setdefault(region, []).append(electrode)
        return out


def build_array(
    region_of_neuron: np.ndarray,
    electrodes_per_region: int = 4,
    units_per_electrode: int = 3,
    seed: int = 0,
) -> VirtualArray:
    """Place electrodes in every region with enough neurons to fill them."""
    rng = np.random.default_rng(seed)
    need = electrodes_per_region * units_per_electrode
    units: dict[str, np.ndarray] = {}
    region: dict[str, str] = {}
    for name in sorted(set(region_of_neuron.tolist()) - {UNPLACED}):
        members = np.flatnonzero(region_of_neuron == name)
        if members.size < need:
            continue
        drawn = rng.choice(members, size=need, replace=False)
        for k, chunk in enumerate(drawn.reshape(electrodes_per_region, units_per_electrode)):
            electrode = f"{name}_{k}"
            units[electrode] = np.sort(chunk)
            region[electrode] = name
    return VirtualArray(units=units, region=region)


def electrode_trains(spikes: FlySpikes, array: VirtualArray) -> dict[str, np.ndarray]:
    """Merged, sorted spike times each electrode hears."""
    return {
        electrode: np.sort(spikes.times[np.isin(spikes.neurons, members)])
        for electrode, members in array.units.items()
    }


def mea_features(
    spikes: FlySpikes, array: VirtualArray, electrodes: list[str] | None = None
) -> dict[str, float | int | None]:
    """The chip feature vector of the virtual recording (optionally a subset of electrodes)."""
    trains = electrode_trains(spikes, array)
    if electrodes is not None:
        trains = {e: trains[e] for e in electrodes}
    return extract_features(trains, spikes.duration_s)
