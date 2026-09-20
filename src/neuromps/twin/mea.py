"""Virtual MEA: place neurons and electrodes in space, and record what each electrode hears.

The real device is a planar grid of electrodes under two culture chambers
separated by a microchannel region. Here each compartment gets its own
rectangular chamber and its own electrode grid at the array's pitch; neurons are
scattered uniformly within their chamber, and an electrode picks up a neuron
whose soma lies within ``detection_radius_um``.

This replaces the placeholder mapping used in HD11, where electrodes listened to
an arbitrary fixed set of neurons. Geometry matters because electrode count,
pitch and detection radius jointly set how many neurons an electrode pools, and
that drives firing rate, synchrony and network-burst counts.

**Not modelled:** the interface electrodes over the microchannel region. Their
signals are axonal rather than somatic, which this soma-distance model does not
represent; the real interface strip is excluded from comparisons instead of
being approximated.
"""

import logging
from dataclasses import asdict, dataclass

import numpy as np

log = logging.getLogger("neuromps.twin")


@dataclass(frozen=True)
class ElectrodeLayout:
    """Electrode grid per compartment, in micrometres."""

    rows: int = 6
    cols: int = 8
    pitch_um: float = 200.0
    detection_radius_um: float = 120.0
    gap_um: float = 600.0
    margin_um: float = 100.0

    @property
    def electrodes_per_compartment(self) -> int:
        return self.rows * self.cols

    @property
    def chamber_width_um(self) -> float:
        """Width of one chamber: the grid plus a margin on each side."""
        return (self.cols - 1) * self.pitch_um + 2 * self.margin_um

    @property
    def chamber_height_um(self) -> float:
        return (self.rows - 1) * self.pitch_um + 2 * self.margin_um

    def as_record(self) -> dict:
        """Layout parameters for provenance."""
        return asdict(self)


def _chamber_origin(layout: ElectrodeLayout, compartment: int) -> float:
    """x of a chamber's left edge; compartment 1 sits past the gap."""
    return compartment * (layout.chamber_width_um + layout.gap_um)


def electrode_positions(layout: ElectrodeLayout) -> dict[str, tuple[float, float]]:
    """Electrode id -> (x, y) in micrometres, for both compartments."""
    positions = {}
    for compartment in (0, 1):
        origin = _chamber_origin(layout, compartment)
        for row in range(layout.rows):
            for col in range(layout.cols):
                positions[f"c{compartment}_r{row}_k{col}"] = (
                    origin + layout.margin_um + col * layout.pitch_um,
                    layout.margin_um + row * layout.pitch_um,
                )
    return positions


def place_neurons(params, layout: ElectrodeLayout, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scatter neurons uniformly inside their own chamber.

    Returns (positions, compartment_of_neuron) with positions in micrometres.
    """
    rng = np.random.default_rng(seed)
    n = params.n_per_compartment
    compartment = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    positions = np.empty((compartment.size, 2))
    for c in (0, 1):
        origin = _chamber_origin(layout, c)
        block = compartment == c
        positions[block, 0] = rng.uniform(origin, origin + layout.chamber_width_um, n)
        positions[block, 1] = rng.uniform(0.0, layout.chamber_height_um, n)
    return positions, compartment


def _audible_neurons(
    electrode: tuple[float, float],
    compartment_id: int,
    neuron_positions: np.ndarray,
    compartment_of_neuron: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Indices of neurons this electrode can hear: same chamber, within radius."""
    same = compartment_of_neuron == compartment_id
    distance = np.hypot(
        neuron_positions[:, 0] - electrode[0], neuron_positions[:, 1] - electrode[1]
    )
    return np.flatnonzero(same & (distance <= radius))


def record_electrodes(
    spike_times: np.ndarray,
    spike_neurons: np.ndarray,
    neuron_positions: np.ndarray,
    compartment_of_neuron: np.ndarray,
    electrodes: dict[str, tuple[float, float]],
    layout: ElectrodeLayout,
) -> dict[str, np.ndarray]:
    """Spike times heard by each electrode, ready for `features/mea.py`."""
    order = np.argsort(spike_times)
    times, neurons = spike_times[order], spike_neurons[order]
    heard = {}
    for name, position in electrodes.items():
        compartment_id = int(name[1])
        audible = _audible_neurons(
            position,
            compartment_id,
            neuron_positions,
            compartment_of_neuron,
            layout.detection_radius_um,
        )
        heard[name] = times[np.isin(neurons, audible)] if audible.size else np.empty(0)
    return heard


def simulate_recording(params, layout: ElectrodeLayout | None = None, seed: int = 0) -> dict:
    """Run the network and return per-electrode spike trains plus the geometry used."""
    from neuromps.twin.network import simulate

    layout = layout or ElectrodeLayout()
    result = simulate(params)
    positions, compartment = place_neurons(params, layout, seed=seed)
    electrodes = electrode_positions(layout)
    heard = record_electrodes(
        result.spike_times, result.spike_neurons, positions, compartment, electrodes, layout
    )
    return {
        "spikes_by_electrode": heard,
        "duration_s": result.duration_s,
        "layout": layout.as_record(),
        "params": params.as_record(),
        "result": result,
    }


def electrode_coupling(heard: dict[str, np.ndarray], duration_s: float) -> dict:
    """Directed coupling between compartments, from what the electrodes heard.

    The real ETL measures coupling from the per-electrode spike trains a device
    detects (`etl/screenneuropharm.py`). Measuring the simulated coupling from
    every neuron instead would give the twin a cleaner population rate than any
    MEA can record, and would make the measure independent of electrode count -
    the one variable an electrode-count experiment has to vary. Same function,
    same input contract as the real recordings.
    """
    from neuromps.features.coupling import directed_coupling

    per_compartment = {
        c: {k: v for k, v in heard.items() if k.startswith(f"c{c}_")} for c in (0, 1)
    }
    return directed_coupling(per_compartment[0], per_compartment[1], duration_s)
