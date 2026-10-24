"""Behavioural readouts of the fly model, from the experiments Shiu et al. 2024 validated.

Each readout is the trial-averaged firing rate of a neuron whose activity the
paper links to a behaviour, under the sensory drive the paper used:

- **feeding**: MN9, the proboscis motor neuron, while sugar-sensing gustatory
  neurons are driven;
- **aversive gating**: how much of that drive driving bitter-sensing neurons at
  the same time removes (1 = all of it, 0 = none);
- **grooming**: aBN1, the antennal grooming interneuron, while JO-CE
  mechanosensory neurons are driven;
- **descending recruitment**: how many descending neurons fire under the sugar
  drive, a measure of how selectively the brain routes one stimulus to one
  motor output.

These are firing rates of model neurons, not behaviour; "feeding" names the
neuron's published role, not something the model does.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from neuromps.fly.model import FlySpikes

NEURON_SETS = Path(__file__).with_name("shiu_neurons.json")
RECRUITED_HZ = 10.0


def neuron_sets(path: Path = NEURON_SETS) -> dict[str, list[int]]:
    """FlyWire root IDs of the published stimulus and readout neurons."""
    return json.loads(path.read_text())["sets"]


def rate_hz(trials: Sequence[FlySpikes], neurons: Sequence[int]) -> float:
    """Mean firing rate of `neurons` (model indices), averaged over trials and neurons."""
    if not trials:
        raise ValueError("no trials")
    wanted = np.asarray(neurons)
    per_trial = [np.isin(t.neurons, wanted).sum() / t.duration_s / wanted.size for t in trials]
    return float(np.mean(per_trial))


def recruited(trials: Sequence[FlySpikes], neurons: Sequence[int], hz: float = RECRUITED_HZ) -> int:
    """How many of `neurons` fire at `hz` or more, averaged over trials."""
    wanted = np.asarray(neurons)
    total = sum(t.duration_s for t in trials)
    spikes = np.concatenate([t.neurons[np.isin(t.neurons, wanted)] for t in trials])
    counts = np.bincount(np.searchsorted(np.sort(wanted), spikes), minlength=wanted.size)
    return int(np.sum(counts / total >= hz))


def gating_index(sugar_hz: float, sugar_bitter_hz: float) -> float | None:
    """Fraction of the sugar-driven MN9 rate that the added bitter drive removes."""
    if sugar_hz <= 0:
        return None
    return float(1.0 - sugar_bitter_hz / sugar_hz)
