"""One simulation each, run in worker processes that load the connectome once.

Every job returns small summaries, never raw spikes: a 10 s whole-brain run
holds tens of millions of spikes, and only the features, readout rates and
region-pair measures leave the worker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from neuromps.fly import behaviour, connectome, direction, electrodes, model

_STATE: dict = {}
ZERO_WHEN_SILENT = (
    "mfr_hz",
    "burst_rate_per_min",
    "pct_spikes_in_bursts",
    "network_burst_rate_per_min",
)


@dataclass(frozen=True)
class Context:
    """What every worker needs besides the connectome."""

    spec: dict
    pairs: pd.DataFrame
    descending: np.ndarray
    # Root (or body) IDs of the stimulus and readout neurons; None reads the
    # published FlyWire sets. The male CNS run passes its matched sets.
    sets: dict | None = None


def _sets() -> dict:
    return _STATE["context"].sets or behaviour.neuron_sets()


def init(context: Context, release: str = "783") -> None:
    """Worker initialiser: load the connectome and the fixed array once per process."""
    if release == "malecns":
        from neuromps.fly import malecns

        con, ann = malecns.cached()
        _STATE.update(context=context, con=con, array=malecns.array(con, ann))
        return
    con = connectome.cached() if release == "783" else connectome.load_630()
    _STATE.update(context=context, con=con)
    if release == "783":
        a = context.spec["design"]["array"]
        _STATE["array"] = electrodes.build_array(
            con.neuropil, a["electrodes_per_region"], a["units_per_electrode"], a["array_seed"]
        )
        names, codes = np.unique(con.neuropil.astype(str), return_inverse=True)
        _STATE["region_names"], _STATE["region_code"] = names, codes


def _base(**changes) -> model.FlyParams:
    return replace(model.FlyParams(), **changes)


def _fill_silent(features: dict) -> dict:
    """A silent array: rate-like features are zero, synchrony and ISI CV undefined."""
    if not features.get("active_electrodes"):
        features = {**features, **dict.fromkeys(ZERO_WHEN_SILENT, 0.0), "silent": True}
    return features


def _region_trains(spikes: model.FlySpikes, units: np.ndarray | None) -> dict[str, dict]:
    """Region -> {region: merged spike times}, from all neurons or only the array's units."""
    times, neurons = spikes.times, spikes.neurons
    if units is not None:
        keep = np.isin(neurons, units)
        times, neurons = times[keep], neurons[keep]
    codes = _STATE["region_code"][neurons]
    order = np.argsort(codes, kind="stable")
    cuts = np.flatnonzero(np.diff(codes[order])) + 1
    out = {}
    for chunk in np.split(order, cuts):
        if chunk.size:
            name = str(_STATE["region_names"][codes[chunk[0]]])
            out[name] = {name: np.sort(times[chunk])}
    return out


def spontaneous(overrides: dict, seed: int, with_direction: bool) -> dict:
    """Array features of one spontaneous recording; for vehicle also the pair asymmetries."""
    ctx, con, array = _STATE["context"], _STATE["con"], _STATE["array"]
    d = ctx.spec["design"]
    params = _base(
        duration_s=d["spontaneous"]["duration_s"],
        background_mean_mv=d["background_mean_mv"],
        seed=seed,
        **overrides,
    )
    spikes = model.simulate(con, params)
    out = {
        "features": _fill_silent(electrodes.mea_features(spikes, array)),
        "spikes": int(spikes.times.size),
    }
    if with_direction:
        for label, units in (("whole_region", None), ("array", array.all_units())):
            trains = _region_trains(spikes, units)
            out[f"asymmetry_{label}"] = direction.spontaneous_asymmetry(
                trains, ctx.pairs, spikes.duration_s
            ).tolist()
    return out


def behaviour_trial(overrides: dict, protocol: str, seed: int) -> dict:
    """Readout rates of one 1 s trial of a published protocol (no background)."""
    ctx, con = _STATE["context"], _STATE["con"]
    sets = _sets()
    rate = ctx.spec["design"]["behaviour"]["stim_rate_hz"]
    drive = {
        "sugar": ("sugar_grn_right", None),
        "sugar_bitter": ("sugar_grn_right", "bitter_grn"),
        "jon_ce": ("jon_ce", None),
    }[protocol]
    excite = con.index_of(sets[drive[0]])
    excite2 = con.index_of(sets[drive[1]]) if drive[1] else []
    params = _base(
        stim_rate_hz=rate, stim2_rate_hz=rate if drive[1] else 0.0, seed=seed, **overrides
    )
    spikes = model.simulate(con, params, excite=excite, excite2=excite2)
    out = {
        name: behaviour.rate_hz([spikes], con.index_of(sets[name]))
        for name in ("mn9", "abn1", "adn1")
    }
    out["dn_recruited"] = behaviour.recruited([spikes], ctx.descending)
    return out


def evoked(region: str, seed: int) -> dict:
    """Pulse `neurons_driven` neurons of one region; early-window response of every region."""
    ctx, con = _STATE["context"], _STATE["con"]
    d, e = ctx.spec["design"], ctx.spec["design"]["evoked"]
    rng = np.random.default_rng(
        seed * 1000 + int(np.flatnonzero(_STATE["region_names"] == region)[0])
    )
    members = np.flatnonzero(con.neuropil == region)
    # A region smaller than the registered count is driven whole; see `small_regions`.
    drive = rng.choice(members, min(e["neurons_driven"], members.size), replace=False)
    params = _base(
        duration_s=e["duration_s"],
        background_mean_mv=d["background_mean_mv"],
        stim_rate_hz=e["rate_hz"],
        stim_period_ms=e["period_ms"],
        stim_width_ms=e["width_ms"],
        seed=seed,
    )
    spikes = model.simulate(con, params, excite=drive)
    out = {}
    for label, units in (("whole_region", None), ("array", _STATE["array"].all_units())):
        keep = ~np.isin(spikes.neurons, drive)
        if units is not None:
            keep &= np.isin(spikes.neurons, units)
        ratio = direction.evoked_log_ratio(
            spikes.times[keep],
            _STATE["region_code"][spikes.neurons[keep]],
            _STATE["region_names"].size,
            spikes.duration_s,
            e["period_ms"],
            tuple(e["window_ms"]),
            tuple(e["reference_window_ms"]),
        )
        out[label] = {
            str(n): float(r) for n, r in zip(_STATE["region_names"], ratio, strict=True) if n
        }
    return out


def replication_trial(seed: int) -> float:
    """MN9 rate under the published sugar drive, on whichever release this worker loaded."""
    con = _STATE["con"]
    sets = behaviour.neuron_sets()
    sugar = [r for r in sets["sugar_grn_right"] if r in set(con.root_ids.tolist())]
    params = _base(
        stim_rate_hz=_STATE["context"].spec["design"]["behaviour"]["stim_rate_hz"], seed=seed
    )
    spikes = model.simulate(con, params, excite=con.index_of(sugar))
    return behaviour.rate_hz([spikes], con.index_of(sets["mn9"]))


def small_regions(con: connectome.Connectome, regions: list[str], neurons_driven: int) -> dict:
    """Regions with fewer neurons than the registered drive: a deviation, reported as such."""
    sizes = {r: int(np.sum(con.neuropil == r)) for r in regions}
    return {r: n for r, n in sizes.items() if n < neurons_driven}
