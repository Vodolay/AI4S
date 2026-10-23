"""Checks on the fly study's chip readout that the registered analysis does not make.

``python -m neuromps.fly.checks`` asks whether the virtual array's network
bursts are coordination or a by-product of firing rate: each electrode's
train is shifted circularly by its own random offset, which keeps every rate
and destroys coincidence, and the burst rate is measured again. Run on the
vehicle and on the lowest GABA-block level, where network bursts first moved.

It also measures each chip feature's false-alarm rate: untreated recordings on
fresh seeds, split at random into two groups of three (the sweep averaged three
seeds per condition), and the fraction of splits whose two means already differ
by the detection threshold. A feature that "detects" early but also fires
between untreated groups is noise, not a sentinel. Everything written is
**simulated**.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from neuromps.features.mea import FeatureParams, _network_burst_rate
from neuromps.fly import connectome, design, electrodes, jobs, model
from neuromps.metrics import update_metrics

log = logging.getLogger("neuromps.fly")
SEED = 900
N_SHIFTS = 20


def shifted(trains: list[np.ndarray], duration_s: float, rng: np.random.Generator) -> list:
    """Each train moved by its own random circular offset: same rates, coincidences broken."""
    return [np.sort((t + rng.uniform(0, duration_s)) % duration_s) for t in trains]


def _one(overrides: dict) -> dict:
    con, array, ctx = jobs._STATE["con"], jobs._STATE["array"], jobs._STATE["context"]
    d = ctx.spec["design"]
    params = jobs._base(
        duration_s=d["spontaneous"]["duration_s"],
        background_mean_mv=d["background_mean_mv"],
        seed=SEED,
        **overrides,
    )
    spikes = model.simulate(con, params)
    trains = [t for t in electrodes.electrode_trains(spikes, array).values() if t.size]
    p, rng = FeatureParams(), np.random.default_rng(SEED)
    real = _network_burst_rate(trains, spikes.duration_s, p)
    null = [
        _network_burst_rate(shifted(trains, spikes.duration_s, rng), spikes.duration_s, p)
        for _ in range(N_SHIFTS)
    ]
    return {
        "network_bursts_per_min": real,
        "shifted_mean": float(np.mean(null)),
        "shifted_max": float(np.max(null)),
        "n_shifts": N_SHIFTS,
    }


VEHICLE_SEEDS = range(1000, 1012)
SPLITS = 500


def _vehicle(seed: int) -> dict:
    con, array, ctx = jobs._STATE["con"], jobs._STATE["array"], jobs._STATE["context"]
    d = ctx.spec["design"]
    params = jobs._base(
        duration_s=d["spontaneous"]["duration_s"],
        background_mean_mv=d["background_mean_mv"],
        seed=seed,
    )
    return jobs._fill_silent(electrodes.mea_features(model.simulate(con, params), array))


def false_alarms(
    per_seed: list[dict], threshold: float, group: int = 3, splits: int = SPLITS, seed: int = 0
) -> dict:
    """Per feature: fraction of random group-vs-group untreated splits crossing the threshold."""
    from neuromps.twin.perturbation import COMPARE_FEATURES, simulated_delta

    rng = np.random.default_rng(seed)
    hits = dict.fromkeys(COMPARE_FEATURES, 0)
    defined = dict.fromkeys(COMPARE_FEATURES, 0)
    spread: dict[str, list[float]] = {k: [] for k in COMPARE_FEATURES}
    for _ in range(splits):
        order = rng.permutation(len(per_seed))
        a, b = order[:group], order[group : 2 * group]
        mean = [
            {k: _nanmean([per_seed[i].get(k) for i in idx]) for k in COMPARE_FEATURES}
            for idx in (a, b)
        ]
        for k, v in simulated_delta(mean[0], mean[1]).items():
            if v is not None:
                spread[k].append(abs(v))
                defined[k] += 1
                hits[k] += int(abs(v) >= threshold)
    rates = {k: (hits[k] / defined[k] if defined[k] else None) for k in COMPARE_FEATURES}
    return {**rates, "q95_abs_log2": {k: _q95(spread[k]) for k in COMPARE_FEATURES}}


def _q95(values: list[float]) -> float | None:
    return round(float(np.quantile(values, 0.95)), 3) if values else None


def _nanmean(values) -> float | None:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def main(argv: list[str] | None = None) -> int:
    """CLI: run the shift check and write it to metrics.json."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from neuromps.fly.study import _context

    spec = design.registration()
    context, _ = _context(spec, connectome.cached())
    conditions = {"vehicle": {}, "gaba_block_0.8": {"gaba_scale": 0.8}}
    with ProcessPoolExecutor(14, initializer=jobs.init, initargs=(context, "783")) as pool:
        shift = pool.map(_one, conditions.values())
        per_seed = list(pool.map(_vehicle, VEHICLE_SEEDS))
        got = dict(zip(conditions, shift, strict=True))
    threshold = spec["design"]["chip_detection"]["abs_log2_threshold"]
    alarms = false_alarms(per_seed, threshold)
    update_metrics(
        "fly_checks",
        {
            "network_bursts_vs_shifted_trains": got,
            "seed": SEED,
            "false_alarm_rate_untreated": {
                "threshold": threshold,
                "group": 3,
                "splits": SPLITS,
                "seeds": len(per_seed),
                "by_feature": alarms,
            },
        },
    )
    print("false alarms at", threshold, alarms)
    for name, entry in got.items():
        print(name, entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
