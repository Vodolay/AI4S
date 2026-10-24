"""Exploratory, not registered: why the tibia flexors stay silent under the walking command.

``python -m neuromps.fly.flexor_diagnostic`` asks, after stages 2a and 2c failed,
whether the flexor motor pools are held down by inhibition or simply receive no
excitation from the neurons the walking command (BDN2) activates. Two measures:
the pools' rates with every inhibitory connection removed, and the excitatory and
inhibitory synapses each pool receives from the neurons active under the command.
Exploration seeds only; nothing here is a test. Everything is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from neuromps.fly import embodied_stage1c, fictive, model, scoring
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import outputs_dir

log = logging.getLogger("neuromps.fly")
SEEDS = range(9300, 9304)
ACTIVE_HZ = 1.0
_STATE: dict = {}


def without_inhibition(con):
    """The same network with every inhibitory connection's synapse count set to zero."""
    return replace(con, synapses=np.where(con.sign < 0, 0, con.synapses))


def _init() -> None:
    fictive._init()
    con = embodied_stage1c._STATE["con"]
    _STATE.update(intact=con, no_inhibition=without_inhibition(con))


def _trial(job) -> dict:
    network, seed = job
    con = _STATE[network]
    spikes = model.simulate(
        con,
        model.FlyParams(duration_s=fictive.DURATION_S, stim_rate_hz=150.0, seed=seed),
        excite=fictive._STATE["drive"]["forward_walking"],
    )
    rates = np.bincount(spikes.neurons, minlength=con.n_neurons) / fictive.DURATION_S
    return {"network": network, "seed": seed, "rates": rates}


def pool_rates(rates: np.ndarray, pools: dict) -> dict:
    """Mean rate over all six legs' flexor pools and over their extensor pools."""
    out = {}
    for kind in fictive.MUSCLES:
        idx = np.concatenate([v for (leg, k), v in pools.items() if k == kind])
        out[kind] = round(float(rates[idx].mean()), 3)
    return out


def inputs_from(con, sources: np.ndarray, targets: np.ndarray) -> dict:
    """Excitatory and inhibitory synapses from `sources` onto `targets`, per target neuron."""
    hit = np.isin(con.pre, sources) & np.isin(con.post, targets)
    exc = int(con.synapses[hit & (con.sign > 0)].sum())
    inh = int(con.synapses[hit & (con.sign < 0)].sum())
    n = max(len(targets), 1)
    inh_by_class = {
        str(c): round(float(con.synapses[hit & (con.sign < 0) & (con.pre_class == c)].sum()) / n, 2)
        for c in np.unique(con.pre_class[hit & (con.sign < 0)])
    }
    return {
        "excitatory_per_neuron": round(exc / n, 2),
        "inhibitory_per_neuron": round(inh / n, 2),
        "inhibitory_by_transmitter_per_neuron": inh_by_class,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: run the exploratory diagnostic and write it to metrics.json."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    jobs = [(n, s) for n in ("intact", "no_inhibition") for s in SEEDS]
    with ProcessPoolExecutor(memory_bounded_workers(per_worker_gb=8.0), initializer=_init) as pool:
        trials = list(pool.map(_trial, jobs))
    _init()
    con, pools = _STATE["intact"], fictive._STATE["pools"]
    rates = {
        n: np.mean([t["rates"] for t in trials if t["network"] == n], axis=0)
        for n in ("intact", "no_inhibition")
    }
    active = np.flatnonzero(rates["intact"] >= ACTIVE_HZ)
    everyone = np.arange(con.n_neurons)
    result = {"label": "EXPLORATORY, not registered: tibia flexor diagnostic (SIMULATED)"}
    result["pool_rates_hz"] = {n: pool_rates(r, pools) for n, r in rates.items()}
    result["active_neurons_under_walking_command"] = int(active.size)
    for kind in fictive.MUSCLES:
        idx = np.concatenate([v for (leg, k), v in pools.items() if k == kind])
        result[f"{kind}_inputs"] = {
            "from_active": inputs_from(con, active, idx),
            "from_all": inputs_from(con, everyone, idx),
            "pool_neurons": int(idx.size),
        }
    out = outputs_dir() / "fly" / "embodied_stage2c_flexor_diagnostic_SIMULATED.json"
    out.write_text(json.dumps(scoring.clean(result), indent=2) + "\n")
    update_metrics("embodied_stage2c_diagnostic", scoring.clean(result))
    print(json.dumps(scoring.clean(result), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
