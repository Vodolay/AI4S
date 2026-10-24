"""How far a sugar stimulus spreads in each connectome model, by neuron class.

The published model (Shiu et al.) on FlyWire responds to driving its sugar
receptor neurons with activity in a few hundred neurons. ``python -m
neuromps.fly.spread`` asks the same of the male CNS model at the published and
at the rescaled weight: how many neurons of each class reach 10 Hz under the
replication's sugar drive. Two 1 s trials each; diagnostic, not registered.
Everything here is **simulated**.
"""

from __future__ import annotations

import json

import numpy as np

from neuromps.fly import behaviour, connectome, malecns, model
from neuromps.metrics import update_metrics

SEEDS = (9100, 9101)
WEIGHTS = {"published": 0.275, "rescaled": 0.1596}
CLASSES = ("cb_intrinsic", "vnc_intrinsic", "descending_neuron", "vnc_motor")


def active(con, excite, w_syn: float) -> np.ndarray:
    """Mean rate over the trials of every neuron; >= 10 Hz counts as active."""
    rates = [
        np.bincount(
            model.simulate(con, model.FlyParams(seed=s, w_syn_mv=w_syn), excite=excite).neurons,
            minlength=con.n_neurons,
        )
        for s in SEEDS
    ]
    return np.mean(rates, axis=0)


def main(argv: list[str] | None = None) -> int:
    """CLI: count active neurons per class in both models and write metrics."""
    fly = connectome.cached()
    sets = behaviour.neuron_sets()
    out = {
        "flywire783": {
            "active_total": int(
                (active(fly, fly.index_of(sets["sugar_grn_right"]), 0.275) >= 10).sum()
            )
        }
    }
    male, ann = malecns.cached()
    matched = malecns.match_sets(
        sets, connectome.read_annotations()[["root_id", "cell_type", "side"]], ann
    )
    excite = male.index_of(matched["sugar_grn_right"])
    superclass = ann["superclass"].fillna("none").to_numpy()
    for label, w in WEIGHTS.items():
        on = active(male, excite, w) >= 10
        entry = {"w_syn_mv": w, "active_total": int(on.sum())}
        for cls in CLASSES:
            entry[cls] = {
                "active": int(on[superclass == cls].sum()),
                "total": int((superclass == cls).sum()),
            }
        out[f"malecns_{label}"] = entry
    update_metrics("fly_spread", out)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
