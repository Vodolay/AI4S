"""How much synaptic input each neuron receives, FlyWire against the male CNS.

The Shiu et al. synaptic weight (0.275 mV per synapse) was set on FlyWire. The
male CNS is reconstructed with its own synapse detection; if it counts more
input synapses per neuron, the same weight drives its neurons harder, and any
stimulus spreads further. ``python -m neuromps.fly.excitability`` compares the
two connectomes as the models use them (excitatory and inhibitory input
synapses per neuron, and synapses per connection), overall and for the central
brain alone, which both reconstruct. Descriptive; nothing is simulated.
"""

from __future__ import annotations

import json

import numpy as np

from neuromps.fly import connectome, malecns
from neuromps.metrics import update_metrics


def input_stats(con: connectome.Connectome, keep: np.ndarray | None = None) -> dict:
    """Median and mean excitatory/inhibitory input synapses per neuron, and per connection."""
    mask = np.ones(con.n_neurons, bool) if keep is None else keep
    sel = mask[con.post]
    exc = np.bincount(
        con.post[sel & (con.sign > 0)], con.synapses[sel & (con.sign > 0)], minlength=con.n_neurons
    )[mask]
    inh = np.bincount(
        con.post[sel & (con.sign < 0)], con.synapses[sel & (con.sign < 0)], minlength=con.n_neurons
    )[mask]
    return {
        "neurons": int(mask.sum()),
        "excitatory_inputs_median": float(np.median(exc)),
        "excitatory_inputs_mean": round(float(exc.mean()), 2),
        "inhibitory_inputs_median": float(np.median(inh)),
        "inhibitory_inputs_mean": round(float(inh.mean()), 2),
        "synapses_per_connection": round(float(con.synapses[sel].mean()), 3),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: compare the two connectomes and write the result to metrics.json."""
    fly = connectome.cached()
    male, ann = malecns.cached()
    brain = ann["superclass"].fillna("").str.startswith(("cb_", "ol_", "visual")).to_numpy()
    out = {
        "flywire783": input_stats(fly),
        "malecns_all": input_stats(male),
        "malecns_brain": input_stats(male, brain),
    }
    out["mean_excitatory_input_ratio_malecns_brain_to_flywire"] = round(
        out["malecns_brain"]["excitatory_inputs_mean"]
        / out["flywire783"]["excitatory_inputs_mean"],
        3,
    )
    update_metrics("fly_excitability", out)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
