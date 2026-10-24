"""What the male CNS connectome already provides for an embodied, closed-loop fly.

``python -m neuromps.fly.roadmap_inventory`` counts, from the MaleCNS v1.0
annotations already fetched: leg motor neurons whose cell type names the muscle
they drive (the neuron-to-muscle map a body model needs), and proprioceptive
sensory neurons of the nerve cord (the feedback path from joints and forces back
into the network). Inventory only; nothing is simulated.
"""

from __future__ import annotations

import json

import pandas as pd

from neuromps.fly import malecns
from neuromps.metrics import update_metrics

MUSCLE_WORDS = (
    "flexor|extensor|adductor|abductor|rotator|reductor|levator|depressor|promotor|remotor"
    "|retractor|protractor"
)
LEGS = {"fl": "front_leg", "ml": "middle_leg", "hl": "hind_leg"}


def inventory(annotations: pd.DataFrame) -> dict:
    """Leg motor neurons typed by muscle, and nerve-cord proprioceptive sensory neurons."""
    traced = annotations[annotations["status"] == "Traced"]
    motor = traced[traced["superclass"].isin(["vnc_motor", "cb_motor"])]
    out: dict = {"leg_motor_neurons": {}}
    for code, leg in LEGS.items():
        rows = motor[motor["subclass"] == code]
        named = rows["type"].fillna("").str.contains(MUSCLE_WORDS, case=False)
        out["leg_motor_neurons"][leg] = {
            "total": int(len(rows)),
            "typed_by_muscle": int(named.sum()),
        }
    out["leg_motor_neurons_total"] = sum(v["total"] for v in out["leg_motor_neurons"].values())
    out["leg_motor_neurons_typed_by_muscle"] = sum(
        v["typed_by_muscle"] for v in out["leg_motor_neurons"].values()
    )
    sensory = traced[traced["superclass"] == "vnc_sensory"]
    out["vnc_proprioceptive_sensory_neurons"] = int(
        (sensory["class"] == "mechanosensory_proprioceptive").sum()
    )
    out["vnc_chordotonal_organ_neurons"] = int((sensory["subclass"] == "chordotonal organ").sum())
    out["vnc_campaniform_sensilla_neurons"] = int(
        (sensory["subclass"] == "campaniform sensilla").sum()
    )
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: count and write the inventory to metrics.json."""
    annotations = pd.read_feather(malecns.RAW / malecns.source.FILES[1])
    found = inventory(annotations)
    update_metrics("fly_roadmap_inventory", found)
    print(json.dumps(found, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
