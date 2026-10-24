"""The roadmap inventory counts what it says, on a hand-made table."""

from __future__ import annotations

import pandas as pd

from neuromps.fly.roadmap_inventory import inventory


def test_inventory_counts_muscle_typed_leg_motor_neurons_and_proprioceptors() -> None:
    table = pd.DataFrame(
        {
            "status": ["Traced"] * 5 + ["Orphan"],
            "superclass": [
                "vnc_motor",
                "vnc_motor",
                "vnc_motor",
                "vnc_sensory",
                "vnc_sensory",
                "vnc_motor",
            ],
            "subclass": ["fl", "fl", "hl", "chordotonal organ", "leg", "fl"],
            "type": [
                "Sternal adductor MN",
                "MNfl01",
                "Tibia flexor MN",
                None,
                None,
                "Fe reductor MN",
            ],
            "class": [
                None,
                None,
                None,
                "mechanosensory_proprioceptive",
                "mechanosensory_tactile",
                None,
            ],
        }
    )
    got = inventory(table)
    assert got["leg_motor_neurons"]["front_leg"] == {"total": 2, "typed_by_muscle": 1}
    assert got["leg_motor_neurons_typed_by_muscle"] == 2
    assert got["vnc_proprioceptive_sensory_neurons"] == 1
    assert got["vnc_chordotonal_organ_neurons"] == 1
