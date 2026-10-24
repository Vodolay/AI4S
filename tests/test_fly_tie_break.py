"""Tie resolution rules, on hand-made numbers."""

from __future__ import annotations

import pandas as pd

from neuromps.fly import tie_break


def _series(motor, chip) -> pd.DataFrame:
    doses = [0.1, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4]
    return pd.DataFrame({"dose": doses, "motor_disruption": motor, "abs_delta_mfr": chip})


def test_motor_first_resolves_to_chip_later() -> None:
    s = _series([0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])
    got = tie_break.resolve(s, 0.5, 0.5)
    assert got == {"motor_onset": 0.28, "chip_onset": 0.36, "resolution": "chip later"}


def test_chip_first_resolves_to_chip_earlier() -> None:
    s = _series([0.1, 0.3, 0.4, 0.4, 0.4, 0.4, 0.9], [0.0, 0.1, 0.6, 0.7, 0.8, 0.9, 0.9])
    assert tie_break.resolve(s, 0.5, 0.5)["resolution"] == "chip earlier"


def test_the_same_fine_dose_stays_tied() -> None:
    s = _series([0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], [0.0, 0.1, 0.2, 0.6, 0.7, 0.8, 0.9])
    assert tie_break.resolve(s, 0.5, 0.5)["resolution"] == "still tied"


def test_fine_conditions_carry_their_dose() -> None:
    conds = tie_break.fine_conditions(
        {"excitability_block": {"knob": "threshold_scale", "levels": [1.24]}},
        {"excitability_block": 1.4},
    )
    assert [(c.name, round(c.dose, 2)) for c in conds] == [
        ("excitability_block_1.24", 0.24),
        ("excitability_block_1.4", 0.4),
    ]
