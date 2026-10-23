"""From the sweep's raw numbers to the registered quantities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

READOUTS = ("mn9_sugar", "mn9_sugar_bitter", "abn1_jon_ce", "dn_recruited")


def log_change(treated: float, vehicle: float) -> float:
    """log2((x + 1) / (x_vehicle + 1)): defined at zero, in rate or count units."""
    return float(np.log2((treated + 1.0) / (vehicle + 1.0)))


def disruption(treated: dict, vehicle: dict) -> float:
    """Mean absolute log change of the four behavioural readouts."""
    return float(np.mean([abs(log_change(treated[k], vehicle[k])) for k in READOUTS]))


def trend(doses, values) -> float | None:
    """Spearman rank correlation of a readout with dose."""
    rho = spearmanr(doses, values)[0]
    return None if not np.isfinite(rho) else float(rho)


def _first(rows: pd.DataFrame, column: str, threshold: float) -> float | None:
    hit = rows[rows[column] >= threshold]
    return None if hit.empty else float(hit["dose"].min())


def sentinel(rows: pd.DataFrame, chip_threshold: float, behaviour_threshold: float) -> dict:
    """Per family: lowest dose the chip detects, lowest that impairs behaviour, which comes first.

    ``chip_first`` is None for a family that never impairs behaviour: there is
    nothing for the chip to have warned about.
    """
    out = {}
    for family, rows_f in rows.groupby("family"):
        chip = _first(rows_f, "abs_delta_mfr", chip_threshold)
        animal = _first(rows_f, "disruption", behaviour_threshold)
        first = None if animal is None else bool(chip is not None and chip <= animal)
        out[family] = {"first_chip_dose": chip, "first_behaviour_dose": animal, "chip_first": first}
    return out
