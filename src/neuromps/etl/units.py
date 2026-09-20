"""Dose-unit normalization, shared so the ETL modules cannot drift apart.

The schema admits `uM` and `nM` (and `mg_l`), so a millimolar dose has to be
**rescaled** to micromolar rather than relabelled. Getting this wrong is silent:
the number stays, the unit changes, and a 5 mM exposure becomes a 5 uM one.
"""

SCHEMA_DOSE_UNITS = ("uM", "nM", "mg_l")
# lowercase source unit -> (schema unit, multiplier applied to the value)
_CONVERSIONS = {
    "nm": ("nM", 1.0),
    "um": ("uM", 1.0),
    "µm": ("uM", 1.0),
    "mm": ("uM", 1000.0),
}


def normalize_dose(value: float | None, unit: str) -> tuple[float | None, str]:
    """Return the dose expressed in a unit the schema allows.

    Args:
        value: dose magnitude, or None when the source records no dose.
        unit: unit as written by the source, in any case.

    Raises:
        ValueError: if the unit is not one this project knows how to convert.
    """
    key = str(unit).strip().lower()
    if key not in _CONVERSIONS:
        raise ValueError(f"unknown dose unit {unit!r}; expected one of {sorted(_CONVERSIONS)}")
    normalized_unit, factor = _CONVERSIONS[key]
    return (None if value is None else float(value) * factor), normalized_unit
