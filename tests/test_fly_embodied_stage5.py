"""Stage 5: the simulated-direction rule and the scoring of comparisons."""

from __future__ import annotations

import json

from neuromps.fly import embodied_stage5 as s5

SPEC = json.loads(s5.REGISTRATION.read_text())
MEDIANS = {"vehicle": {"m": 0.30}, "f_low": {"m": 0.29}, "f_high": {"m": 0.05}, "g_1": {"m": 0.9}}
DOSES = {"f": {"f_low": 0.2, "f_high": 0.6}, "g": {"g_1": 0.5}}


def test_the_direction_is_taken_at_the_first_dose_crossing_the_line() -> None:
    assert s5.simulated_direction("f", "m", MEDIANS, DOSES) == "decrease"
    assert s5.simulated_direction("g", "m", MEDIANS, DOSES) == "increase"
    flat = {"vehicle": {"m": 0.3}, "h_1": {"m": 0.31}}
    assert s5.simulated_direction("h", "m", flat, {"h": {"h_1": 0.5}}) == "none"


def _row(category: str, agrees: bool, family: str = "gaba_block") -> dict:
    return {"category": category, "agrees": agrees, "family": family, "movement": "proboscis_sugar"}


def test_too_few_primary_comparisons_are_uninterpretable() -> None:
    got = s5.score([_row("transmitter-neuron", True)], SPEC)
    assert {v["outcome"] for v in got.values()} == {"uninterpretable"}


def test_majority_agreement_passes_v1() -> None:
    rows = [
        _row("pharmacological", True),
        _row("pharmacological", True),
        _row("receptor-genetic", False, "ach_block"),
    ]
    got = s5.score(rows, SPEC)
    assert got["V1_the_pharmacology_agrees"]["outcome"] == "pass"
    assert got["V2_disinhibition_and_feeding_agree"]["outcome"] == "pass"
