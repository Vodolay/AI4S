"""Stage 2f scoring: the precondition, the conditional lesion test, the registered rules."""

from __future__ import annotations

import json

import pandas as pd

from neuromps.fly import embodied_stage2f as s2f


def _table(dng100_hz: float, rhythm: float) -> pd.DataFrame:
    rows = []
    for name, (base, _) in s2f.CONDITIONS.items():
        for seed in range(8):
            fw = name.startswith("forward_walking")
            rows.append(
                {
                    "condition": name,
                    "drive": "forward_walking" if fw or name == "lesion" else base,
                    "seed": seed,
                    "dng100_hz": dng100_hz,
                    "rhythmicity_L1_flexor": rhythm if fw else 0.0,
                    "phase_L1flex_L1ext": 0.5,
                    "phase_L1_R1": 0.5,
                    "phase_L1_L2": 0.5,
                    "phase_L1_L3": 0.0,
                    "period_L1_flexor_s": 0.1,
                }
            )
    return pd.DataFrame(rows)


SPEC = json.loads(s2f.REGISTRATION.read_text())


def test_a_walking_pattern_passes_everything() -> None:
    got = s2f.score(_table(50.0, 0.6), SPEC)
    assert all(v["outcome"] == "pass" for v in got.values()), got


def test_without_the_command_on_the_rest_is_uninterpretable() -> None:
    got = s2f.score(_table(6.0, 0.6), SPEC)
    assert got["S0_the_command_is_on"]["outcome"] == "fail"
    assert {v["outcome"] for k, v in got.items() if k != "S0_the_command_is_on"} == {
        "uninterpretable"
    }


def test_the_lesion_test_cannot_pass_without_a_rhythm() -> None:
    got = s2f.score(_table(50.0, 0.0), SPEC)
    assert got["S1_the_walking_command_makes_a_rhythm"]["outcome"] == "fail"
    assert got["S6_the_rhythm_needs_the_three_neuron_circuit"]["outcome"] == "uninterpretable"
