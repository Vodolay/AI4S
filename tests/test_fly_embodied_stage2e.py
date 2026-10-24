"""Stage 2e scoring rules, on synthetic replicate tables."""

from __future__ import annotations

import json

import pandas as pd

from neuromps.fly import embodied_stage2e as s2e


def _table(walking: bool) -> pd.DataFrame:
    rows = []
    for name in s2e.CONDITIONS:
        for seed in range(8):
            fw = name in ("forward_walking", "forward_walking_150")
            rows.append(
                {
                    "condition": name,
                    "drive": s2e.CONDITIONS[name][0],
                    "seed": seed,
                    "rhythmicity_L1_flexor": 0.6 if fw and walking else 0.1,
                    "phase_L1flex_L1ext": 0.5 if walking else None,
                    "phase_L1_R1": 0.45 if walking else None,
                    "phase_L1_L2": 0.55 if walking else None,
                    "phase_L1_L3": 0.05 if walking else None,
                    "period_L1_flexor_s": 0.1 if walking else None,
                    "rate_L1_flexor": 0.0 if name == "none" else 20.0,
                }
            )
    return pd.DataFrame(rows)


def test_a_walking_pattern_passes_every_rule_and_a_silent_one_fails_them() -> None:
    spec = json.loads(s2e.REGISTRATION.read_text())
    good = s2e.score(_table(True), spec)
    assert all(v["outcome"] == "pass" for v in good.values()), good
    bad = s2e.score(_table(False), spec)
    passing = {k for k, v in bad.items() if v["outcome"] == "pass"}
    assert passing == {"R0_silent_without_drive"}
