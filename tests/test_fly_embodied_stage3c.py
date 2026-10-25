"""Stage 3c: command mapping, tuning, readouts and scoring (flexion is a smaller angle)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neuromps.fly import embodied_stage3c as s3c

SPEC = json.loads(s3c.REGISTRATION.read_text())


def test_net_drive_maps_to_the_servo_range() -> None:
    assert s3c.command(1.0) == s3c.EXTENDED and s3c.command(-1.0) == s3c.FLEXED
    assert s3c.command(0.0) == 0.0 and abs(s3c.command(0.5) - 0.65) < 1e-9


def test_extension_tuned_claws_fire_as_the_tibia_extends() -> None:
    assert s3c.claw_rate(0.0, 0.0, "extension") == 0.0
    assert s3c.claw_rate(s3c.EXTENDED, 0.0, "extension") == s3c.RATE_MAX_HZ
    assert s3c.claw_rate(s3c.FLEXED, 0.0, "flexion") == s3c.RATE_MAX_HZ


def test_readouts_time_the_return_from_a_flexing_push() -> None:
    tibia = np.full(1000, 1.2)
    tibia[500:520] = np.linspace(1.2, 0.6, 20)
    tibia[520:540] = np.linspace(0.6, 1.18, 20)
    got = s3c.readouts(tibia)
    assert abs(got["excursion_rad"] - 0.6) < 1e-9
    assert got["recovery_ms"] == float(np.flatnonzero(tibia[520:] >= 1.15)[0])


def _table(rec: dict, pre: float = 1.2, exc: float = 0.6) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"arm": a, "seed": k, "pre_rad": pre, "excursion_rad": exc, "recovery_ms": r}
            for a, r in rec.items()
            for k in range(16)
        ]
    )


def test_a_faster_contingent_return_passes_and_no_hold_is_uninterpretable() -> None:
    rec = {"open": 30, "resistance": 20, "assistance": 35, "sham": 29}
    assert all(v["outcome"] == "pass" for v in s3c.score(_table(rec), SPEC).values())
    got = s3c.score(_table(rec, pre=0.1), SPEC)
    assert {got[k]["outcome"] for k in got if not k.startswith("K0")} == {"uninterpretable"}
