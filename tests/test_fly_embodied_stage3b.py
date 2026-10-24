"""Stage 3b: the recovery readouts and the scoring rules, on synthetic inputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neuromps.fly import embodied_stage3b as s3b

SPEC = json.loads(s3b.REGISTRATION.read_text())


def test_readouts_time_the_return_after_the_push() -> None:
    tibia = np.full(1000, 0.6)
    tibia[500:520] = np.linspace(0.6, 1.2, 20)
    tibia[520:550] = np.linspace(1.2, 0.62, 30)
    got = s3b.readouts(tibia)
    assert abs(got["pre_rad"] - 0.6) < 1e-9
    assert abs(got["excursion_rad"] - 0.6) < 1e-9
    assert got["recovery_ms"] == float(np.flatnonzero(tibia[520:] <= 0.65)[0])


def test_the_unperturbed_arm_runs_without_the_push() -> None:
    job = s3b._job("resistance_unperturbed", "grooming", 1, perturbed=False)
    assert job[0] == "resistance" and job[5] is None and job[6] == "resistance_unperturbed"


def _table(rec: dict[str, float], excursion: float = 0.6) -> pd.DataFrame:
    rows = []
    for arm in ("open", "resistance", "assistance", "sham"):
        for drive in ("grooming", "none"):
            for seed in range(16):
                rows.append(
                    {
                        "arm": arm,
                        "drive": drive,
                        "seed": seed,
                        "excursion_rad": excursion,
                        "recovery_ms": rec[arm],
                        "time_near_head": 0.9 if drive == "grooming" else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_a_faster_contingent_return_passes() -> None:
    got = s3b.score(_table({"open": 30, "resistance": 20, "assistance": 35, "sham": 29}), SPEC)
    assert all(v["outcome"] == "pass" for v in got.values()), got


def test_a_failed_push_makes_the_reflex_tests_uninterpretable() -> None:
    got = s3b.score(
        _table({"open": 30, "resistance": 20, "assistance": 35, "sham": 29}, 0.05), SPEC
    )
    assert {got[k]["outcome"] for k in got if k[:2] in ("M1", "M2", "M3")} == {"uninterpretable"}
