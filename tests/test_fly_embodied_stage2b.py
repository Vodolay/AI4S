"""Stage 2b scoring rules and the pool-to-command mapping, on synthetic inputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neuromps.fly import embodied_stage2b as s2b


def test_extensor_drive_commands_extension_and_antagonist_drive_retraction() -> None:
    n = 5
    traces = {p: np.zeros(n) for p in s2b.POOLS}
    traces["rostrum_protractor"] = np.full(n, 40.0)
    traces["haustellum_flexor"] = np.full(n, 10.0)
    cmd = s2b.commands(traces, r_full=20.0)
    np.testing.assert_allclose(cmd["rostrum"], -1.24)
    np.testing.assert_allclose(cmd["haustellum"], 0.5 * 0.7)


def test_onset_is_the_first_time_at_half_the_peak() -> None:
    assert s2b.onset_s(np.array([0, 0, -0.2, -0.6, -1.0])) == 3 * s2b.DT_S
    assert s2b.onset_s(np.zeros(4)) is None


def _table(peaks: dict[str, float], act_ok: bool = True) -> pd.DataFrame:
    rows = []
    for r_full in s2b.R_FULL:
        for drive, peak in peaks.items():
            for seed in range(8):
                row = {"drive": drive, "seed": seed, "r_full": r_full, "peak_extension_mm": peak}
                for p in s2b.SCORED:
                    extending = p in ("rostrum_protractor", "haustellum_extensor")
                    row[f"act_{p}"] = 0.8 if extending == act_ok else 0.1
                    row[f"rate_{p}"] = 0.0 if drive == "none" else 5.0
                rows.append(row)
    return pd.DataFrame(rows)


def test_the_registered_rules_pass_a_feeding_pattern_and_fail_without_gating() -> None:
    spec = json.loads(s2b.REGISTRATION.read_text())
    good = {"sugar": 0.3, "sugar_bitter": 0.05, "bitter": 0.0, "jon_ce": 0.0, "none": 0.0}
    got = s2b.score(_table(good), spec, e_full=0.35)
    assert all(v["outcome"] == "pass" for v in got.values()), got
    ungated = good | {"sugar_bitter": 0.3}
    got = s2b.score(_table(ungated, act_ok=False), spec, e_full=0.35)
    assert got["G3_bitter_gates_extension"]["outcome"] == "fail"
    assert got["G2_both_joints_are_driven_to_extend"]["outcome"] == "fail"
    assert got["G5_the_activation_scale_does_not_decide_it"]["outcome"] == "fail"
