"""Stage 4e: the fraction readout and its scoring, on synthetic inputs."""

from __future__ import annotations

import json

import pandas as pd

from neuromps.fly import embodied_stage4d as s4d
from neuromps.fly import embodied_stage4e as s4e

SPEC = json.loads(s4e.REGISTRATION.read_text())


def test_fresh_draws() -> None:
    assert len(s4e.SEEDS) == 32 and not set(s4e.SEEDS) & set(s4d.SEEDS)


def test_fraction_counts_feasible_draws_at_the_rhythm_line() -> None:
    trials = pd.DataFrame(
        {
            "condition": ["a"] * 4,
            "feasible": [True, True, True, False],
            "score": [0.95, 0.3, 0.1, 0.9],
        }
    )
    assert s4e.fractions(trials)["a"] == 2 / 3


def _conds():
    names = ["vehicle"] + [f"{f}_{k}" for k in (1, 2) for f in s4d.FAMILIES]
    families = ["vehicle"] + list(s4d.FAMILIES) * 2
    doses = [0.0] + [0.2] * 5 + [0.6] * 5
    return pd.DataFrame({"condition": names, "family": families, "dose": doses}), names


def test_the_registered_pattern_passes() -> None:
    conds, names = _conds()
    frac = pd.Series(0.7, index=names)
    for f in ("gaba_block", "glucl_block"):
        frac[f"{f}_1"] = frac[f"{f}_2"] = 0.05
    frac["ach_block_2"] = 0.0
    chip = {"gaba_block": 0.6, "glucl_block": 0.6, "ach_block": 0.2, "gaba_potentiate": 0.6}
    got, _ = s4e.score(conds, frac, 32, chip, SPEC)
    assert [got[n]["outcome"] for n in s4e.PREDICTIONS] == ["pass"] * 4


def test_walking_that_fails_with_the_chip_is_not_caught_first() -> None:
    conds, names = _conds()
    frac = pd.Series(0.7, index=names)
    frac["ach_block_1"] = frac["ach_block_2"] = 0.0
    got, _ = s4e.score(conds, frac, 32, {"ach_block": 0.2}, SPEC)
    assert got["E1_acetylcholine_block_is_caught_first"]["outcome"] == "fail"
    got, _ = s4e.score(conds, frac.mul(0.1), 32, {"ach_block": 0.2}, SPEC)
    assert got["E1_acetylcholine_block_is_caught_first"]["outcome"] == "uninterpretable"
