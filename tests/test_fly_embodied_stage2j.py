"""Stage 2j: fresh draws, per-draw shuffles, and the predictions read from stage 2g's scores."""

from __future__ import annotations

import json

import numpy as np

from neuromps.fly import embodied_stage2i as s2i
from neuromps.fly import embodied_stage2j as s2j


def test_fresh_draws() -> None:
    assert len(s2j.SEEDS) == 32 and not set(s2j.SEEDS) & set(s2i.SEEDS)


def test_each_draw_gets_its_own_permutation_of_the_same_values() -> None:
    sizes = np.arange(1.0, 11.0)
    got = s2j.shuffled(sizes, [1, 2, 3])
    assert got.shape == (10, 3)
    for k in range(3):
        assert sorted(got[:, k]) == sorted(sizes)
    assert not np.array_equal(got[:, 0], got[:, 1])
    assert np.array_equal(got[:, 0], s2j.shuffled(sizes, [1])[:, 0])


def _scores(p0="pass", p1="pass", p2="pass", p5="pass") -> dict:
    names = json.loads(s2j.s2g.REGISTRATION.read_text())["predictions"]
    out = {n: {"outcome": "pass", "value": 1} for n in names}
    for prefix, outcome in (("P0", p0), ("P1", p1), ("P2", p2), ("P5", p5)):
        name = next(n for n in names if n.startswith(prefix))
        out[name]["outcome"] = outcome
    return out


def test_predictions_read_the_arms_scores() -> None:
    spec = json.loads(s2j.REGISTRATION.read_text())
    arms = {
        "volume": _scores(),
        "volume_shuffled": _scores(p2="fail", p5="fail"),
        "volume_cord": _scores(p2="fail"),
    }
    got = s2j.predictions(arms, spec)
    assert [got[n]["outcome"] for n in spec["predictions"]] == ["pass", "pass", "pass", "fail"]
    arms["volume_shuffled"] = _scores(p0="fail", p2="fail", p5="fail")
    got = s2j.predictions(arms, spec)
    assert got["J1_circuit_necessity_needs_the_neurons_own_volumes"]["outcome"] == "uninterpretable"
