"""Stage 5c scoring: flexion under silencing, extension under activation."""

from __future__ import annotations

import json

import pandas as pd

from neuromps.fly import embodied_stage5c as s5c


def test_the_published_postures_pass_and_their_absence_fails() -> None:
    spec = json.loads(s5c.REGISTRATION.read_text())
    angle = {
        "intact": 0.63,
        "silence_13A": 0.9,
        "silence_13B": 0.65,
        "activate_13A": 0.5,
        "activate_13B": 0.6,
    }
    table = pd.DataFrame(
        [{"arm": a, "seed": k, s5c.TIBIA: v} for a, v in angle.items() for k in range(8)]
    )
    got = [v["outcome"] for v in s5c.score(table, spec).values()]
    assert got == ["pass", "fail", "pass", "fail"]
