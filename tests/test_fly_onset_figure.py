"""The onsets figure: onset and direction per family, on synthetic inputs."""

from __future__ import annotations

import pandas as pd

from neuromps.fly import embodied_stage4d as s4d
from neuromps.fly import onset_figure as of


def test_onset_takes_the_lowest_changed_dose_and_its_direction() -> None:
    names = ["vehicle"] + [f"{f}_{k}" for f in s4d.FAMILIES for k in (1, 2)]
    families = ["vehicle"] + [f for f in s4d.FAMILIES for _ in (1, 2)]
    doses = [0.0] + [0.2, 0.6] * len(s4d.FAMILIES)
    conds = pd.DataFrame({"condition": names, "family": families, "dose": doses})
    values = pd.Series(0.5, index=names)
    values["gaba_block_1"] = 0.1
    values["ach_block_2"] = 1.0
    got = of.onset_and_direction(conds, values)
    assert got["gaba_block"] == (0.2, "down")
    assert got["ach_block"] == (0.6, "up")
    assert got["glucl_block"] == (None, None)
