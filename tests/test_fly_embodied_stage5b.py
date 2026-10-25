"""Stage 5b: populations, silencing, the direction rule and scoring, on synthetic inputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neuromps.fly import connectome
from neuromps.fly import embodied_stage5b as s5b


def test_population_is_one_lineage_in_the_left_front_neuromere() -> None:
    table = pd.DataFrame(
        {
            "trumanHl": ["13A", "13A", "13B", None, "13A"],
            "somaNeuromere": ["T1", "T2", "T1", "T1", "T1"],
            "somaSide": ["L", "L", "L", "L", "R"],
        }
    )
    np.testing.assert_array_equal(s5b.population(table, "13A"), [0])
    np.testing.assert_array_equal(s5b.population(table, "13B"), [2])


def test_silencing_removes_only_the_populations_outputs() -> None:
    con = connectome.Connectome(
        root_ids=np.arange(3),
        pre=np.array([0, 1, 2]),
        post=np.array([1, 2, 0]),
        synapses=np.array([5, 6, 7]),
        sign=np.ones(3, int),
        pre_class=np.array(["gaba"] * 3, dtype=object),
        neuropil=np.array(["A"] * 3, dtype=object),
    )
    np.testing.assert_array_equal(s5b.silenced(con, np.array([1])).synapses, [5, 0, 7])


def test_scoring_reads_decreases_against_the_intact_reach() -> None:
    spec = json.loads(s5b.REGISTRATION.read_text())
    depth = {
        "intact": 0.9,
        "silence_13A": 0.3,
        "silence_13B": 0.88,
        "activate_13A": 0.2,
        "activate_13B": 1.5,
    }
    table = pd.DataFrame(
        [{"arm": a, "seed": k, "depth_mm": d} for a, d in depth.items() for k in range(8)]
    )
    got = s5b.score(table, spec)
    outcomes = [v["outcome"] for v in got.values()]
    assert outcomes == ["pass", "pass", "fail", "pass", "fail"]
    table["depth_mm"] = 0.1
    assert (
        s5b.score(table, spec)["I1_silencing_13A_reduces_grooming"]["outcome"] == "uninterpretable"
    )
