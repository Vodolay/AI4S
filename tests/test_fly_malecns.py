"""The male CNS model: signs from its own transmitter calls, neuron sets matched by type."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuromps.fly import malecns


def test_signs_follow_the_registered_rule() -> None:
    nt = np.array(
        ["acetylcholine", "gaba", "glutamate", "histamine", "unclear", None], dtype=object
    )
    assert malecns.signs(nt).tolist() == [1, -1, -1, -1, 1, 1]


def test_histamine_is_inhibitory_but_no_knob_acts_on_it() -> None:
    nt = np.array(["histamine", "gaba", "acetylcholine"], dtype=object)
    sign = malecns.signs(nt)
    assert malecns.classes(sign, nt).tolist() == ["other_inh", "gaba", "ach"]


def test_a_set_is_matched_by_type_and_side() -> None:
    flywire = pd.DataFrame(
        {
            "root_id": [1, 2, 3],
            "cell_type": ["LB3c", "LB3c", "MN9x"],
            "side": ["left", "left", "right"],
        }
    )
    male = pd.DataFrame(
        {
            "bodyId": [10, 11, 12, 13, 14],
            "type": ["LB3c", "LB3c", "LB3c", "MN9x", "MN9x"],
            "flywireType": ["LB3", None, None, "MN9x", "MN9x"],
            "rootSide": ["L", "R", "L", None, None],
            "somaSide": [None, None, None, "L", "R"],
            "superclass": ["cb_sensory", "cb_sensory", "cb_sensory", "cb_motor", "cb_motor"],
        }
    )
    got = malecns.match_sets({"sugar": [1, 2], "mn9": [3]}, flywire, male)
    assert got == {"sugar": [10, 12], "mn9": [14]}


def test_a_type_also_matches_through_the_flywire_cross_reference() -> None:
    flywire = pd.DataFrame({"root_id": [3], "cell_type": ["CB0701"], "side": ["right"]})
    male = pd.DataFrame(
        {
            "bodyId": [13, 14],
            "type": ["MN9", "MN9"],
            "flywireType": ["CB0701", "CB0701"],
            "rootSide": [None, None],
            "somaSide": ["L", "R"],
            "superclass": ["cb_motor", None],
        }
    )
    assert malecns.match_sets({"mn9": [3]}, flywire, male) == {"mn9": [14]}
