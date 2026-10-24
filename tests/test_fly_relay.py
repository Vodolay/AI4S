"""Matching descending neurons across connectomes, by type and side."""

from __future__ import annotations

import pandas as pd

from neuromps.fly import relay


def test_neurons_pair_by_type_and_side_in_id_order_and_extras_share() -> None:
    flywire = pd.DataFrame(
        {
            "root_id": [11, 10, 20, 30],
            "cell_type": ["DNa", "DNa", "DNb", "DNc"],
            "side": ["left", "left", "right", "left"],
        }
    )
    male = pd.DataFrame(
        {
            "bodyId": [3, 1, 2, 7, 9],
            "names": [{"DNa"}, {"DNa"}, {"DNa"}, {"DNb"}, {"DNz"}],
            "somaSide": ["L", "L", "L", "R", "L"],
        }
    )
    got = relay.match(flywire, male)
    # DNa left: FlyWire 10, 11; male 1, 2, 3 -> 1<-10, 2<-11, 3<-10 (in turn)
    assert got == {1: 10, 2: 11, 3: 10, 7: 20}


def test_coverage_counts_both_sides_of_the_match() -> None:
    pairs = {1: 10, 2: 11, 3: 10}
    assert relay.coverage(pairs, flywire_ids=[10, 11, 12], male_ids=[1, 2, 3, 4]) == {
        "flywire_matched": 2,
        "flywire_total": 3,
        "male_receiving": 3,
        "male_total": 4,
    }


def test_stage1e_scoring_reads_the_registered_rules() -> None:
    import json

    rows = []
    for scale in (20.0, 10.0, 40.0):
        for drive, near in (("jon_ce", 0.9), ("jon_ce_lesion", 0.2), ("sugar", 0.1), ("none", 0.0)):
            for seed in range(8):
                rows.append({"drive": drive, "seed": seed, "r_full": scale, "time_near_head": near})
    pools = pd.DataFrame(
        [
            {"drive": d, "front": f, "middle": 1.0, "hind": 1.0}
            for d, f in (("jon_ce", 10.0), ("jon_ce_lesion", 2.0), ("sugar", 2.0), ("none", 0.0))
        ]
    )
    spec = json.loads(relay.REGISTRATION.read_text())
    got = relay.score(pd.DataFrame(rows), pools, grooming_rate=40.0, spec=spec)
    assert all(v["outcome"] == "pass" for v in got.values())
    gated = relay.score(pd.DataFrame(rows), pools, grooming_rate=2.0, spec=spec)
    assert gated["E1_antennal_input_holds_the_leg_at_the_head"]["outcome"] == "uninterpretable"


def test_stage1f_scoring_reads_the_registered_rules() -> None:
    import json

    from neuromps.fly import embodied_stage1f as s1f

    rows = []
    for scale in (20.0, 10.0, 40.0):
        for drive, d in (("jon_ce", 0.55), ("jon_ce_lesion", 1.5), ("sugar", 0.95), ("none", 1.53)):
            for seed in range(8):
                rows.append({"drive": drive, "seed": seed, "r_full": scale, "min_distance_mm": d})
    pools = pd.DataFrame(
        [
            {"drive": d, "front": f, "middle": 1.0, "hind": 1.0}
            for d, f in (("jon_ce", 10.0), ("jon_ce_lesion", 2.0), ("sugar", 0.7), ("none", 0.0))
        ]
    )
    spec = json.loads(s1f.REGISTRATION.read_text())
    rates = {"jon_ce": 8.5, "sugar": 0.5, "none": 0.0}
    got = s1f.score(pd.DataFrame(rows), pools, rates, spec)
    assert all(v["outcome"] == "pass" for v in got.values())
    gated = s1f.score(pd.DataFrame(rows), pools, {"jon_ce": 0.1, "sugar": 0.5, "none": 0.0}, spec)
    assert gated["F1_antennal_input_reaches_deepest"]["outcome"] == "uninterpretable"
