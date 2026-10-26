"""Recovery of wells and rval for components whose level-0 rows are missing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuromps.etl import toxcast_recover as tr


def _dump(acids=(2449, 2450, 2451), wells=4, batches=2):
    """Two batches of runs; acid 2449 has no level 0, and well 1 of batch 0 failed QC."""
    mc0, mc3, m0 = [], [], 1000
    for b in range(batches):
        for a in acids:
            for w in range(wells):
                conc = [0.002, 1.0, 10.0, 25.0][w]
                well = {
                    "apid": f"d{b}",
                    "srcf": f"TC_MW 10{b}-1_x.csv",
                    "rowi": w + 1,
                    "coli": 1,
                    "wllt": "n" if w == 0 else "t",
                    "spid": "s",
                    "conc": conc,
                }
                rval = 10.0 * w + b
                if a != 2449:
                    mc0.append({"m0id": m0, "acid": a, "rval": rval, **well})
                bval, pval = 2.0, -100.0
                if not (b == 0 and w == 1):
                    mc3.append(
                        {
                            "m0id": m0,
                            "acid": a,
                            "aeid": a + 10,
                            "conc": conc,
                            "bval": bval,
                            "pval": pval,
                            "resp": (rval - bval) / (pval - bval) * 100,
                        }
                    )
                m0 += 1
        m0 += 50
    return pd.DataFrame(mc0), pd.DataFrame(mc3)


def test_runs_and_batches() -> None:
    mc0, mc3 = _dump()
    r = tr.runs(mc0, mc3, acids=(2449, 2450, 2451))
    assert r["batch"].nunique() == 2
    assert list(r["acid"]) == [2449, 2450, 2451] * 2


def test_missing_component_gets_the_reference_well_and_rval() -> None:
    mc0, mc3 = _dump()
    rec = tr.recovered(mc0, mc3)
    miss = rec[rec["acid"] == 2449].sort_values("m0id")
    assert miss["ref_m0id"].notna().all()
    assert list(miss["rowi"]) == [1, 3, 4, 1, 2, 3, 4]
    assert np.allclose(miss["conc"], miss["conc_l3"])
    assert np.allclose(miss["rval"], [0, 20, 30, 1, 11, 21, 31])
    assert (miss["wllt"] == "n").sum() == 2


def test_checks_pass_on_present_components() -> None:
    mc0, mc3 = _dump()
    mc1 = mc0[["m0id", "acid"]].assign(cndx=1)
    rec = tr.recovered(mc0, mc3)
    got = tr.checks(mc0, mc1, mc3, rec)
    assert got["positional_same_well"] == {2450: 1.0, 2451: 1.0}
    assert got["rval_identity_where_level0"] == 1.0
    assert set(got["positional_conc_agrees"].values()) == {1.0}
    assert got["run_starts_predicted"] == "4/4"
    assert got["final_rows_with_well"] == 1.0


def test_present_components_keep_their_own_wells() -> None:
    mc0, mc3 = _dump()
    rec = tr.recovered(mc0, mc3)
    own = rec[rec["acid"] == 2451]
    assert (own["ref_m0id"] == own["m0id"]).all()


def test_baseline_check_detects_a_shuffled_date() -> None:
    rec = pd.DataFrame(
        {
            "acid": [1] * 4,
            "apid": ["a", "a", "b", "b"],
            "wllt": ["n", "n", "n", "n"],
            "rval": [1.0, 3.0, 10.0, 30.0],
            "bval": [2.0, 2.0, 2.0, 2.0],
            "ref_m0id": [1.0, 2.0, 3.0, 4.0],
        }
    )
    mc1 = pd.DataFrame({"m0id": [1, 2, 3, 4], "cndx": [0, 0, 0, 0]})
    assert tr.baseline_check(rec, mc1) == {1: 0.5}
