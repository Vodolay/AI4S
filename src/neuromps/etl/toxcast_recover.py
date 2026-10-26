"""Well identities and raw values for every ToxCast acute-MEA electrophysiology component.

The invitrodb v4.3 dump's level-0 table (mc0) lacks 7 of the 15 electrophysiology components
(among them firing rate, spike number and burst number), though level 3 (mc3) has their rows.
Two regularities of the dump recover the wells (decision D29 in report/DECISION_LOG.md):

- level-0 ids are allocated in runs, one per component, in component order; within a batch
  every component's run has the same length and the same well order, and runs are contiguous.
  So a component's run starts at the reference component's start + (acid - reference) x length,
  and a row's well is the reference run's well at the same position;
- level 2 is 'none' (cval = rval) and level 3 is resp = (cval - bval) / (pval - bval) x 100, so
  rval = bval + resp x (pval - bval) / 100.

Both are checked on the components that do have level 0 (`checks`); nothing here looks at the
spread of any response.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuromps.etl.toxcast_wells_summary import physical_plate

EPHYS = tuple(range(2447, 2462))
REFERENCE = 2450
WELL = ["apid", "srcf", "rowi", "coli", "wllt", "spid", "conc"]


def runs(mc0: pd.DataFrame, mc3: pd.DataFrame, acids=EPHYS) -> pd.DataFrame:
    """Runs of consecutive level-0 ids per component, with a batch number per component cycle."""
    have = set(mc0["acid"]) & set(acids)
    ids = pd.concat(
        [
            mc0.loc[mc0["acid"].isin(have), ["m0id", "acid"]],
            mc3.loc[mc3["acid"].isin(set(acids) - have), ["m0id", "acid"]],
        ]
    ).sort_values("m0id", ignore_index=True)
    run = (ids["acid"] != ids["acid"].shift()).cumsum()
    out = ids.groupby(run).agg(acid=("acid", "first"), start=("m0id", "min"), end=("m0id", "max"))
    out = out.reset_index(drop=True)
    out["batch"] = (out["acid"] <= out["acid"].shift(fill_value=np.iinfo(np.int64).max)).cumsum()
    return out


def batch_layout(r: pd.DataFrame, mc0: pd.DataFrame, reference: int = REFERENCE) -> pd.DataFrame:
    """Per batch: the reference run's start and length, from its complete level-0 ids."""
    ref = r[r["acid"] == reference].set_index("batch")
    return pd.DataFrame({"start": ref["start"], "length": ref["end"] - ref["start"] + 1})


def assign(mc3: pd.DataFrame, layout: pd.DataFrame, reference: int = REFERENCE) -> pd.Series:
    """Each level-3 row's reference level-0 id (NaN when it falls in no batch's run)."""
    starts, lengths = layout["start"].to_numpy(), layout["length"].to_numpy()
    out = np.full(len(mc3), np.nan)
    m0, acid = mc3["m0id"].to_numpy(), mc3["acid"].to_numpy()
    for s, n in zip(starts, lengths, strict=True):
        own = s + (acid - reference) * n
        hit = (m0 >= own) & (m0 < own + n)
        out[hit] = (m0[hit] - own[hit] + s).astype(float)
    return pd.Series(out, index=mc3.index, name="ref_m0id")


def raw_value(mc3: pd.DataFrame) -> pd.Series:
    """rval recovered from level 3: bval + resp x (pval - bval) / 100."""
    return mc3["bval"] + mc3["resp"] * (mc3["pval"] - mc3["bval"]) / 100.0


def positional(mc0: pd.DataFrame, mc3: pd.DataFrame) -> pd.DataFrame:
    """Every electrophysiology level-3 row with the reference run's well at its position."""
    rows = mc3[mc3["acid"].isin(EPHYS)].copy()
    rows["ref_m0id"] = assign(rows, batch_layout(runs(mc0, mc3), mc0))
    return _with_wells(rows, mc0[mc0["acid"] == REFERENCE], mc0)


def _with_wells(rows: pd.DataFrame, source: pd.DataFrame, mc0: pd.DataFrame) -> pd.DataFrame:
    src = source.set_index("m0id")
    src = src.assign(plate=physical_plate(src.reset_index()).to_numpy())
    ref = rows["ref_m0id"].to_numpy()
    wells = src.reindex(ref)[[*WELL, "plate"]].reset_index(drop=True)
    out = rows[["m0id", "acid", "aeid", "ref_m0id", "bval"]].reset_index(drop=True)
    out = pd.concat([out, wells], axis=1)
    out["conc_l3"] = rows["conc"].to_numpy()
    out["rval"] = raw_value(rows).to_numpy()
    out["level0_present"] = out["acid"].isin(set(mc0["acid"]))
    return out


def recovered(mc0: pd.DataFrame, mc3: pd.DataFrame) -> pd.DataFrame:
    """Rows of components with level 0 keep their own wells; the others get positional wells."""
    pos = positional(mc0, mc3)
    pos = pos[~pos["level0_present"]]
    rows = mc3[mc3["acid"].isin(set(EPHYS) & set(mc0["acid"]))].copy()
    rows["ref_m0id"] = rows["m0id"].astype(float)
    own = _with_wells(rows, mc0, mc0)
    return pd.concat([own, pos], ignore_index=True).sort_values(["acid", "m0id"], ignore_index=True)


def baseline_check(rec: pd.DataFrame, mc1: pd.DataFrame) -> dict[int, float]:
    """Per component: the fraction of dates whose stored bval is the median rval of the date's
    vehicle wells and test wells at concentration index 1-2 (the level-3 method), as mapped."""
    cndx = mc1.set_index("m0id")["cndx"]
    df = rec.assign(cndx=cndx.reindex(rec["ref_m0id"].astype("int64")).to_numpy())
    base = df[(df["wllt"] == "n") | ((df["wllt"] == "t") & df["cndx"].isin([1, 2]))]
    got = base.groupby(["acid", "apid"])["rval"].median()
    stored = df.groupby(["acid", "apid"])["bval"].first().reindex(got.index)
    ok = pd.Series(np.isclose(got, stored, rtol=1e-6, atol=1e-9), index=got.index)
    return {int(a): round(float(v), 4) for a, v in ok.groupby(level="acid").mean().items()}


def checks(mc0: pd.DataFrame, mc1: pd.DataFrame, mc3: pd.DataFrame, rec: pd.DataFrame) -> dict:
    """Run contiguity, rval identity, and per component: positional same-well fraction (where
    level 0 exists), concentration agreement and the baseline check, positional and final."""
    r = runs(mc0, mc3)
    present = sorted(set(mc0["acid"]) & set(EPHYS))
    contiguous = total = 0
    for _, g in r.groupby("batch"):
        g = g.set_index("acid")
        n = g.loc[REFERENCE, "end"] - g.loc[REFERENCE, "start"] + 1
        for a in present:
            total += 1
            contiguous += int(g.loc[a, "start"] == g.loc[REFERENCE, "start"] + (a - REFERENCE) * n)
    pos = positional(mc0, mc3)
    truth = mc0.set_index("m0id").reindex(pos["m0id"].to_numpy())
    keys = ["apid", "srcf", "rowi", "coli"]
    same = (truth[keys].to_numpy() == pos[keys].to_numpy()).all(axis=1)
    rv = np.isclose(truth["rval"].to_numpy(float), pos["rval"].to_numpy(float), rtol=1e-6)
    conc = np.isclose(pos["conc"].to_numpy(float), pos["conc_l3"].to_numpy(float), rtol=1e-3)
    by = pos["acid"].to_numpy()
    return {
        "batches": int(r["batch"].nunique()),
        "run_starts_predicted": f"{contiguous}/{total}",
        "rval_identity_where_level0": float(rv[pos["level0_present"].to_numpy()].mean()),
        "positional_same_well": {int(a): round(float(same[by == a].mean()), 4) for a in present},
        "positional_conc_agrees": {
            int(a): round(float(conc[by == a].mean()), 4) for a in sorted(set(by))
        },
        "positional_baseline_check": baseline_check(pos, mc1[mc1["acid"] == REFERENCE]),
        "final_rows_with_well": float(rec["apid"].notna().mean()),
        "final_baseline_check": baseline_check(rec, mc1),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: write every electrophysiology row with its well and rval, and the checks."""
    import json

    from neuromps.paths import REPO_ROOT

    d = REPO_ROOT / "data" / "interim" / "toxcast_mea"
    mc0 = pd.read_parquet(d / "mc0_mea_acute.parquet")
    mc3 = pd.read_parquet(d / "mc3_mea_acute.parquet")
    mc1 = pd.read_parquet(d / "mc1_mea_acute.parquet")
    rec = recovered(mc0, mc3)
    got = checks(mc0, mc1, mc3, rec)
    rec.to_parquet(d / "ephys_wells_recovered.parquet", index=False)
    (d / "recovery_checks.json").write_text(json.dumps(got, indent=2) + "\n")
    print(json.dumps(got, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
