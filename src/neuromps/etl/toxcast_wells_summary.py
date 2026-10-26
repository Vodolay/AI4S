"""Counts over the extracted ToxCast acute-MEA well tables (no statistics on responses).

Per assay component: wells by type, vehicle (``wllt = 'n'``) wells, plates
(``apid``), vehicle wells per plate, and which chemicals the vehicle wells hold,
so vehicle wells can be identified both by well type and by sample.
"""

from collections import Counter

import pandas as pd

VEHICLE_WLLT = "n"
# In the acute MEA assays apid is the exposure date and pools several 48-well
# MEA plates. The physical plate (Axion "MW xxxx-xx") is only named in the
# source file names, in two formats:
#   "TC_MW 1073-16_20150729_20150811_13_00(000)_Neural Statistics Compiler(000).csv"
#   "TC_20190529_MW69-0102_13_00(000)(000).csv"
# Plate names recur on other dates, so a physical plate is keyed by apid + plate.
_SRCF_PLATE = r"MW ?(\d+-\d+)"
# What rval is for these assays, as stated in the EPA assay description PDFs in
# data/raw/toxcast_mea/assay_docs/ (AEID2454-2482, 2540, 2541; read 2026-10-26).
RVAL_NOTE = {
    "mea_endpoints": (
        "Section 1.2 of each of the 15 electrophysiology PDFs: 'The recordings before and after "
        "chemical treatment are used to calculate a percent change in the <metric> in each well.' "
        "A 40-min baseline recording precedes a 40-min recording after exposure. So mc0.rval is a "
        "per-well percent change from that well's own baseline, not a raw rate."
    ),
    "viability_endpoints": (
        "acute_LDH (2540) and acute_AB (2541): section 2.6 says the endpoint measures LDH activity "
        "by spectrophotometry and resazurin reduction by fluorescence; the PDFs give no percent "
        "change wording, and their neutral-control medians are on the plate-reader scale."
    ),
    "processing": (
        "All 17 PDFs: level 2 method 'none' (cval = rval); level 3 resp.pc with "
        "bval.apid.nwllslowconc.med (plate-wise median of wllt='n' wells and test wells at "
        "cndx 1-2). The 15 electrophysiology endpoints use pval.neg.100, i.e. "
        "resp = (cval - bval) / (-100 - bval) * 100; LDH uses pval.apid.pwlls.med; AB uses "
        "pval.zero and resp.multneg1. Level 4 bmad from wllt='n' wells. The dump's mc3_aeid "
        "table gives the assignments as data (see mc3_methods_by_aeid in the summary)."
    ),
    "neutral_control": "PDFs: 'Neutral vehicle control: DMSO or water'.",
    "apid": (
        "apid is the exposure date and pools several MEA plates (bval is computed per apid); "
        "the physical plate is parsed from srcf, see vehicle_wells_per_srcf_plate."
    ),
}


def sample_names(tables: dict[str, pd.DataFrame]) -> pd.Series:
    """spid -> chemical name (NaN when the sample has no linked chemical)."""
    s = tables.get("sample", pd.DataFrame(columns=["spid", "chid"]))
    c = tables.get("chemical", pd.DataFrame(columns=["chid", "chnm"]))
    m = s[["spid", "chid"]].merge(c[["chid", "chnm"]], on="chid", how="left")
    return m.drop_duplicates("spid").set_index("spid")["chnm"]


def _per_plate(vehicle: pd.DataFrame, key: str = "apid") -> dict:
    n = vehicle.groupby(key).size()
    if n.empty:
        return {"median": None, "min": None, "max": None, "counts": {}}
    counts = {str(k): int(v) for k, v in sorted(Counter(n.tolist()).items())}
    return {"median": float(n.median()), "min": int(n.min()), "max": int(n.max()), "counts": counts}


def physical_plate(mc0: pd.DataFrame) -> pd.Series:
    """``<apid>_MW<plate>`` per well, from ``srcf`` (NaN where the name has no plate)."""
    mw = mc0["srcf"].fillna("").str.extract(_SRCF_PLATE, expand=False)
    return (mc0["apid"].astype(str) + "_MW" + mw).where(mw.notna())


def physical_plate_counts(mc0: pd.DataFrame) -> dict:
    """Plates and vehicle wells per plate when plates are keyed by ``srcf``, not ``apid``."""
    if "srcf" not in mc0.columns:
        return {}
    df = mc0.assign(plate=physical_plate(mc0))
    veh = df[df["wllt"] == VEHICLE_WLLT].dropna(subset=["plate"])
    per_apid = df.groupby("apid")["plate"].nunique()
    return {
        "n_srcf_plates": int(df["plate"].nunique()),
        "n_wells_srcf_unparsed": int(df["plate"].isna().sum()),
        "srcf_plates_per_apid": {str(k): int(v) for k, v in per_apid.value_counts().items()},
        "vehicle_wells_per_srcf_plate": _per_plate(veh, "plate"),
    }


def component_summary(acid: int, mc0: pd.DataFrame, names: pd.Series) -> dict:
    """Well, plate and vehicle counts for one assay component's mc0 rows.

    A well's label is its chemical name when its spid has a ``sample`` row, else
    the spid itself (invitrodb gives vehicle wells spids such as ``DMSO`` with no
    sample row). ``wllt_of_vehicle_labels`` shows every well type carrying a
    label seen in a vehicle well, to cross-check well type against content.
    """
    labels = mc0["spid"].map(names).fillna(mc0["spid"])
    veh = mc0[mc0["wllt"] == VEHICLE_WLLT]
    veh_labels = labels[veh.index]
    of_veh = mc0[labels.isin(set(veh_labels))].assign(label=labels)
    wllt_of_vehicle_labels = {
        str(k): {str(w): int(n) for w, n in g["wllt"].value_counts().items()}
        for k, g in of_veh.groupby("label")
    }
    plates_veh = veh["apid"].nunique()
    return {
        "acid": int(acid),
        "n_wells": len(mc0),
        "wllt_counts": {str(k): int(v) for k, v in mc0["wllt"].value_counts().items()},
        "n_plates": int(mc0["apid"].nunique()),
        "n_vehicle_wells": len(veh),
        "n_vehicle_wells_wllq1": int((veh["wllq"] == 1).sum()),
        "n_vehicle_rval_null": int(veh["rval"].isna().sum()),
        "n_plates_with_vehicle": int(plates_veh),
        "n_plates_without_vehicle": int(mc0["apid"].nunique() - plates_veh),
        "vehicle_wells_per_plate": _per_plate(veh),
        "vehicle_well_labels": {str(k): int(v) for k, v in veh_labels.value_counts().items()},
        "wllt_of_vehicle_labels": wllt_of_vehicle_labels,
        "vehicle_conc_values": sorted({float(x) for x in veh["conc"].dropna().unique()})[:20],
        **physical_plate_counts(mc0),
    }


def method_assignments(tables: dict[str, pd.DataFrame], level: int) -> dict[str, list[str]]:
    """aeid (or acid) -> method names in execution order, from ``mc{level}_*`` tables."""
    key = "aeid" if level >= 3 else "acid"
    assign, methods = tables.get(f"mc{level}_{key}"), tables.get(f"mc{level}_methods")
    mid, name = f"mc{level}_mthd_id", f"mc{level}_mthd"
    if assign is None or methods is None or not {key, mid} <= set(assign.columns):
        return {}
    m = assign.merge(methods[[mid, name]], on=mid, how="left")
    if "exec_ordr" in m.columns:
        m = m.sort_values([key, "exec_ordr"])
    return {str(int(k)): g[name].astype(str).tolist() for k, g in m.groupby(key)}


def level_coverage(tables: dict[str, pd.DataFrame], acid: int) -> dict:
    """Rows per level for one acid, and mc3 rows whose m0id is absent from mc0.

    A non-zero ``n_mc3_rows_without_mc0`` means the dump's level-0 table lacks
    wells that later levels were computed from.
    """
    out = {}
    for t in ("mc1", "mc2", "mc3"):
        if t in tables:
            out[f"n_{t}_rows"] = int((tables[t]["acid"] == acid).sum())
    if "mc3" in tables:
        m3 = tables["mc3"][tables["mc3"]["acid"] == acid]
        out["n_mc3_rows_without_mc0"] = int((~m3["m0id"].isin(tables["mc0"]["m0id"])).sum())
    return out


def summarise(tables: dict[str, pd.DataFrame]) -> dict:
    """Row counts per table and per-endpoint vehicle/plate counts."""
    comp = tables["assay_component"].set_index("acid")["assay_component_name"]
    ace = tables["assay_component_endpoint"]
    names, mc0 = sample_names(tables), tables["mc0"]
    endpoints = []
    for acid in sorted(comp.index):
        rec = {"assay_component_name": comp[acid]}
        e = ace[ace["acid"] == acid]
        rec["endpoints"] = [
            {
                "aeid": int(r.aeid),
                "name": r.assay_component_endpoint_name,
                "normalized_data_type": r.normalized_data_type,
            }
            for r in e.itertuples()
        ]
        rec.update(component_summary(acid, mc0[mc0["acid"] == acid], names))
        rec.update(level_coverage(tables, acid))
        endpoints.append(rec)
    return {
        "row_counts": {t: len(df) for t, df in tables.items()},
        "components": endpoints,
        "components_without_mc0": [c["acid"] for c in endpoints if c["n_wells"] == 0],
        "mc2_methods_by_acid": method_assignments(tables, 2),
        "mc3_methods_by_aeid": method_assignments(tables, 3),
        "rval_note": RVAL_NOTE,
    }
