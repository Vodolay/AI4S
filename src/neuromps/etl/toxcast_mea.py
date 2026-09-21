"""Reduce the ToxCast mc5-6 table to the CCTE_Shafer MEA endpoints and inventory them.

Run: ``python -m neuromps.etl.toxcast_mea``

Two facts drive this module (both verified against the data, see data/SOURCES.md):

1. MEA rows must be selected by assay-endpoint name, not by AEID range: AEIDs
   2454-2541 also contain unrelated assays.
2. The sign of ``top`` means opposite things in the two assays, so it is mapped
   per assay (OQ-1, resolved). ``activity_change`` is the change in the measured
   quantity: positive = the quantity went up.

   - acute: ``-top``. Normalized against a positive control of -100, so +100
     means activity was abolished.
   - dev: ``+top``. Percent of control on AUC-based metrics, so -100 means every
     metric collapsed to zero.

   Derived from cytotoxicity anchors (chemicals that kill cells must reduce
   electrical activity) over 16 acute and 221 dev chemicals, and re-checked on
   every run by :func:`check_direction_convention`. The 4 viability endpoints
   (LDH, AB) carry their own annotated direction and get no ``activity_change``.
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.fetch.manifest import utc_now
from neuromps.metrics import update_metrics
from neuromps.paths import outputs_dir

SOURCE = "toxcast_mea"
MEA_PREFIX = "CCTE_Shafer_MEA_"
HIT_THRESHOLD = 0.9
VIABILITY_ENDPOINTS = ("LDH", "AB")
# Multiplier taking `top` to a change in the measured quantity; see module docstring.
ACTIVITY_SIGN = {"acute": -1, "dev": 1}
# Endpoints where the measured quantity moves opposite to network activity
# (longer intervals mean less activity), used only for reporting.
INTERVAL_ENDPOINTS = (
    "interburst_interval_mean",
    "inter_network_spike_interval_mean",
    "per_burst_interspike_interval",
)
REPO_ROOT = Path(__file__).resolve().parents[3]
MC56_CSV = (
    REPO_ROOT
    / "data/raw/toxcast_mea/summary/mc5-6_winning_model_fits-flags_invitrodbv4_3_AUG2024.csv"
)
INTERIM_PARQUET = REPO_ROOT / "data/interim/toxcast_mea_mc56.parquet"

log = logging.getLogger("neuromps.etl")
CITATION = (
    "U.S. EPA Center for Computational Toxicology and Exposure. ToxCast Database: invitrodb "
    "version 4.3. doi:10.23645/epacomptox.6062623.v14"
)
PLATFORM = {
    "vendor": "Axion Biosystems",
    "system": "Maestro 768-channel",
    "plate_format": "48-well",
    "electrodes_per_unit": 16,
    "sampling_rate_hz": 12500.0,
}


def select_mea(chunk: pd.DataFrame) -> pd.DataFrame:
    """Keep only CCTE_Shafer MEA rows, tagged with assay type and endpoint name."""
    out = chunk[chunk["aenm"].str.startswith(MEA_PREFIX, na=False)].copy()
    out["assay"] = out["aenm"].str.slice(len(MEA_PREFIX)).str.split("_").str[0]
    out["endpoint"] = out["aenm"].str.replace(f"{MEA_PREFIX}(acute|dev)_", "", regex=True)
    out["is_hit"] = out["hitc"] >= HIT_THRESHOLD
    out["is_viability"] = out["endpoint"].isin(VIABILITY_ENDPOINTS)
    sign = out["assay"].map(ACTIVITY_SIGN)
    out["activity_change"] = (sign * out["top"]).where(~out["is_viability"])
    return out


def load_mc56(csv_path: Path = MC56_CSV, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Stream the 2.55 GB mc5-6 CSV and return only the MEA rows."""
    chunks = [select_mea(c) for c in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)]
    return pd.concat(chunks, ignore_index=True)


def check_direction_convention(df: pd.DataFrame, min_agreement: float = 0.9) -> dict[str, float]:
    """Re-derive the sign convention from cytotoxicity anchors; raise if it disagrees.

    Chemicals that are hits in a viability endpoint (LDH/AB) must show reduced
    electrical activity. Interval endpoints are excluded: intervals lengthen as
    activity dies. Returns the agreement fraction per assay.
    """
    agreement = {}
    for assay, part in df.groupby("assay"):
        cytotoxic = set(part.loc[part["is_viability"] & part["is_hit"], "chid"])
        act = part[
            ~part["is_viability"]
            & ~part["endpoint"].isin(INTERVAL_ENDPOINTS)
            & part["is_hit"]
            & part["chid"].isin(cytotoxic)
            & part["activity_change"].notna()
        ]
        if act.empty:
            continue
        frac = float((act["activity_change"] < 0).mean())
        agreement[str(assay)] = round(frac, 3)
        if frac < min_agreement:
            raise ValueError(
                f"{assay}: only {frac:.1%} of activity hits for cytotoxic chemicals show reduced "
                f"activity (n={len(act)}); the ACTIVITY_SIGN mapping may be wrong"
            )
    return agreement


def endpoint_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per assay endpoint: coverage, hit counts and direction of change.

    ``hits_quantity_up``/``down`` count hits where the measured quantity rose or
    fell. For interval endpoints a rise means *less* network activity.
    For the viability endpoints both counts are 0 (no signed value).
    """
    g = df.groupby(["aeid", "assay", "endpoint"], as_index=False).agg(
        chemicals=("chid", "nunique"),
        samples=("spid", "size"),
        hits=("is_hit", "sum"),
        cutoff=("coff", "first"),
        conc_min=("conc_min", "median"),
        conc_max=("conc_max", "median"),
    )
    hits = df[df["is_hit"] & df["activity_change"].notna()]
    for name, mask in (("up", hits["activity_change"] > 0), ("down", hits["activity_change"] < 0)):
        counts = hits[mask].groupby("aeid").size().rename(f"hits_quantity_{name}")
        g = g.merge(counts, on="aeid", how="left")
        g[f"hits_quantity_{name}"] = g[f"hits_quantity_{name}"].fillna(0).astype(int)
    g["hit_rate"] = (g["hits"] / g["samples"]).round(3)
    return g.sort_values(["assay", "endpoint"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Write the interim MEA parquet, the endpoint table and the metrics section."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=MC56_CSV)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.csv.exists():
        raise SystemExit(f"missing {args.csv}; run: python -m neuromps.fetch.runner --full")
    df = load_mc56(args.csv)
    agreement = check_direction_convention(df)
    log.info("direction check (cytotoxic chemicals with reduced activity): %s", agreement)
    INTERIM_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERIM_PARQUET, index=False)
    summary = endpoint_summary(df)
    endpoint_table = outputs_dir() / "tables" / "toxcast_mea_endpoints.csv"
    endpoint_table.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(endpoint_table, index=False)
    acute, dev = (df[df["assay"] == a] for a in ("acute", "dev"))
    update_metrics(
        "toxcast_mea_inventory",
        {
            "source_file": args.csv.name,
            "rows_mea": int(len(df)),
            "endpoints_total": int(df["aeid"].nunique()),
            "endpoints_acute": int(acute["aeid"].nunique()),
            "endpoints_dev": int(dev["aeid"].nunique()),
            "chemicals_acute": int(acute["chid"].nunique()),
            "chemicals_dev": int(dev["chid"].nunique()),
            "hits_acute": int(acute["is_hit"].sum()),
            "hits_dev": int(dev["is_hit"].sum()),
            "hit_threshold": HIT_THRESHOLD,
            "direction_check_agreement": agreement,
        },
    )
    log.info(
        "wrote %s, %s and metrics section 'toxcast_mea_inventory'",
        INTERIM_PARQUET.relative_to(REPO_ROOT),
        endpoint_table.relative_to(REPO_ROOT),
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


VEHICLES = {"dmso", "ethanol", "dmso/ethanol", "acoh", "hcl", "water", "media"}
CONTROL_MECHANISM = {"bicuculline": "gabaa_antagonist", "tetrodotoxin": "na_channel_blocker"}
MIN_CONCENTRATIONS = 3


def _compound_block(chnm: object, spid: object, casn: object) -> dict:
    """Name and role for a tested chemical or a control.

    Control rows carry no `chnm`; their identity is in `spid` (DMSO, Ethanol,
    AcOH, HCl as vehicles; Bicuculline as the reference GABA-A antagonist).
    """
    named = None if chnm is None or pd.isna(chnm) else str(chnm).strip()
    name = named or str(spid).strip()
    key = name.lower()
    if named:
        role, mechanism = "test", None
    elif key in VEHICLES:
        role, mechanism = "vehicle_control", None
    else:
        role, mechanism = "positive_control", CONTROL_MECHANISM.get(key)
    return {
        "name": name,
        "role": role,
        "mechanism_class": mechanism,
        "cas": _cas(casn),
        "dose_unit": "uM",
    }


def _slug(value: object) -> str:
    """Lowercase id fragment safe for record_id (spids contain '/', e.g. 'DMSO/Ethanol')."""
    return re.sub(r"[^a-z0-9_.-]+", "-", str(value).strip().lower()).strip("-") or "unknown"


def _cas(value: object) -> str | None:
    """Return a CAS number if it looks like one (the table also holds 'NOCAS' placeholders)."""
    text = str(value).strip()
    return text if re.fullmatch(r"[0-9]{2,7}-[0-9]{2}-[0-9]", text) else None


def build_curve_records(df: pd.DataFrame, hashes: dict[str, str], source_file: str) -> list[dict]:
    """Convert selected MEA rows into `fitted_curve` schema records."""
    sha = hashes.get(Path(source_file).name, "")
    records = []
    for row in df.itertuples():
        records.append(
            {
                "record_id": f"{SOURCE}:aeid{row.aeid}:{_slug(row.spid)}",
                "source": SOURCE,
                "license": "CC0-1.0",
                "citation": CITATION,
                "data_level": "fitted_curve",
                "cell_source": {"species": "rat", "system": "primary_cortical"},
                "platform": dict(PLATFORM),
                "unit_id": f"aeid{row.aeid}",
                "compound": _compound_block(row.chnm, row.spid, row.casn),
                "curve": {
                    "endpoint": row.aenm,
                    "model": None if pd.isna(row.modl) else str(row.modl),
                    "n_concentrations": int(row.nconc),
                    "conc_min": _f(row.conc_min),
                    "conc_max": _f(row.conc_max),
                    "ac50": _f(row.ac50),
                    "top": _f(row.top),
                    "activity_change": _f(row.activity_change),
                    "cutoff": _f(row.coff),
                    "hitcall": float(np.clip(row.hitc, 0.0, 1.0)),
                },
                "qc": {
                    "usable": int(row.nconc) >= MIN_CONCENTRATIONS,
                    "flags": (
                        [] if int(row.nconc) >= MIN_CONCENTRATIONS else ["incomplete_dose_series"]
                    ),
                },
                "provenance": {
                    "source_files": [
                        {"path": f"{SOURCE}/summary/{Path(source_file).name}", "sha256": sha}
                    ],
                    "transform": ["select_mea()", "activity_change per assay (see OQ-1)"],
                    "generated_by": "neuromps.etl.toxcast_mea",
                    "generated_at": utc_now(),
                },
            }
        )
    return records


def _f(value: object) -> float | None:
    """Float or None for NaN."""
    return None if value is None or pd.isna(value) else float(value)
