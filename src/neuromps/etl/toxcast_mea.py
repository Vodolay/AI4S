"""Reduce the ToxCast mc5-6 table to the CCTE_Shafer MEA endpoints and inventory them.

Run: ``python -m neuromps.etl.toxcast_mea``

Two facts drive this module (both verified against the data, see data/SOURCES.md):

1. MEA rows must be selected by assay-endpoint name, not by AEID range: AEIDs
   2454-2541 also contain unrelated assays.
2. The physical direction of ``top`` is NOT resolved yet. tcpl normalizes each
   endpoint in its own stated fitting direction (``resp.pc`` with a positive or
   negative control value), so the sign cannot be interpreted globally: a single
   flip makes the cytotoxicity endpoints (LDH must rise when cells die) come out
   backwards. Until the per-endpoint direction is read from the EPA assay
   annotations, this module reports the raw ``top`` and counts hits by sign only.
   Do not map ``top`` to "activity increased/decreased" downstream. See
   data/SOURCES.md, open question OQ-1.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from neuromps.metrics import update_metrics

MEA_PREFIX = "CCTE_Shafer_MEA_"
HIT_THRESHOLD = 0.9
REPO_ROOT = Path(__file__).resolve().parents[3]
MC56_CSV = (
    REPO_ROOT
    / "data/raw/toxcast_mea/summary/mc5-6_winning_model_fits-flags_invitrodbv4_3_AUG2024.csv"
)
INTERIM_PARQUET = REPO_ROOT / "data/interim/toxcast_mea_mc56.parquet"
ENDPOINT_TABLE = REPO_ROOT / "outputs/tables/toxcast_mea_endpoints.csv"
log = logging.getLogger("neuromps.etl")


def select_mea(chunk: pd.DataFrame) -> pd.DataFrame:
    """Keep only CCTE_Shafer MEA rows, tagged with assay type and endpoint name."""
    out = chunk[chunk["aenm"].str.startswith(MEA_PREFIX, na=False)].copy()
    out["assay"] = out["aenm"].str.slice(len(MEA_PREFIX)).str.split("_").str[0]
    out["endpoint"] = out["aenm"].str.replace(f"{MEA_PREFIX}(acute|dev)_", "", regex=True)
    out["is_hit"] = out["hitc"] >= HIT_THRESHOLD
    return out


def load_mc56(csv_path: Path = MC56_CSV, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Stream the 2.55 GB mc5-6 CSV and return only the MEA rows."""
    chunks = [select_mea(c) for c in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)]
    return pd.concat(chunks, ignore_index=True)


def endpoint_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per assay endpoint: coverage, hit counts, and hit counts by sign of ``top``.

    The sign is reported as-is; see the module docstring for why it is not
    translated into "activity up/down" here.
    """
    g = df.groupby(["aeid", "assay", "endpoint"], as_index=False).agg(
        chemicals=("chid", "nunique"),
        samples=("spid", "size"),
        hits=("is_hit", "sum"),
        cutoff=("coff", "first"),
        conc_min=("conc_min", "median"),
        conc_max=("conc_max", "median"),
    )
    hits = df[df["is_hit"]]
    pos = hits[hits["top"] > 0].groupby("aeid").size().rename("hits_top_positive")
    g = g.merge(pos, on="aeid", how="left")
    g["hits_top_positive"] = g["hits_top_positive"].fillna(0).astype(int)
    g["hits_top_negative"] = g["hits"] - g["hits_top_positive"]
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
    INTERIM_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERIM_PARQUET, index=False)
    summary = endpoint_summary(df)
    ENDPOINT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ENDPOINT_TABLE, index=False)
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
        },
    )
    log.info(
        "wrote %s, %s and metrics section 'toxcast_mea_inventory'",
        INTERIM_PARQUET.relative_to(REPO_ROOT),
        ENDPOINT_TABLE.relative_to(REPO_ROOT),
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
