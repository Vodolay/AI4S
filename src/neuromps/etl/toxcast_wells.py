"""Well-level ToxCast acute-MEA rows (tcpl levels 0-3) from the invitrodb v4.3 MySQL dump.

Run (``-I`` because the dump is untrusted input; nothing in it is executed)::

    python -I -m neuromps.etl.toxcast_wells \
        --dump data/raw/toxcast_mea/mysql/invitrodb_v4_3.sql.gz --out data/interim/toxcast_mea

One streaming pass over the gzip text. ``CREATE TABLE`` statements are kept as
the schema; INSERT rows of the small dimension tables are parsed whole; rows of
the well tables (``mc0``-``mc3``) are kept only for the assay components whose
name contains ``CCTE_Shafer_MEA_acute``. mysqldump writes tables in name order,
so ``assay_component`` and ``assay_component_endpoint`` (which define that set)
are complete before ``mc0`` starts; the extractor raises if a dump breaks that.

This module only extracts and counts. It makes no statement about noise or
effects; that analysis is pre-registered separately.
"""

import argparse
import json
import logging
import re
import subprocess
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from neuromps.etl.toxcast_wells_summary import summarise
from neuromps.fetch.manifest import utc_now
from neuromps.fetch.sqldump import TableSchema, insert_table, parse_create_table, parse_insert

PATTERN = "CCTE_Shafer_MEA_acute"
WELL_TABLES = ("mc0", "mc1", "mc2", "mc3")
DIM_TABLES = (
    "assay",
    "assay_component",
    "assay_component_endpoint",
    "assay_source",
    "chemical",
    "sample",
    "mc2_acid",
    "mc2_methods",
    "mc3_aeid",
    "mc3_methods",
)
_PROGRESS_BYTES = 5 << 30
log = logging.getLogger("neuromps.etl")


class Extractor:
    """Feed dump lines (bytes) in order; collect schema and the selected rows."""

    def __init__(self, pattern: str = PATTERN):
        self.pattern = pattern
        self.schemas: dict[str, TableSchema] = {}
        self.create_sql: list[str] = []
        self.rows: dict[str, list[tuple]] = {}
        self.insert_lines: Counter = Counter()
        self.insert_bytes: Counter = Counter()
        self.keys: dict[str, frozenset[int]] | None = None
        self._prefilter: re.Pattern | None = None
        self._create_buf: list[str] = []

    def feed(self, line: bytes) -> None:
        """Consume one line of the dump."""
        if self._create_buf or line.startswith(b"CREATE TABLE"):
            self._feed_create(line.decode("utf-8", "replace"))
            return
        table = insert_table(line)
        if table is None:
            return
        self.insert_lines[table] += 1
        self.insert_bytes[table] += len(line)
        if table in DIM_TABLES:
            self._add(table, self._parse(table, line))
        elif table in WELL_TABLES:
            self._feed_well(table, line)

    def _feed_create(self, text: str) -> None:
        self._create_buf.append(text.rstrip("\n"))
        if text.rstrip().endswith(";"):
            stmt = "\n".join(self._create_buf)
            self._create_buf = []
            self.create_sql.append(stmt)
            schema = parse_create_table(stmt)
            self.schemas[schema.name] = schema

    def _parse(self, table: str, line: bytes) -> list[tuple]:
        cols, rows = parse_insert(line.decode("utf-8"))
        if cols is not None and cols != self.schemas[table].columns:
            raise ValueError(f"{table}: INSERT column list differs from CREATE TABLE")
        self.schemas[table].check_rows(rows)
        return rows

    def _add(self, table: str, rows: list[tuple]) -> None:
        self.rows.setdefault(table, []).extend(rows)

    def _feed_well(self, table: str, line: bytes) -> None:
        if self.keys is None:
            self._freeze_keys()
        if not self._prefilter.search(line):
            return
        col = "acid" if "acid" in self.schemas[table].columns else "aeid"
        idx, keep = self.schemas[table].columns.index(col), self.keys[col]
        rows = [r for r in self._parse(table, line) if r[idx] in keep]
        if rows:
            self._add(table, rows)

    def _freeze_keys(self) -> None:
        acids = frozenset(select_acids(self.frame("assay_component"), self.pattern))
        if not acids:
            raise ValueError(f"no assay_component name contains {self.pattern!r} before mc0")
        ace = self.frame("assay_component_endpoint")
        aeids = frozenset(int(a) for a in ace.loc[ace["acid"].isin(acids), "aeid"])
        self.keys = {"acid": acids, "aeid": aeids}
        alt = b"|".join(str(a).encode() for a in sorted(acids | aeids))
        self._prefilter = re.compile(rb"[(,](?:" + alt + rb")[,)]")
        log.info("selected acids %s, aeids %s", sorted(acids), sorted(aeids))

    def frame(self, table: str) -> pd.DataFrame:
        """Rows collected for ``table`` as a DataFrame, numeric columns coerced."""
        schema = self.schemas.get(table)
        if schema is None:
            raise KeyError(f"table {table!r} not in dump schema")
        df = pd.DataFrame(self.rows.get(table, []), columns=list(schema.columns))
        for c in schema.numeric:
            df[c] = pd.to_numeric(df[c])
        return df

    def finish(self) -> None:
        """Raise if the dump ended mid-statement or never reached the well tables."""
        if self._create_buf:
            raise ValueError("dump ended inside a CREATE TABLE statement")
        if self.keys is None:
            raise ValueError("dump has no INSERT rows for any of " + ", ".join(WELL_TABLES))

    def frames(self) -> dict[str, pd.DataFrame]:
        """Every collected (or declared) dimension and well table."""
        return {t: self.frame(t) for t in DIM_TABLES + WELL_TABLES if t in self.schemas}


def select_acids(components: pd.DataFrame, pattern: str) -> list[int]:
    """acid of every assay component whose name contains ``pattern``."""
    hit = components["assay_component_name"].fillna("").str.contains(pattern, regex=False)
    return sorted(int(a) for a in components.loc[hit, "acid"])


def filter_tables(frames: dict[str, pd.DataFrame], acids: Iterable[int]) -> dict[str, pd.DataFrame]:
    """Keep the rows linked to ``acids``; each table is cut by keys of the ones before it.

    Method catalogues (``mc*_methods``) are kept whole.
    """
    out: dict[str, pd.DataFrame] = {}
    keys: dict[str, set] = {"acid": set(acids)}

    def cut(table: str, col: str, derive: tuple[str, ...] = ()) -> None:
        if table not in frames:
            return
        df = frames[table]
        out[table] = df[df[col].isin(keys[col])].reset_index(drop=True)
        for k in derive:
            keys[k] = set(out[table][k].dropna())

    cut("assay_component", "acid", ("aid",))
    cut("assay_component_endpoint", "acid", ("aeid",))
    cut("assay", "aid", ("asid",))
    cut("assay_source", "asid")
    for t in WELL_TABLES:
        cut(t, "acid" if t in frames and "acid" in frames[t].columns else "aeid")
    keys["spid"] = set(out["mc0"]["spid"].dropna()) if "mc0" in out else set()
    cut("sample", "spid", ("chid",))
    cut("chemical", "chid")
    cut("mc2_acid", "acid")
    cut("mc3_aeid", "aeid")
    out.update({t: frames[t] for t in ("mc2_methods", "mc3_methods") if t in frames})
    return out


def stream_lines(path: Path) -> Iterator[bytes]:
    """Lines of a gzip file, decompressed by ``zcat`` in a separate process."""
    proc = subprocess.Popen(["zcat", "--", str(path)], stdout=subprocess.PIPE, bufsize=1 << 24)
    yield from proc.stdout
    if proc.wait() != 0:
        raise RuntimeError(f"zcat exited with {proc.returncode} on {path}")


def extract(lines: Iterable[bytes], pattern: str = PATTERN) -> Extractor:
    """Run the extractor over ``lines``, logging progress every 5 GiB."""
    ex, seen, mark = Extractor(pattern), 0, _PROGRESS_BYTES
    for line in lines:
        ex.feed(line)
        seen += len(line)
        if seen >= mark:
            log.info(
                "  %.0f GiB read; kept %s",
                seen / (1 << 30),
                {t: len(ex.rows.get(t, [])) for t in WELL_TABLES},
            )
            mark += _PROGRESS_BYTES
    ex.finish()
    return ex


def write_outputs(ex: Extractor, out_dir: Path, provenance: dict) -> dict:
    """Write schema text, one parquet per filtered table, joined views and summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "schema.sql.txt").write_text("\n\n".join(ex.create_sql) + "\n")
    tables = filter_tables(ex.frames(), ex.keys["acid"])
    for t, df in tables.items():
        df.to_parquet(out_dir / f"{t}_mea_acute.parquet", index=False)
    components(tables).to_parquet(out_dir / "components.parquet", index=False)
    samples(tables).to_parquet(out_dir / "samples.parquet", index=False)
    summary = {
        **provenance,
        "pattern": ex.pattern,
        "dump_tables": sorted(ex.schemas),
        "insert_lines": dict(sorted(ex.insert_lines.items())),
        "insert_bytes": dict(sorted(ex.insert_bytes.items())),
        **summarise(tables),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def components(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per endpoint: aeid, acid, endpoint and component names, assay name."""
    ace, ac = tables["assay_component_endpoint"], tables["assay_component"]
    df = ace.merge(ac[["acid", "aid", "assay_component_name"]], on="acid", how="left")
    if "assay" in tables:
        df = df.merge(tables["assay"][["aid", "assay_name"]], on="aid", how="left")
    return df


def samples(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Samples used in the extracted mc0 rows, with chemical name, CAS and DTXSID."""
    return tables["sample"].merge(tables["chemical"], on="chid", how="left")


def main(argv: list[str] | None = None) -> None:
    """CLI: stream the dump once and write the extracts."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pattern", default=PATTERN)
    ap.add_argument("--dump-sha256", default=None, help="recorded in summary.json")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = utc_now()
    ex = extract(stream_lines(a.dump), a.pattern)
    prov = {
        "dump": a.dump.name,
        "dump_bytes": a.dump.stat().st_size,
        "dump_sha256": a.dump_sha256,
        "started_at": started,
        "finished_at": utc_now(),
    }
    summary = write_outputs(ex, a.out, prov)
    log.info("row counts: %s", summary["row_counts"])


if __name__ == "__main__":
    main()
