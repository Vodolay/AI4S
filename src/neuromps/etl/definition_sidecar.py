"""The sidecar table: every spike-time record's features under every definition set.

`neuromps.parquet` holds each record's features under the reference definitions
(``neuromps``). The sidecar, ``features_by_definition.parquet`` beside it, holds the
same records under every set in `features/definitions.py`, long-form:

    record_id, source, definition, feature, value

It is additive: the asset's records are not touched, and the ``neuromps`` rows must
equal the asset's own ``features.*`` columns exactly (:func:`check_reference`), which
is what proves the sidecar re-reads the same spikes the asset was built from. A value
the extractor leaves undefined (no active electrode, no burst) is NaN, as in the asset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.etl.spike_trains import SpikeRecord, Unit, load, units
from neuromps.features.definitions import REFERENCE, REGISTRY, extract
from neuromps.features.mea import FEATURE_NAMES

log = logging.getLogger("neuromps.etl")

FILE_NAME = "features_by_definition.parquet"
COLUMNS = ["record_id", "source", "definition", "feature", "value"]
# The quick tier reads one unit per source and this many of its records, the ones with
# the fewest spikes, each in its own process: enough to reach every loader and definition
# in a few seconds (STTC over 60 electrodes costs seconds a record), not to cover the tier.
QUICK_RECORDS = 2
SLOW_FIRST = (
    "sharf",
    "mateus",
    "kapucu",
    "vassallo",
    "brewer",
    "snp_single",
    "trujillo",
    "snp_joint",
)


def _value(value: object) -> float:
    """A feature value as a float; undefined is NaN."""
    return float("nan") if value is None else float(value)


def smallest(records: list[SpikeRecord], rank: int) -> list[SpikeRecord]:
    """The record with the ``rank``-th fewest spikes (ties by id), or none if there are fewer."""
    ordered = sorted(records, key=lambda r: (sum(len(t) for t in r.channels.values()), r.record_id))
    return ordered[rank : rank + 1]


def _job(args: tuple[Unit, Path, int | None]) -> list[tuple]:
    """Rows for one unit under every definition set, in its own process.

    ``rank`` None takes every record of the unit; an integer takes one record, the one
    :func:`smallest` picks (the quick share).
    """
    unit, raw_root, rank = args
    try:
        records = load(unit, raw_root)
    except Exception:  # noqa: BLE001 - one unreadable unit must not stop the table
        log.exception("could not read %s %s", unit.loader, unit.key[:1])
        return []
    rows = []
    for record in records if rank is None else smallest(records, rank):
        for definition in REGISTRY.values():
            computed = extract(definition, record.channels, record.duration_s, record.noisy)
            rows += [
                (record.record_id, record.source, definition.name, f, _value(computed.get(f)))
                for f in FEATURE_NAMES
            ]
    return rows


def first_unit_per_source(found: list[Unit]) -> list[Unit]:
    """The quick tier's share: each source's first unit only."""
    seen: set[str] = set()
    out = []
    for unit in found:
        if unit.source not in seen:
            seen.add(unit.source)
            out.append(unit)
    return out


def build_table(
    raw_root: Path,
    hashes: dict[str, str],
    full: bool,
    quick: bool = False,
    processes: int | None = None,
) -> pd.DataFrame:
    """The long table over every unit this tier has (the quick share when ``quick``)."""
    import multiprocessing as mp

    from neuromps.parallel import worker_processes

    found = units(raw_root, hashes, full)
    found = first_unit_per_source(found) if quick else found
    # Slowest loaders first, so the pool does not end waiting on one large archive member.
    found = sorted(found, key=lambda u: SLOW_FIRST.index(u.loader))
    ranks = range(QUICK_RECORDS) if quick else [None]
    jobs = [(unit, Path(raw_root), rank) for unit in found for rank in ranks]
    log.info(
        "definition sidecar: %d units from %d sources", len(jobs), len({u.source for u in found})
    )
    if not jobs:
        return pd.DataFrame(columns=COLUMNS)
    with mp.Pool(min(processes or worker_processes(), len(jobs))) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    table = pd.DataFrame([row for batch in results for row in batch], columns=COLUMNS)
    return sort_table(table)


def sort_table(table: pd.DataFrame) -> pd.DataFrame:
    """Records in id order, definitions in registry order, features in schema order."""
    order = {
        "definition": {name: i for i, name in enumerate(REGISTRY)},
        "feature": {name: i for i, name in enumerate(FEATURE_NAMES)},
    }
    keys = table.assign(
        _d=table["definition"].map(order["definition"]), _f=table["feature"].map(order["feature"])
    )
    keys = keys.sort_values(["source", "record_id", "_d", "_f"], kind="stable")
    return keys.drop(columns=["_d", "_f"]).reset_index(drop=True)


def check_reference(table: pd.DataFrame, asset: pd.DataFrame) -> dict:
    """Whether every ``neuromps`` value equals the asset's own ``features.*`` value exactly.

    Two NaNs count as equal (an undefined feature in both). A record the sidecar has
    and the asset lacks is a failure too: it would mean the record ids disagree.
    """
    ours = table[table["definition"] == REFERENCE].pivot(
        index="record_id", columns="feature", values="value"
    )
    theirs = asset.set_index("record_id")
    missing = sorted(set(ours.index) - set(theirs.index))
    common = ours.index.intersection(theirs.index)
    mismatched = []
    for feature in FEATURE_NAMES:
        a = ours.loc[common, feature].to_numpy(dtype=float)
        b = theirs.loc[common, f"features.{feature}"].to_numpy(dtype=float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        mismatched += [(rid, feature) for rid in common[~same]]
    return {
        "n_records": int(len(ours)),
        "n_values": int(len(common) * len(FEATURE_NAMES)),
        "n_records_missing_from_asset": len(missing),
        "n_values_mismatched": len(mismatched),
        "examples": [f"{rid}:{feature}" for rid, feature in mismatched[:5]] + missing[:5],
        "pass": bool(len(ours) > 0 and not missing and not mismatched),
    }


def counts(table: pd.DataFrame, asset: pd.DataFrame) -> dict:
    """Rows per source and definition, and records covered against the asset's spike-time ones."""
    spike_records = asset[asset["data_level"] == "spike_times"].groupby("source").size()
    out = {}
    for source, rows in table.groupby("source"):
        out[str(source)] = {
            "records": int(rows["record_id"].nunique()),
            "asset_spike_time_records": int(spike_records.get(source, 0)),
            "rows_by_definition": {
                name: int((rows["definition"] == name).sum()) for name in REGISTRY
            },
        }
    return out


def write_table(table: pd.DataFrame, directory: Path) -> Path:
    """Write the sidecar beside the asset; returns its path."""
    path = Path(directory) / FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path
