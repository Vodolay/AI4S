"""Reading the built asset, for code that must tolerate a partial one.

A ``--quick`` run fetches only the small-tier sources, so the asset it builds
can contain no multi-compartment records at all. The ``device.*`` columns are
then *absent* rather than null: the flattened table only grows a column when
some record carries the block. Analyses that need compartments therefore ask
for them through :func:`compartment_records`, which hands back an empty but
column-complete frame in that case, so a compartment analysis skips itself
instead of raising ``KeyError`` on a fresh checkout.
"""

import pandas as pd

COMPARTMENT_COLUMN = "device.compartment_id"


def compartment_records(records: pd.DataFrame) -> pd.DataFrame:
    """Records that belong to a device compartment; empty when the asset has none.

    The returned frame always carries ``device.compartment_id``, so callers can
    apply their own masks to it without checking whether the column exists.
    """
    if COMPARTMENT_COLUMN not in records.columns:
        empty = records.iloc[:0].copy()
        empty[COMPARTMENT_COLUMN] = pd.Series(dtype="object")
        return empty
    return records[records[COMPARTMENT_COLUMN].notna()]
