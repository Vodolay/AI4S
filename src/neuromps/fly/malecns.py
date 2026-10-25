"""The male CNS connectome (MaleCNS v1.0) as a second fly brain for the replication.

Built the way `preregistration/fly_malecns_replication.json` registers it:
every body with status 'Traced', each connection signed by the presynaptic
body's consensus transmitter, and the published FlyWire neuron sets matched by
cell type and side. MaleCNS has no per-neuron neuropil among the files fetched,
so every neuron's region is left unplaced and the virtual array is drawn from
central-brain intrinsic neurons instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.fetch import malecns as source
from neuromps.fly.connectome import UNPLACED, Connectome, transmitter_class
from neuromps.fly.electrodes import VirtualArray
from neuromps.paths import RAW_ROOT

log = logging.getLogger("neuromps.fly")
RAW = RAW_ROOT / source.SOURCE
INHIBITORY = ("gaba", "glutamate", "histamine")
SIDE = {"left": "L", "right": "R"}


def signs(nt: np.ndarray) -> np.ndarray:
    """-1 for GABA, glutamate and histamine, +1 for everything else, missing included."""
    labels = pd.Series(nt, dtype=object).fillna("").to_numpy()
    return np.where(np.isin(labels, INHIBITORY), -1, 1)


def classes(sign: np.ndarray, nt: np.ndarray) -> np.ndarray:
    """Knob class per connection; histamine stays in `other_inh`."""
    return transmitter_class(sign, nt)


def _side(row: pd.Series) -> str | None:
    side = row["rootSide"] if str(row["superclass"]).endswith("sensory") else row["somaSide"]
    return None if pd.isna(side) else str(side)


def _names(row: pd.Series) -> set[str]:
    """A body's own type and the FlyWire types MaleCNS cross-references it to."""
    names = {str(row["type"])} if pd.notna(row["type"]) else set()
    if pd.notna(row["flywireType"]):
        names |= {t.strip() for t in str(row["flywireType"]).split(",")}
    return names


def match_sets(sets: dict, flywire: pd.DataFrame, male: pd.DataFrame) -> dict[str, list[int]]:
    """MaleCNS body IDs for each FlyWire set: same cell type, same side.

    A type matches by MaleCNS's own name or by its `flywireType` cross-reference.
    The registration named only the first; it finds no MN9, which MaleCNS
    calls "MN9" and cross-references to FlyWire's CB0701 (see DEVIATIONS).
    """
    fw = flywire.set_index("root_id")
    sided = male.assign(side=male.apply(_side, axis=1), names=male.apply(_names, axis=1))
    out = {}
    for name, ids in sets.items():
        wanted = {(fw.at[i, "cell_type"], SIDE.get(fw.at[i, "side"])) for i in ids if i in fw.index}
        hit = sided[
            [
                any((t, side) in wanted for t in names)
                for names, side in zip(sided["names"], sided["side"], strict=True)
            ]
        ]
        out[name] = sorted(int(b) for b in hit["bodyId"])
    return out


DEVIATIONS = {
    "neuron_sets": (
        "types also matched through MaleCNS's flywireType cross-reference, not only by type "
        "name: by name alone no body matches FlyWire MN9 (CB0701). Decided before any "
        "MaleCNS simulation."
    )
}


def _annotations(raw: Path) -> pd.DataFrame:
    ann = pd.read_feather(raw / source.FILES[1])
    return ann[ann["status"] == "Traced"].reset_index(drop=True)


def load(raw: Path = RAW) -> tuple[Connectome, pd.DataFrame]:
    """The MaleCNS model connectome and its annotations (Traced bodies only)."""
    ann = _annotations(raw)
    ids = ann["bodyId"].to_numpy(np.int64)
    weights = pd.read_feather(raw / source.FILES[0])
    log.info("MaleCNS weights columns: %s", list(weights.columns))
    pre_col, post_col, w_col = _weight_columns(weights)
    index = pd.Series(np.arange(ids.size), index=ids)
    keep = weights[pre_col].isin(index.index) & weights[post_col].isin(index.index)
    keep &= weights[pre_col] != weights[post_col]
    w = weights.loc[keep]
    nt = pd.read_feather(raw / source.FILES[2]).set_index("body")["consensus_nt"]
    body_nt = nt.reindex(ids).to_numpy(dtype=object)
    pre = index.loc[w[pre_col]].to_numpy()
    sign = signs(body_nt)[pre]
    con = Connectome(
        root_ids=ids,
        pre=pre,
        post=index.loc[w[post_col]].to_numpy(),
        synapses=w[w_col].to_numpy(),
        sign=sign,
        pre_class=classes(sign, body_nt[pre]),
        neuropil=np.full(ids.size, UNPLACED, dtype=object),
    )
    log.info("MaleCNS: %d neurons, %d connections", con.n_neurons, con.pre.size)
    return con, ann


def _weight_columns(weights: pd.DataFrame) -> tuple[str, str, str]:
    cols = list(weights.columns)
    pre = next(c for c in cols if "pre" in c.lower())
    post = next(c for c in cols if "post" in c.lower())
    weight = next(c for c in cols if c.lower() in ("weight", "count", "syn_count", "synweight"))
    return pre, post, weight


def array(con: Connectome, ann: pd.DataFrame, n_units: int = 544, seed: int = 0) -> VirtualArray:
    """Single-unit electrodes drawn from central-brain intrinsic neurons."""
    pool = np.flatnonzero(ann["superclass"].to_numpy(dtype=object) == "cb_intrinsic")
    drawn = np.sort(np.random.default_rng(seed).choice(pool, n_units, replace=False))
    units = {f"cb_{k}": np.array([u]) for k, u in enumerate(drawn)}
    return VirtualArray(units=units, region=dict.fromkeys(units, "cb_intrinsic"))


def cached(raw: Path = RAW, cache_dir: Path | None = None) -> tuple[Connectome, pd.DataFrame]:
    """:func:`load`, kept in `data/interim/fly/` and rebuilt when an input file changes."""
    from neuromps.fly.connectome import CLASSES, _from_arrays
    from neuromps.paths import INTERIM_ROOT
    from neuromps.qc.provenance import sha256_file

    cache_dir = cache_dir or INTERIM_ROOT / "fly"
    key = "|".join(sha256_file(raw / name)[:16] for name in source.FILES)
    path, ann_path = cache_dir / "connectome_malecns.npz", cache_dir / "malecns_bodies.parquet"
    if path.exists() and ann_path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["key"]) == key:
                return _from_arrays(stored), pd.read_parquet(ann_path)
    con, ann = load(raw)
    cache_dir.mkdir(parents=True, exist_ok=True)
    codes = {c: k for k, c in enumerate(CLASSES)}
    np.savez(
        path,
        key=key,
        root_ids=con.root_ids,
        pre=con.pre.astype(np.int32),
        post=con.post.astype(np.int32),
        synapses=con.synapses.astype(np.int32),
        sign=con.sign.astype(np.int8),
        pre_class=np.array([codes[c] for c in con.pre_class], dtype=np.int8),
        neuropil=con.neuropil.astype(str),
    )
    columns = ["bodyId", "type", "flywireType", "rootSide", "somaSide", "superclass"]
    ann[columns].to_parquet(ann_path)
    return con, ann[columns]


SIZE_COLUMNS = (":ID(Body-ID)", "size:long")


def neuron_sizes(raw_root: Path = RAW_ROOT, cache_dir: Path | None = None) -> pd.Series:
    """Voxel count per body ID, from the neuPrint neuron table (`malecns_sizes` source).

    Only the two columns are read from the 4.65 GB table; the result is kept as a small
    parquet file in `data/interim/fly/` and rebuilt when the table changes.
    """
    import pyarrow.feather as feather

    from neuromps.fetch import malecns_sizes
    from neuromps.paths import interim_dir
    from neuromps.qc.provenance import sha256_file

    table = Path(raw_root) / malecns_sizes.REL
    cache = Path(cache_dir or interim_dir() / "fly") / "malecns_sizes.parquet"
    stamp = cache.with_suffix(".source")
    if not (cache.exists() and stamp.exists() and stamp.read_text() == _stat(table)):
        frame = feather.read_table(table, columns=list(SIZE_COLUMNS)).to_pandas()
        frame.columns = ["body", "size"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache, index=False)
        stamp.write_text(_stat(table))
        log.info(
            "neuron sizes: %d bodies from %s (sha256 %s)",
            len(frame),
            table,
            sha256_file(table)[:12],
        )
    frame = pd.read_parquet(cache)
    return pd.Series(frame["size"].to_numpy(), index=frame["body"].to_numpy(), name="size")


def _stat(path: Path) -> str:
    s = Path(path).stat()
    return f"{s.st_size}:{int(s.st_mtime)}"
