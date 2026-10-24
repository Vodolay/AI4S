"""The FlyWire 783 connectome as the fly model needs it: who connects to whom, by what.

Each connection carries its sign from Shiu et al. 2024 (excitatory +1,
inhibitory -1, GABA and glutamate counted as inhibitory) and a transmitter class
that decides which drug knob acts on it. The class needs both the sign and the
neuron's predicted transmitter, because the two come from different predictions
and disagree for about 6% of presynaptic neurons; such a neuron is left to no
transmitter-specific knob rather than assigned by one source over the other.

Every table must come from the 783 release. Root IDs change when a neuron is
edited, so a table from another release matches only part of the model's
neurons; :func:`check_release` refuses one that covers less than 99%.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.fetch import flywire783
from neuromps.paths import RAW_ROOT

log = logging.getLogger("neuromps.fly")
RAW = RAW_ROOT / flywire783.SOURCE
MIN_COVERAGE = 0.99
UNPLACED = ""
CLASSES = ("ach", "gaba", "glut", "other_exc", "other_inh")


class ReleaseMismatch(ValueError):
    """A table's neurons are not the model's: probably another FlyWire release."""


@dataclass(frozen=True)
class Connectome:
    """Neurons by model index, connections as parallel arrays."""

    root_ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    synapses: np.ndarray
    sign: np.ndarray
    pre_class: np.ndarray
    neuropil: np.ndarray

    @property
    def n_neurons(self) -> int:
        """Number of model neurons."""
        return int(self.root_ids.size)

    def index_of(self, root_ids) -> np.ndarray:
        """Model indices of FlyWire root IDs; raises KeyError for one not in the model."""
        lookup = pd.Series(np.arange(self.n_neurons), index=self.root_ids)
        missing = [r for r in root_ids if r not in lookup.index]
        if missing:
            raise KeyError(f"{len(missing)} root ID(s) not in the model, e.g. {missing[:2]}")
        return lookup.loc[list(root_ids)].to_numpy()


def transmitter_class(sign: np.ndarray, top_nt: np.ndarray) -> np.ndarray:
    """Class of each connection from its sign and its presynaptic neuron's transmitter."""
    nt = pd.Series(top_nt, dtype=object).fillna("").to_numpy()
    out = np.where(sign > 0, "other_exc", "other_inh").astype(object)
    out[(sign > 0) & (nt == "acetylcholine")] = "ach"
    out[(sign < 0) & (nt == "gaba")] = "gaba"
    out[(sign < 0) & (nt == "glutamate")] = "glut"
    return out


def check_release(table: str, model_ids: np.ndarray, table_ids: np.ndarray) -> float:
    """Fraction of model neurons the table covers; raise below :data:`MIN_COVERAGE`."""
    coverage = float(np.isin(model_ids, table_ids).mean())
    if coverage < MIN_COVERAGE:
        raise ReleaseMismatch(
            f"{table} covers {coverage:.1%} of the model's neurons (need {MIN_COVERAGE:.0%}): "
            "is it from another FlyWire release than 783?"
        )
    return coverage


def primary_neuropil(counts: pd.DataFrame, root_ids: np.ndarray) -> np.ndarray:
    """Region where each neuron receives most of its input synapses ('' if none)."""
    placed = counts[counts["neuropil"].notna() & (counts["neuropil"] != "None")]
    placed = placed[placed["post_pt_root_id"].isin(root_ids)]
    top = placed.sort_values("count", ascending=False).drop_duplicates("post_pt_root_id")
    region = top.set_index("post_pt_root_id")["neuropil"]
    return region.reindex(root_ids).fillna(UNPLACED).to_numpy(dtype=object)


def neuropil_flow(con: Connectome) -> pd.Series:
    """Synapses from neurons placed in region A onto neurons placed in region B."""
    a, b = con.neuropil[con.pre], con.neuropil[con.post]
    keep = (a != UNPLACED) & (b != UNPLACED)
    frame = pd.DataFrame({"source": a[keep], "target": b[keep], "synapses": con.synapses[keep]})
    return frame.groupby(["source", "target"])["synapses"].sum()


def read_annotations(raw: Path = RAW) -> pd.DataFrame:
    """Per-neuron annotations from the CC-BY Zenodo archive, read inside the zip."""
    path = raw / flywire783.local_name(flywire783.ANNOTATION_ZIP).split("/", 1)[1]
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.endswith(flywire783.ANNOTATION_MEMBER))
        with archive.open(member) as handle:
            return pd.read_csv(handle, sep="\t", low_memory=False)


def load(raw: Path = RAW) -> Connectome:
    """Build the connectome from the fetched 783 files, checking they share one release."""
    root_ids = pd.read_csv(raw / "Completeness_783.csv", index_col=0).index.to_numpy(np.int64)
    con = pd.read_parquet(raw / "Connectivity_783.parquet")
    ann = read_annotations(raw).drop_duplicates("root_id").set_index("root_id")
    counts = pd.read_feather(raw / "per_neuron_neuropil_count_post_783.feather")
    check_release("annotations", root_ids, ann.index.to_numpy())
    check_release("neuropil counts", root_ids, counts["post_pt_root_id"].unique())
    pre = con["Presynaptic_Index"].to_numpy()
    sign = con["Excitatory"].to_numpy()
    top_nt = ann["top_nt"].reindex(root_ids).to_numpy(dtype=object)
    out = Connectome(
        root_ids=root_ids,
        pre=pre,
        post=con["Postsynaptic_Index"].to_numpy(),
        synapses=con["Connectivity"].to_numpy(),
        sign=sign,
        pre_class=transmitter_class(sign, top_nt[pre]),
        neuropil=primary_neuropil(counts, root_ids),
    )
    log.info("fly connectome: %d neurons, %d connections", out.n_neurons, out.pre.size)
    return out


def load_630(raw: Path = RAW) -> Connectome:
    """The release the paper used, with signs only: for the replication check, not for drugs.

    The 783 annotations do not cover 630 root IDs, so no transmitter class or
    region is assigned; every knob but the threshold leaves this model unchanged.
    """
    root_ids = pd.read_csv(
        raw / "2023_03_23_completeness_630_final.csv", index_col=0
    ).index.to_numpy(np.int64)
    con = pd.read_parquet(raw / "2023_03_23_connectivity_630_final.parquet")
    sign = con["Excitatory"].to_numpy()
    return Connectome(
        root_ids=root_ids,
        pre=con["Presynaptic_Index"].to_numpy(),
        post=con["Postsynaptic_Index"].to_numpy(),
        synapses=con["Connectivity"].to_numpy(),
        sign=sign,
        pre_class=np.where(sign > 0, "other_exc", "other_inh").astype(object),
        neuropil=np.full(root_ids.size, UNPLACED, dtype=object),
    )


INPUTS = (
    "Completeness_783.csv",
    "Connectivity_783.parquet",
    "flywire_annotations-v3.2.0.zip",
    "per_neuron_neuropil_count_post_783.feather",
)


def cached(raw: Path = RAW, cache_dir: Path | None = None) -> Connectome:
    """:func:`load`, kept in `data/interim/fly/` and rebuilt when an input file changes."""
    from neuromps.paths import INTERIM_ROOT
    from neuromps.qc.provenance import sha256_file

    cache_dir = cache_dir or INTERIM_ROOT / "fly"
    key = "|".join(sha256_file(raw / name)[:16] for name in INPUTS)
    path = cache_dir / "connectome_783.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["key"]) == key:
                return _from_arrays(stored)
    con = load(raw)
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
    return con


def _from_arrays(stored) -> Connectome:
    return Connectome(
        root_ids=stored["root_ids"],
        pre=stored["pre"],
        post=stored["post"],
        synapses=stored["synapses"],
        sign=stored["sign"],
        pre_class=np.asarray(CLASSES, dtype=object)[stored["pre_class"]],
        neuropil=stored["neuropil"].astype(object),
    )


def release_coverage(raw: Path = RAW) -> dict:
    """Fraction of the model's neurons each auxiliary table covers (what `load` checks)."""
    root_ids = pd.read_csv(raw / "Completeness_783.csv", index_col=0).index.to_numpy(np.int64)
    annotations = read_annotations(raw)["root_id"].to_numpy()
    counts = pd.read_feather(raw / "per_neuron_neuropil_count_post_783.feather")
    return {
        "annotations": round(check_release("annotations", root_ids, annotations), 4),
        "neuropil_counts": round(
            check_release("neuropil counts", root_ids, counts["post_pt_root_id"].unique()), 4
        ),
        "minimum_required": MIN_COVERAGE,
    }


def rewire(con: Connectome, seed: int = 0) -> Connectome:
    """A degree-preserving control: postsynaptic partners permuted across all connections.

    Every neuron keeps its numbers of outgoing and incoming connections, and every
    connection keeps its weight, sign and transmitter class with its presynaptic neuron.
    """
    post = np.random.default_rng(seed).permutation(con.post)
    return Connectome(
        root_ids=con.root_ids,
        pre=con.pre,
        post=post,
        synapses=con.synapses,
        sign=con.sign,
        pre_class=con.pre_class,
        neuropil=con.neuropil,
    )


def make_excitatory(con: Connectome, which: np.ndarray) -> Connectome:
    """The same network with the connections where `which` is true made excitatory."""
    return replace(con, sign=np.where(np.asarray(which, dtype=bool), 1, con.sign))


def subnetwork(con: Connectome, keep: np.ndarray) -> Connectome:
    """The neurons where `keep` is true, renumbered, with the connections among them only."""
    keep = np.asarray(keep, dtype=bool)
    new_index = np.full(con.n_neurons, -1)
    new_index[keep] = np.arange(int(keep.sum()))
    inside = keep[con.pre] & keep[con.post]
    return Connectome(
        root_ids=con.root_ids[keep],
        pre=new_index[con.pre[inside]],
        post=new_index[con.post[inside]],
        synapses=con.synapses[inside],
        sign=con.sign[inside],
        pre_class=con.pre_class[inside],
        neuropil=con.neuropil[keep],
    )
