"""Tab 3: the data asset and the standard it validates against."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app import loaders

BADGES = {
    "too_few_active_electrodes": "#D55E00",
    "replicate_divergence": "#E69F00",
    "outlier_feature_value": "#E69F00",
    "incomplete_dose_series": "#56B4E9",
    "excluded_well_upstream": "#CC79A7",
    "noisy_electrodes_present": "#009E73",
}


def render(stamps: dict[str, float]) -> None:
    """Dataset card, filterable records with QC badges, schema and provenance.

    The card, the schema and the provenance panel read `outputs/metrics.json`,
    which travels with the repository; only the record table needs the built
    asset, which does not. A clone that has not run the pipeline yet therefore
    shows everything except the table, and says why - rather than showing an
    empty tab and implying there is nothing there.
    """
    metrics = loaders.load_metrics(stamps["metrics"])
    records = loaders.load_asset(stamps["asset"])

    _dataset_card(metrics)
    st.divider()
    if records.empty:
        st.info(
            "The built asset is not in this checkout, so the record table is unavailable. "
            "Run `.venv/bin/python run_all.py --quick` to build it; the numbers above come from "
            "`outputs/metrics.json`, which is committed."
        )
    else:
        _records_table(records)
    st.divider()
    # The standard and the provenance machinery are what make the asset worth
    # having, and they are also the part a first-time reader does not need open.
    # They are one click away rather than occupying the page.
    with st.expander("The standard, and how provenance is enforced"):
        left, right = st.columns(2)
        with left:
            _schema_panel(stamps)
        with right:
            _provenance_panel(metrics, stamps)


def _dataset_card(metrics: dict) -> None:
    """The headline counts, read from metrics.json rather than recounted here."""
    asset = metrics.get("data_asset", {})
    counts = asset.get("counts", {})
    st.subheader("Dataset card")
    columns = st.columns(5)
    columns[0].metric("Records", f"{asset.get('n_records', 0):,}")
    columns[1].metric("Sources", len(counts.get("by_source", {})))
    columns[2].metric("Usable", f"{counts.get('by_usable', {}).get('True', 0):,}")
    columns[3].metric("Species", len(counts.get("by_species", {})))
    columns[4].metric("Mechanism classes", max(len(counts.get("by_mechanism_class", {})) - 1, 0))

    rows = []
    for source, n in sorted(counts.get("by_source", {}).items(), key=lambda kv: -kv[1]):
        rows.append({"source": source, "records": n})
    card = pd.DataFrame(rows)
    licenses = _licenses()
    card["license"] = card["source"].map(licenses).fillna("see CITATIONS.md")
    st.dataframe(card, hide_index=True, width="stretch")
    st.caption(
        f"Generated {asset.get('generated_at', 'unknown')} at conformance level 2: every record's "
        "provenance hash matched data/raw/MANIFEST.json, or the build would have refused to write."
    )


def _licenses() -> dict[str, str]:
    """Per-source licenses, as the fetchers recorded them at download time."""
    manifest = loaders.REPO_ROOT / "data" / "raw" / "MANIFEST.json"
    if not manifest.exists():
        return {}
    sources = json.loads(manifest.read_text())
    sources = sources.get("sources", sources)
    return {
        name: entry.get("license", "") for name, entry in sources.items() if isinstance(entry, dict)
    }


def _records_table(records: pd.DataFrame) -> None:
    """A filterable view of the harmonized records, with QC flags as badges."""
    st.subheader("Harmonized records")
    columns = st.columns(4)
    source = columns[0].selectbox("Source", ["all", *sorted(records["source"].unique())])
    species = columns[1].selectbox(
        "Species", ["all", *sorted(records["cell_source.species"].dropna().unique())]
    )
    level = columns[2].selectbox("Data level", ["all", *sorted(records["data_level"].unique())])
    usable_only = columns[3].checkbox("Usable only", value=True)

    view = records
    if source != "all":
        view = view[view["source"] == source]
    if species != "all":
        view = view[view["cell_source.species"] == species]
    if level != "all":
        view = view[view["data_level"] == level]
    if usable_only:
        view = view[view["qc.usable"]]

    st.caption(f"{len(view):,} records match. Showing the first 200.")
    shown = view.head(200)
    display = shown[
        [
            "record_id",
            "source",
            "cell_source.species",
            "data_level",
            "compound.name",
            "compound.mechanism_class",
            "features.mfr_hz",
            "qc.usable",
        ]
    ].rename(columns=lambda c: c.split(".")[-1])
    st.dataframe(display, hide_index=True, width="stretch", height=260)
    _flag_badges(view)
    _provenance_expander(shown)


def _flag_badges(view: pd.DataFrame) -> None:
    """QC flags present in the current selection, coloured by flag."""
    # qc.flags is a semicolon-separated string, not a list: iterating it directly
    # counts characters, which is what the first version of this panel did.
    counts: dict[str, int] = {}
    for flags in view["qc.flags"].dropna():
        for flag in _split_flags(flags):
            counts[flag] = counts.get(flag, 0) + 1
    if not counts:
        st.caption("No QC flags in this selection.")
        return
    badges = " ".join(
        f"<span style='background:{BADGES.get(flag, '#888')};color:white;padding:2px 8px;"
        f"border-radius:10px;margin-right:6px;font-size:0.8em'>{flag}: {n}</span>"
        for flag, n in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    st.markdown(f"**QC flags in this selection** {badges}", unsafe_allow_html=True)
    st.caption("Flags are informational: a flagged record stays in the asset with its reason.")


def _split_flags(flags) -> list[str]:
    """The flags on one record, however the column stores them."""
    if isinstance(flags, str):
        return [part.strip() for part in flags.split(";") if part.strip()]
    return [str(part) for part in list(flags) if str(part).strip()]


def _provenance_expander(shown: pd.DataFrame) -> None:
    """The source hash chain for one record, on demand."""
    with st.expander("Provenance for one record (source file, SHA256, transform)"):
        if shown.empty:
            st.caption("No records selected.")
            return
        record_id = st.selectbox("Record", shown["record_id"].tolist(), key="prov_record")
        row = shown[shown["record_id"] == record_id].iloc[0]
        chain = row.get("provenance.source_files")
        transform = row.get("provenance.transform")
        st.json(
            {
                "record_id": record_id,
                "source_files": _as_list(chain),
                "transform": _as_list(transform),
            }
        )


def _as_list(value) -> list:
    """Numpy arrays and lists both arrive here; JSON needs a list."""
    if value is None:
        return []
    return [dict(v) if hasattr(v, "keys") else v for v in list(value)]


def _schema_panel(stamps: dict[str, float]) -> None:
    """The record schema, and what conformance level 2 requires."""
    st.subheader("The standard")
    schema = loaders.load_schema(stamps["metrics"])
    properties = list(schema.get("properties", {}))
    st.caption(f"`schema/neuromps_schema.json` - {len(properties)} top-level fields.")
    st.code("\n".join(properties) or "schema not found", language="text")
    st.markdown(
        "Records carry an explicit **data level** and compare only within a level: "
        "`spike_times` supports the nine-feature vector, `fitted_curve` does not."
    )


def _provenance_panel(metrics: dict, stamps: dict[str, float]) -> None:
    """A worked level-2 check, and the bug that made the rule worth enforcing."""
    st.subheader("Provenance check (level 2)")
    provenance = loaders.load_provenance(stamps["asset"])
    checked = provenance.get("n_records_checked") or metrics.get("data_asset", {}).get("n_records")
    st.success(
        f"Level 2 passed for {checked:,} records at build time." if checked else "No record."
    )
    example = (provenance.get("sources") or provenance.get("files") or [{}])[0]
    st.json(example if example else provenance, expanded=False)
    st.markdown(
        "**Why this is enforced, not documented.** A `--quick` fetch once marked a 582 MB archive "
        "`skipped`, which erased its files from the manifest; 580 records then failed this check "
        "and the build aborted rather than shipping an asset whose provenance did not resolve. "
        "See *Known limitations* and the HD7 entry in `report/LAB_NOTEBOOK.md`."
    )
    _download_button()


def _download_button() -> None:
    """Offer the asset and its provenance for download."""
    if loaders.ASSET.exists():
        st.download_button(
            "Download the asset (parquet)",
            data=loaders.ASSET.read_bytes(),
            file_name="neuromps.parquet",
            mime="application/octet-stream",
        )
    if loaders.PROVENANCE.exists():
        st.download_button(
            "Download provenance.json",
            data=loaders.PROVENANCE.read_bytes(),
            file_name="provenance.json",
            mime="application/json",
        )
