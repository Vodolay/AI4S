"""NeuroMPS demo: the data asset, mechanism prediction, and the digital twin.

Run with `streamlit run app/app.py`, after `python run_all.py --quick` has
written the asset, the metrics and the precomputed demo inputs.

Nothing here computes a reported number. Everything shown comes from
`metrics.json` or from `demo/`, both produced by scripts under `src/`, so the
app and the technical report cannot disagree. Those are the full run's files
where they exist, and a `--quick` run's otherwise, with a line on the page
saying which. The single
exception is the "Run one live simulation" button in Tab 3, which is there to
show that the twin is a real simulator and not a lookup table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]

# This file is `app/app.py`, so the name `app` is ambiguous: it is both the
# package holding the tabs and this script. A runner puts the script's own
# directory on `sys.path` and registers the script under its file stem, and
# both of those make `app` resolve to this half-executed file - an ImportError
# under `pytest`, and unbounded recursion once the import is retried. Undo
# both before importing, rather than renaming the entry point that the README,
# the Makefile and every screenshot caption already point at.
_HERE = str(Path(__file__).resolve().parent)
while _HERE in sys.path:
    sys.path.remove(_HERE)
if str(REPO_ROOT) not in sys.path:  # `streamlit run` does not add the repo root
    sys.path.insert(0, str(REPO_ROOT))
# A hosted copy (Streamlit Community Cloud) installs app/requirements.txt and never
# `pip install -e .`, so `neuromps` is found in src/ instead of site-packages.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))
_running_script = sys.modules.get("app")
if _running_script is not None and not hasattr(_running_script, "__path__"):
    del sys.modules["app"]  # the runner keeps its own reference to this module

from app import loaders, tab_data, tab_fly, tab_models, tab_twin  # noqa: E402

BANNER = (
    "**Research decision-support only.** Nothing in this app is an efficacy, safety, "
    "diagnostic or clinical claim."
)


# Slug per tab, in display order. The slugs exist because the numbers moved:
# the data tab was first and is now last, so `?tab=1` no longer means what it
# meant in an older screenshot caption. A slug says which tab it wants.
TABS = (
    ("models", "1 · Mechanism & transfer"),
    ("twin", "2 · Digital twin & inference"),
    ("data", "3 · Data asset & standard"),
    ("fly", "4 · From chip to animal"),
)


def _requested_tab() -> int | None:
    """The index the `?tab=` value asks for, or None.

    Accepts a slug or a 1-based number. The running server hands the value back
    as a string; the AppTest harness hands back a list of strings, and accepting
    only one of those silently disables the deep link in whichever context was
    not checked.
    """
    value = st.query_params.get("tab")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    value = str(value).strip().lower()
    for index, (slug, _) in enumerate(TABS):
        if value == slug:
            return index
    if value.isdigit() and 1 <= int(value) <= len(TABS):
        return int(value) - 1
    return None


def main() -> None:
    """Draw the page."""
    st.set_page_config(page_title="NeuroMPS demo", layout="wide", page_icon="🧠")
    st.title("NeuroMPS")
    st.caption(
        "A provenance-tracked MEA data asset, and a two-compartment spiking-network twin "
        "calibrated on it."
    )
    st.warning(BANNER)

    if note := loaders.run_root_note():
        st.info(note)

    stamps = loaders.stamps()
    if stamps["demo"] == 0.0:
        st.info(loaders.MISSING_DEMO)

    # The models lead and the asset follows: a reader wants to know what was
    # found before how the records were shelved.
    renderers = [tab_models.render, tab_twin.render, tab_data.render, tab_fly.render]
    # `?tab=twin` renders one tab on its own. Streamlit tabs cannot be selected
    # programmatically, so this is what makes a tab linkable - and screenshottable.
    requested = _requested_tab()
    if requested is not None:
        st.subheader(TABS[requested][1])
        renderers[requested](stamps)
        return
    labels = [label for _, label in TABS]
    for tab, render in zip(st.tabs(labels), renderers, strict=True):
        with tab:
            render(stamps)

    st.divider()
    st.caption(
        "Every number here is read from outputs/metrics.json, written by scripts under src/. "
        "See report/TECHNICAL_REPORT.md for how each was produced, and report/LAB_NOTEBOOK.md "
        "for what was tried and abandoned."
    )


main()
