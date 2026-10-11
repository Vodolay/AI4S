"""Generate the Kaggle notebook version of the Streamlit demo.

A Kaggle notebook cannot host the Streamlit app: the container has no inbound
networking, so there is no port for a browser to reach, and a saved notebook has
no live kernel to answer a callback. What it can do is emit self-contained
output - Vega-Lite specifications with their data embedded - which stays
interactive in the saved notebook without anything running behind it.

That port is cheap here only because `demo/precompute.py` already decided that
"no interaction runs a simulator". Five of the app's six controls select among
precomputed grid points, so they become client-side selections; the sixth runs
Brian2, which in a notebook is simply a cell.

The notebook is generated rather than hand-written so it cannot drift from the
artifacts it reads. Everything it shows is loaded at run time from the files in
`DATA_FILES`, not baked in here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("neuromps.demo")

# What the Kaggle dataset must contain. Paths are relative to the repository
# root; `build_dataset_dir` copies them into one flat folder to upload.
DATA_FILES = (
    "outputs/demo/meta.json",
    "outputs/demo/grid.parquet",
    "outputs/demo/real_records.parquet",
    "outputs/demo/real_posteriors.npz",
    # The raster cell loads one of these by the name in grid.parquet's `run_file`.
    "outputs/demo/runs/",
    "outputs/metrics.json",
    "data/processed/neuromps.parquet",
)
NOTEBOOK_NAME = "neuromps_demo.ipynb"
# Kaggle mounts an attached dataset here; the local checkout keeps the same
# files where the pipeline wrote them.
SEARCH_ROOTS = ("/kaggle/input", ".", "..")


def _md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [f"{line}\n" for line in lines]}


def _as_source_lines(source: str) -> list[str]:
    """Cell source split the way nbformat stores it: no newline on the last line."""
    lines = source.strip("\n").splitlines()
    return [f"{line}\n" for line in lines[:-1]] + lines[-1:] if lines else []


def _formatted(source: str) -> str:
    """Run a cell through `ruff format`, so generated code cannot fail the linter.

    The cells are string literals in this module, and hand-matching them to the
    formatter is a losing game: the generated notebook is linted like any other
    file, and a stray blank line in a literal here failed CI while every local
    check passed. Formatting at generation time removes the class. If ruff is not
    installed the source is emitted unchanged rather than the build failing.
    """
    import subprocess
    import sys

    # `python -m ruff`, not the `ruff` on PATH: in a virtualenv the binary is in
    # `.venv/bin` and PATH usually is not set, so a PATH lookup silently skipped
    # the formatting - and then CI, which does have it, failed on the result.
    done = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", "cell.py", "-"],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout if done.returncode == 0 and done.stdout.strip() else source


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        # nbformat convention: every line ends in a newline except the last.
        # Appending one to the last line too makes `ruff format --check` want the
        # file reformatted, which fails CI on a file nothing hand-edits.
        "source": _as_source_lines(_formatted(source)),
    }


SETUP = '''
# Find the data whether this runs on Kaggle or in a checkout of the repository.
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _find(name, roots=("/kaggle/input", ".", "..", "../..")):
    """First match for `name` under any root. Kaggle mounts datasets under
    /kaggle/input/<dataset-name>/; a checkout has them under outputs/demo and
    data/processed."""
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for candidate in sorted(base.rglob(name)):
            return candidate
    raise FileNotFoundError(
        f"{name} not found. On Kaggle, attach the NeuroMPS demo dataset; "
        "locally, run `python run_all.py` first (outputs/demo is written by the demo stage)."
    )

META = json.loads(_find("meta.json").read_text())
GRID = pd.read_parquet(_find("grid.parquet"))
RECORDS = pd.read_parquet(_find("real_records.parquet"))
POSTERIORS = np.load(_find("real_posteriors.npz"))["samples"]
METRICS = json.loads(_find("metrics.json").read_text())
ASSET = pd.read_parquet(_find("neuromps.parquet"))

print(f"asset            {len(ASSET):,} records")
print(f"simulated grid   {len(GRID)} points, {META['simulated_duration_s']:g}s each, "
      f"{META['n_per_compartment']} neurons per compartment")
print(f"real recordings  {len(RECORDS)} with posteriors of shape {POSTERIORS.shape}")
'''

ASSET_VIEW = """
import altair as alt

# Vega-Lite specifications carry their own data, so these stay interactive in the
# saved notebook with no kernel behind them.
by_source = (
    ASSET.groupby(["source", "data_level"]).size().reset_index(name="records")
)
chart = (
    alt.Chart(by_source)
    .mark_bar()
    .encode(
        x=alt.X("records:Q", scale=alt.Scale(type="symlog"), title="records (log scale)"),
        y=alt.Y("source:N", sort="-x", title=None),
        color=alt.Color("data_level:N", title="data level"),
        tooltip=["source", "data_level", "records"],
    )
    .properties(height=220, title="REAL records by source and data level")
)
chart
"""

ASSET_NOTE = """
levels = ASSET["data_level"].value_counts()
spikes = int(levels.get("spike_times", 0))
print(f"{spikes:,} records hold spike times and can be recomputed from scratch.")
print(f"{len(ASSET) - spikes:,} hold someone else's fitted summary and cannot.")
print()
# The schema nests cell provenance, so species is a dotted field rather than a
# bare column - the kind of thing that only shows up when the notebook is run.
print("Species:", dict(ASSET["cell_source.species"].value_counts()))
"""

MODELS_VIEW = """
task1 = METRICS["mechanism_baselines"]["task1_within_platform"]
task1b = METRICS["mechanism_baselines"]["task1b_independent_platform"]

rows = []
for label, block in (("within platform", task1), ("unseen platform", task1b)):
    for name, model in block["models"].items():
        low, high = model["balanced_accuracy_ci95"]
        rows.append(
            {
                "setting": label,
                "model": name.replace("_", " "),
                "balanced accuracy": model["balanced_accuracy_mean"],
                "low": low,
                "high": high,
                "chance": block["chance_balanced_accuracy"],
            }
        )
scores = pd.DataFrame(rows)

base = alt.Chart(scores).encode(y=alt.Y("model:N", title=None))
chart = (
    base.mark_point(filled=True, size=90).encode(
        x=alt.X("balanced accuracy:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=list(scores.columns),
    )
    + base.mark_rule().encode(x="low:Q", x2="high:Q")
    + base.mark_tick(color="firebrick", thickness=2).encode(x="chance:Q")
).facet(row=alt.Row("setting:N", title=None)).properties(
    title="Mechanism classification, REAL data (red tick = chance)"
)
chart
"""

TWIN_VIEW = """
# The app's three sliders snap to this grid. Here the same choice is a dropdown
# inside the chart, which works client-side in the saved notebook.
features = [c for c in GRID.columns if c.startswith("c0_")]
long = GRID.melt(
    id_vars=["forward_backward_ratio", "gabaa_scale", "ampa_scale", "silent"],
    value_vars=features,
    var_name="feature",
    value_name="value",
)
long["feature"] = long["feature"].str.removeprefix("c0_")

pick = alt.binding_select(options=sorted(long["feature"].unique()), name="feature ")
selection = alt.selection_point(fields=["feature"], bind=pick, value="mfr_hz")

chart = (
    alt.Chart(long)
    .mark_line(point=True)
    .encode(
        x=alt.X("gabaa_scale:Q", title="gabaa_scale  (1.0 = untreated)"),
        y=alt.Y("value:Q", title="SIMULATED feature value"),
        color=alt.Color("ampa_scale:N", title="ampa_scale"),
        tooltip=["feature", "gabaa_scale", "ampa_scale", "forward_backward_ratio", "value"],
    )
    .add_params(selection)
    .transform_filter(selection)
    .properties(width=260, height=200, title="SIMULATED twin: one grid point per marker")
    .facet(column=alt.Column("forward_backward_ratio:N", title="forward/backward ratio"))
)
chart
"""

TWIN_RASTER = """
import matplotlib.pyplot as plt

# One grid point's spike raster. Change `POINT` and re-run if you are running
# this notebook rather than reading it. Spikes are stored per compartment.
POINT = 2
row = GRID.iloc[POINT]
run = np.load(_find(row["run_file"]))
offset = int(META["n_per_compartment"])
fig, ax = plt.subplots(figsize=(9, 3.2))
for c, colour in ((0, "#0072B2"), (1, "#D55E00")):
    ax.plot(
        run[f"c{c}_times"],
        run[f"c{c}_neurons"] + (0 if c == 0 else offset),
        "|",
        markersize=2,
        color=colour,
        label=f"compartment {c}",
    )
ax.legend(loc="upper right", fontsize=8, markerscale=4)
ax.set_xlabel("time (s)")
ax.set_ylabel("neuron")
ax.set_title(
    f"SIMULATED raster - gabaa_scale {row['gabaa_scale']:g}, "
    f"ampa_scale {row['ampa_scale']:g}, ratio {row['forward_backward_ratio']:g}",
    fontsize=10,
)
plt.show()
"""

POSTERIOR_VIEW = """
# The posterior's inferred GABA-A scale for each real recording, against dose.
names = META["parameter_names"]
# NaN where rejection sampling timed out before drawing the full set, so the
# median must ignore them rather than propagate.
gabaa = np.nanmedian(POSTERIORS[:, :, names.index("gabaa_scale")], axis=1)
doses = RECORDS["dose_value"].fillna(0.0).to_numpy(dtype=float)
doses = np.where(RECORDS["dose_unit"].fillna("uM").to_numpy() == "nM", doses / 1000.0, doses)

inferred = pd.DataFrame(
    {
        "dose_uM": doses,
        "gabaa_scale": gabaa,
        "device": RECORDS["device_id"],
        "phase": RECORDS["phase"],
    }
)
chart = (
    alt.Chart(inferred)
    .mark_circle(size=70, opacity=0.7)
    .encode(
        x=alt.X("dose_uM:Q", title="bicuculline dose (uM)"),
        y=alt.Y("gabaa_scale:Q", title="inferred gabaa_scale  (1.0 = untreated)"),
        color=alt.Color("phase:N"),
        tooltip=["device", "dose_uM", "gabaa_scale", "phase"],
    )
    .properties(
        width=520,
        height=260,
        title="Posterior trained on SIMULATIONS, applied to REAL recordings",
    )
)
chart + alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
"""


def code_cells() -> list[str]:
    """Every code cell's source, in order. Tested for syntax, so a typo here fails a test."""
    return [SETUP, ASSET_VIEW, ASSET_NOTE, MODELS_VIEW, TWIN_VIEW, TWIN_RASTER, POSTERIOR_VIEW]


def build_notebook() -> dict:
    """The notebook document, as nbformat 4 JSON."""
    cells = [
        _md(
            "# NeuroMPS — the demo, as a notebook",
            "",
            "A provenance-tracked dataset of public MEA recordings, and a calibrated",
            "two-compartment spiking-network twin. This is the Streamlit demo's content in a form",
            "Kaggle can host: the app is a server, and a notebook has no port for a browser to",
            "reach, so the interactive parts here are Vega-Lite charts that carry their own data",
            "and keep working in saved output with no kernel behind them.",
            "",
            "> **Research decision-support only.** Nothing here is an efficacy, safety, diagnostic",
            "> or clinical claim, and nothing here should be used to make one.",
            "",
            "Every figure is computed from the attached files at run time. Nothing is typed in.",
            "The full analysis, technical report and the Streamlit app are in the repository.",
        ),
        _md("## Setup"),
        _code(SETUP),
        _md(
            "## 1. The data asset",
            "",
            "Eight public sources under licences permitting reuse, every record carrying its",
            "source, licence and a checksum of the file it came from. The split that matters is",
            "not the count but the **data level**: only records holding spike times can be",
            "recomputed through one common feature definition, and almost all of the modelling",
            "uses those alone.",
        ),
        _code(ASSET_VIEW),
        _code(ASSET_NOTE),
        _md(
            "## 2. Can you read a drug's mechanism from the features?",
            "",
            "Every model works on the **change** between a treated well and its own baseline, and",
            "splits group by culture so a model cannot recognise the culture instead of the",
            "treatment. Within one platform there is real signal. On a platform the model has",
            "never seen, there is close to none — and that second number is the more informative",
            "of the two.",
        ),
        _code(MODELS_VIEW),
        _md(
            "## 3. The digital twin",
            "",
            "Two compartments of leaky integrate-and-fire neurons with spike-frequency adaptation,",
            "wired through a directional pathway, measured by virtual electrodes through the same",
            "feature code the real recordings use. The grid below was simulated in advance; use",
            "the dropdown to change which feature is plotted.",
        ),
        _code(TWIN_VIEW),
        _md(
            "### One grid point's raster",
            "",
            "Static, because a raster is a picture rather than a selection. Change `POINT` and",
            "re-run the cell if you are running this notebook rather than reading it.",
        ),
        _code(TWIN_RASTER),
        _md(
            "## 4. Running the twin backwards on real recordings",
            "",
            "A posterior trained only on simulations, applied to real bicuculline recordings. Two",
            "things to look for, and both are negative results: untreated recordings do not come",
            "back at the neutral value of 1.0, and the inferred scale does not track dose. The",
            "report's §18 shows the second is a property of the measurement rather than of the",
            "estimator — no feature in this set moves monotonically with dose.",
        ),
        _code(POSTERIOR_VIEW),
        _md(
            "## What this does not show",
            "",
            "- Nothing clinical, and no efficacy or safety claim.",
            "- Mechanism accuracy is modest within a platform and near chance on an unseen one.",
            "- Most classification rests on 73 treated/baseline pairs from 50 cultures.",
            "- The twin matches the real band on some features and not others.",
            "",
            f"The simulated grid here was run at {'{'}META['n_per_compartment']{'}'} neurons per",
            "compartment — printed by the setup cell, and worth comparing against the calibrated",
            "density reported in the technical report, since cell density is what sets the twin's",
            "synchrony regime.",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path) -> Path:
    """Write the notebook, creating its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_notebook(), indent=1) + "\n")
    return path


def build_dataset_dir(destination: Path, root: Path | None = None) -> list[Path]:
    """Copy everything in `DATA_FILES` into one folder, ready to upload to Kaggle.

    The notebook finds its inputs by name with `rglob`, so the layout inside the
    dataset does not matter; what matters is that every file it reads is present.
    A missing file is reported rather than skipped silently, because the failure
    it causes appears only for whoever opens the notebook next.
    """
    import shutil

    root = Path(root or ".")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for name in DATA_FILES:
        source = root / name
        if source.is_dir():
            target = destination / Path(name.rstrip("/")).name
            shutil.copytree(source, target, dirs_exist_ok=True)
            copied.extend(sorted(target.rglob("*")))
        elif source.exists():
            target = destination / source.name
            shutil.copy2(source, target)
            copied.append(target)
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"not built yet, run the pipeline first: {', '.join(missing)}")
    return copied


def main(argv: list[str] | None = None) -> int:
    """CLI: write the notebook and say what to upload with it."""
    import argparse

    from neuromps.paths import outputs_dir

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="accepted for pipeline symmetry")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--dataset-dir", type=Path, default=None, help="stage the Kaggle dataset here")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = write_notebook(args.output or (outputs_dir() / "demo" / NOTEBOOK_NAME))
    log.info("wrote %s", path)
    if args.dataset_dir:
        files = build_dataset_dir(args.dataset_dir)
        size = sum(f.stat().st_size for f in files if f.is_file())
        log.info("staged %d files (%.1f MB) in %s", len(files), size / 1e6, args.dataset_dir)
        log.info("upload that folder as a private Kaggle dataset and attach it to the notebook")
    else:
        log.info("pass --dataset-dir to stage the files the notebook needs:")
        for name in DATA_FILES:
            log.info("  [%s] %s", "ok " if Path(name).exists() else "MISSING", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
