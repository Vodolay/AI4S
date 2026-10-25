"""The drug onsets of stages 4a-4e in one figure: each readout's first change, and the chip's.

``python -m neuromps.fly.onset_figure`` reads recorded outputs only (no simulation): the
movement medians of stages 4b and 4c on their combined dose series, the walking trials of
stage 4e (fraction of rhythmic draws), and the chip's two runs. Per drug family it finds the
lowest dose at which each readout's |log2| change reaches 0.5, and whether the readout fell
or rose there. Descriptive: every onset shown was scored in a registered stage. Everything
here is **simulated**.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from neuromps.fly import embodied_stage4a as s4a
from neuromps.fly import embodied_stage4d as s4d
from neuromps.fly import embodied_stage4e as s4e
from neuromps.fly import scoring
from neuromps.metrics import update_metrics
from neuromps.paths import figures_dir, outputs_dir

log = logging.getLogger("neuromps.fly")
READOUTS = {
    "proboscis_sugar": "proboscis, sugar",
    "proboscis_sugar_bitter": "proboscis, sugar + bitter",
    "reach_depth": "grooming reach",
    "walking": "walking rhythm (fraction of draws)",
    "index": "three-movement index",
}
# Okabe-Ito order; each readout also has its own marker so identity is never colour alone.
STYLE = {
    "proboscis_sugar": ("#E69F00", "o"),
    "proboscis_sugar_bitter": ("#56B4E9", "s"),
    "reach_depth": ("#009E73", "^"),
    "walking": ("#CC79A7", "D"),
    "index": ("#0072B2", "P"),
}
FAMILY_LABELS = {
    "gaba_block": "GABA-A block",
    "gaba_potentiate": "GABA-A potentiation",
    "glucl_block": "GluCl block",
    "ach_block": "ACh block",
    "excitability_block": "excitability block",
}


def onset_and_direction(conds: pd.DataFrame, values: pd.Series) -> dict:
    """Per family: (lowest dose with |log2((x + 0.01)/(x_vehicle + 0.01))| >= 0.5, 'down'/'up')."""
    change = np.log2((values + s4a.EPS_MM) / (values["vehicle"] + s4a.EPS_MM))
    out = {}
    for family in s4d.FAMILIES:
        rows = conds[conds["family"] == family].sort_values("dose")
        hit = [
            (float(d), "down" if change[c] < 0 else "up")
            for c, d in zip(rows["condition"], rows["dose"], strict=True)
            if abs(change[c]) >= s4a.IMPAIR
        ]
        out[family] = hit[0] if hit else (None, None)
    return out


def index_onsets(conds: pd.DataFrame, medians: pd.DataFrame) -> dict:
    """Per family: the three-movement index onset (no single direction)."""
    index = s4a.disruption(medians, s4a.EPS_MM).loc[conds["condition"]].to_numpy()
    got = s4a.onset(conds.assign(md=index), "md", s4a.IMPAIR)
    return {f: (got.get(f), None) for f in s4d.FAMILIES}


def _chip_first_run(conds: pd.DataFrame) -> dict:
    coarse = pd.read_csv(outputs_dir() / "fly" / "conditions_SIMULATED.csv")
    levels = json.loads((outputs_dir() / "fly" / "embodied_stage4c_SIMULATED.json").read_text())
    known = set(coarse["condition"])
    fine = conds.set_index("condition")
    rows = [
        {
            "condition": c,
            "family": fine.loc[c, "family"],
            "dose": fine.loc[c, "dose"],
            "abs_delta_mfr": v["chip_first_run"],
        }
        for c, v in levels["levels"].items()
        if "chip_first_run" in v and c not in known
    ]
    keep = ["condition", "family", "dose", "abs_delta_mfr"]
    table = pd.concat([coarse[keep], pd.DataFrame(rows)], ignore_index=True)
    return s4a.onset(table, "abs_delta_mfr", s4a.CHIP)


def collect() -> dict:
    """Every onset the figure shows, from recorded outputs."""
    conds = s4d.conditions()
    medians = s4d._movement_medians()
    walking = s4e.fractions(pd.read_csv(outputs_dir() / "fly" / "embodied_stage4e_SIMULATED.csv"))
    readouts = {m: onset_and_direction(conds, medians[m]) for m in s4a.MOVEMENTS}
    readouts["walking"] = onset_and_direction(conds, walking)
    readouts["index"] = index_onsets(conds, medians)
    chip = s4a.onset(s4d._chip_series(), "abs_delta_mfr", s4a.CHIP)
    return {
        "readouts": readouts,
        "chip_replication": chip,
        "chip_first_run": _chip_first_run(conds),
    }


def plot(got: dict, path) -> None:
    """One row per family: readout onsets (filled: fell; open: rose) against the chip's."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    never = 1.12
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    families = list(FAMILY_LABELS)
    offsets = np.linspace(-0.28, 0.28, len(READOUTS))
    for r, family in enumerate(families):
        y = len(families) - 1 - r
        ax.axhspan(y - 0.42, y + 0.42, color="#F2F2F2" if r % 2 else "white", zorder=0, lw=0)
        chip = got["chip_replication"][family]
        if chip is not None:
            ax.plot([chip, chip], [y - 0.4, y + 0.4], color="black", lw=2.0, zorder=3)
        first = got["chip_first_run"][family]
        if first is not None:
            ax.plot([first, first], [y - 0.4, y + 0.4], color="#7F7F7F", lw=1.2, ls="--", zorder=2)
        for k, (key, label) in enumerate(READOUTS.items()):
            dose, direction = got["readouts"][key][family]
            colour, marker = STYLE[key]
            x = never if dose is None else dose
            ax.scatter(
                x,
                y + offsets[k],
                s=42,
                marker=marker,
                facecolor="#D9D9D9" if dose is None else "white" if direction == "up" else colour,
                edgecolor=colour,
                linewidth=1.4,
                zorder=4,
                label=label if r == 0 else None,
            )
    ax.axvline(1.06, color="#BFBFBF", lw=0.8)
    ax.set_yticks(range(len(families)), [FAMILY_LABELS[f] for f in reversed(families)], fontsize=7)
    ax.set_xticks(
        [0, 0.2, 0.4, 0.6, 0.8, 1.0, never], ["0", "0.2", "0.4", "0.6", "0.8", "1.0", "never"]
    )
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(-0.02, never + 0.06)
    ax.set_xlabel(
        "dose at which the readout first changes (fraction of the knob's range)", fontsize=7
    )
    ax.plot([], [], color="black", lw=2.0, label="chip flags (replication)")
    ax.plot([], [], color="#7F7F7F", lw=1.2, ls="--", label="chip flags (first run)")
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title(
        "SIMULATED fly: when each behaviour first changes under a drug knob, and when the chip"
        " flags it\n(filled: the readout fell; open: it rose; grey fill: no change at any dose;"
        " left of the black bar: behaviour changed first)",
        fontsize=7.5,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI: collect the onsets, write them to metrics and draw the figure."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    got = collect()
    record = {
        "label": "SIMULATED drug onsets per readout (descriptive, from stages 4b-4e)",
        "onsets": {
            key: {f: (None if d is None else round(d, 4)) for f, (d, _) in v.items()}
            for key, v in got["readouts"].items()
        },
        "directions": {
            key: {f: d for f, (_, d) in v.items()} for key, v in got["readouts"].items()
        },
        "chip_replication": got["chip_replication"],
        "chip_first_run": got["chip_first_run"],
    }
    update_metrics("embodied_onsets", scoring.clean(record))
    plot(got, figures_dir() / "embodied_onsets_SIMULATED.png")
    print(json.dumps(scoring.clean(record), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
