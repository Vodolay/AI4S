"""Figures of the chip-to-animal study. Every title says the data are simulated."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Okabe-Ito, colourblind-safe
COLOURS = {
    "gaba_block": "#D55E00",
    "glucl_block": "#E69F00",
    "gaba_potentiate": "#0072B2",
    "ach_block": "#009E73",
    "excitability_block": "#CC79A7",
}
LABELS = {
    "gaba_block": "GABA block",
    "glucl_block": "GluCl block",
    "gaba_potentiate": "GABA potentiation",
    "ach_block": "ACh block",
    "excitability_block": "threshold raised",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def dose_panels(frame, path: Path) -> None:
    """Chip readouts and behaviour readouts against dose, one line per mechanism."""
    plt = _plt()
    columns = [
        ("delta.mfr_hz", "chip: log2 change in firing rate"),
        ("delta.synchrony_index", "chip: log2 change in synchrony"),
        ("mn9_sugar", "MN9 | sugar (Hz)"),
        ("mn9_sugar_bitter", "MN9 | sugar + bitter (Hz)"),
        ("abn1_jon_ce", "aBN1 | JO-CE (Hz)"),
        ("disruption", "behavioural disruption index"),
    ]
    vehicle = frame[frame["family"] == "vehicle"].iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))
    for ax, (column, label) in zip(axes.flat, columns, strict=False):
        for family, colour in COLOURS.items():
            rows = frame[frame["family"] == family].sort_values("dose")
            if rows.empty:
                continue
            x = np.r_[0.0, rows["dose"]]
            y = np.r_[
                vehicle[column] if column.startswith("delta") is False else 0.0,
                rows[column].astype(float),
            ]
            ax.plot(x, y, marker="o", ms=2.5, lw=1.2, color=colour, label=LABELS[family])
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("knob change from untreated", fontsize=6)
    axes.flat[0].legend(fontsize=5.5, frameon=False)
    fig.suptitle(
        "SIMULATED fly brain (FlyWire 783): chip readout vs behavioural readouts", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def direction_panel(direction: dict, path: Path) -> None:
    """True region-pair asymmetry against the two measurements, whole-region signals."""
    plt = _plt()
    points = direction["whole_region"]["points"]
    truth = np.asarray(points["truth"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), sharex=True)
    for ax, key, title in (
        (axes[0], "spontaneous", "spontaneous cross-correlation"),
        (axes[1], "evoked", "pulsed stimulation, 5-30 ms"),
    ):
        y = np.asarray([np.nan if v is None else v for v in points[key]], dtype=float)
        ax.scatter(truth, y, s=3, alpha=0.5, color="#0072B2", linewidths=0)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        s = direction["whole_region"][key]
        ax.set_title(
            f"{title}\nsign agreement (|asym|>=0.5): {s['sign_agreement_strong']}", fontsize=7
        )
        ax.set_xlabel("connectome asymmetry (synapses)", fontsize=6.5)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("measured asymmetry", fontsize=6.5)
    fig.suptitle(
        "SIMULATED fly brain: can activity recover wiring direction between regions?", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def all_figures(tables: dict, result: dict, out_dir: Path) -> None:
    """Write every figure of the study."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dose_panels(tables["conditions"], out_dir / "fly_chip_to_animal_dose_SIMULATED.png")
    direction_panel(result["direction"], out_dir / "fly_direction_SIMULATED.png")


def sentinel_ladder(panels: list[tuple[str, dict]], path: Path) -> None:
    """Per mechanism: first dose that impairs behaviour, and first each chip feature detects.

    `panels` is a list of (title, verdicts), verdicts as written by `fly.confirm.verdicts`.
    A missing marker means the readout never reached its threshold.
    """
    plt = _plt()
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.2 * len(panels), 2.6), sharey=True, sharex=True, squeeze=False
    )
    families = list(COLOURS)
    marks = (
        ("behaviour", "#000000", "s", "behaviour impaired"),
        ("network_bursts", "#009E73", "o", "chip: network-burst rate"),
        ("firing_rate", "#D55E00", "D", "chip: firing rate"),
    )
    for ax, (title, v) in zip(axes[0], panels, strict=True):
        for row, family in enumerate(families):
            entry = v["network_bursts"].get(family)
            if entry is None:
                continue
            doses = {
                "behaviour": entry["first_behaviour_dose"],
                "network_bursts": entry["first_chip_dose"],
                "firing_rate": v["firing_rate"][family]["first_chip_dose"],
            }
            for k, (key, colour, marker, label) in enumerate(marks):
                if doses[key] is not None:
                    ax.scatter(
                        doses[key],
                        row + (k - 1) * 0.18,
                        color=colour,
                        marker=marker,
                        s=22,
                        label=label if row == 0 else None,
                        zorder=3,
                    )
        ax.set_yticks(range(len(families)), [LABELS[f] for f in families], fontsize=6.5)
        ax.set_xlabel("knob change from untreated (first dose reaching threshold)", fontsize=6)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.invert_yaxis()
    axes[0][0].legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.suptitle("SIMULATED fly brain: which readout raises the alarm first?", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
