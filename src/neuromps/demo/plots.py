"""Figures the demo app draws from precomputed data.

These return a matplotlib Figure rather than writing a file, because the app
renders them inline; the report's figures are written by the plotting functions
that already live beside each analysis. Every title says whether what is shown
is real or simulated, the same rule the report follows.
"""

from __future__ import annotations

import numpy as np

COLOURS = {"c0": "#0072B2", "c1": "#D55E00", "real": "#009E73", "simulated": "#CC79A7"}


def _figure(width: float, height: float):
    """A figure created without touching the global pyplot state machine."""
    from matplotlib.figure import Figure

    return Figure(figsize=(width, height), dpi=110)


def raster(snapshot: dict, seconds: float, title: str):
    """Spike raster for both compartments of one SIMULATED recording."""
    figure = _figure(7.0, 3.2)
    axes = figure.subplots(2, 1, sharex=True)
    for index, (compartment, label) in enumerate(
        (("c0", "compartment 0"), ("c1", "compartment 1"))
    ):
        times = np.asarray(snapshot[f"{compartment}_times"])
        neurons = np.asarray(snapshot[f"{compartment}_neurons"])
        axes[index].scatter(times, neurons, s=1.4, color=COLOURS[compartment], linewidths=0)
        axes[index].set_ylabel(label, fontsize=8)
        axes[index].tick_params(labelsize=8)
        if times.size == 0:
            axes[index].text(
                0.5, 0.5, "silent", transform=axes[index].transAxes, ha="center", fontsize=9
            )
    axes[1].set_xlabel("time (s)", fontsize=8)
    axes[0].set_xlim(0, seconds)
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    return figure


def features_against_real(simulated: dict, real: dict, title: str):
    """Simulated features against the distribution of the REAL baseline recordings."""
    names = [n for n in real if n in simulated and simulated[n] is not None]
    figure = _figure(7.0, 0.55 * len(names) + 1.4)
    axes = figure.subplots(len(names), 1, sharex=False) if len(names) > 1 else [figure.subplots()]
    for axis, name in zip(axes, names, strict=True):
        values = np.asarray(real[name], dtype=float)
        values = values[np.isfinite(values)]
        axis.scatter(
            values,
            np.random.default_rng(0).normal(0, 0.035, values.size),
            s=7,
            alpha=0.45,
            color=COLOURS["real"],
            linewidths=0,
            label="real recordings",
        )
        axis.axvline(simulated[name], color=COLOURS["simulated"], lw=2, label="simulated")
        axis.set_yticks([])
        axis.set_ylabel(name, rotation=0, ha="right", fontsize=8)
        axis.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, loc="upper right")
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    return figure


def posterior_marginals(samples: np.ndarray, names: list[str], truth, title: str):
    """Posterior marginals, with the value that generated the run marked."""
    figure = _figure(7.5, 2.4)
    axes = figure.subplots(1, len(names))
    for index, (axis, name) in enumerate(zip(np.atleast_1d(axes), names, strict=True)):
        axis.hist(samples[:, index], bins=22, color=COLOURS["simulated"], alpha=0.85)
        if truth is not None:
            axis.axvline(truth[index], color="black", ls="--", lw=1.2)
        axis.set_title(name, fontsize=7.5)
        axis.tick_params(labelsize=7)
        axis.set_yticks([])
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    return figure


def drug_knob_posterior(samples: np.ndarray, name: str, expected: str, title: str):
    """Posterior over one drug knob for REAL recordings, with the expected direction."""
    figure = _figure(6.4, 2.6)
    axis = figure.subplots()
    axis.hist(samples, bins=26, color=COLOURS["real"], alpha=0.85)
    axis.axvline(1.0, color="black", ls="--", lw=1.2, label="untreated (scale = 1)")
    axis.set_xlabel(f"inferred {name}", fontsize=9)
    axis.set_ylabel("posterior samples", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.legend(fontsize=8)
    figure.suptitle(f"{title}\nexpected direction: {expected}", fontsize=9)
    figure.tight_layout()
    return figure


def transfer_bars(rows: list[dict], chance: float, title: str):
    """Balanced accuracy with 95% intervals for the REAL rat->human transfer."""
    figure = _figure(6.4, 3.0)
    axis = figure.subplots()
    labels = [row["label"] for row in rows]
    values = [row["value"] for row in rows]
    lower = [row["value"] - row["ci95"][0] for row in rows]
    upper = [row["ci95"][1] - row["value"] for row in rows]
    positions = np.arange(len(rows))
    axis.bar(positions, values, color=COLOURS["real"], alpha=0.85)
    axis.errorbar(positions, values, yerr=[lower, upper], fmt="none", ecolor="black", capsize=4)
    axis.axhline(chance, color="black", ls=":", lw=1, label=f"chance = {chance}")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8, rotation=12, ha="right")
    axis.set_ylabel("balanced accuracy", fontsize=9)
    axis.set_ylim(0, 1)
    axis.tick_params(labelsize=8)
    axis.legend(fontsize=8)
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    return figure
