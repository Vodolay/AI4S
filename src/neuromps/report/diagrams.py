"""Draw the three model diagrams: the classifier, the twin, and the inversion.

The app, the video and the report all need to show what the models actually are.
Drawn once here and read from `outputs/metrics.json`, they cannot disagree with
each other or with the numbers beside them -- which is the failure a hand-drawn
diagram invites, and the reason the architecture figure is generated too.

Nothing here computes a result. Every number on these figures is looked up by
key, so a diagram built against metrics that lack the key fails loudly instead
of displaying a stale literal.

Usage:
    python -m neuromps.report.diagrams
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("neuromps.report")

# Wong's colourblind-safe palette, as the rest of the figures use.
INK = "#222222"
MUTED = "#555555"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
YELLOW = "#F0E442"
PANEL = "#FFFFFF"
SHADE = "#DCEBF7"
WARN = "#FBE3D6"

# figure -> (drawing function, the top-level metrics keys it needs).
#
# The requirement is declared rather than discovered, because the quick tier does
# not fetch `screenneuropharm_bic` and so never calibrates: `twin_hd12_calibration`
# is simply absent there, and the twin stage already falls back to the full run's
# calibration file for everything else. A diagram whose tier has no numbers is
# skipped out loud. A diagram whose numbers are present but malformed still
# raises - that distinction is the point, and collapsing it would turn a stale
# figure into a silent one.
PRINT_SCALE = 0.8

FIGURES = {
    "mechanism_classifier_real.png": ("draw_classifier", ("mechanism_baselines",)),
    "twin_architecture_simulated.png": (
        "draw_twin",
        ("twin_hd12_calibration", "twin_hd13_perturbation"),
    ),
    "inference_network_simulated.png": (
        "draw_inference",
        ("twin_hd14_inference", "twin_hd14b_recalibration"),
    ),
}

# The inversion panel's verdicts. Only D31's registered adoption rule
# (`twin_hd14b_recalibration.adopted_parameters`) earns the first.
QUOTED = "calibrated: quoted"
NOT_QUOTED = "not calibrated"


def metric(metrics: dict, path: str):
    """Value at a dotted path in metrics.json, raising if it is not there.

    The diagrams quote results. A missing key must stop the build, not fall back
    to a default that would be drawn as though it had been measured.
    """
    node = metrics
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def _box(ax, x, y, w, h, title, body="", fill=PANEL, mono=True, fontsize=8.2):
    """One labelled box. Returns its centre, so arrows can be aimed at it."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.014",
            linewidth=1.1,
            edgecolor=INK,
            facecolor=fill,
        )
    )
    top = y + h - 0.052 if body else y + h / 2
    ax.text(
        x + w / 2,
        top,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        family="monospace" if mono else "sans-serif",
        color=INK,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.045,
            body,
            ha="center",
            va="center",
            fontsize=fontsize - 1.3,
            color=INK,
        )
    return (x + w / 2, y + h / 2)


def _arrow(ax, start, end, label="", colour=INK, width=1.1, style="-|>", rad=0.0, dash=None):
    """An arrow between two points, optionally labelled at its midpoint."""
    from matplotlib.patches import FancyArrowPatch

    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=width,
        color=colour,
        shrinkA=2,
        shrinkB=2,
        connectionstyle=f"arc3,rad={rad}",
    )
    if dash:
        patch.set_linestyle(dash)
    ax.add_patch(patch)
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.028,
            label,
            ha="center",
            fontsize=6.6,
            color=MUTED,
        )


def _canvas(width, height, title, subtitle=""):
    """A blank figure with the title every project figure carries.

    Returns the axes' aspect ratio with it: the axes span 0-1 in both directions
    on a figure that is not square, so a circle of equal radii is drawn as an
    ellipse unless the y radius is stretched by this factor.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Drawn smaller than the nominal size so that text prints larger: the report
    # shows these about 5.3 in wide, and at the nominal 10 in their labels printed
    # at 3-4 pt. Layout is in axes fractions, so only the text-to-box ratio moves.
    fig, ax = plt.subplots(figsize=(width * PRINT_SCALE, height * PRINT_SCALE))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.975, title, ha="center", va="top", fontsize=11, fontweight="bold", color=INK)
    if subtitle:
        ax.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=8, color=MUTED)
    return fig, ax, width / height


def _footer(ax, text):
    """The caveat line, in the same place on every diagram."""
    ax.text(0.5, 0.022, text, ha="center", fontsize=7.0, style="italic", color=MUTED)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    log.info("wrote %s", path)
    return path


# --- 1. the mechanism classifier ----------------------------------------------

FEATURES = [
    "mfr_hz",
    "burst_rate_per_min",
    "burst_duration_s",
    "spikes_per_burst",
    "pct_spikes_in_bursts",
    "network_burst_rate_per_min",
    "synchrony_index",
    "isi_cv",
    "active_electrodes",
]


def _delta_vector(ax, x, y, w, h):
    """The nine-feature log-ratio vector, drawn as the column the model sees."""
    from matplotlib.patches import Rectangle

    cell = h / len(FEATURES)
    for index, name in enumerate(FEATURES):
        top = y + h - (index + 1) * cell
        ax.add_patch(Rectangle((x, top), w, cell, linewidth=0.7, edgecolor=INK, facecolor=PANEL))
        ax.text(x + 0.010, top + cell / 2, name, va="center", fontsize=5.5, family="monospace")
    ax.text(
        x + w / 2,
        y + h + 0.040,
        "Δ = log2(exposure / baseline)",
        ha="center",
        fontsize=7.2,
        fontweight="bold",
    )


def _grouped_folds(ax, y, n_pairs, n_groups, n_splits, aspect):
    """Why the scores look low: wells of one culture never straddle a split."""
    import numpy as np
    from matplotlib.patches import Rectangle

    ax.text(
        0.5,
        y + 0.145,
        f"Cross-validation groups by culture: {n_pairs} pairs from {n_groups} cultures, "
        f"{n_splits} folds",
        ha="center",
        fontsize=7.6,
        fontweight="bold",
    )
    width, gap = 0.165, 0.028
    left = (1 - (n_splits * width + (n_splits - 1) * gap)) / 2
    rng = np.random.default_rng(7)
    per_fold = n_groups // n_splits
    for fold in range(n_splits):
        x = left + fold * (width + gap)
        ax.add_patch(Rectangle((x, y), width, 0.085, linewidth=0.9, edgecolor=INK, facecolor=SHADE))
        ax.text(x + width / 2, y + 0.100, f"fold {fold + 1}", ha="center", fontsize=6.6)
        # One marker per culture, so "grouped by culture" is visible rather than
        # asserted: a culture is a unit, and it sits entirely in one fold.
        spots = rng.uniform([x + 0.012, y + 0.014], [x + width - 0.012, y + 0.071], (per_fold, 2))
        ax.scatter(*spots.T, s=9, color=BLUE, linewidths=0, zorder=3)
    ax.text(
        0.5,
        y - 0.045,
        "One dot is one culture. Every well of a culture sits in one fold; splitting them would "
        "let the model score by recognising the culture.",
        ha="center",
        fontsize=6.5,
        color=MUTED,
    )


def draw_classifier(metrics: dict, path: Path) -> Path:
    """How one mechanism call is made, end to end."""
    m = lambda key: metric(metrics, key)  # noqa: E731 - local shorthand
    task = "mechanism_baselines.task1_within_platform"
    classes = list(m(f"{task}.class_counts"))
    accuracy = m(f"{task}.models.logistic_regression.balanced_accuracy_mean")
    low, high = m(f"{task}.models.logistic_regression.balanced_accuracy_ci95")
    chance = m(f"{task}.chance_balanced_accuracy")

    fig, ax, aspect = _canvas(
        10.0, 4.8, "How a mechanism is called", "REAL recordings, Kapucu 2022"
    )
    _box(ax, 0.010, 0.558, 0.150, 0.215, "one culture", "baseline well\nexposed well", mono=False)
    _delta_vector(ax, 0.250, 0.515, 0.210, 0.300)
    _box(
        ax,
        0.520,
        0.580,
        0.215,
        0.170,
        "logistic regression",
        "L2 penalty\none linear boundary",
        fill=SHADE,
        mono=False,
        fontsize=7.8,
    )
    _box(
        ax,
        0.780,
        0.455,
        0.215,
        0.380,
        f"{len(classes)} classes",
        "\n".join(name.replace("_", " ") for name in classes),
        mono=False,
    )
    _arrow(ax, (0.165, 0.665), (0.245, 0.665))
    ax.text(0.205, 0.695, "paired,\nper well", ha="center", fontsize=6.2, color=MUTED)
    _arrow(ax, (0.465, 0.665), (0.515, 0.665))
    _arrow(ax, (0.740, 0.665), (0.775, 0.665))
    _grouped_folds(
        ax, 0.235, m(f"{task}.n_pairs"), m(f"{task}.n_groups"), m(f"{task}.n_splits"), aspect
    )
    _footer(
        ax,
        f"Balanced accuracy {accuracy} on [{low}, {high}], against a chance of {chance}. "
        "The interval is wide because the study is small, and it is reported that way.",
    )
    return _save(fig, path)


# --- 2. the twin ---------------------------------------------------------------


def _compartment(ax, x, y, w, h, label, n_neurons, seed):
    """One chamber: a readable sample of cells, 80 per cent excitatory."""
    import numpy as np
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((x, y), w, h, linewidth=1.2, edgecolor=INK, facecolor=PANEL, zorder=1))
    rng = np.random.default_rng(seed)
    # A sample of the population, not the population: 967 dots is mud.
    drawn = 80
    points = rng.uniform([x + 0.014, y + 0.070], [x + w - 0.014, y + h - 0.075], size=(drawn, 2))
    inhibitory = rng.random(drawn) < 0.2
    ax.scatter(*points[~inhibitory].T, s=8, color=BLUE, zorder=2, linewidths=0)
    ax.scatter(*points[inhibitory].T, s=10, color=ORANGE, marker="s", zorder=2, linewidths=0)
    ax.text(x + w / 2, y + h - 0.030, label, ha="center", fontsize=8.2, fontweight="bold", zorder=3)
    ax.text(
        x + w / 2,
        y + 0.026,
        f"{n_neurons:,} neurons",
        ha="center",
        fontsize=6.8,
        color=MUTED,
        zorder=3,
    )


def _electrodes(ax, x, y, w, aspect):
    """The virtual electrodes and their listening catchments, captioned once below."""
    from matplotlib.patches import Ellipse

    reach = 0.024
    for index in range(4):
        cx = x + w * (index + 0.5) / 4
        ax.add_patch(Ellipse((cx, y), 0.010, 0.010 * aspect, facecolor=INK, zorder=3))
        ax.add_patch(
            Ellipse(
                (cx, y),
                2 * reach,
                2 * reach * aspect,
                facecolor="none",
                edgecolor=GREEN,
                linewidth=0.9,
                linestyle=":",
            )
        )


def _excitation_legend(ax, x, y):
    """Which marker is which, rather than leaving the colours to be guessed."""
    ax.scatter([x], [y], s=8, color=BLUE, linewidths=0)
    ax.text(x + 0.016, y, "excitatory (80%)", va="center", fontsize=6.4, color=MUTED)
    ax.scatter([x + 0.175], [y], s=10, color=ORANGE, marker="s", linewidths=0)
    ax.text(x + 0.191, y, "inhibitory (20%)", va="center", fontsize=6.4, color=MUTED)


def draw_twin(metrics: dict, path: Path) -> Path:
    """What the digital twin is: two chambers, a directional pathway, one extractor."""
    m = lambda key: metric(metrics, key)  # noqa: E731 - local shorthand
    draw = "twin_hd12_calibration.best.draw"
    neurons = m(f"{draw}.n_per_compartment")
    in_degree = m(f"{draw}.intra_in_degree")
    radius = m(f"{draw}.detection_radius_um")
    # The layout's own pitch, not twice the radius: the radius is capped at half
    # the pitch and sits below the cap, so deriving one from the other would
    # print a pitch the simulator never used.
    pitch = round(m("twin_hd13_perturbation.simulation.layout.pitch_um"))

    fig, ax, aspect = _canvas(10.0, 5.0, "The two-compartment digital twin", "SIMULATED")
    _excitation_legend(ax, 0.340, 0.145)
    for left, index in ((0.050, 0), (0.465, 1)):
        _compartment(ax, left, 0.45, 0.28, 0.30, f"compartment {index}", neurons, seed=index)
        _electrodes(ax, left, 0.390, 0.28, aspect)
        _arrow(ax, (left + 0.065, 0.775), (left + 0.215, 0.775), colour=BLUE, rad=-0.75, width=1.0)
        # Beside the arc rather than under it, where the arc ran through the words.
        ax.text(
            left + 0.225,
            0.800,
            f"recurrent\nin-degree {in_degree}",
            ha="left",
            va="center",
            fontsize=6.6,
            color=BLUE,
        )
    _arrow(ax, (0.335, 0.635), (0.460, 0.635), colour=ORANGE, width=2.6)
    _arrow(ax, (0.460, 0.565), (0.335, 0.565), colour=MUTED, width=0.8)
    # Not "axon diode": the backward pathway is never closed. The two differ by a
    # ratio the inversion samples, and at ratio 1 they are equal.
    ax.text(
        0.3975,
        0.665,
        "directional\npathway",
        ha="center",
        va="bottom",
        fontsize=7.4,
        fontweight="bold",
        linespacing=1.1,
    )
    ax.text(0.3975, 0.600, "forward", ha="center", fontsize=5.8, color=ORANGE)
    ax.text(0.3975, 0.530, "back", ha="center", fontsize=5.8, color=MUTED)
    ax.text(0.3975, 0.475, "ratio adjustable", ha="center", fontsize=5.8, color=MUTED)
    _box(ax, 0.795, 0.520, 0.190, 0.190, "features/mea.py", "the nine features", fill=SHADE)
    _arrow(ax, (0.745, 0.615), (0.790, 0.615))
    ax.text(
        0.5,
        0.295,
        f"Dotted rings are the electrodes' {radius} µm listening catchments, capped at half "
        f"the {pitch} µm pitch so that no two can hear the same neuron.",
        ha="center",
        fontsize=6.6,
        color=MUTED,
    )
    ax.text(
        0.5,
        0.230,
        "The ratio between the two pathway strengths, and the cross-chamber in-degree, are not "
        "set: they are what the inversion recovers.",
        ha="center",
        fontsize=6.8,
        color=INK,
    )
    _footer(
        ax,
        "The twin is scored through the same extractor as the real recordings, so it cannot look "
        "good by being measured differently.",
    )
    return _save(fig, path)


# --- 3. the inversion ----------------------------------------------------------


def _prior_block(ax, x, y, w, h, bounds):
    """The parameters the inversion is over, with the ranges drawn from."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.014",
            linewidth=1.1,
            edgecolor=INK,
            facecolor=PANEL,
        )
    )
    ax.text(x + w / 2, y + h - 0.035, "prior", ha="center", fontsize=8.2, fontweight="bold")
    for index, (name, (low, high)) in enumerate(sorted(bounds.items())):
        row = y + h - 0.080 - index * 0.032
        ax.text(x + 0.010, row, name, fontsize=5.5, family="monospace", va="center")
        ax.text(
            x + w - 0.010,
            row,
            f"{low} – {high}",
            fontsize=5.5,
            ha="right",
            va="center",
            color=MUTED,
        )


def _uniformity_panel(ax, y, recalibration):
    """The honesty gate: which posteriors may be read as credible intervals.

    Round 1 judged this on its own 300 held-out runs. D31 re-ran the check on
    fresh simulations, so each box shows the round-1 model and the recalibrated
    ensemble on the same runs, and only the registered rule decides the verdict.
    """
    from matplotlib.patches import Rectangle

    before, after = (recalibration["results"][model]["H1000"] for model in ("M0", "M2"))
    adopted = set(recalibration["adopted_parameters"])
    ax.text(
        0.5,
        y + 0.140,
        f"Before any interval is quoted, on {before['n']:,} fresh simulations: "
        "round-1 model → recalibrated ensemble",
        ha="center",
        fontsize=7.4,
        fontweight="bold",
    )
    names = sorted(before["ks"])
    width, gap = 0.172, 0.010  # wide enough for "contraction a → b" at this size
    left = (1 - (len(names) * width + (len(names) - 1) * gap)) / 2
    for index, name in enumerate(names):
        passes = name in adopted
        x = left + index * (width + gap)
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                0.110,
                linewidth=0.9,
                edgecolor=INK,
                facecolor=SHADE if passes else WARN,
            )
        )
        ax.text(x + width / 2, y + 0.084, name, ha="center", fontsize=5.5, family="monospace")
        ax.text(
            x + width / 2,
            y + 0.059,
            f"contraction {before['contraction'][name]['contraction']} → "
            f"{after['contraction'][name]['contraction']}",
            ha="center",
            fontsize=5.4,
            color=MUTED,
        )
        ax.text(
            x + width / 2,
            y + 0.035,
            f"KS p {before['ks'][name]['ks_p']} → {after['ks'][name]['ks_p']}",
            ha="center",
            fontsize=5.4,
            color=MUTED,
        )
        ax.text(
            x + width / 2,
            y + 0.011,
            QUOTED if passes else NOT_QUOTED,
            ha="center",
            fontsize=5.4,
            fontweight="normal" if passes else "bold",
            color=INK if passes else ORANGE,
        )


def inference_caption(metrics: dict) -> str:
    """The inversion diagram's caveat: D31 on fresh simulations, not round 1's 300 runs."""
    recalibration = metric(metrics, "twin_hd14b_recalibration")
    fresh = recalibration["results"]["M0"]["H1000"]
    failing = [
        name for name, value in fresh["ks"].items() if value["ks_p"] < recalibration["alpha"]
    ]
    adopted = recalibration["adopted_parameters"]
    if not adopted:
        verdict = "no parameter passes every registered check, so no interval is quoted."
    elif len(adopted) == 1:
        verdict = (
            f"only {adopted[0]} passes every registered check, so only its interval is quoted."
        )
    else:
        verdict = (
            f"only {', '.join(adopted)} pass every registered check, so only their intervals "
            "are quoted."
        )
    return (
        f"On {fresh['n']:,} fresh simulations the round-1 model fails rank uniformity for "
        f"{len(failing)} of {len(fresh['ks'])} parameters.\nAfter the registered recalibration "
        + verdict
    )


def draw_inference(metrics: dict, path: Path) -> Path:
    """Running the twin backwards: parameters recovered from features."""
    m = lambda key: metric(metrics, key)  # noqa: E731 - local shorthand
    hd14 = "twin_hd14_inference"
    bounds = m(f"{hd14}.prior_bounds")
    summaries = m(f"{hd14}.summary_names")
    simulations = m(f"{hd14}.n_simulations")
    silent = m(f"{hd14}.fraction_silent_simulations")
    flow = m(f"{hd14}.density_estimator")

    fig, ax, _ = _canvas(10.0, 5.6, "Running the twin backwards", "SIMULATED")
    _prior_block(ax, 0.010, 0.600, 0.235, 0.230, bounds)
    _box(
        ax,
        0.325,
        0.640,
        0.110,
        0.165,
        "the twin",
        f"{simulations:,} draws\n{silent:.0%} silent",
        fill=SHADE,
    )
    _box(ax, 0.520, 0.640, 0.110, 0.165, "summaries", f"{len(summaries)} features\nper draw")
    _box(
        ax,
        0.700,
        0.475,
        0.290,
        0.315,
        "neural posterior estimator",
        f"a {flow['family']}:\n{flow['n_autoregressive_transforms']} transforms, "
        f"{flow['hidden_units_per_layer']} hidden units,\n{flow['n_parameters']:,} weights, "
        "trained on\n(parameters, features) pairs\nsimulated by the twin",
        fill=SHADE,
        mono=False,
        fontsize=7.8,
    )
    ax.text(0.010, 0.868, "TRAINED ONCE", fontsize=6.6, fontweight="bold", color=MUTED)
    _arrow(ax, (0.250, 0.7225), (0.320, 0.7225), "sample")
    _arrow(ax, (0.440, 0.7225), (0.515, 0.7225), "simulate")
    _arrow(ax, (0.635, 0.7225), (0.695, 0.7225), "train")
    ax.text(0.010, 0.540, "THEN USED", fontsize=6.6, fontweight="bold", color=BLUE)
    _box(
        ax, 0.465, 0.495, 0.165, 0.115, "a real recording", "its features", mono=False, fontsize=7.4
    )
    _arrow(ax, (0.635, 0.5525), (0.695, 0.5525), "ask", colour=BLUE, width=1.3)
    _box(
        ax,
        0.700,
        0.315,
        0.290,
        0.135,
        "posterior",
        f"{len(bounds)} parameters, with intervals",
        mono=False,
        fontsize=7.8,
    )
    _arrow(ax, (0.845, 0.472), (0.845, 0.448), colour=BLUE, width=1.3)
    _uniformity_panel(ax, 0.120, m("twin_hd14b_recalibration"))
    _footer(ax, inference_caption(metrics))
    return _save(fig, path)


def main(argv: list[str] | None = None) -> int:
    """CLI: redraw the three model diagrams from the current metrics."""
    import argparse
    import json

    from neuromps.paths import figures_dir, metrics_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="accepted for pipeline symmetry")
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = json.loads(metrics_path().read_text())
    for name, (function, required) in FIGURES.items():
        missing = [key for key in required if key not in metrics]
        if missing:
            log.warning("skipping %s: this run's metrics have no %s", name, ", ".join(missing))
            continue
        globals()[function](metrics, figures_dir() / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
