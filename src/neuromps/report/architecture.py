"""Draw the system diagram, so it is generated rather than drawn by hand.

Every other figure in this report is emitted by a script; an architecture
diagram maintained in a drawing tool would be the one picture that could drift
from the code without anything noticing. The boxes name real modules and the
arrows name what actually passes between them.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("neuromps.report")

# (x, y, width, height, title, body). Three rows: ingest, use, publish.
BOXES = [
    (
        0.02,
        0.70,
        0.21,
        0.26,
        "fetch/",
        "eight public deposits\nlicense re-checked\nURL, SHA256, time",
    ),
    (
        0.28,
        0.70,
        0.21,
        0.26,
        "etl/ + qc/",
        "one record per well\nQC flags, never applied\nJSON Schema, level 2",
    ),
    (
        0.54,
        0.70,
        0.21,
        0.26,
        "features/mea.py",
        "the nine-feature vector\nthe only extractor\n(§3 measures it)",
    ),
    (
        0.80,
        0.70,
        0.18,
        0.26,
        "the data asset",
        "21,887 records\nexplicit data level\nprovenance enforced",
    ),
    (
        0.14,
        0.34,
        0.24,
        0.24,
        "models/",
        "mechanism, transfer\nconformal sets\ngrouped CV, cluster boot",
    ),
    (0.50, 0.34, 0.24, 0.24, "twin/", "Brian2 network\nvirtual electrodes\nsbi posterior"),
    (0.32, 0.02, 0.24, 0.22, "outputs/", "metrics.json, figures\nreport with verified\ncitations"),
]
SHADED = {"the data asset", "outputs/"}
# Arrows along the ingestion chain, and between the rows. All straight: a
# diagonal is unambiguous where a routed polyline would have to cross a box.
ARROWS = [
    ((0.23, 0.83), (0.28, 0.83), "spike files"),
    ((0.49, 0.83), (0.54, 0.83), "features"),
    ((0.75, 0.83), (0.80, 0.83), "records"),
    ((0.86, 0.70), (0.30, 0.58), "the asset"),
    ((0.89, 0.70), (0.63, 0.58), "calibration targets"),
    ((0.26, 0.34), (0.40, 0.24), ""),
    ((0.62, 0.34), (0.48, 0.24), "results"),
]


def draw(path: Path) -> Path:
    """Write the architecture figure, creating its directory if it is not there.

    This is the first thing the models stage draws, so on a fresh quick run
    nothing has made `outputs/quick/figures/` yet.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.10, 1)
    ax.axis("off")
    for x, y, w, h, title, body in BOXES:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                linewidth=1.1,
                edgecolor="#333333",
                facecolor="#DCEBF7" if title in SHADED else "#FFFFFF",
            )
        )
        ax.text(
            x + w / 2,
            y + h - 0.055,
            title,
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            family="monospace",
        )
        ax.text(x + w / 2, y + h / 2 - 0.045, body, ha="center", va="center", fontsize=7.2)
    for (x0, y0), (x1, y1), label in ARROWS:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                color="#333333",
                shrinkA=0,
                shrinkB=0,
            )
        )
        if label:
            horizontal = abs(y1 - y0) < 0.01
            # Diagonal labels sit near their destination rather than at the
            # midpoint, where the two cross-row arrows would collide.
            where = 0.5 if horizontal else 0.72
            ax.text(
                x0 + where * (x1 - x0),
                y0 + where * (y1 - y0) + (0.035 if horizontal else 0.022),
                label,
                ha="center",
                fontsize=6.3,
                color="#444444",
            )
    # The constraint the project is built around, drawn rather than only stated:
    # the twin scores itself through the extractor the real recordings use.
    ax.add_patch(
        FancyArrowPatch(
            (0.56, 0.58),
            (0.63, 0.70),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.3,
            color="#0072B2",
            linestyle="--",
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=-0.3",
        )
    )
    # The dashed arrow is explained in the report caption rather than
    # labelled here: any text beside it collides with a cross-row label.
    ax.text(
        0.50,
        -0.075,
        "Provenance is enforced at the asset boundary: the build refuses to emit an asset if any "
        "record's hash disagrees with the download manifest.",
        ha="center",
        fontsize=7.4,
        style="italic",
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI: redraw the architecture figure."""
    import argparse

    from neuromps.paths import figures_dir

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="accepted for pipeline symmetry")
    p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = draw(figures_dir() / "architecture.png")
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
