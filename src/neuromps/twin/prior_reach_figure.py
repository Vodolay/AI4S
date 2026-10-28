"""Figure of the prior-reach study (D38): SIMULATED pools against the REAL recordings."""

import numpy as np

from neuromps.twin import prior_reach as pr
from neuromps.twin.inference import SUMMARY_NAMES

# Okabe-Ito
COLOURS = {"hd14": "#E69F00", "new": "#0072B2", "real": "#D55E00", "gap": "#999999"}
Z_CLIP = (-5.0, 10.0)


def _feature_panel(ax, pools: dict) -> None:
    """Per summary, the HD14 pool, the new pool and the real recordings on HD14's scale."""
    offsets = {"hd14": -0.27, "new": 0.0, "real": 0.27}
    positions = np.arange(len(SUMMARY_NAMES))
    for key, offset in offsets.items():
        data = [np.clip(pools[key][:, j], *Z_CLIP) for j in range(len(SUMMARY_NAMES))]
        parts = ax.boxplot(
            data,
            positions=positions + offset,
            widths=0.22,
            showfliers=False,
            whis=(1, 99),
            patch_artist=True,
            medianprops={"color": "black", "lw": 0.8},
        )
        for box in parts["boxes"]:
            box.set(facecolor=COLOURS[key], alpha=0.8, lw=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(SUMMARY_NAMES, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("z on HD14's scale\n(whiskers 1-99%)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    handles = [ax.add_patch(_patch(COLOURS[k])) for k in ("hd14", "new", "real")]  # legend proxies
    ax.legend(
        handles,
        ["HD14 prior (simulated)", "widened prior (simulated)", "real recordings"],
        fontsize=6,
        frameon=False,
        loc="upper right",
        ncol=3,
    )


def _patch(colour: str):
    from matplotlib.patches import Rectangle

    return Rectangle((0, 0), 0, 0, facecolor=colour, alpha=0.8)


def _ecdf(ax, values: np.ndarray, colour: str, label: str, ls: str = "-") -> None:
    ordered = np.sort(values)
    ax.step(
        ordered,
        np.arange(1, len(ordered) + 1) / len(ordered),
        color=colour,
        lw=1.2,
        ls=ls,
        label=label,
    )


def _distance_panel(ax, pools: dict, n: int) -> None:
    """Distance from each real recording to its nearest simulation, both pools at equal size."""
    old = pr.equal_size(pools["hd14"], n)
    _ecdf(ax, pr.nearest_distance(pools["real"], old), COLOURS["hd14"], "real -> HD14 pool")
    _ecdf(
        ax, pr.nearest_distance(pools["real"], pools["new"]), COLOURS["new"], "real -> widened pool"
    )
    held = pools["hd14"][-pr.HD14_HELD :]
    gap = pr.nearest_distance(held, pools["hd14"][: -pr.HD14_HELD])
    _ecdf(ax, gap, COLOURS["gap"], "HD14 held-out -> HD14 pool", ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("distance to nearest simulation (HD14 scale)", fontsize=7)
    ax.set_ylabel("cumulative share", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False, loc="lower right")


def _share_panel(ax, payload: dict) -> None:
    """Which summaries carry the distance to the nearest simulation, before and after."""
    before = payload["diagnostic"]["distance_share_of_total"]
    after = payload["reach"]["remaining_distance_share_of_total"]
    top = sorted(SUMMARY_NAMES, key=lambda n: -max(before[n], after[n]))[:8][::-1]
    y = np.arange(len(top))
    ax.barh(y + 0.2, [before[n] for n in top], height=0.4, color=COLOURS["hd14"], label="HD14 pool")
    ax.barh(
        y - 0.2, [after[n] for n in top], height=0.4, color=COLOURS["new"], label="widened pool"
    )
    ax.set_yticks(y)
    ax.set_yticklabels(top, fontsize=6)
    ax.set_xlabel("share of squared real-to-nearest distance", fontsize=7)
    ax.tick_params(axis="x", labelsize=6)
    ax.legend(fontsize=6, frameon=False, loc="lower right")


def plot(reference: dict, new: dict, real_x: np.ndarray, payload: dict, path) -> None:
    """Three panels: summaries, nearest distances, and which summaries carry them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = pr.reference_scale(reference["x"])
    pools = {
        "hd14": pr.zscore(reference["x"], scale),
        "new": pr.zscore(new["x"], scale),
        "real": pr.zscore(real_x, scale),
    }
    fig = plt.figure(figsize=(7.6, 6.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0), hspace=0.75, wspace=0.45)
    _feature_panel(fig.add_subplot(grid[0, :]), pools)
    _distance_panel(fig.add_subplot(grid[1, 0]), pools, len(new["x"]))
    _share_panel(fig.add_subplot(grid[1, 1]), payload)
    r1 = payload["predictions"]["R1_reach"]
    fig.suptitle(
        "SIMULATED twin pools (HD14 prior and the widened D38 prior) against 113 REAL "
        "two-chamber recordings\n"
        f"median real-to-nearest distance {r1['hd14_pool_median']} (HD14) vs "
        f"{r1['new_pool_median']} (widened), {len(new['x']):,} simulations each; "
        f"R1 {r1['outcome']}",
        fontsize=7.5,
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
