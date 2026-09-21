"""HD11 validation: does the simulated network behave like the real cultures?

Two checks, both written to ``outputs/twin/`` and labelled **simulated**:

1. Spontaneous activity: simulated features against the range the real
   recordings show, via the same ``features/mea.py`` the real data uses.
2. Directionality: a sweep over ``forward_ratio`` and over the balance of drive
   between compartments, which is how the HD11 caveat was found - coupling
   asymmetry follows the compartments' relative firing rate, not only the
   pathway.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from neuromps.features.mea import extract_features
from neuromps.metrics import update_metrics
from neuromps.twin.mea import electrode_coupling
from neuromps.twin.network import NetworkParams, simulate

log = logging.getLogger("neuromps.twin")
REPO_ROOT = Path(__file__).resolve().parents[3]
TWIN_DIR = REPO_ROOT / "outputs" / "twin"
ELECTRODES_PER_COMPARTMENT = 16

# Ranges the real recordings occupy, for orientation (not fitted here; HD12 calibrates).
REAL_RANGES = {
    "mfr_hz": (1.0, 6.5),
    "burst_rate_per_min": (1.5, 15.0),
    "network_burst_rate_per_min": (0.0, 25.0),
    "synchrony_index": (0.04, 0.8),
}


NEURONS_PER_ELECTRODE = 8


def electrode_spikes(
    result,
    per_compartment: int = ELECTRODES_PER_COMPARTMENT,
    neurons_per_electrode: int = NEURONS_PER_ELECTRODE,
    seed: int = 0,
) -> dict:
    """Assign neurons to virtual electrodes within their compartment.

    Each electrode listens to a fixed number of neurons, as a real electrode
    hears only the cells near it. Pooling every neuron into a fixed electrode
    count instead would make the per-electrode firing rate grow with network
    size, which is an artefact of the mapping rather than the network.

    A placeholder until the MEA simulator (HD12) places electrodes spatially.
    """
    rng = np.random.default_rng(seed)
    spikes = {}
    for c in (0, 1):
        in_compartment = np.flatnonzero(result.compartment_of_neuron == c)
        for electrode in range(per_compartment):
            listened = rng.choice(
                in_compartment,
                size=min(neurons_per_electrode, in_compartment.size),
                replace=False,
            )
            selected = np.isin(result.spike_neurons, listened)
            spikes[f"c{c}_e{electrode}"] = np.sort(result.spike_times[selected])
    return spikes


def simulated_features(params: NetworkParams) -> dict:
    """Features of one simulated recording, from the shared extractor."""
    result = simulate(params)
    heard = electrode_spikes(result)
    features = extract_features(heard, result.duration_s)
    coupling = electrode_coupling(heard, result.duration_s)
    return {
        "params": params.as_record(),
        "n_spikes": int(result.spike_times.size),
        "features_simulated": features,
        "coupling_simulated": coupling,
    }


def in_real_range(features: dict) -> dict:
    """Which simulated features land inside the range the real recordings span."""
    verdict = {}
    for name, (low, high) in REAL_RANGES.items():
        value = features.get(name)
        verdict[name] = None if value is None else bool(low <= value <= high)
    return verdict


def directionality_sweep(
    ratios: tuple[float, ...], drive_scales: tuple[float, ...], seeds: tuple[int, ...], duration_s
) -> list[dict]:
    """Coupling asymmetry across pathway ratios and drive balances."""
    rows = []
    for ratio in ratios:
        for scale in drive_scales:
            values = []
            for seed in seeds:
                params = NetworkParams(
                    n_per_compartment=150,
                    duration_s=duration_s,
                    seed=seed,
                    inter_in_degree=9.0,
                    forward_ratio=ratio,
                    background_scale_target=scale,
                )
                result = simulate(params)
                out = electrode_coupling(electrode_spikes(result), result.duration_s)
                spikes = result.spikes_by_compartment()
                rates = {
                    c: len(t) / params.n_per_compartment / duration_s for c, t in spikes.items()
                }
                values.append({**out, "rate_c0": rates[0], "rate_c1": rates[1]})
            rows.append(
                {
                    "forward_ratio": ratio,
                    "background_scale_target": scale,
                    "asymmetry_mean": _mean(values, "asymmetry"),
                    "lag_ms_mean": _mean(values, "lag_ms"),
                    "peak_correlation_mean": _mean(values, "peak_correlation"),
                    "rate_ratio_c1_over_c0": round(
                        float(
                            np.mean([v["rate_c1"] for v in values])
                            / max(np.mean([v["rate_c0"] for v in values]), 1e-9)
                        ),
                        3,
                    ),
                    "n_seeds": len(seeds),
                }
            )
    return rows


def _mean(values: list[dict], key: str) -> float | None:
    """Mean of a key, ignoring the runs where the measure was undefined."""
    present = [v[key] for v in values if v.get(key) is not None]
    return round(float(np.mean(present)), 3) if present else None


def plot_sweep(rows: list[dict], path, title: str) -> None:
    """Asymmetry against drive balance, one line per pathway ratio (simulated)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    ratios = sorted({row["forward_ratio"] for row in rows})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, ratio in enumerate(ratios):
        subset = [
            r for r in rows if r["forward_ratio"] == ratio and r["asymmetry_mean"] is not None
        ]
        subset.sort(key=lambda r: r["background_scale_target"])
        ax.plot(
            [r["background_scale_target"] for r in subset],
            [r["asymmetry_mean"] for r in subset],
            "o-",
            color=colours[i % len(colours)],
            label=f"forward ratio {ratio:g}",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("background drive into the target compartment (relative)")
    ax.set_ylabel("coupling asymmetry (source -> target)")
    import textwrap

    ax.set_title("\n".join(textwrap.wrap(title, 64)), fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run the HD11 validation and write simulated outputs and metrics."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="short runs for CI")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    TWIN_DIR.mkdir(parents=True, exist_ok=True)
    duration = 20.0 if args.quick else 60.0
    seeds = (0, 1) if args.quick else (0, 1, 2)

    baseline = simulated_features(
        NetworkParams(n_per_compartment=250, duration_s=duration, seed=args.seed)
    )
    baseline["within_real_range"] = in_real_range(baseline["features_simulated"])
    log.info("simulated baseline features: %s", baseline["features_simulated"])

    sweep = directionality_sweep(
        ratios=(1.0, 4.0, 16.0), drive_scales=(1.0, 0.85, 0.7), seeds=seeds, duration_s=duration
    )
    results = {
        "label": "SIMULATED - not measured data",
        "duration_s": duration,
        "baseline": baseline,
        "directionality_sweep": sweep,
        "finding": (
            "coupling asymmetry tracks the compartments' relative firing rate as much as the "
            "pathway ratio: with equal drive a strongly directed pathway can read as reversed, "
            "because the target compartment also receives the pathway's excitation"
        ),
    }
    (TWIN_DIR / "hd11_validation_simulated.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_sweep(
        sweep,
        TWIN_DIR / "hd11_directionality_simulated.png",
        "SIMULATED two-compartment network: coupling asymmetry tracks drive, not wiring\n"
        "at fixed wiring, drive moves the measure further than the pathway ratio does",
    )
    update_metrics("twin_hd11_validation", results)
    log.info("wrote %s", TWIN_DIR / "hd11_validation_simulated.json")
    print(
        json.dumps(
            {
                "baseline_features": baseline["features_simulated"],
                "within_real_range": baseline["within_real_range"],
            },
            indent=2,
        )
    )
    for row in sweep:
        print(
            f"  ratio {row['forward_ratio']:>5} drive {row['background_scale_target']:>5} "
            f"-> asym {row['asymmetry_mean']}, lag {row['lag_ms_mean']} ms, "
            f"rate ratio {row['rate_ratio_c1_over_c0']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
