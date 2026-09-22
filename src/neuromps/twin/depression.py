"""Does short-term synaptic depression close the twin's calibration gap?

§7.3 calibrates five features against the real band and lands three of them
inside it, leaving `pct_spikes_in_bursts` and `synchrony_index` outside: the
simulated network bursts too hard and too together. The mechanism usually
blamed for that is missing, not mis-tuned. Spike-frequency adaptation acts on
each cell separately and cannot stop a burst the population is driving;
vesicle depletion at the recurrent excitatory synapses can, because the more
the population fires the less each spike delivers.

So this asks the narrow version of the question rather than rebuilding the
twin: holding the existing calibration fixed, does adding Tsodyks-Markram
depression to the recurrent excitatory synapses move those two features toward
the real band, and at what cost to the three already inside it?

Holding the calibration fixed is the point and also the caveat. The weights
were tuned without depression, so a fair comparison would re-run the search
over the enlarged space; what this can show is whether the mechanism pushes in
the direction the diagnosis predicts, which is what decides whether that search
is worth running. `depression_u = 0` is the control and reproduces §7.3 exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.parallel import worker_processes
from neuromps.paths import twin_dir

log = logging.getLogger("neuromps.twin")
REPO_ROOT = Path(__file__).resolve().parents[3]

# The two features §7.3 leaves outside the real band.
GAP_FEATURES = ("pct_spikes_in_bursts", "synchrony_index")
# Depletion fraction per spike, and how fast resources return (ms). The
# recovery range brackets the 500-800 ms usually quoted for cortical synapses.
DEPRESSION_GRID = (
    (0.0, 600.0),  # control: no depression, which is §7.3 as it stands
    (0.2, 300.0),
    (0.2, 600.0),
    (0.4, 300.0),
    (0.4, 600.0),
    (0.4, 900.0),
    (0.6, 600.0),
    (0.6, 900.0),
)


def _score_job(args) -> dict:
    """One configuration at one seed, scored by the calibration's own scorer."""
    from neuromps.twin.calibrate import score_features, simulate_chamber_features

    drawn, duration_s, seed, targets = args
    try:
        simulated = simulate_chamber_features(drawn, duration_s, seed)
    except Exception:  # noqa: BLE001 - one bad configuration must not stop the sweep
        log.exception("depression sweep failed for %s", drawn)
        return {"seed": seed, "score": None, "per_feature": {}, "features": {}}
    scored = score_features(simulated["mean"], targets)
    return {
        "seed": seed,
        "score": scored["score"],
        "per_feature": scored["per_feature"],
        "features": {k: v for k, v in simulated["mean"].items()},
    }


def sweep(
    targets: dict,
    grid: tuple[tuple[float, float], ...] = DEPRESSION_GRID,
    seeds: tuple[int, ...] = (0, 1, 2),
    duration_s: float = 40.0,
    processes: int | None = None,
) -> pd.DataFrame:
    """Score every depression setting on top of the existing calibration."""
    import multiprocessing as mp

    from neuromps.twin.calibrate import calibrated_params

    base = dict(calibrated_params())
    jobs, keys = [], []
    for u, tau in grid:
        drawn = {**base, "depression_u": u, "depression_tau_rec_ms": tau}
        for seed in seeds:
            jobs.append((drawn, duration_s, seed, targets))
            keys.append((u, tau))
    processes = processes or worker_processes()
    log.info("scoring %d depression settings on %d processes", len(jobs), processes)
    with mp.Pool(processes) as pool:
        results = pool.map(_score_job, jobs, chunksize=1)
    rows = []
    for (u, tau), out in zip(keys, results, strict=True):
        rows.append(
            {
                "depression_u": u,
                "depression_tau_rec_ms": tau,
                "is_control": u == 0.0,
                "seed": out["seed"],
                "score": out["score"],
                **{f"penalty_{k}": v for k, v in out["per_feature"].items()},
                **{f"value_{k}": v for k, v in out["features"].items()},
            }
        )
    return pd.DataFrame(rows)


def summarise(scored: pd.DataFrame) -> dict:
    """Which settings beat the control, and on which features."""
    if scored.empty or scored["score"].isna().all():
        return {"status": "no usable simulations"}
    by_setting = (
        scored.groupby(["depression_u", "depression_tau_rec_ms", "is_control"])
        .mean(numeric_only=True)
        .reset_index()
    )
    control = by_setting[by_setting["is_control"]]
    if control.empty:
        return {"status": "no control run"}
    control_row = control.iloc[0]
    settings = {}
    for _, row in by_setting.iterrows():
        name = f"u{row['depression_u']:g}_tau{row['depression_tau_rec_ms']:g}"
        settings[name] = {
            "depression_u": float(row["depression_u"]),
            "depression_tau_rec_ms": float(row["depression_tau_rec_ms"]),
            "score": _round(row.get("score")),
            "penalties": {
                column[len("penalty_") :]: _round(row.get(column))
                for column in by_setting.columns
                if column.startswith("penalty_")
            },
        }
        settings[name]["features"] = {
            column[len("value_") :]: _round(row.get(column))
            for column in by_setting.columns
            if column.startswith("value_")
        }
    control_name = f"u{control_row['depression_u']:g}_tau{control_row['depression_tau_rec_ms']:g}"
    best_name = min(
        (n for n, v in settings.items() if v["score"] is not None),
        key=lambda n: settings[n]["score"],
        default=None,
    )
    out = {
        "control": control_name,
        "best": best_name,
        "best_beats_control": None
        if best_name is None
        else bool(settings[best_name]["score"] < settings[control_name]["score"]),
        "settings": settings,
    }
    for feature in GAP_FEATURES:
        control_penalty = settings[control_name]["penalties"].get(feature)
        improved = [
            name
            for name, entry in settings.items()
            if name != control_name
            and entry["penalties"].get(feature) is not None
            and control_penalty is not None
            and entry["penalties"][feature] < control_penalty
        ]
        out[f"{feature}_improves_under"] = sorted(improved)
        out[f"{feature}_control_penalty"] = control_penalty
        out[f"{feature}_best_penalty"] = _round(
            min(
                (
                    e["penalties"][feature]
                    for e in settings.values()
                    if e["penalties"].get(feature) is not None
                ),
                default=None,
            )
        )
    return out


def _round(value) -> float | None:
    """Null-safe rounding for a metrics block."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), 4)


def main(argv: list[str] | None = None) -> int:
    """CLI: test the depression hypothesis against the calibration's own scorer."""
    import argparse
    import json

    from neuromps.metrics import update_metrics
    from neuromps.paths import asset_path, for_reading
    from neuromps.twin.calibrate import real_targets

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="two settings, one seed, short runs")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asset = for_reading(asset_path())
    if not asset.exists():
        log.warning("no asset at %s; skipping", asset)
        update_metrics("twin_hd18_depression", {"status": "no asset"})
        return 0
    targets = real_targets(pd.read_parquet(asset))
    if not targets.get("by_feature"):
        log.warning("no real calibration band in this asset; skipping")
        update_metrics("twin_hd18_depression", {"status": "no real reference in this asset"})
        return 0
    grid = DEPRESSION_GRID[:2] if args.quick else DEPRESSION_GRID
    scored = sweep(
        targets,
        grid=grid,
        seeds=(0,) if args.quick else (0, 1, 2),
        duration_s=10.0 if args.quick else 40.0,
    )
    result = {
        "label": "SIMULATED - a model comparison, not measured data",
        "question": (
            "does Tsodyks-Markram depression on recurrent excitatory synapses move "
            "pct_spikes_in_bursts and synchrony_index toward the real band"
        ),
        "caveat": (
            "the weights were calibrated without depression, so this shows the direction "
            "the mechanism pushes, not what a re-calibrated model would score"
        ),
        "duration_s": 10.0 if args.quick else 40.0,
        "n_settings": int(len(grid)),
        "n_seeds": 1 if args.quick else 3,
        **summarise(scored),
    }
    twin_dir().mkdir(parents=True, exist_ok=True)
    (twin_dir() / "hd18_depression_simulated.json").write_text(json.dumps(result, indent=2) + "\n")
    scored.to_csv(twin_dir() / "hd18_depression_simulated.csv", index=False)
    update_metrics("twin_hd18_depression", result)
    print(json.dumps({k: v for k, v in result.items() if k != "settings"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
