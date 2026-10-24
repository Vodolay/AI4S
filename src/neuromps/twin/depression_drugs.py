"""Does vesicle depletion make the twin's firing rate fall under GABA-A block?

Real GABA-A antagonist pairs lower firing and burst rates; the calibrated twin,
which has no short-term synaptic depression, raises firing rate (HD13 'differ').
`preregistration/twin_depression_under_block.json` registers the test: run the
HD13 knobs and a GABA-A dose series under each HD18 depression setting, and ask
whether some setting (i) lowers firing rate at the strongest block, (ii) makes
both burst rates fall with dose, and (iii) keeps at least the control's
agreement with real chips; then replicate the chosen setting on fresh seeds.

The weights stay as calibrated without depression (the HD18 caveat), so a pass
shows the mechanism *can* produce the real direction. Everything here is
**simulated**.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neuromps.metrics import update_metrics
from neuromps.parallel import worker_processes
from neuromps.paths import REPO_ROOT, metrics_path, twin_dir
from neuromps.twin.depression import DEPRESSION_GRID

log = logging.getLogger("neuromps.twin")
REGISTRATION = REPO_ROOT / "preregistration" / "twin_depression_under_block.json"
DOSE_SERIES = (1.0, 0.8, 0.6, 0.4, 0.2)
OTHER_KNOBS = {
    "gabaa_block": ("gabaa_scale", 0.2),
    "gabaa_potentiate": ("gabaa_scale", 2.0),
    "ampa_block": ("ampa_scale", 0.4),
    "sodium_block": ("threshold_scale", 1.6),
}
DURATION_S = 40.0


def label(u: float, tau: float) -> str:
    """Setting label as HD18 writes it."""
    return f"u{u:g}_tau{tau:g}"


def _conditions() -> dict[str, dict]:
    out = {f"gabaa_{g:g}": {"gabaa_scale": g} for g in DOSE_SERIES}
    out.update(
        {
            "gabaa_potentiate": {"gabaa_scale": 2.0},
            "ampa_block": {"ampa_scale": 0.4},
            "sodium_block": {"threshold_scale": 1.6},
        }
    )
    return out


def _job(args) -> dict:
    """One setting, one condition, one seed: HD13's features of one recording."""
    from neuromps.twin.calibrate import calibrated_params
    from neuromps.twin.mea import ElectrodeLayout
    from neuromps.twin.network import NetworkParams
    from neuromps.twin.perturbation import _features_for

    (u, tau), condition, overrides, seed = args
    tuned = calibrated_params()
    network_keys = set(NetworkParams().as_record())
    layout_keys = set(ElectrodeLayout().as_record())
    layout = ElectrodeLayout(**{k: v for k, v in tuned.items() if k in layout_keys})
    kwargs = {
        "n_per_compartment": 250,
        "duration_s": DURATION_S,
        "seed": seed,
        **{k: v for k, v in tuned.items() if k in network_keys},
        "depression_u": u,
        "depression_tau_rec_ms": tau,
        **overrides,
    }
    features = _features_for(NetworkParams(**kwargs), layout, seed)
    return {
        "setting": label(u, tau),
        "condition": condition,
        "seed": seed,
        **{k: v for k, v in features.items() if k != "silent"},
    }


def run(grid, seeds, processes: int) -> pd.DataFrame:
    """Every (setting, condition, seed) recording's features."""
    jobs = [(s, c, o, seed) for s in grid for c, o in _conditions().items() for seed in seeds]
    with ProcessPoolExecutor(processes) as pool:
        return pd.DataFrame(list(pool.map(_job, jobs)))


def _delta(rows: pd.DataFrame, condition: str, feature: str) -> float:
    """Seed-matched log2(treated / untreated), averaged over seeds (HD13's rule)."""
    from neuromps.twin.perturbation import simulated_delta

    base = rows[rows["condition"] == "gabaa_1"].set_index("seed")
    treat = rows[rows["condition"] == condition].set_index("seed")
    values = [
        simulated_delta(treat.loc[s].to_dict(), base.loc[s].to_dict()).get(feature)
        for s in treat.index
        if s in base.index
    ]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else float("nan")


def rule_i(rows: pd.DataFrame) -> float:
    """Mean delta log2 mfr_hz at the strongest block of the dose series."""
    return _delta(rows, f"gabaa_{min(DOSE_SERIES):g}", "mfr_hz")


def rule_ii(rows: pd.DataFrame) -> dict:
    """Spearman of each burst rate with block strength over the dose series."""
    series = rows[rows["condition"].isin([f"gabaa_{g:g}" for g in DOSE_SERIES])]
    block = 1 - series["condition"].str.removeprefix("gabaa_").astype(float)
    out = {}
    for feature in ("burst_rate_per_min", "network_burst_rate_per_min"):
        rho = spearmanr(block, series[feature].astype(float))[0]
        out[feature] = None if not np.isfinite(rho) else round(float(rho), 4)
    out["passes"] = all(v is not None and v <= -0.5 for v in out.values())
    return out


def agreement(rows: pd.DataFrame, references: dict) -> dict:
    """Per knob: HD13's agreement table on seed-averaged deltas, and the agreeing count."""
    from neuromps.twin.perturbation import COMPARE_FEATURES, KNOBS, agreement_table

    condition = {
        "gabaa_block": f"gabaa_{min(DOSE_SERIES):g}",
        "gabaa_potentiate": "gabaa_potentiate",
        "ampa_block": "ampa_block",
        "sodium_block": "sodium_block",
    }
    out = {}
    for knob in KNOBS:
        delta = {f: _delta(rows, condition[knob], f) for f in COMPARE_FEATURES}
        delta = {f: (None if not np.isfinite(v) else round(v, 3)) for f, v in delta.items()}
        table = agreement_table(delta, references[knob])
        out[knob] = {
            "agreeing": int(sum(v["agrees"] is True for v in table.values())),
            "scored": int(sum(v["agrees"] is not None for v in table.values())),
            "delta": delta,
        }
    return out


def references() -> dict:
    """Real direction per HD13 knob, computed exactly as HD13 does."""
    from neuromps.models.dataset import load_pairs
    from neuromps.twin.perturbation import KNOBS, real_directions

    pairs = load_pairs()
    return {k: real_directions(pairs, v["mechanism"], 1000, seed=0) for k, v in KNOBS.items()}


def score_settings(frame: pd.DataFrame, refs: dict) -> dict:
    """Rules i-iii per setting."""
    out = {}
    for setting, rows in frame.groupby("setting"):
        agree = agreement(rows, refs)
        out[setting] = {
            "i_mfr_delta_at_block": round(rule_i(rows), 4),
            "ii_dose_trend": rule_ii(rows),
            "agreement": agree,
            "agreeing_total": sum(a["agreeing"] for a in agree.values()),
        }
    control = out[label(*DEPRESSION_GRID[0])]["agreeing_total"]
    for entry in out.values():
        entry["i"] = entry["i_mfr_delta_at_block"] < 0
        entry["iii"] = entry["agreeing_total"] >= control
        entry["qualifies"] = bool(entry["i"] and entry["ii_dose_trend"]["passes"] and entry["iii"])
    return out


def _hd18_scores() -> dict:
    settings = json.loads(metrics_path().read_text())["twin_hd18_depression"]["settings"]
    return {name: entry["score"] for name, entry in settings.items()}


def main(argv: list[str] | None = None) -> int:
    """CLI: run the registered test and write it to metrics.json."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processes", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    processes = args.processes or worker_processes(cap=18)
    refs = references()
    frame = run(DEPRESSION_GRID, tuple(spec["design"]["seeds"]), processes)
    scored = score_settings(frame, refs)
    control = label(*DEPRESSION_GRID[0])
    qualifying = [s for s, e in scored.items() if e["qualifies"] and s != control]
    v1 = len(qualifying) >= 1
    v2 = not scored[control]["i"]
    chosen, replication = None, None
    if qualifying:
        hd18 = _hd18_scores()
        chosen = min(qualifying, key=lambda s: hd18.get(s, float("inf")))
        grid = [g for g in DEPRESSION_GRID if label(*g) == chosen]
        fresh = run(grid, tuple(spec["design"]["fresh_seeds"]), processes)
        replication = {
            "i_mfr_delta_at_block": round(rule_i(fresh), 4),
            "ii_dose_trend": rule_ii(fresh),
        }
        replication["passes"] = bool(
            replication["i_mfr_delta_at_block"] < 0 and replication["ii_dose_trend"]["passes"]
        )
        frame = pd.concat([frame, fresh.assign(replication=True)], ignore_index=True)
    predictions = {
        "V1_some_setting_qualifies": {
            "value": len(qualifying),
            "outcome": "pass" if v1 else "fail",
        },
        "V2_the_control_does_not": {
            "value": scored[control]["i_mfr_delta_at_block"],
            "outcome": "pass" if v2 else "fail",
        },
        "V3_the_chosen_setting_replicates": {
            "value": chosen,
            "outcome": "n/a"
            if replication is None
            else ("pass" if replication["passes"] else "fail"),
        },
    }
    result = {
        "label": "SIMULATED twin: depression under drug knobs",
        "settings": scored,
        "qualifying": qualifying,
        "chosen": chosen,
        "replication": replication,
        "predictions": predictions,
        "duration_s": DURATION_S,
    }
    out = twin_dir()
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "depression_under_block_simulated.csv", index=False)
    (out / "depression_under_block_simulated.json").write_text(json.dumps(result, indent=2) + "\n")
    update_metrics("twin_depression_under_block", result)
    for name, entry in predictions.items():
        print(f"{name:36s} {entry['outcome']:>5}  {entry['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
