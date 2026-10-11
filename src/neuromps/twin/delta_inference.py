"""Paired inference: infer the drug knob from a change, not from an absolute state.

§7.5 found the absolute-summary posterior inferring `gabaa_scale` = 0.378 on
*untreated* chips, where 1.0 is neutral, and showing no bicuculline dose trend.
The diagnosis was that the knob absorbs general model mismatch: the estimator
sees one real recording's features and has no way to separate "this culture is
unlike the simulator" from "this culture had a drug applied".

Pharmacology is never quantified that way. A compound's effect is measured
against the same preparation's own vehicle baseline. This module trains the
same kind of estimator on that quantity instead:

    p(theta_drug | x_treated - x_baseline, x_baseline)

Each training pair is *one simulated chip recorded twice* - the same
connectivity and threshold draw, a fresh noise realisation
(`NetworkParams.noise_seed`), neutral knob for the baseline and a prior draw
for the treated run. Conditioning on the baseline as well as the difference
lets the estimator use how the culture behaved untreated, which is exactly the
information the absolute version had to guess at.

The real side has the matching structure: each of the 16 bicuculline devices
contributes one untreated recording and several dosed ones.

Whether this fixes anything is a question with three prearranged answers, all
reported whichever way they come out: does the dose trend appear, does a pair
whose treated recording *is* the baseline return the knob to 1.0, and does the
posterior contract more than the absolute version's 0.34.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.parallel import worker_processes
from neuromps.paths import twin_dir
from neuromps.twin.inference import (
    PARAMETER_NAMES,
    PRIOR_BOUNDS,
    SUMMARY_NAMES,
    simulate_summary,
)

log = logging.getLogger("neuromps.twin")
REPO_ROOT = Path(__file__).resolve().parents[3]

DRUG_PARAMETER = "gabaa_scale"
NEUTRAL_VALUE = 1.0
DRUG_INDEX = PARAMETER_NAMES.index(DRUG_PARAMETER)
# The baseline recording's own summaries are given to the estimator alongside
# the difference, so the context is twice the summary width.
CONTEXT_NAMES = tuple(
    [f"delta_{n}" for n in SUMMARY_NAMES] + [f"baseline_{n}" for n in SUMMARY_NAMES]
)


def drug_prior():
    """Uniform prior over the drug knob alone."""
    import torch
    from sbi.utils import BoxUniform

    low, high = PRIOR_BOUNDS[DRUG_PARAMETER]
    return BoxUniform(
        low=torch.tensor([low], dtype=torch.float32),
        high=torch.tensor([high], dtype=torch.float32),
    )


def nuisance_prior():
    """Uniform prior over everything except the drug knob."""
    import torch
    from sbi.utils import BoxUniform

    names = [n for n in PARAMETER_NAMES if n != DRUG_PARAMETER]
    low = torch.tensor([PRIOR_BOUNDS[n][0] for n in names], dtype=torch.float32)
    high = torch.tensor([PRIOR_BOUNDS[n][1] for n in names], dtype=torch.float32)
    return BoxUniform(low=low, high=high)


def theta_with_drug(nuisance: np.ndarray, drug_value: float) -> np.ndarray:
    """Assemble a full parameter vector from nuisance draws and one knob value."""
    names = [n for n in PARAMETER_NAMES if n != DRUG_PARAMETER]
    values = dict(zip(names, np.asarray(nuisance, dtype=float), strict=True))
    values[DRUG_PARAMETER] = float(drug_value)
    return np.array([values[n] for n in PARAMETER_NAMES], dtype=float)


def context_vector(baseline: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """What the estimator conditions on: the change, and the state it started from."""
    baseline = np.asarray(baseline, dtype=float)
    treated = np.asarray(treated, dtype=float)
    return np.concatenate([treated - baseline, baseline], axis=-1)


def _paired_job(args) -> np.ndarray:
    """One chip recorded twice: untreated, then with the knob moved."""
    nuisance, drug_value, n_per_compartment, duration_s, seed = args
    try:
        # One chip - shared `seed`, so shared connectivity and thresholds -
        # recorded twice under different noise, as two sessions on one device.
        baseline = simulate_summary(
            theta_with_drug(nuisance, NEUTRAL_VALUE),
            n_per_compartment,
            duration_s,
            seed,
            noise_seed=2 * seed + 1,
        )
        treated = simulate_summary(
            theta_with_drug(nuisance, drug_value),
            n_per_compartment,
            duration_s,
            seed,
            noise_seed=2 * seed + 2,
        )
    except Exception:  # noqa: BLE001 - one bad draw must not kill the sweep
        log.exception("paired simulation failed for nuisance=%s drug=%s", nuisance, drug_value)
        return np.full(2 * len(SUMMARY_NAMES), np.nan)
    return np.concatenate([baseline, treated])


def run_paired_simulations(
    n_pairs: int,
    n_per_compartment: int | None = None,
    duration_s: float = 10.0,
    seed: int = 0,
    processes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw chips and knob values, simulate each chip twice. Failed pairs are dropped."""
    import multiprocessing as mp

    import torch

    torch.manual_seed(seed)
    nuisance = nuisance_prior().sample((n_pairs,)).numpy()
    drug = drug_prior().sample((n_pairs,)).numpy().reshape(-1)
    jobs = [(nuisance[i], drug[i], n_per_compartment, duration_s, seed + i) for i in range(n_pairs)]
    processes = processes or worker_processes()
    log.info("simulating %d pairs (%d runs) on %d processes", n_pairs, 2 * n_pairs, processes)
    with mp.Pool(processes) as pool:
        results = pool.map(_paired_job, jobs, chunksize=4)
    stacked = np.vstack(results)
    usable = np.isfinite(stacked).all(axis=1)
    if not usable.all():
        log.warning("dropping %d failed pairs", int((~usable).sum()))
    width = len(SUMMARY_NAMES)
    return drug[usable], stacked[usable, :width], stacked[usable, width:]


def train_delta_posterior(drug: np.ndarray, context: np.ndarray, max_epochs: int | None = None):
    """Neural posterior estimator for the knob given the change and the baseline."""
    import torch
    from sbi.inference import SNPE

    trainer = SNPE(prior=drug_prior())
    trainer.append_simulations(
        torch.as_tensor(np.asarray(drug, dtype=np.float32).reshape(-1, 1)),
        torch.as_tensor(np.asarray(context, dtype=np.float32)),
    )
    density = trainer.train(max_num_epochs=max_epochs) if max_epochs else trainer.train()
    return trainer.build_posterior(density)


def _sample(posterior, context: np.ndarray, n_samples: int, max_seconds: float) -> np.ndarray:
    """Posterior draws for one conditioning vector, time-capped as §7.5's are."""
    import torch

    try:
        samples = posterior.sample(
            (n_samples,),
            x=torch.as_tensor(np.asarray(context, dtype=np.float32)),
            show_progress_bars=False,
            max_sampling_time=max_seconds,
            return_partial_on_timeout=True,
        )
    except RuntimeError:  # nothing accepted within the cap: no draws, not a crash
        return np.empty(0)
    samples = samples.detach().numpy() if hasattr(samples, "detach") else np.asarray(samples)
    return np.asarray(samples).reshape(-1)


def posterior_medians(
    posterior, context: np.ndarray, n_samples: int = 500, max_seconds: float = 15.0
) -> list[dict]:
    """Median and 90% interval of the knob for each conditioning vector."""
    out = []
    for row in np.atleast_2d(context):
        samples = _sample(posterior, row, n_samples, max_seconds)
        if samples.size == 0:
            nan = float("nan")
            out.append({"median": nan, "ci90": [nan, nan], "n_effective_samples": 0})
            continue
        out.append(
            {
                "median": round(float(np.median(samples)), 4),
                "ci90": [
                    round(float(np.percentile(samples, 5)), 4),
                    round(float(np.percentile(samples, 95)), 4),
                ],
                "n_effective_samples": int(samples.size),
            }
        )
    return out


def held_out_checks(
    posterior, drug: np.ndarray, context: np.ndarray, n_samples: int = 200, seed: int = 0
) -> dict:
    """Rank calibration and contraction on pairs the estimator never saw.

    The same two tests §7.5 applies to the absolute posterior, so the two are
    comparable: a calibrated posterior puts the truth at a uniformly
    distributed rank, and contraction says how much narrower the posterior is
    than the prior it started from.
    """
    from scipy.stats import kstest

    low, high = PRIOR_BOUNDS[DRUG_PARAMETER]
    prior_sd = (high - low) / np.sqrt(12)
    ranks, spreads, errors = [], [], []
    for truth, row in zip(np.asarray(drug).reshape(-1), np.atleast_2d(context), strict=True):
        samples = _sample(posterior, row, n_samples, 10.0)
        if samples.size < 2:
            continue
        ranks.append(float((samples < truth).mean()))
        spreads.append(float(np.std(samples)))
        errors.append(float(np.median(samples) - truth))
    if not ranks:
        return {"n_checked": 0, "note": "no usable held-out pairs"}
    contraction = 1.0 - float(np.mean(spreads)) / prior_sd
    uniformity = kstest(ranks, "uniform")
    return {
        "n_checked": len(ranks),
        "contraction": round(contraction, 4),
        "rank_uniformity_ks_p": round(float(uniformity.pvalue), 4),
        "rank_uniform": bool(uniformity.pvalue > 0.05),
        "median_bias": round(float(np.mean(errors)), 4),
        "note": "contraction and rank uniformity, as §7.5 reports them for the absolute posterior",
    }


def real_pairs(summaries: pd.DataFrame) -> pd.DataFrame:
    """Each dosed recording beside its own device's untreated recording.

    A device with no untreated recording contributes nothing: the point of this
    estimator is that it is never asked to judge a culture it has not seen
    untreated.
    """
    from neuromps.models.dataset import doses_in_um

    rows = []
    for device, group in summaries.groupby("device_id"):
        doses = doses_in_um(group)
        untreated = group[doses <= 0]
        if untreated.empty:
            log.info("device %s has no untreated recording; skipped", device)
            continue
        baseline = untreated.iloc[0][list(SUMMARY_NAMES)].to_numpy(dtype=float)
        for dose, (_, row) in zip(doses, group.iterrows(), strict=True):
            treated = row[list(SUMMARY_NAMES)].to_numpy(dtype=float)
            rows.append(
                {
                    "device_id": device,
                    "dose_value": float(dose),  # uM, whatever unit the recording used
                    "is_the_baseline_itself": bool(np.array_equal(treated, baseline)),
                    "context": context_vector(baseline, treated),
                }
            )
    return pd.DataFrame(rows)


def neutral_recovery(pairs: pd.DataFrame, medians: list[dict], seed: int = 0) -> dict:
    """Where the estimator puts a pair whose treated recording *is* the baseline.

    The direct analogue of §7.5's failing check. With no change to explain, a
    well-behaved estimator should return the knob to 1.0; the absolute-summary
    posterior put untreated chips at 0.3781 instead.
    """
    from neuromps.models.bootstrap import ci95, group_bootstrap_indices

    mask = pairs["is_the_baseline_itself"].to_numpy(dtype=bool)
    if mask.sum() < 3:
        return {"median": None, "note": "too few untreated recordings"}
    values = np.array([m["median"] for m in medians], dtype=float)[mask]
    groups = pairs["device_id"].to_numpy()[mask]
    draws = [
        float(np.median(values[idx]))
        for idx in group_bootstrap_indices(groups, np.random.default_rng(seed), 1000)
    ]
    interval = ci95(np.asarray(draws))
    return {
        "median": round(float(np.median(values)), 4),
        "median_ci95": interval,
        "n_untreated": int(mask.sum()),
        "neutral_value": NEUTRAL_VALUE,
        "includes_neutral": bool(interval[0] <= NEUTRAL_VALUE <= interval[1]),
        "bootstrap_resamples": 1000,
    }


def dose_trend(pairs: pd.DataFrame, medians: list[dict], seed: int = 0) -> dict:
    """Rank correlation of the inferred knob with dose, clustered over devices."""
    from scipy.stats import spearmanr

    from neuromps.models.bootstrap import group_bootstrap_metric

    treated = ~pairs["is_the_baseline_itself"].to_numpy(dtype=bool)
    doses = pairs["dose_value"].to_numpy(dtype=float)[treated]
    values = np.array([m["median"] for m in medians], dtype=float)[treated]
    groups = pairs["device_id"].to_numpy()[treated]
    if len(values) < 4 or len(set(doses)) < 2:
        return {"n_recordings": int(len(values)), "note": "too few dosed recordings"}

    def metric(idx: np.ndarray) -> float:
        if len(set(doses[idx])) < 2 or len(set(values[idx])) < 2:
            return float("nan")
        return float(spearmanr(doses[idx], values[idx]).statistic)

    observed = spearmanr(doses, values)
    boot = group_bootstrap_metric(metric, groups, 1000, seed)
    interval = boot.get("ci95")
    return {
        "spearman_rho": round(float(observed.statistic), 4),
        "naive_p_value": round(float(observed.pvalue), 4),
        "spearman_rho_ci95": interval,
        "excludes_zero": None
        if not interval or interval[0] is None
        else bool(interval[0] > 0 or interval[1] < 0),
        "expected_direction": "falls with dose (a GABA-A antagonist lowers the knob)",
        "n_recordings": int(len(values)),
        "n_devices": int(len(set(groups))),
        "bootstrap_resamples": 1000,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: train the paired estimator and put the three questions to it."""
    import argparse
    import json

    import pandas as pd

    from neuromps.metrics import update_metrics
    from neuromps.paths import asset_path, for_reading
    from neuromps.twin.real_inference import real_summaries

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="a handful of pairs, for the path test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, default=None, help="override the pair count")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    n_pairs = args.pairs or (10 if args.quick else 1500)
    duration_s = 5.0 if args.quick else 10.0
    drug, baseline_x, treated_x = run_paired_simulations(
        n_pairs, duration_s=duration_s, seed=args.seed
    )
    if len(drug) < 10:
        log.warning("only %d usable pairs; not training", len(drug))
        update_metrics("twin_hd17_delta_inference", {"status": "too few usable pairs"})
        return 0
    context = context_vector(baseline_x, treated_x)
    held_out = max(1, len(drug) // 5)
    posterior = train_delta_posterior(
        drug[:-held_out], context[:-held_out], max_epochs=20 if args.quick else None
    )
    checks = held_out_checks(
        posterior, drug[-held_out:], context[-held_out:], n_samples=100 if args.quick else 200
    )

    result = {
        "label": "posterior trained on PAIRED SIMULATIONS, applied to REAL recordings",
        "question": (
            "does inferring the drug knob from a change against the same chip's own baseline "
            "fix what §7.5 found, where absolute summaries put untreated chips at 0.3781"
        ),
        "drug_parameter": DRUG_PARAMETER,
        "n_pairs_simulated": int(len(drug)),
        "n_pairs_held_out": int(held_out),
        "duration_s": duration_s,
        "seed": args.seed,
        "held_out_checks": checks,
    }

    asset = for_reading(asset_path())
    if asset.exists():
        summaries = real_summaries(pd.read_parquet(asset))
        pairs = real_pairs(summaries) if not summaries.empty else pd.DataFrame()
    else:
        pairs = pd.DataFrame()
    if pairs.empty:
        result["real_data"] = {"status": "no paired real recordings in this asset"}
    else:
        if args.quick:
            # The path test only has to reach every branch; scoring all 113
            # recordings is most of its cost on a two-vCPU runner.
            pairs = pairs.groupby("device_id", group_keys=False).head(2).reset_index(drop=True)
        stacked = np.vstack(pairs["context"].to_numpy())
        medians = posterior_medians(posterior, stacked, n_samples=100 if args.quick else 500)
        sampled = np.array([m["n_effective_samples"] > 0 for m in medians], dtype=bool)
        pairs = pairs[sampled].reset_index(drop=True)
        medians = [m for m, keep in zip(medians, sampled, strict=True) if keep]
        result["real_data"] = {
            "n_recordings_unsampled": int((~sampled).sum()),
            "n_recordings": int(len(pairs)),
            "n_devices": int(pairs["device_id"].nunique()),
            "gabaa_scale_at_baseline": neutral_recovery(pairs, medians, args.seed),
            "gabaa_scale_dose_trend": dose_trend(pairs, medians, args.seed),
        }
    update_metrics("twin_hd17_delta_inference", result)
    (twin_dir() / "hd17_delta_inference_simulated.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
