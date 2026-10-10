"""Simulation-based inference: recover chip parameters from MEA summary features.

Neural posterior estimation over five parameters — the forward/backward ratio of
the inter-compartment pathway, its strength, and the three drug knobs — from
summary features of a simulated recording.

**The summary vector deliberately carries `log_rate_ratio`.** HD11 showed that
coupling asymmetry tracks the compartments' relative firing rate as much as the
pathway, so a posterior trained on coupling alone would infer direction from
excitability. Including the rate imbalance lets the estimator condition on it
instead of confusing it with connectivity; whether that is enough is an
empirical question the posterior checks answer, not an assumption.

Missing measurements become explicit sentinels with an accompanying indicator
(`coupling_measured`), never NaN, so a silent network is a datum rather than a
hole the estimator has to guess about.
"""

import json
import logging
from pathlib import Path

import numpy as np

from neuromps.features.mea import extract_features
from neuromps.parallel import worker_processes
from neuromps.paths import simulation_cache, twin_dir
from neuromps.twin.mea import (
    ElectrodeLayout,
    electrode_coupling,
    electrode_positions,
    place_neurons,
    record_electrodes,
)
from neuromps.twin.network import NetworkParams, simulate

# Used only when no calibration is on disk; the calibrated density otherwise.
DEFAULT_N_PER_COMPARTMENT = 250
log = logging.getLogger("neuromps.twin")
REPO_ROOT = Path(__file__).resolve().parents[3]
POSTERIOR_NAME = "hd14_posterior_simulated.pt"
DENSITY_ESTIMATOR = "maf"  # a masked autoregressive flow

PARAMETER_NAMES = (
    "log10_forward_ratio",
    "inter_in_degree",
    "gabaa_scale",
    "ampa_scale",
    "threshold_scale",
)
PRIOR_BOUNDS = {
    "log10_forward_ratio": (0.0, 1.3),  # ratio 1 to ~20, sampled geometrically
    "inter_in_degree": (2.0, 20.0),
    "gabaa_scale": (0.1, 2.0),
    "ampa_scale": (0.4, 1.4),
    # The network has a sharp excitability cliff: at threshold_scale >= 1.12 it
    # is silent in 98% of draws, and a silent recording carries no information
    # about connectivity. The first prior ran to 1.5 and spent 64% of its
    # simulations there; the bound is restricted to keep the informative regime.
    # See the lab notebook, HD14.
    "threshold_scale": (0.85, 1.15),
}
FEATURE_KEYS = (
    "mfr_hz",
    "burst_rate_per_min",
    "pct_spikes_in_bursts",
    "network_burst_rate_per_min",
    "synchrony_index",
    "isi_cv",
    "active_electrodes",
)
SUMMARY_NAMES = tuple(
    [f"c0_{k}" for k in FEATURE_KEYS]
    + [f"c1_{k}" for k in FEATURE_KEYS]
    + ["log_rate_ratio", "peak_correlation", "lag_ms", "asymmetry", "coupling_measured"]
)
MISSING = 0.0


def prior_bounds_tensor():
    """Prior box as torch tensors, in `PARAMETER_NAMES` order."""
    import torch

    low = torch.tensor([PRIOR_BOUNDS[name][0] for name in PARAMETER_NAMES], dtype=torch.float32)
    high = torch.tensor([PRIOR_BOUNDS[name][1] for name in PARAMETER_NAMES], dtype=torch.float32)
    return low, high


def build_prior():
    """Uniform prior over the box, for sbi."""
    from sbi.utils import BoxUniform

    low, high = prior_bounds_tensor()
    return BoxUniform(low=low, high=high)


def params_from_vector(
    theta,
    n_per_compartment: int | None = None,
    duration_s: float = 10.0,
    seed: int = 0,
    noise_seed: int | None = None,
) -> NetworkParams:
    """Turn a parameter vector into network parameters, on top of the calibration.

    ``noise_seed`` is passed through so that two draws sharing ``seed`` can be
    the same simulated chip under different noise, which paired inference needs
    (`twin/delta_inference.py`). Left unset it changes nothing.
    """

    values = dict(zip(PARAMETER_NAMES, np.asarray(theta, dtype=float), strict=True))
    # Inference runs thousands of simulations, so it uses a smaller network than
    # the calibration fitted and the explicit size wins. That is a tractability
    # choice, not a modelling one; `inference_calibration` is the single
    # definition of which calibrated values do apply, and the cache is keyed on
    # the same thing.
    tuned = inference_calibration()
    # None means "the density the twin was calibrated at", which is what the
    # pipeline wants; an explicit value is for callers that deliberately want a
    # cheap network, such as tests.
    size = tuned.pop("n_per_compartment", DEFAULT_N_PER_COMPARTMENT)
    if n_per_compartment is not None:
        size = n_per_compartment
    return NetworkParams(
        n_per_compartment=int(size),
        duration_s=duration_s,
        seed=seed,
        noise_seed=noise_seed,
        **tuned,
        forward_ratio=float(10 ** values["log10_forward_ratio"]),
        inter_in_degree=float(values["inter_in_degree"]),
        gabaa_scale=float(values["gabaa_scale"]),
        ampa_scale=float(values["ampa_scale"]),
        threshold_scale=float(values["threshold_scale"]),
    )


def default_layout() -> ElectrodeLayout:
    """The layout the twin was calibrated on, so simulated and real geometry match.

    An earlier version used a smaller 4x6 grid for speed; the real chambers have
    48 electrodes with 42-44 typically active, which put the electrode-count
    summaries outside the simulated range and made the posterior extrapolate on
    them. Geometry has to match the data it will be applied to.
    """
    from neuromps.twin.calibrate import calibrated_params

    tuned = calibrated_params()
    keys = set(ElectrodeLayout().as_record())
    return ElectrodeLayout(**{k: v for k, v in tuned.items() if k in keys})


def summary_vector(features_by_compartment: dict, coupling: dict) -> np.ndarray:
    """Fixed-length, finite summary of one recording."""
    values = []
    for compartment in (0, 1):
        features = features_by_compartment.get(compartment) or {}
        for key in FEATURE_KEYS:
            value = features.get(key)
            values.append(float(value) if value is not None and np.isfinite(value) else MISSING)
    rate0 = (features_by_compartment.get(0) or {}).get("mfr_hz") or 0.0
    rate1 = (features_by_compartment.get(1) or {}).get("mfr_hz") or 0.0
    log_ratio = float(np.log2((rate0 + 1e-3) / (rate1 + 1e-3)))
    measured = coupling.get("asymmetry") is not None
    values += [
        log_ratio,
        float(coupling.get("peak_correlation") or MISSING),
        float(coupling.get("lag_ms") or MISSING),
        float(coupling.get("asymmetry") or MISSING),
        1.0 if measured else 0.0,
    ]
    return np.asarray(values, dtype=float)


def simulate_summary(
    theta,
    n_per_compartment: int | None = None,
    duration_s: float = 10.0,
    seed: int = 0,
    layout: ElectrodeLayout | None = None,
    noise_seed: int | None = None,
) -> np.ndarray:
    """Run one simulation and summarise it exactly as a real recording would be."""
    params = params_from_vector(theta, n_per_compartment, duration_s, seed, noise_seed)
    layout = layout or default_layout()
    result = simulate(params)
    positions, compartment = place_neurons(params, layout, seed=seed)
    heard = record_electrodes(
        result.spike_times,
        result.spike_neurons,
        positions,
        compartment,
        electrode_positions(layout),
        layout,
    )
    features = {
        c: extract_features(
            {k: v for k, v in heard.items() if k.startswith(f"c{c}_")}, result.duration_s
        )
        for c in (0, 1)
    }
    coupling = electrode_coupling(heard, result.duration_s)
    return summary_vector(features, coupling)


def within_prior(theta: np.ndarray) -> np.ndarray:
    """Mask of draws lying inside the current prior box.

    Lets a cache generated under a wider prior be reused by rejection when a
    bound is tightened, instead of re-running the simulator.
    """
    theta = np.atleast_2d(theta)
    inside = np.ones(len(theta), dtype=bool)
    for i, name in enumerate(PARAMETER_NAMES):
        low, high = PRIOR_BOUNDS[name]
        inside &= (theta[:, i] >= low) & (theta[:, i] <= high)
    return inside


def save_simulations(path: Path, theta: np.ndarray, x: np.ndarray, metadata: dict) -> None:
    """Cache simulations with the names and geometry they were generated under."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **metadata,
        "electrodes_per_compartment": default_layout().electrodes_per_compartment,
        "summary_pipeline_sha256": summary_pipeline_fingerprint(),
        "calibration_sha256": calibration_fingerprint(),
    }
    np.savez_compressed(
        path,
        theta=np.asarray(theta, dtype=float),
        x=np.asarray(x, dtype=float),
        parameter_names=np.array(PARAMETER_NAMES),
        summary_names=np.array(SUMMARY_NAMES),
        metadata=np.array(json.dumps(metadata)),
    )


def load_simulations(path: Path) -> dict:
    """Read a simulation cache, including the names it was written with."""
    with np.load(path, allow_pickle=False) as data:
        return {
            "theta": data["theta"],
            "x": data["x"],
            "parameter_names": [str(n) for n in data["parameter_names"]],
            "summary_names": [str(n) for n in data["summary_names"]],
            "metadata": json.loads(str(data["metadata"])),
        }


def summary_pipeline_fingerprint() -> str:
    """Hash of the source of every function that decides what a summary vector means.

    Summary *names* guard the meaning of each dimension and the electrode count
    guards their scale, but neither notices a changed definition: HD16 changed
    coupling from all-neuron to electrode-heard spikes without either guard
    firing. This is deliberately conservative - reformatting one of these
    functions invalidates the cache - because a stale cache is a silent wrong
    answer while a needless resimulation only costs time.
    """
    import hashlib
    import inspect

    from neuromps.features import coupling as coupling_module
    from neuromps.features import mea as mea_module
    from neuromps.twin import mea as twin_mea

    sources = [
        inspect.getsource(f)
        for f in (
            simulate_summary,
            summary_vector,
            twin_mea.electrode_coupling,
            twin_mea.record_electrodes,
            mea_module.extract_features,
            coupling_module.directed_coupling,
        )
    ]
    return hashlib.sha256("".join(sources).encode()).hexdigest()[:16]


def inference_calibration() -> dict:
    """The calibrated values inference applies to its simulations.

    This includes `n_per_compartment`. Inference used to override cell density
    with a smaller network for tractability, which meant the posterior was
    learned on a twin several times sparser than the one §7.3 reports as
    calibrated - and density is what sets the twin's synchrony regime, so that
    was not a neutral saving. Measured, the calibrated network costs 1.55x per
    simulation rather than the 6x its neuron count suggests, most of the work
    being fixed overhead, so the gap is closed instead of documented. The cache
    is keyed on this dictionary, and now correctly invalidates when density
    changes, because the simulations do depend on it.
    """
    from neuromps.twin.calibrate import calibrated_params

    keys = set(NetworkParams().as_record())
    return {k: v for k, v in calibrated_params().items() if k in keys}


def calibration_fingerprint() -> str:
    """Hash of the calibration the simulations were generated under."""
    import hashlib

    payload = json.dumps(inference_calibration(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def cache_is_usable() -> bool:
    """Whether the cache on disk can be reused, logging the reason when it cannot.

    A stale cache is not a user error, it is an artifact that has been overtaken,
    and the right answer is to regenerate it. `check_cache_matches_code` raises so
    that a caller *reading* the cache - `real_inference` - refuses rather than
    quietly comparing against a superseded model. Here the caller is about to
    write it, so the pipeline rebuilds instead of aborting: `run_all.py` passes no
    `--rebuild`, so raising here left the whole run dead on any machine whose
    cache predated a change to the summary code.
    """
    try:
        check_cache_matches_code(load_simulations(simulation_cache()))
    except SystemExit as reason:
        log.warning("%s - rebuilding instead", str(reason).replace("; rerun with --rebuild", ""))
        return False
    return True


def check_cache_matches_code(cached: dict) -> None:
    """Refuse a cache produced under different summaries or a different layout.

    The summary names guard the meaning of each dimension; the electrode count
    guards their scale. HD15 found the second one the hard way: a posterior
    trained on a 24-electrode layout extrapolated when shown real 48-electrode
    chambers.
    """
    if list(cached["summary_names"]) != list(SUMMARY_NAMES):
        raise SystemExit("cached simulations use different summaries; rerun with --rebuild")
    cached_electrodes = cached["metadata"].get("electrodes_per_compartment")
    expected = default_layout().electrodes_per_compartment
    if cached_electrodes != expected:
        raise SystemExit(
            f"cached simulations used {cached_electrodes} electrodes per compartment, "
            f"the current layout has {expected}; rerun with --rebuild"
        )
    cached_fingerprint = cached["metadata"].get("summary_pipeline_sha256")
    current = summary_pipeline_fingerprint()
    if cached_fingerprint != current:
        raise SystemExit(
            f"cached simulations were produced by summary code {cached_fingerprint}, "
            f"the current code is {current}; rerun with --rebuild"
        )
    # Nothing else in the cache changes when the twin is recalibrated - same
    # prior, same summaries, same layout - so without this the simulations of a
    # superseded model are reused and the run looks clean.
    cached_calibration = cached["metadata"].get("calibration_sha256")
    expected_calibration = calibration_fingerprint()
    if cached_calibration != expected_calibration:
        raise SystemExit(
            f"cached simulations were produced under calibration {cached_calibration}, "
            f"the current calibration is {expected_calibration}; rerun with --rebuild"
        )


def _simulate_job(args) -> np.ndarray:
    """Worker entry point: one simulation, in its own process."""
    theta, n_per_compartment, duration_s, seed = args
    try:
        return simulate_summary(theta, n_per_compartment, duration_s, seed)
    except Exception:  # noqa: BLE001 - a failed draw must not kill the sweep
        log.exception("simulation failed for theta=%s", theta)
        return np.full(len(SUMMARY_NAMES), np.nan)


def run_simulations(
    n_simulations: int,
    n_per_compartment: int | None = None,
    duration_s: float = 10.0,
    seed: int = 0,
    processes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw from the prior and simulate, in parallel. Failed draws are dropped."""
    import multiprocessing as mp

    import torch

    torch.manual_seed(seed)
    prior = build_prior()
    theta = prior.sample((n_simulations,)).numpy()
    jobs = [(theta[i], n_per_compartment, duration_s, seed + i) for i in range(n_simulations)]
    processes = processes or worker_processes()
    log.info("simulating %d draws on %d processes", n_simulations, processes)
    with mp.Pool(processes) as pool:
        results = pool.map(_simulate_job, jobs, chunksize=4)
    x = np.vstack(results)
    usable = np.isfinite(x).all(axis=1)
    if not usable.all():
        log.warning("dropping %d failed simulations", int((~usable).sum()))
    return theta[usable], x[usable]


def estimator_architecture(posterior) -> dict:
    """The density estimator inside a trained posterior, read off the network itself.

    Recorded from the artifact rather than from the call that built it, so what
    the report states is what was trained - including after a library upgrade.
    """
    flow = getattr(posterior.posterior_estimator, "net", posterior.posterior_estimator)
    kinds = [type(module).__name__ for module in flow.modules()]
    made = [module for module in flow.modules() if type(module).__name__ == "MADE"]
    widths = sorted(
        {
            int(layer.weight.shape[0])
            for block in made
            for layer in block.modules()
            if type(layer).__name__ == "MaskedLinear"
        }
    )
    return {
        "family": "masked autoregressive flow",
        "requested": DENSITY_ESTIMATOR,
        "n_autoregressive_transforms": kinds.count("MaskedAffineAutoregressiveTransform"),
        "n_permutations": kinds.count("RandomPermutation"),
        "hidden_units_per_layer": max(widths) if widths else None,
        "n_parameters": int(sum(t.numel() for t in flow.parameters())),
    }


TIMING_DRAWS = 1000
TIMING_OBSERVATIONS = 20
SIMULATED_S = 10.0  # the duration every full-tier training simulation runs for


def holdout_size(n_simulations: int) -> int:
    """Simulations held out from training: a tenth, and never fewer than twenty."""
    return max(20, int(0.1 * n_simulations))


def amortized_timing(
    posterior,
    observations: np.ndarray,
    simulate=None,
    simulated_s: float | None = None,
    n_draws: int = TIMING_DRAWS,
    n_observations: int | None = None,
) -> dict:
    """Wall-clock cost of a full posterior once the estimator is trained.

    The point of amortized inference is that the simulator is not run again: a
    new recording costs one pass through the flow and the sampling. Timed per
    held-out observation and summarised by the median, so one slow first call
    does not set the figure. When a simulator is given it is timed once beside
    it, so the two costs can be put side by side. A wall clock depends on the
    machine, so the thread count is recorded with it.
    """
    import time

    import torch

    seconds = []
    for observation in observations[:n_observations]:
        start = time.perf_counter()
        posterior.sample(
            (n_draws,),
            x=torch.as_tensor(observation, dtype=torch.float32),
            show_progress_bars=False,
        )
        seconds.append(time.perf_counter() - start)
    out = {
        "posterior_draws": int(n_draws),
        "n_observations": len(seconds),
        "median_seconds_per_posterior": round(float(np.median(seconds)), 4),
        "torch_threads": int(torch.get_num_threads()),
    }
    if simulate is not None:
        start = time.perf_counter()
        simulate()
        out["forward_simulation_seconds"] = round(time.perf_counter() - start, 2)
        out["forward_simulated_duration_s"] = simulated_s
    return out


def reference_simulator(duration_s: float = SIMULATED_S):
    """One forward run at the centre of the prior, warmed up before it is timed.

    The untimed short run absorbs Brian2's one-time code generation, which a
    sweep of thousands of draws pays once and a single timed run should not.
    """
    centre = np.array([sum(PRIOR_BOUNDS[name]) / 2 for name in PARAMETER_NAMES])
    simulate_summary(centre, duration_s=1.0, seed=0)
    return lambda: simulate_summary(centre, duration_s=duration_s, seed=0)


def _timing_arguments(quick: bool) -> dict:
    """How much `amortized_timing` asks for: the quick tier times a token sample only.

    Its posterior is trained for five epochs and rejection-samples slowly, so the
    full-size request - twenty posteriors of 1,000 draws - ran a fresh clone's
    quick pass far past its ten minutes. Its timings are never reported.
    """
    if quick:
        return {"n_observations": 2, "n_draws": 100}
    return {
        "n_observations": TIMING_OBSERVATIONS,
        "simulate": reference_simulator(),
        "simulated_s": SIMULATED_S,
    }


def describe_saved_posterior() -> dict:
    """Add the trained estimator's architecture and timing to this run's metrics.

    For a run whose posterior already exists: it reads the saved network, times
    it on the held-out simulations it was checked against, and writes nothing
    else, so the stage's other results are left as they were.
    """
    import torch

    from neuromps.metrics import update_metrics
    from neuromps.paths import metrics_path

    posterior = torch.load(twin_dir() / POSTERIOR_NAME, weights_only=False)
    cached = load_simulations(simulation_cache())
    x = cached["x"][within_prior(cached["theta"])]
    held_out = x[-holdout_size(len(x)) :]
    section = json.loads(metrics_path().read_text())["twin_hd14_inference"]
    section["density_estimator"] = estimator_architecture(posterior)
    section["amortized_inference"] = amortized_timing(
        posterior, held_out, **_timing_arguments(quick=False)
    )
    update_metrics("twin_hd14_inference", section)
    return {key: section[key] for key in ("density_estimator", "amortized_inference")}


def train_posterior(theta: np.ndarray, x: np.ndarray, max_epochs: int | None = None):
    """Train a neural posterior estimator (NPE) on cached simulations."""
    import torch
    from sbi.inference import NPE

    # Named rather than left to the library default: the report states the
    # architecture, and a default that changed in an sbi upgrade would change the
    # model without changing a line here.
    inference = NPE(prior=build_prior(), density_estimator=DENSITY_ESTIMATOR)
    inference.append_simulations(
        torch.as_tensor(theta, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32)
    )
    kwargs = {"max_num_epochs": max_epochs} if max_epochs else {}
    density_estimator = inference.train(**kwargs)
    return inference.build_posterior(density_estimator)


def posterior_ranks(
    posterior, theta: np.ndarray, x: np.ndarray, n_samples: int = 200
) -> np.ndarray:
    """Rank of each true parameter within posterior samples, for coverage checking.

    Simulation-based calibration: if the posterior is well calibrated, ranks are
    uniform over [0, n_samples]. Systematic skew means bias; a U shape means the
    posterior is over-confident, a hump means it is too wide.
    """
    import torch

    ranks = np.empty((len(theta), theta.shape[1]), dtype=int)
    for i, (true_value, observation) in enumerate(zip(theta, x, strict=True)):
        samples = posterior.sample(
            (n_samples,),
            x=torch.as_tensor(observation, dtype=torch.float32),
            show_progress_bars=False,
        ).numpy()
        ranks[i] = (samples < true_value).sum(axis=0)
    return ranks


def contraction(posterior, theta: np.ndarray, x: np.ndarray, n_samples: int = 200) -> dict:
    """How much narrower the posterior is than the prior, per parameter.

    1 means the observation pinned the parameter down; 0 means it said nothing
    beyond the prior. Reported per parameter because the twin's parameters are
    not equally identifiable.
    """
    import torch

    widths = []
    for observation in x:
        samples = posterior.sample(
            (n_samples,),
            x=torch.as_tensor(observation, dtype=torch.float32),
            show_progress_bars=False,
        ).numpy()
        widths.append(samples.std(axis=0))
    posterior_sd = np.mean(widths, axis=0)
    prior_sd = np.array(
        [(high - low) / np.sqrt(12) for low, high in (PRIOR_BOUNDS[n] for n in PARAMETER_NAMES)]
    )
    return {
        name: {
            "prior_sd": round(float(prior_sd[i]), 4),
            "posterior_sd": round(float(posterior_sd[i]), 4),
            "contraction": round(float(1 - posterior_sd[i] / prior_sd[i]), 3),
        }
        for i, name in enumerate(PARAMETER_NAMES)
    }


def plot_ranks(ranks: np.ndarray, n_samples: int, path, title: str) -> None:
    """Rank histograms: uniform means calibrated, U-shaped means over-confident."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Drawn near the width it prints at: at 13 in it was shown at under 4 and its
    # labels printed at about 2 pt. Takes effect on the next inference run.
    fig, axes = plt.subplots(1, len(PARAMETER_NAMES), figsize=(7.6, 2.4), sharey=True)
    for i, (ax, name) in enumerate(zip(axes, PARAMETER_NAMES, strict=True)):
        ax.hist(ranks[:, i], bins=10, range=(0, n_samples), color="#0072B2")
        expected = len(ranks) / 10
        ax.axhline(expected, color="black", ls="--", lw=1)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("rank", fontsize=8)
    axes[0].set_ylabel("count", fontsize=8)
    fig.suptitle(title, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Simulate (or reuse the cache), train the posterior, and check it."""
    import argparse

    import torch

    from neuromps.metrics import update_metrics

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="few simulations, short training")
    p.add_argument("--simulations", type=int, default=None)
    p.add_argument("--rebuild", action="store_true", help="ignore the simulation cache")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--describe-saved",
        action="store_true",
        help="record the saved posterior's architecture and timing in metrics, nothing else",
    )
    args = p.parse_args(argv)
    if args.describe_saved:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        print(json.dumps(describe_saved_posterior(), indent=2))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    twin_dir().mkdir(parents=True, exist_ok=True)
    # Training a neural posterior estimator is stochastic: weight initialisation,
    # batching and the train/validation split all draw from torch's global RNG.
    # Without this seed, rerunning the same code on the same cache moved the
    # reported contraction by ~0.02 and the rank-uniformity p-value for
    # threshold_scale from 0.46 to 0.01 - enough to flip a stated conclusion.
    torch.manual_seed(args.seed)
    # The quick tier exists to prove the path runs, not to train a usable
    # posterior: on a cold clone this simulation is the single largest item in
    # the whole pipeline, and its contraction numbers are never reported.
    n_simulations = args.simulations or (30 if args.quick else 3000)

    if simulation_cache().exists() and not args.rebuild and cache_is_usable():
        cached = load_simulations(simulation_cache())
        theta, x = cached["theta"], cached["x"]
        inside = within_prior(theta)
        if not inside.all():
            log.info(
                "cache holds %d draws outside the current prior; keeping %d by rejection",
                int((~inside).sum()),
                int(inside.sum()),
            )
            theta, x = theta[inside], x[inside]
        log.info("reusing %d cached simulations", len(theta))
    else:
        theta, x = run_simulations(
            n_simulations, duration_s=5.0 if args.quick else 10.0, seed=args.seed
        )
        save_simulations(simulation_cache(), theta, x, metadata={"n_requested": n_simulations})

    holdout = holdout_size(len(theta))
    train_theta, train_x = theta[:-holdout], x[:-holdout]
    test_theta, test_x = theta[-holdout:], x[-holdout:]
    if args.quick:
        # Coverage checking costs two posterior samples per holdout point, and
        # rejection sampling is slow whenever the estimator puts mass outside the
        # prior. The full run evaluates every holdout point; --quick evaluates a
        # subset to stay inside the run_all budget, and the rank test records how
        # many points it saw (`n_evaluated`).
        test_theta, test_x = test_theta[:60], test_x[:60]
    posterior = train_posterior(train_theta, train_x, max_epochs=5 if args.quick else None)
    torch.save(posterior, twin_dir() / POSTERIOR_NAME)

    n_samples = 100 if args.quick else 200
    ranks = posterior_ranks(posterior, test_theta, test_x, n_samples=n_samples)
    shrink = contraction(posterior, test_theta, test_x, n_samples=n_samples)
    silent = float(np.mean(x[:, SUMMARY_NAMES.index("c0_mfr_hz")] == 0.0))
    results = {
        "label": "SIMULATED - inference trained on simulations only",
        "n_simulations": int(len(theta)),
        # Recorded because 7.3 shows cell density is what sets the twin's
        # synchrony regime: a posterior learned at a different density from the
        # calibration is a posterior for a different model.
        "n_per_compartment": params_from_vector(
            np.zeros(len(PARAMETER_NAMES)), duration_s=1.0
        ).n_per_compartment,
        "n_train": int(len(train_theta)),
        "n_holdout": int(holdout),
        "seed": int(args.seed),
        "posterior_samples": n_samples,
        "fraction_silent_simulations": round(silent, 3),
        "contraction": shrink,
        "rank_uniformity_ks_p": _rank_uniformity(ranks, n_samples),
        "prior_bounds": {k: list(v) for k, v in PRIOR_BOUNDS.items()},
        "summary_names": list(SUMMARY_NAMES),
        "density_estimator": estimator_architecture(posterior),
        "amortized_inference": amortized_timing(posterior, test_x, **_timing_arguments(args.quick)),
    }
    plot_ranks(
        ranks,
        n_samples,
        twin_dir() / "hd14_posterior_ranks_simulated.png",
        "SIMULATED posterior calibration: rank of the true value among posterior samples\n"
        "flat = calibrated, U-shaped = over-confident, humped = too wide",
    )
    update_metrics("twin_hd14_inference", results)
    print(
        json.dumps(
            {
                k: results[k]
                for k in (
                    "n_simulations",
                    "fraction_silent_simulations",
                    "contraction",
                    "rank_uniformity_ks_p",
                )
            },
            indent=2,
        )
    )
    return 0


def _rank_uniformity(ranks: np.ndarray, n_samples: int) -> dict:
    """Departure of the rank distribution from uniform, per parameter.

    Both the KS statistic and its p-value are reported. The **statistic** is what
    to compare across runs: a p-value falls as the holdout grows for the same
    amount of miscalibration, so comparing p-values between a small and a large
    evaluation set reads a power increase as a quality drop.
    """
    from scipy.stats import kstest

    out = {}
    for i, name in enumerate(PARAMETER_NAMES):
        scaled = ranks[:, i] / max(n_samples, 1)
        result = kstest(scaled, "uniform")
        out[name] = {
            "ks_statistic": round(float(result.statistic), 4),
            "ks_p": round(float(result.pvalue), 4),
            "n_evaluated": int(len(scaled)),
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())


def learning_curve(
    theta: np.ndarray,
    x: np.ndarray,
    sizes: tuple[int, ...],
    seed: int = 0,
    n_samples: int = 100,
    holdout: int = 120,
) -> list[dict]:
    """Contraction and calibration against training-set size.

    Distinguishes two explanations for a poorly recovered parameter: too few
    simulations, or summaries that do not carry the information. If the curve
    rises with size, simulate more; if it is flat, the summaries are the limit.
    """
    test_theta, test_x = theta[-holdout:], x[-holdout:]
    pool_theta, pool_x = theta[:-holdout], x[:-holdout]
    rows = []
    for size in sizes:
        if size > len(pool_theta):
            continue
        posterior = train_posterior(pool_theta[:size], pool_x[:size])
        ranks = posterior_ranks(posterior, test_theta, test_x, n_samples=n_samples)
        rows.append(
            {
                "n_train": int(size),
                "contraction": contraction(posterior, test_theta, test_x, n_samples=n_samples),
                "rank_uniformity_ks_p": _rank_uniformity(ranks, n_samples),
            }
        )
        log.info("  learning curve: n=%d done", size)
    return rows
