"""Recalibrating the twin's inverse model: a larger budget, then an ensemble (D31).

HD14's posterior (`neuromps.twin.inference`, metrics `twin_hd14_inference`)
fails the rank-uniformity check for two of its five parameters. This module
runs the registered test of two remedies, as registered in
`preregistration/twin_inference_recalibration.json`:

- **M1**, the same flow retrained on HD14's simulations plus up to 15,000 fresh
  ones (budget alone);
- **M2**, an equal-weight ensemble of five such flows (the estimator the slide
  adopts if it passes);
- **M3**, the M2 recipe on HD14's own training set (exploratory).

All are judged with HD14's untouched posterior (**M0**) on 1,000 fresh held-out
simulations. Everything that defines a simulation - prior, simulator, summaries,
the cache fingerprints - is imported from `neuromps.twin.inference` unchanged;
this module only decides *which* seeds are simulated and how the results are
judged. All of it is simulated data.
"""

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from neuromps.paths import interim_dir
from neuromps.twin.inference import (
    PARAMETER_NAMES,
    PRIOR_BOUNDS,
    SUMMARY_NAMES,
    _simulate_job,
    check_cache_matches_code,
    draw_parameters,
    load_simulations,
    save_simulations,
    train_posterior,
)

log = logging.getLogger("neuromps.twin.recalibration")

CHUNK_SIZE = 250
SIMULATED_S = 10.0
METRICS_KEY = "twin_hd14b_recalibration"
REGISTRATION = "preregistration/twin_inference_recalibration.json"
# Registered: no training chunk starts after this moment (a running one finishes).
DEADLINE = datetime(2026, 10, 28, 5, 0, tzinfo=ZoneInfo("America/New_York"))
MAX_PROCESSES = 5
M1_SEED = 3101
M2_SEEDS = (3111, 3112, 3113, 3114, 3115)
M3_SEEDS = (3121, 3122, 3123, 3124, 3125)
EVALUATION_SEED = 3200
COVERAGE_LEVELS = (0.5, 0.8, 0.95)
ALPHA = 0.05
# Registered (amended before any simulation): 2 SE flagged exactly calibrated ranks in a
# quarter of held-out sets across the 15 parameter-level cells; 3 SE does in about 2%.
COVERAGE_SE = 3.0
# rank/200 lands exactly on an interval edge; without this, rounding of (1 - 0.95)/2
# drops rank 5 from the 95% interval and biases that level low.
EDGE_TOLERANCE = 1e-9
N_POSTERIOR_SAMPLES = 200
PRIMARY_N = 300
NEAR_REAL_N = 300
CONTRACTION_MARGIN = 0.05
TIMING_LIMIT_S = 1.0


@dataclass(frozen=True)
class SimulationSet:
    """A registered set of simulations: its seeds and its size."""

    name: str
    draw_base: int
    network_base: int
    n: int


HELD_OUT = SimulationSet("heldout", draw_base=32000, network_base=32_000_000, n=1000)
TRAIN = SimulationSet("train", draw_base=31000, network_base=31_000_000, n=15000)


# --- which simulations ------------------------------------------------------------


def n_chunks(sim_set: SimulationSet, chunk_size: int = CHUNK_SIZE) -> int:
    """Number of chunks the set is simulated in."""
    return math.ceil(sim_set.n / chunk_size)


def chunk_seeds(
    sim_set: SimulationSet, k: int, chunk_size: int = CHUNK_SIZE
) -> tuple[int, list[int]]:
    """Prior-draw seed and per-row network seeds of chunk ``k``."""
    start = k * chunk_size
    size = min(chunk_size, sim_set.n - start)
    return sim_set.draw_base + k, [sim_set.network_base + start + j for j in range(size)]


def chunk_parameters(
    sim_set: SimulationSet, k: int, chunk_size: int = CHUNK_SIZE
) -> tuple[np.ndarray, list[int]]:
    """Prior draws and network seeds of chunk ``k``, through HD14's own `draw_parameters`."""
    draw_seed, seeds = chunk_seeds(sim_set, k, chunk_size)
    return draw_parameters(len(seeds), draw_seed), seeds


def chunk_path(sim_set: SimulationSet, k: int, root: Path | None = None) -> Path:
    """Where chunk ``k`` of a set is cached (git-ignored, under data/interim)."""
    root = Path(root) if root is not None else interim_dir() / "recalibration"
    return root / f"{sim_set.name}_chunk_{k:03d}_simulated.npz"


def completed_prefix(sim_set: SimulationSet, root: Path | None = None) -> list[int]:
    """Indices of the chunks on disk, up to the first missing one.

    Training always uses a prefix of the registered chunk sequence, so where the
    run was cut cannot depend on what any chunk contains.
    """
    done = []
    for k in range(n_chunks(sim_set)):
        if not chunk_path(sim_set, k, root).exists():
            break
        done.append(k)
    return done


def may_start_chunk(now: datetime, deadline: datetime | None) -> bool:
    """The registered truncation rule: no chunk starts at or after the deadline."""
    return deadline is None or now < deadline


def simulate_chunk(sim_set: SimulationSet, k: int, processes: int, root: Path | None = None):
    """Simulate one chunk in parallel and cache it with HD14's fingerprints."""
    import multiprocessing as mp

    theta, seeds = chunk_parameters(sim_set, k)
    jobs = [(theta[i], None, SIMULATED_S, seed) for i, seed in enumerate(seeds)]
    with mp.Pool(processes) as pool:
        x = np.vstack(pool.map(_simulate_job, jobs, chunksize=1))
    draw_seed, _ = chunk_seeds(sim_set, k)
    metadata = {
        "set": sim_set.name,
        "chunk": k,
        "draw_seed": draw_seed,
        "network_seeds": [seeds[0], seeds[-1]],
        "n_requested": len(seeds),
        "duration_s": SIMULATED_S,
        "label": "SIMULATED",
    }
    path = chunk_path(sim_set, k, root)
    # Written under another name and renamed, so a chunk is on disk complete or
    # not at all: `completed_prefix` counts files, and a reader must never see a
    # half-written one.
    partial = path.with_name(path.stem + ".partial.npz")
    save_simulations(partial, theta, x, metadata)
    partial.replace(path)
    return path


def simulate_set(
    sim_set: SimulationSet,
    processes: int = MAX_PROCESSES,
    deadline: datetime | None = None,
    root: Path | None = None,
    max_chunks: int | None = None,
) -> list[int]:
    """Simulate every missing chunk in order, starting none after ``deadline``.

    ``max_chunks`` reproduces a recorded run: a rerun from a clean clone
    simulates exactly the chunks the registered run completed, no more.
    """
    for k in range(min(n_chunks(sim_set), max_chunks or n_chunks(sim_set))):
        if chunk_path(sim_set, k, root).exists():
            continue
        now = datetime.now(tz=ZoneInfo("America/New_York"))
        if not may_start_chunk(now, deadline):
            log.info("%s: deadline reached before chunk %d; stopping", sim_set.name, k)
            break
        log.info("%s: simulating chunk %d of %d", sim_set.name, k + 1, n_chunks(sim_set))
        simulate_chunk(sim_set, k, processes, root)
    return completed_prefix(sim_set, root)


def load_set(sim_set: SimulationSet, root: Path | None = None) -> dict:
    """The completed prefix of a set, checked against the current code, failures dropped."""
    thetas, xs = [], []
    chunks = completed_prefix(sim_set, root)
    for k in chunks:
        cached = load_simulations(chunk_path(sim_set, k, root))
        check_cache_matches_code(cached)
        thetas.append(cached["theta"])
        xs.append(cached["x"])
    if not chunks:
        empty = np.empty((0, len(PARAMETER_NAMES)))
        return {"theta": empty, "x": np.empty((0, len(SUMMARY_NAMES))), "chunks": 0, "failed": 0}
    theta, x = np.vstack(thetas), np.vstack(xs)
    finite = np.isfinite(x).all(axis=1)
    return {
        "theta": theta[finite],
        "x": x[finite],
        "chunks": len(chunks),
        "failed": int((~finite).sum()),
    }


# --- calibration statistics -------------------------------------------------------


def coverage_floor(level: float, n: int) -> float:
    """Lowest coverage not flagged as over-confident: three binomial standard errors."""
    return level - COVERAGE_SE * math.sqrt(level * (1 - level) / n)


def coverage_from_ranks(
    ranks: np.ndarray, n_samples: int, levels: tuple[float, ...] = COVERAGE_LEVELS
) -> dict:
    """Coverage of central marginal credible intervals, read off SBC ranks.

    The truth lies inside the central interval at level L when its rank among
    the posterior samples sits between the (1 - L)/2 and (1 + L)/2 quantiles.
    Only coverage *below* nominal is flagged: that is over-confidence; a too-wide
    posterior is reported but is not what the check is for.
    """
    scaled = np.asarray(ranks, dtype=float) / n_samples
    n = len(scaled)
    out = {}
    for i, name in enumerate(PARAMETER_NAMES[: scaled.shape[1]]):
        out[name] = {}
        for level in levels:
            low = (1 - level) / 2 - EDGE_TOLERANCE
            high = (1 + level) / 2 + EDGE_TOLERANCE
            inside = (scaled[:, i] >= low) & (scaled[:, i] <= high)
            coverage = float(inside.mean())
            floor = coverage_floor(level, n)
            out[name][level_key(level)] = {
                "coverage": round(coverage, 4),
                "floor": round(floor, 4),
                "over_confident": bool(coverage < floor),
                "n": int(n),
            }
    return out


def holm_adjust(p_values: dict) -> dict:
    """Holm step-down adjusted p values (reported beside the registered rule, not used)."""
    names = sorted(p_values, key=p_values.get)
    m = len(names)
    adjusted, running = {}, 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i) * p_values[name]))
        adjusted[name] = round(running, 4)
    return adjusted


def level_key(level: float) -> str:
    """Dot-free key for a credible level, so report citations can path into it."""
    return f"L{round(level * 100)}"


def ranks_and_widths(
    posterior, theta: np.ndarray, x: np.ndarray, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per observation: SBC rank of each true value and the posterior's standard deviation.

    The definitions of `inference.posterior_ranks` and `inference.contraction`
    (a test holds them equal), drawn once instead of twice, and kept per
    observation so any subset can be summarised without sampling again.
    """
    import torch

    ranks = np.empty((len(theta), theta.shape[1]), dtype=int)
    widths = np.empty((len(theta), theta.shape[1]), dtype=float)
    for i, (true_value, observation) in enumerate(zip(theta, x, strict=True)):
        samples = posterior.sample(
            (n_samples,),
            x=torch.as_tensor(observation, dtype=torch.float32),
            show_progress_bars=False,
        ).numpy()
        ranks[i] = (samples < true_value).sum(axis=0)
        widths[i] = samples.std(axis=0)
    return ranks, widths


def ranks_and_contraction(posterior, theta: np.ndarray, x: np.ndarray, n_samples: int):
    """SBC ranks and the contraction summary, from one sample per observation."""
    ranks, widths = ranks_and_widths(posterior, theta, x, n_samples)
    return ranks, contraction_from_widths(widths.mean(axis=0))


def contraction_from_widths(posterior_sd: np.ndarray) -> dict:
    """`inference.contraction`'s summary of mean posterior widths."""
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


# --- estimators -------------------------------------------------------------------


def ensemble_posterior(members: list):
    """Equal-weight mixture of trained posteriors (sbi's EnsemblePosterior)."""
    from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior

    return EnsemblePosterior(list(members))


def train_members(
    theta: np.ndarray, x: np.ndarray, seeds: tuple[int, ...], max_epochs: int | None = None
) -> list:
    """One HD14-recipe flow per registered seed, each trained on the same pool."""
    import torch

    members = []
    for seed in seeds:
        torch.manual_seed(seed)
        members.append(train_posterior(theta, x, max_epochs=max_epochs))
        log.info("trained ensemble member with seed %d", seed)
    return members


# --- near the real recordings -----------------------------------------------------


def nearest_to_real(
    pool_x: np.ndarray, real_x: np.ndarray, scale_from: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` pool simulations nearest to any real recording, by summaries only.

    Distances are Euclidean on summaries z-scored with ``scale_from``'s mean and
    standard deviation. Returns the selected pool indices and, per real
    recording, the distance to its nearest pool simulation.
    """
    mean = scale_from.mean(axis=0)
    sd = scale_from.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    pool_z = (pool_x - mean) / sd
    real_z = (real_x - mean) / sd
    distance = np.linalg.norm(pool_z[:, None, :] - real_z[None, :, :], axis=2)
    nearest_real = distance.min(axis=1)
    selected = np.argsort(nearest_real, kind="stable")[:k]
    return selected, distance.min(axis=0)


# --- command line -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Simulate the registered sets, or train and evaluate on what is cached."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="token run on the quick cache")
    p.add_argument(
        "--simulate",
        choices=["heldout", "train", "all"],
        default=None,
        help="simulate the registered sets (held-out first), then stop",
    )
    p.add_argument("--processes", type=int, default=MAX_PROCESSES)
    p.add_argument("--no-deadline", action="store_true", help="ignore the registered cut-off")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.simulate:
        deadline = None if args.no_deadline else DEADLINE
        if args.simulate in ("heldout", "all"):
            log.info("held-out chunks done: %s", simulate_set(HELD_OUT, args.processes))
        if args.simulate in ("train", "all"):
            done = simulate_set(TRAIN, args.processes, deadline)
            log.info("training chunks done: %d of %d", len(done), n_chunks(TRAIN))
        return 0
    if not args.quick:
        _reproduce_simulations(args.processes)
    from neuromps.twin.recalibration_run import run

    print(json.dumps(run(quick=args.quick), indent=2, default=str))
    return 0


def recorded_train_chunks() -> int | None:
    """How many training chunks the recorded run used, if there is a record."""
    from neuromps.paths import metrics_path

    if not metrics_path().exists():
        return None
    section = json.loads(metrics_path().read_text()).get(METRICS_KEY) or {}
    return (section.get("budget") or {}).get("new_train_chunks_completed")


def _reproduce_simulations(processes: int) -> None:
    """On a full run, simulate whatever the recorded run had and this clone lacks.

    With a record, exactly its chunks; without one, the registered rule (no
    chunk after the deadline). On the machine that holds the cache this is a
    no-op; on a clean clone it is several hours of simulation.
    """
    recorded = recorded_train_chunks()
    simulate_set(HELD_OUT, processes)
    deadline = None if recorded else DEADLINE
    simulate_set(TRAIN, processes, deadline=deadline, max_chunks=recorded)


if __name__ == "__main__":
    raise SystemExit(main())
