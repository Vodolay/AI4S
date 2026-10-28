"""Tests for the registered recalibration of the twin's inverse model (D31)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from neuromps.twin import recalibration as rc
from neuromps.twin.inference import PARAMETER_NAMES, PRIOR_BOUNDS, draw_parameters

HD14_NETWORK_SEEDS = set(range(0, 3000)) | set(range(1000, 4000))  # seed and seed + 1000


def _all_network_seeds(sim_set: rc.SimulationSet) -> list[int]:
    return [s for k in range(rc.n_chunks(sim_set)) for s in rc.chunk_seeds(sim_set, k)[1]]


# --- seeding -----------------------------------------------------------------------


def test_registered_sets_match_the_registration() -> None:
    assert (rc.HELD_OUT.draw_base, rc.HELD_OUT.network_base, rc.HELD_OUT.n) == (
        32000,
        32_000_000,
        1000,
    )
    assert (rc.TRAIN.draw_base, rc.TRAIN.network_base, rc.TRAIN.n) == (31000, 31_000_000, 15000)
    assert rc.CHUNK_SIZE == 250


def test_network_seeds_are_unique_and_disjoint_across_sets_and_from_hd14() -> None:
    held = _all_network_seeds(rc.HELD_OUT)
    train = _all_network_seeds(rc.TRAIN)
    assert len(held) == len(set(held)) == rc.HELD_OUT.n
    assert len(train) == len(set(train)) == rc.TRAIN.n
    # The network also uses seed + 1000 for its delays, so ranges must not touch.
    shifted = {s + 1000 for s in held + train}
    assert not (set(held) | set(train) | shifted) & HD14_NETWORK_SEEDS
    assert not set(held) & set(train)
    assert not {s + 1000 for s in held} & set(train)


def test_draw_seeds_differ_from_hd14_and_between_sets() -> None:
    held = {rc.chunk_seeds(rc.HELD_OUT, k)[0] for k in range(rc.n_chunks(rc.HELD_OUT))}
    train = {rc.chunk_seeds(rc.TRAIN, k)[0] for k in range(rc.n_chunks(rc.TRAIN))}
    assert 0 not in held | train
    assert not held & train


def test_chunk_parameters_come_from_the_prior_and_are_reproducible() -> None:
    small = rc.SimulationSet("t", draw_base=77, network_base=10_000, n=6)
    theta_a, seeds_a = rc.chunk_parameters(small, 0, chunk_size=4)
    theta_b, _ = rc.chunk_parameters(small, 0, chunk_size=4)
    np.testing.assert_array_equal(theta_a, theta_b)
    np.testing.assert_array_equal(theta_a, draw_parameters(4, 77))
    assert seeds_a == [10_000, 10_001, 10_002, 10_003]
    theta_last, seeds_last = rc.chunk_parameters(small, 1, chunk_size=4)
    assert len(theta_last) == 2 and seeds_last == [10_004, 10_005]
    for i, name in enumerate(PARAMETER_NAMES):
        low, high = PRIOR_BOUNDS[name]
        assert np.all((theta_a[:, i] >= low) & (theta_a[:, i] <= high))


def test_no_chunk_starts_after_the_deadline() -> None:
    deadline = datetime(2026, 10, 28, 5, 0, tzinfo=timezone(timedelta(hours=-4)))
    assert rc.may_start_chunk(deadline - timedelta(minutes=1), deadline)
    assert not rc.may_start_chunk(deadline, deadline)
    assert rc.may_start_chunk(deadline + timedelta(days=9), None)


def test_completed_prefix_stops_at_the_first_gap(tmp_path) -> None:
    small = rc.SimulationSet("t", draw_base=1, network_base=1, n=1000)
    for k in (0, 1, 3):
        rc.chunk_path(small, k, root=tmp_path).parent.mkdir(parents=True, exist_ok=True)
        rc.chunk_path(small, k, root=tmp_path).write_bytes(b"")
    assert rc.completed_prefix(small, root=tmp_path) == [0, 1]


# --- coverage ----------------------------------------------------------------------


def test_coverage_from_ranks_counts_the_central_interval() -> None:
    ranks = np.array([[0], [10], [50], [100], [150], [190], [200]])
    out = rc.coverage_from_ranks(ranks, n_samples=200, levels=(0.5, 0.95))
    # central 50%: 50 <= r <= 150 -> ranks 50, 100, 150
    assert out[PARAMETER_NAMES[0]]["0.5"]["coverage"] == pytest.approx(3 / 7, abs=1e-4)
    # central 95%: 5 <= r <= 195 -> all but 0 and 200
    assert out[PARAMETER_NAMES[0]]["0.95"]["coverage"] == pytest.approx(5 / 7, abs=1e-4)


def test_coverage_flags_over_confidence_only() -> None:
    n = 1000
    narrow = np.zeros((n, 1), dtype=int)  # every truth outside: over-confident
    wide = np.full((n, 1), 100)  # every truth at the median: too wide, not a failure
    assert rc.coverage_from_ranks(narrow, 200, (0.8,))[PARAMETER_NAMES[0]]["0.8"]["over_confident"]
    assert not rc.coverage_from_ranks(wide, 200, (0.8,))[PARAMETER_NAMES[0]]["0.8"][
        "over_confident"
    ]


def test_uniform_ranks_are_not_flagged() -> None:
    rng = np.random.default_rng(0)
    ranks = rng.integers(0, 201, size=(1000, len(PARAMETER_NAMES)))
    out = rc.coverage_from_ranks(ranks, 200, rc.COVERAGE_LEVELS)
    assert not any(cell["over_confident"] for p in out.values() for cell in p.values())
    for p in out.values():
        assert p["0.8"]["coverage"] == pytest.approx(0.8, abs=0.04)


def test_coverage_tolerance_is_three_binomial_standard_errors() -> None:
    assert rc.coverage_floor(0.8, 400) == pytest.approx(0.8 - 3 * np.sqrt(0.8 * 0.2 / 400))


def test_interval_edges_survive_floating_point_rounding() -> None:
    """(1 - 0.95)/2 rounds above 0.025; rank 5 of 200 must still count as inside."""
    ranks = np.array([[5], [195], [4], [196]])
    out = rc.coverage_from_ranks(ranks, 200, (0.95,))
    assert out[PARAMETER_NAMES[0]]["0.95"]["coverage"] == pytest.approx(0.5)


def test_holm_adjustment() -> None:
    adjusted = rc.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


# --- ranks and contraction from one set of samples --------------------------------


class _FixedPosterior:
    """A stand-in posterior that returns the same samples for every observation."""

    def __init__(self, samples: np.ndarray) -> None:
        self.samples = samples

    def sample(self, shape, x=None, show_progress_bars=False):
        import torch

        return torch.as_tensor(self.samples[: shape[0]], dtype=torch.float32)


def test_one_pass_matches_the_hd14_rank_and_contraction_functions() -> None:
    from neuromps.twin.inference import contraction, posterior_ranks

    rng = np.random.default_rng(1)
    low = np.array([PRIOR_BOUNDS[n][0] for n in PARAMETER_NAMES])
    high = np.array([PRIOR_BOUNDS[n][1] for n in PARAMETER_NAMES])
    samples = rng.uniform(low, high, size=(50, 5)).astype(np.float32)
    theta = rng.uniform(low, high, size=(8, 5)).astype(np.float32)
    x = rng.normal(size=(8, 19))
    posterior = _FixedPosterior(samples)
    ranks, shrink = rc.ranks_and_contraction(posterior, theta, x, n_samples=50)
    np.testing.assert_array_equal(ranks, posterior_ranks(posterior, theta, x, n_samples=50))
    assert shrink == contraction(posterior, theta, x, n_samples=50)


# --- ensemble ---------------------------------------------------------------------


def test_ensemble_samples_mix_every_member() -> None:
    """Every member contributes about an equal share of the ensemble's draws."""
    import torch

    from neuromps.twin.inference import SUMMARY_NAMES

    rng = np.random.default_rng(0)
    theta = draw_parameters(80, 5)
    x = rng.normal(size=(80, len(SUMMARY_NAMES)))
    members = rc.train_members(theta, x, seeds=(1, 2, 3), max_epochs=1)
    drawn = []
    for i, member in enumerate(members):
        original = member.sample

        def counting(shape, *args, _i=i, _original=original, **kwargs):
            drawn.append((_i, int(torch.Size(shape).numel())))
            return _original(shape, *args, **kwargs)

        member.sample = counting
    torch.manual_seed(0)
    ensemble = rc.ensemble_posterior(members)
    draws = ensemble.sample((600,), x=torch.zeros(len(SUMMARY_NAMES)), show_progress_bars=False)
    assert tuple(draws.shape) == (600, 5)
    per_member = {i: sum(n for j, n in drawn if j == i) for i in range(3)}
    assert sum(per_member.values()) == 600
    assert all(140 < n < 260 for n in per_member.values())


def test_ensemble_training_uses_each_registered_seed(monkeypatch) -> None:
    seen = []

    def fake_train(theta, x, max_epochs=None):
        import torch

        seen.append(int(torch.initial_seed()))
        return _FixedPosterior(np.zeros((10, 5), dtype=np.float32))

    monkeypatch.setattr(rc, "train_posterior", fake_train)
    members = rc.train_members(np.zeros((4, 5)), np.zeros((4, 19)), seeds=(5, 6, 7))
    assert seen == [5, 6, 7] and len(members) == 3


# --- near-real subset -------------------------------------------------------------


def test_near_real_subset_selects_by_summary_distance() -> None:
    pool = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [1.0, 1.0], [9.0, 9.0]])
    real = np.array([[0.2, 0.1], [9.5, 9.5]])
    index, distance = rc.nearest_to_real(pool, real, scale_from=pool, k=3)
    assert set(index) == {0, 3, 4}
    assert distance.shape == (2,)
    assert np.all(distance >= 0)
