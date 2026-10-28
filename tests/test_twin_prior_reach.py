"""Tests for the registered prior-reach study of the twin (D38)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from neuromps.twin import prior_reach as pr
from neuromps.twin import recalibration as rc
from neuromps.twin.inference import PARAMETER_NAMES, PRIOR_BOUNDS, SUMMARY_NAMES, draw_parameters

REGISTRATION = Path(__file__).resolve().parents[1] / "preregistration" / "twin_prior_reach.json"
HD14_NETWORK_SEEDS = set(range(0, 4000))  # seeds 0-2999 and seed + 1000 for delays


def _seeds(sim_set: rc.SimulationSet) -> list[int]:
    return [s for k in range(rc.n_chunks(sim_set)) for s in rc.chunk_seeds(sim_set, k)[1]]


# --- the registered box and seeds -----------------------------------------------------


def test_the_new_box_widens_three_parameters_and_keeps_two() -> None:
    changed = {n for n in PARAMETER_NAMES if pr.NEW_BOUNDS[n] != PRIOR_BOUNDS[n]}
    assert changed == {"log10_forward_ratio", "ampa_scale", "threshold_scale"}
    for name in PARAMETER_NAMES:
        low, high = pr.NEW_BOUNDS[name]
        assert low <= PRIOR_BOUNDS[name][0] and high >= PRIOR_BOUNDS[name][1]


@pytest.mark.skipif(not REGISTRATION.exists(), reason="registration not written yet")
def test_constants_match_the_registration() -> None:
    reg = json.loads(REGISTRATION.read_text())
    assert {k: list(v) for k, v in pr.NEW_BOUNDS.items()} == reg["prior"]["new_bounds"]
    seeds = reg["seeds"]
    for sim_set, key in ((pr.HELD_OUT, "held_out"), (pr.TRAIN, "train")):
        assert sim_set.draw_base == seeds[key]["draw_base"]
        assert sim_set.network_base == seeds[key]["network_base"]
        assert sim_set.n == seeds[key]["n"]
    assert seeds["flow_training_seed"] == pr.TRAIN_SEED
    assert seeds["evaluation_sampling_seed"] == pr.EVALUATION_SEED
    assert pr.MAX_PROCESSES <= 12


def test_budget_is_at_most_three_thousand() -> None:
    assert pr.HELD_OUT.n + pr.TRAIN.n == pr.MAX_NEW == 3000


def test_cut_off_is_six_in_the_morning_new_york_time() -> None:
    assert pr.DEADLINE.isoformat() == "2026-10-29T06:00:00-04:00"
    before = pr.DEADLINE - timedelta(seconds=1)
    assert rc.may_start_chunk(before, pr.DEADLINE)
    assert not rc.may_start_chunk(pr.DEADLINE.astimezone(UTC), pr.DEADLINE)


def test_network_seeds_are_disjoint_from_hd14_d31_and_each_other() -> None:
    held, train = _seeds(pr.HELD_OUT), _seeds(pr.TRAIN)
    assert len(set(held)) == pr.HELD_OUT.n and len(set(train)) == pr.TRAIN.n
    mine = set(held) | set(train)
    mine_shifted = mine | {s + 1000 for s in mine}
    d31 = set(_seeds(rc.HELD_OUT)) | set(_seeds(rc.TRAIN))
    d31 |= {s + 1000 for s in d31}
    assert not mine_shifted & HD14_NETWORK_SEEDS
    assert not mine_shifted & d31
    assert not {s + 1000 for s in held} & set(train)
    assert not {s + 1000 for s in train} & set(held)


def test_draw_seeds_differ_from_every_earlier_set() -> None:
    mine = {pr.HELD_OUT.draw_base + k for k in range(rc.n_chunks(pr.HELD_OUT))}
    mine |= {pr.TRAIN.draw_base + k for k in range(rc.n_chunks(pr.TRAIN))}
    d31 = {rc.HELD_OUT.draw_base + k for k in range(rc.n_chunks(rc.HELD_OUT))}
    d31 |= {rc.TRAIN.draw_base + k for k in range(rc.n_chunks(rc.TRAIN))}
    assert len(mine) == rc.n_chunks(pr.HELD_OUT) + rc.n_chunks(pr.TRAIN)
    assert 0 not in mine and not mine & d31


def test_chunk_draws_come_from_the_new_box_and_are_reproducible() -> None:
    small = rc.SimulationSet("t", draw_base=77, network_base=10_000, n=6)
    theta_a, seeds = pr.chunk_parameters(small, 0, chunk_size=4)
    theta_b, _ = pr.chunk_parameters(small, 0, chunk_size=4)
    np.testing.assert_array_equal(theta_a, theta_b)
    np.testing.assert_array_equal(theta_a, draw_parameters(4, 77, pr.NEW_BOUNDS))
    assert seeds == [10_000, 10_001, 10_002, 10_003]
    assert pr.within_box(theta_a, pr.NEW_BOUNDS).all()


def test_within_box() -> None:
    inside = np.array([[0.0, 2.0, 0.1, 0.4, 0.85]])
    outside = np.array([[-0.1, 2.0, 0.1, 0.4, 0.85]])
    assert pr.within_box(inside, PRIOR_BOUNDS).all()
    assert not pr.within_box(outside, PRIOR_BOUNDS).any()
    assert pr.within_box(outside, pr.NEW_BOUNDS).all()


# --- chunks on disk -------------------------------------------------------------------


def _fake_job(args):
    return np.zeros(len(SUMMARY_NAMES))


def test_a_chunk_records_its_box_and_reloads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr, "_simulate_job", _fake_job)
    small = rc.SimulationSet("t", draw_base=11, network_base=500, n=3)
    path = pr.simulate_chunk(small, 0, processes=1, root=tmp_path)
    assert "simulated" in path.name and not list(tmp_path.glob("*.partial.npz"))
    loaded = pr.load_set(small, root=tmp_path)
    assert loaded["chunks"] == 1 and len(loaded["theta"]) == 3
    np.testing.assert_array_equal(loaded["theta"], draw_parameters(3, 11, pr.NEW_BOUNDS))


def test_a_chunk_from_another_box_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr, "_simulate_job", _fake_job)
    monkeypatch.setattr(pr, "NEW_BOUNDS", dict(PRIOR_BOUNDS))
    small = rc.SimulationSet("t", draw_base=11, network_base=500, n=3)
    pr.simulate_chunk(small, 0, processes=1, root=tmp_path)
    monkeypatch.undo()
    with pytest.raises(SystemExit, match="prior box"):
        pr.load_set(small, root=tmp_path)


def test_simulate_set_stops_at_the_deadline(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pr, "_simulate_job", _fake_job)
    small = rc.SimulationSet("t", draw_base=11, network_base=500, n=3)
    past = datetime(2000, 1, 1, tzinfo=UTC)
    assert pr.simulate_set(small, processes=1, deadline=past, root=tmp_path) == []
    assert pr.simulate_set(small, processes=1, deadline=None, root=tmp_path) == [0]


# --- distances ------------------------------------------------------------------------


def test_scale_comes_from_the_reference_only() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(5.0, 2.0, size=(400, 3))
    reference[:, 2] = 1.0  # a constant column keeps unit scale, as in D31
    mean, sd = pr.reference_scale(reference)
    np.testing.assert_allclose(mean, reference.mean(axis=0))
    assert sd[2] == 1.0
    values = rng.normal(size=(5, 3))
    np.testing.assert_allclose(pr.zscore(values, (mean, sd)), (values - mean) / sd)


def test_matches_the_d31_standardisation() -> None:
    from neuromps.twin.recalibration_run import _zscore

    rng = np.random.default_rng(1)
    reference = rng.normal(size=(50, 4))
    values = rng.normal(size=(7, 4))
    scale = pr.reference_scale(reference)
    np.testing.assert_allclose(pr.zscore(values, scale), _zscore(values, reference))


def test_nearest_distance() -> None:
    pool = np.array([[0.0, 0.0], [3.0, 4.0]])
    query = np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(pr.nearest_distance(query, pool), [1.0, 0.0, 5.0])


def test_distance_shares_sum_to_one_and_name_the_gap() -> None:
    pool = np.zeros((4, 3))
    query = np.array([[0.0, 3.0, 0.0], [0.0, 4.0, 1.0]])
    shares = pr.distance_shares(query, pool, names=("a", "b", "c"))
    assert sum(shares["share_of_total"].values()) == pytest.approx(1.0)
    assert max(shares["share_of_total"], key=shares["share_of_total"].get) == "b"


def test_feature_positions_flag_values_outside_the_pool() -> None:
    pool = np.tile(np.arange(100.0)[:, None], (1, 2))
    real = np.array([[50.0, 500.0], [50.0, 600.0]])
    out = pr.feature_positions(real, pool, pr.reference_scale(pool), names=("a", "b"))
    assert out["a"]["share_above_q99"] == 0.0
    assert out["b"]["share_above_q99"] == 1.0


# --- predictions ----------------------------------------------------------------------


def test_r1_passes_at_exactly_half_and_fails_above() -> None:
    assert pr.score_r1(1.0, 2.0, n_new=3000, reproduced=True)["outcome"] == "pass"
    assert pr.score_r1(1.01, 2.0, n_new=3000, reproduced=True)["outcome"] == "fail"


def test_r1_is_uninterpretable_without_budget_or_reproduction() -> None:
    assert pr.score_r1(0.1, 2.0, n_new=599, reproduced=True)["outcome"] == "uninterpretable"
    assert pr.score_r1(0.1, 2.0, n_new=3000, reproduced=False)["outcome"] == "uninterpretable"


def test_r2_share_is_high_for_the_same_pool_and_zero_for_a_distant_one() -> None:
    rng = np.random.default_rng(2)
    hd14 = rng.normal(size=(1000, 3))
    same = pr.old_region_share(hd14, rng.normal(size=(900, 3)), n_held=100)
    far = pr.old_region_share(hd14, rng.normal(size=(900, 3)) + 50.0, n_held=100)
    assert same["share"] > 0.6 and far["share"] == 0.0
    assert same["n_reference"] == 900


def test_r2_rule() -> None:
    assert pr.score_r2(0.5, n_new=3000, reproduced=True)["outcome"] == "pass"
    assert pr.score_r2(0.49, n_new=3000, reproduced=True)["outcome"] == "fail"
    assert pr.score_r2(0.9, n_new=100, reproduced=True)["outcome"] == "uninterpretable"


def test_r3_rule() -> None:
    assert pr.score_r3(0.05, n_held=600, n_train=2400)["outcome"] == "pass"
    assert pr.score_r3(0.049, n_held=600, n_train=2400)["outcome"] == "fail"
    assert pr.score_r3(0.9, n_held=599, n_train=2400)["outcome"] == "uninterpretable"
    assert pr.score_r3(0.9, n_held=600, n_train=999)["outcome"] == "uninterpretable"


def test_equal_size_comparison_takes_a_prefix_of_hd14() -> None:
    x = np.arange(30.0).reshape(10, 3)
    np.testing.assert_array_equal(pr.equal_size(x, 4), x[:4])
    np.testing.assert_array_equal(pr.equal_size(x, 40), x)


# --- edges of the box -----------------------------------------------------------------


class _FixedPosterior:
    """Returns the same raw samples for every observation."""

    def __init__(self, samples: np.ndarray) -> None:
        self.samples = samples

    def sample(self, shape, x=None, show_progress_bars=False, reject_outside_prior=True):
        import torch

        return torch.as_tensor(self.samples[: shape[0]], dtype=torch.float32)


def test_edge_mass_counts_draws_beyond_each_bound() -> None:
    centre = [np.mean(PRIOR_BOUNDS[n]) for n in PARAMETER_NAMES]
    samples = np.tile(centre, (10, 1))
    samples[:4, 4] = 2.0  # threshold_scale above the box in 4 of 10 draws
    samples[:1, 0] = -1.0  # forward ratio below in 1 of 10
    out = pr.edge_mass(_FixedPosterior(samples), np.zeros((3, 19)), PRIOR_BOUNDS, n_draws=10)
    assert out["by_parameter"]["threshold_scale"]["median_share_above"] == pytest.approx(0.4)
    assert out["by_parameter"]["log10_forward_ratio"]["median_share_below"] == pytest.approx(0.1)
    assert out["median_acceptance"] == pytest.approx(0.6)


# --- the run module -------------------------------------------------------------------


def _pools(rng, shift: float = 0.0) -> tuple[dict, dict]:
    reference = {"x": rng.normal(size=(400, 4))}
    x = rng.normal(size=(300, 4)) + shift
    new = {"x": x, "held": {"x": x[:100], "failed": 0, "chunks": 1}}
    new["train"] = {"x": x[100:], "failed": 1, "chunks": 2}
    return reference, new


def test_scale_check_is_near_one_for_the_same_distribution() -> None:
    from neuromps.twin.prior_reach_run import scale_check

    rng = np.random.default_rng(3)
    reference, new = _pools(rng)
    out = scale_check(reference, new, rng.normal(size=(30, 4)))
    assert 0.6 < out["new_pool_scale"] < 1.6 and 0.6 < out["pooled_scale"] < 1.6
    assert out["all_at_most_threshold"] is False


def test_budget_counts_failures_and_the_cut() -> None:
    from neuromps.twin.prior_reach_run import budget

    _, new = _pools(np.random.default_rng(4))
    out = budget(new)
    assert out["new_simulations_usable"] == 300 and out["new_simulations_finished"] == 301
    assert out["truncated_by_cut_off"] is True and out["held_out_usable"] == 100


def test_rescore_reproduces_the_recorded_outcomes() -> None:
    from neuromps.twin.prior_reach_run import rescore_section

    section = {
        "assumption_checks": {"A1_hd14_cache_reproduced": {"passed": True}},
        "budget": {"new_simulations_usable": 3000, "held_out_usable": 600},
        "reach": {"R1": {"new_pool_median": 1.0, "hd14_pool_median": 2.5}},
        "old_region": {"share": 0.4},
        "calibration": {"n_train": 2400, "near_real": {"ks": {"threshold_scale": {"ks_p": 0.2}}}},
    }
    got = rescore_section(section)["predictions"]
    assert [got[k]["outcome"] for k in sorted(got)] == ["pass", "fail", "pass"]


def test_reading_follows_the_registered_rule() -> None:
    from neuromps.twin.prior_reach_run import reading

    fail, ok = {"outcome": "fail"}, {"outcome": "pass"}
    assert reading(ok, ok, ok).startswith("prior range")
    assert reading(fail, ok, ok).startswith("misspecification")
    assert reading({"outcome": "uninterpretable"}, ok, ok).startswith("uninterpretable")
