"""Tests for the registered settling-versus-steady-noise analysis (synthetic inputs only)."""

import json

import numpy as np
import pandas as pd
import pytest

from neuromps.features.mea import extract_features
from neuromps.models import hpsc_settling as st
from neuromps.models import hpsc_split_half as sh

# ------------------------------------------------------------------ registration


def test_constants_match_the_registration():
    reg = json.loads(st.REGISTRATION.read_text())
    method = reg["method"]
    assert f"{st.N_WINDOWS} consecutive windows" in method["windows"]
    assert f"{st.RESAMPLES:,} resamples" in method["bootstrap"]
    assert str(st.SEED) in method["bootstrap"]
    assert f"fewer than {st.MIN_WELLS} wells" in reg["uninterpretable"]
    assert "at least 0.5" in reg["predictions"]["T3_settling_carries_the_human_spread"]["test"]
    assert st.T3_SHARE == 0.5
    assert "more than a third" in reg["assumption_check"]["check"] and st.QUAD_DROP == 1 / 3
    assert "exceeds one half" in reg["assumption_check"]["check"] and st.QUAD_MAJORITY == 0.5
    assert reg["outputs"]["metrics_key"] == st.ROOT
    assert reg["outputs"]["results_file"].startswith(f"outputs/{st.RESULTS_NAME}")
    assert reg["outputs"]["figure"] == f"outputs/figures/{st.FIGURE_NAME}"
    assert set(reg["predictions"]) == set(st.PREDICTIONS)


def test_reuses_the_split_half_pool_and_seed():
    assert st.SEED == sh.SEED and st.RESAMPLES == sh.RESAMPLES
    assert st.SPECIES == sh.SPECIES


# ------------------------------------------------------------------ electrodes and windows


def _channels(rng, n=12, duration=60.0, rate=5.0, quiet=2):
    out = {f"A1_{i}": np.sort(rng.uniform(0, duration, int(rate * duration))) for i in range(n)}
    for j in range(quiet):  # below the active threshold over the whole recording
        out[f"A1_q{j}"] = np.array([1.0, 2.0])
    return out


def test_electrode_set_is_the_extractors_active_set_over_the_whole_recording():
    rng = np.random.default_rng(1)
    channels = _channels(rng)
    got = st.electrode_set(channels, 60.0)
    assert len(got) == extract_features(channels, 60.0)["active_electrodes"] == 12
    assert not any(name.startswith("A1_q") for name in got)
    assert got == sorted(got)


def test_window_counts_cut_into_equal_windows_with_the_boundary_starting_the_next():
    channels = {"e": np.array([0.0, 9.99, 10.0, 25.0, 59.99, 60.0])}
    counts = st.window_counts(channels, ["e"], 60.0)
    assert counts.shape == (1, st.N_WINDOWS)
    # Window length 10 s: a spike at 10.0 starts window 1; the last spike (60.0) ends window 5.
    np.testing.assert_array_equal(counts[0], [2, 1, 1, 0, 0, 2])


def test_window_counts_conserve_every_spike_of_the_set():
    rng = np.random.default_rng(2)
    channels = _channels(rng, n=5, duration=90.0)
    names = st.electrode_set(channels, 90.0)
    counts = st.window_counts(channels, names, 90.0)
    assert counts.sum() == sum(channels[n].size for n in names)


def test_log_rate_is_the_mean_electrode_rate_with_a_one_spike_floor():
    counts = np.array([[10, 0, 4, 2, 2, 2], [30, 0, 0, 2, 2, 2]])
    y = st.log_rates(counts, window_s=5.0)
    assert y[0] == pytest.approx(np.log2((10 + 30) / 2 / 5.0))
    # An empty window floors at one spike over all electrodes: 1 / (2 x 5 s).
    assert y[1] == pytest.approx(np.log2(1 / (2 * 5.0)))
    assert y[2] == pytest.approx(np.log2(4 / 2 / 5.0))


# ------------------------------------------------------------------ trend and decomposition


def test_trend_on_a_straight_line_has_zero_residual():
    x = np.arange(6) - 2.5
    b, s = st.trend(1.0 - 0.3 * x)
    assert b == pytest.approx(-0.3) and s == pytest.approx(0.0, abs=1e-12)


def test_trend_residual_sd_uses_two_degrees_of_freedom():
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    b, s = st.trend(y)
    x = np.arange(6) - 2.5
    resid = y - y.mean() - b * x
    assert b == pytest.approx(np.polyfit(x, y, 1)[0])
    assert s == pytest.approx(np.sqrt(np.sum(resid**2) / 4))


def test_split_change_splits_into_trend_and_residual():
    x = np.arange(6) - 2.5
    y = 0.2 * x + np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.0])
    got = st.decompose(y)
    assert got["d"] == pytest.approx(y[3:].mean() - y[:3].mean())
    assert got["d_trend"] == pytest.approx(3 * got["b"])
    assert got["d_resid"] == pytest.approx(got["d"] - got["d_trend"])


def test_on_a_straight_line_the_split_half_change_is_all_trend():
    x = np.arange(6) - 2.5
    got = st.decompose(2.0 + 0.4 * x)
    assert got["d"] == pytest.approx(got["d_trend"]) and got["d_resid"] == pytest.approx(0)


def test_quadratic_check_flags_a_curve_and_not_a_noisy_line():
    x = np.arange(6) - 2.5
    curved = st.decompose(0.5 * x**2)
    assert curved["quad_lowers"] is True
    line = st.decompose(0.3 * x + np.array([0.05, -0.05, 0.05, -0.05, 0.05, -0.05]))
    assert line["quad_lowers"] is False


# ------------------------------------------------------------------ bootstrap and predictions


def _wells(rng, n, slope_sd, noise, species):
    x = np.arange(6) - 2.5
    rows = []
    for i in range(n):
        y = rng.normal(0, slope_sd) * x + rng.normal(0, noise, 6)
        row = {"species": species, "unit": f"{species}:{i}", "defined": True, "window_s": 300.0}
        row = {**row, "n_electrodes": 12, **st.decompose(y)}
        rows.append({**row, "y": list(y), "rate_hz": list(2.0**y)})
    return rows


def _table(rng, n_h=24, n_r=38, h_slope=0.3, r_slope=0.1, h_noise=0.2, r_noise=0.1):
    rows = _wells(rng, n_h, h_slope, h_noise, "human") + _wells(rng, n_r, r_slope, r_noise, "rat")
    return pd.DataFrame(rows)


def test_bootstrap_draws_wells_as_the_split_half_interval_does():
    rng = np.random.default_rng(3)
    table = _table(rng)
    got = st.bootstrap(table, resamples=300, seed=st.SEED)
    groups = {s: [np.array([b]) for b in table.loc[table.species == s, "b"]] for s in st.SPECIES}
    ref = sh.ratio_interval(groups["human"], groups["rat"], 300, st.SEED)
    assert got["sd_b_ratio"]["ci95"] == ref["ci95"]
    assert got["sd_b_ratio"]["point"] == ref["R"]


def test_predictions_pass_when_human_settles_and_wobbles_more():
    table = _table(np.random.default_rng(4))
    preds = st.analyse(table, resamples=400)["predictions"]
    assert preds["T1_human_settling_more_uneven"]["outcome"] == "pass"
    assert preds["T2_human_steady_noise_larger"]["outcome"] == "pass"
    assert preds["T3_settling_carries_the_human_spread"]["outcome"] == "pass"
    assert all(p["pass"] is True for p in preds.values())


def test_t1_and_t2_fail_when_the_species_match():
    table = _table(np.random.default_rng(5), h_slope=0.0, r_slope=0.0, h_noise=0.1)
    preds = st.analyse(table, resamples=400)["predictions"]
    assert preds["T1_human_settling_more_uneven"]["outcome"] == "fail"
    assert preds["T2_human_steady_noise_larger"]["outcome"] == "fail"
    assert preds["T1_human_settling_more_uneven"]["pass"] is False


def _no_trend_shape() -> np.ndarray:
    """A window pattern that moves the split-half change and has zero slope."""
    x = np.arange(6) - 2.5
    c = np.r_[-np.ones(3), np.ones(3)] / 3
    return c - (c @ x) / (x @ x) * x


def test_t3_fails_when_the_change_has_no_trend_part():
    rng = np.random.default_rng(11)
    shape = _no_trend_shape()
    rows = []
    for s, n in (("human", 24), ("rat", 38)):
        for i in range(n):
            y = rng.normal(0, 3) * shape + rng.normal(0, 0.01, 6)
            rows.append({"species": s, "unit": f"{s}:{i}", "defined": True, **st.decompose(y)})
    t3 = st.analyse(pd.DataFrame(rows), resamples=50)["predictions"]
    t3 = t3["T3_settling_carries_the_human_spread"]
    assert t3["outcome"] == "fail" and t3["share_human"] < 0.1


def test_white_noise_share_is_27_over_35_and_matches_simulation():
    assert st.white_noise_share() == pytest.approx(27 / 35)
    rng = np.random.default_rng(12)
    got = [st.decompose(rng.normal(0, 1, 6)) for _ in range(20000)]
    d, dt = np.array([g["d"] for g in got]), np.array([g["d_trend"] for g in got])
    assert np.var(dt) / np.var(d) == pytest.approx(27 / 35, abs=0.02)


def test_slope_spread_expected_from_window_noise_alone():
    x = np.arange(6) - 2.5
    s = np.array([0.2, 0.4])
    assert st.noise_slope_var(s) == pytest.approx(np.mean(s**2) / (x @ x))


def test_too_few_wells_makes_every_prediction_uninterpretable():
    table = _table(np.random.default_rng(6), n_h=9)
    preds = st.analyse(table, resamples=100)["predictions"]
    assert {p["outcome"] for p in preds.values()} == {"uninterpretable"}
    assert not any(p["pass"] for p in preds.values())


def test_t3_share_is_the_variance_ratio_and_rat_is_reported():
    table = _table(np.random.default_rng(7))
    got = st.analyse(table, resamples=100)
    t3 = got["predictions"]["T3_settling_carries_the_human_spread"]
    human = table[table.species == "human"]
    assert t3["share_human"] == pytest.approx(
        np.var(human.d_trend, ddof=1) / np.var(human.d, ddof=1), abs=1e-4
    )
    assert t3["share_rat"] is not None


def test_wells_needed_use_the_split_half_formula_on_the_residual_part():
    table = _table(np.random.default_rng(8))
    got = st.analyse(table, resamples=100)["wells_needed_residual_only"]
    sd_h = float(np.std(table.loc[table.species == "human", "d_resid"], ddof=1))
    assert got["human"]["down_20pct"] == sh.wells_needed(round(sd_h, 4), 0.2)
    assert "illustrative" in got["note"].lower()


def test_assumption_check_reads_the_trend_as_curved_when_most_wells_curve():
    y = 0.5 * (np.arange(6) - 2.5) ** 2
    rows = [
        {"species": s, "unit": f"{s}:{i}", "defined": True, **st.decompose(y + 0.01 * i)}
        for s in st.SPECIES
        for i in range(12)
    ]
    check = st.analyse(pd.DataFrame(rows), resamples=50)["assumption_check"]
    assert check["share_lowered"]["human"] == 1.0 and check["linear"] is False


# ------------------------------------------------------------------ inputs


def test_usable_units_come_from_the_split_half_results(tmp_path):
    path = tmp_path / "hpsc_split_half_REAL.json"
    wells = [
        {"pool": "primary", "unit": "P:A1", "usable": True},
        {"pool": "primary", "unit": "P:A2", "usable": False},
        {"pool": "replication", "unit": "Q:A1", "usable": True},
    ]
    path.write_text(json.dumps({"wells": wells}))
    assert st.usable_units(path) == {"P:A1"}


def test_well_record_on_a_synthetic_well():
    rng = np.random.default_rng(9)
    channels = _channels(rng, n=10, duration=120.0)
    row = st.well_record(channels, 120.0)
    assert row["n_electrodes"] == 10 and row["window_s"] == pytest.approx(20.0)
    assert len(row["rate_hz"]) == st.N_WINDOWS and row["defined"] is True
    assert row["d_trend"] == pytest.approx(3 * row["b"])


def test_plot_writes_a_real_data_figure(tmp_path):
    rng = np.random.default_rng(10)
    table = _table(rng)
    payload = st.analyse(table, resamples=50)
    title = st.plot(payload, table, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").exists()
    assert "REAL" in title and "Kapucu 2022" in title and "one lab" in title


# ------------------------------------------------------------------ post hoc readings


def test_first_window_check_drops_window_zero():
    y = np.array([3.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    rows = []
    for s in st.SPECIES:
        for i in range(4):
            yy = y + np.r_[0, 0, 0, 0.1 * i, 0.1 * i, 0.1 * i]
            rows.append({"species": s, "unit": f"{s}:{i}", "defined": True, "y": list(yy)})
    by = st._values(pd.DataFrame(rows))
    got = st.first_window_check(by)
    # Without window 0 the change is mean(y3..5) - mean(y1..2) = 0.1 i; window 0 drops by 1.
    assert got["human"]["sd_d_without_window_0"] == pytest.approx(
        np.std([0, 0.1, 0.2, 0.3], ddof=1), abs=1e-4
    )
    assert got["human"]["median_step_0_to_1"] == pytest.approx(-1.0)
    assert got["ratio_sd_d_without_window_0"] == pytest.approx(1.0)


def test_assumption_check_reports_the_scatter_around_a_curve():
    x = np.arange(6) - 2.5
    rows = [
        {"species": s, "unit": f"{s}:{i}", "defined": True, **st.decompose(0.2 * x**2 + e)}
        for s, scale in (("human", 0.2), ("rat", 0.1))
        for i, e in enumerate(np.random.default_rng(13).normal(0, scale, (12, 6)))
    ]
    check = st.assumption_check(st._values(pd.DataFrame(rows)))
    assert check["median_s_quad"]["human"] > check["median_s_quad"]["rat"]
    assert check["median_s_quad_ratio"] == pytest.approx(
        check["median_s_quad"]["human"] / check["median_s_quad"]["rat"], rel=1e-3
    )
