"""Tests for the registered early-network-state replication (synthetic inputs only)."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from neuromps.features.mea import extract_features
from neuromps.models import hpsc_network_state as ns
from neuromps.models import hpsc_settling as st
from neuromps.models import hpsc_split_half as sh

# ------------------------------------------------------------------ registration


def test_constants_match_the_registration():
    reg = json.loads(ns.REGISTRATION.read_text())
    method = reg["method"]
    assert f"{ns.N_WINDOWS} consecutive windows" in method["windows"]
    assert f"{ns.RESAMPLES:,} bootstrap resamples" in method["interval"]
    assert str(ns.SEED) in method["interval"]
    assert "k = 1..5" in method["late_trend"] and list(ns.LATE_K) == [1, 2, 3, 4, 5]
    for short, feature in ns.EARLY.items():
        assert f"{short} = {feature}" in method["early_state"]
    assert f"fewer than {ns.MIN_RECORDINGS} scored recordings" in reg["uninterpretable"]
    assert f"at least {ns.MIN_WELLS} wells" in reg["uninterpretable"]
    assert f"more than {round(ns.MAX_ZERO_SHARE * 100)}%" in reg["uninterpretable"]
    assert f"above {ns.AGE_PARTIAL}" in reg["assumption_check"]["check"]
    assert reg["outputs"]["metrics_key"] == ns.ROOT
    assert reg["outputs"]["results_file"].startswith(f"outputs/{ns.RESULTS_NAME}")
    assert reg["outputs"]["figure"] == f"outputs/figures/{ns.FIGURE_NAME}"
    assert set(reg["predictions"]) == set(ns.PREDICTIONS)


def test_reuses_the_split_half_pool_and_the_settling_windows():
    assert ns.SEED == sh.SEED == st.SEED and ns.RESAMPLES == sh.RESAMPLES
    assert ns.SPECIES == sh.SPECIES and ns.N_WINDOWS == st.N_WINDOWS


# ------------------------------------------------------------------ windows and per recording


def _channels(rng, n=12, duration=60.0, rate=5.0, quiet=2):
    out = {f"A1_{i}": np.sort(rng.uniform(0, duration, int(rate * duration))) for i in range(n)}
    for j in range(quiet):
        out[f"A1_q{j}"] = np.array([1.0, 2.0])
    return out


def test_window_zero_matches_the_settling_cut():
    channels = {"e": np.array([0.0, 9.99, 10.0, 25.0, 59.99, 60.0])}
    w0, length = ns.window_zero(channels, 60.0)
    assert length == pytest.approx(10.0)
    assert list(w0["e"]) == [0.0, 9.99]  # a spike on the edge starts window 1
    assert st.window_counts(channels, ["e"], 60.0)[0, 0] == len(w0["e"])


def test_early_state_is_the_shared_extractor_on_window_zero():
    rng = np.random.default_rng(1)
    channels = _channels(rng, duration=120.0)
    row = ns.recording_state(channels, 120.0)
    w0 = {k: v[v < 20.0] for k, v in channels.items()}
    ref = extract_features(w0, 20.0)
    for short, feature in ns.EARLY.items():
        assert row[short] == ref[feature]
    assert row["active0"] == ref["active_electrodes"]


def test_late_slope_ignores_window_zero_and_is_the_ols_slope():
    y = np.array([9.0, 1.0, 1.5, 2.0, 2.5, 3.0])
    assert ns.late_slope(y) == pytest.approx(0.5)
    noisy = np.array([0.0, 1.0, 3.0, 2.0, 5.0, 4.0])
    assert ns.late_slope(noisy) == pytest.approx(np.polyfit(np.arange(1, 6), noisy[1:], 1)[0])


def test_late_trend_uses_the_settling_rate_and_floor():
    rng = np.random.default_rng(2)
    channels = _channels(rng, duration=120.0)
    row = ns.recording_state(channels, 120.0)
    ref = st.well_record(channels, 120.0)
    assert row["y"] == pytest.approx(ref["y"])
    assert row["late_b"] == pytest.approx(ns.late_slope(np.array(ref["y"])))
    assert row["n_electrodes"] == ref["n_electrodes"]


def test_a_silent_recording_is_unscored():
    # One spike, outside window 0: nothing active in window 0 nor over the recording.
    row = ns.recording_state({"e": np.array([30.0])}, 60.0)
    assert row["late_b"] is None and row["nb0"] is None and row["mfr0"] is None


# ------------------------------------------------------------------ statistics


def test_spearman_matches_scipy_with_ties_and_is_nan_when_constant():
    x = np.array([0, 0, 0, 1, 2, 3, 3, 5], dtype=float)
    y = np.array([3, 1, 2, 0, -1, -2, -2, -5], dtype=float)
    assert ns.spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)
    assert np.isnan(ns.spearman(np.zeros(5), np.arange(5.0)))


def test_partial_spearman_removes_a_shared_age_trend():
    rng = np.random.default_rng(3)
    div = np.repeat(np.arange(28, 70, 3), 10).astype(float)
    x = div + rng.normal(0, 1, div.size)
    y = -div + rng.normal(0, 1, div.size)
    assert ns.spearman(x, y) < -0.9
    assert abs(ns.partial_spearman(x, y, div)) < 0.2
    # Within an age a relation survives the adjustment.
    z = rng.normal(0, 1, div.size)
    assert ns.partial_spearman(z + 0.1 * div, -z, div) < -0.8


def _rows(rng, species, n_wells, per_well, slope_nb, slope_mfr=0.0, zero_share=0.0, div0=28):
    rows = []
    for w in range(n_wells):
        for j in range(per_well):
            nb = 0.0 if rng.uniform() < zero_share else rng.gamma(2.0, 2.0)
            mfr = rng.gamma(2.0, 1.0)
            late = slope_nb * nb + slope_mfr * mfr + rng.normal(0, 0.5)
            rows.append(
                {
                    "pool": "replication",
                    "species": species,
                    "plate": f"{species}_P{w % 2}",
                    "well": f"A{w}",
                    "unit": f"{species}_P{w % 2}:A{w}",
                    "div": div0 + 3 * j,
                    "nb0": nb,
                    "sync0": 0.1 * nb + rng.normal(0, 0.1),
                    "mfr0": mfr,
                    "late_b": late,
                }
            )
    return rows


def _table(rng, human_nb=-1.0, rat_nb=0.0, human_mfr=0.0, n_wells=12, per_well=6, zero=0.0):
    rows = _rows(rng, "human", n_wells, per_well, human_nb, human_mfr, zero)
    rows += _rows(rng, "rat", n_wells, per_well, rat_nb, div0=21)
    return pd.DataFrame(rows)


def test_bootstrap_draws_whole_wells_and_is_reproducible():
    rng = np.random.default_rng(4)
    rows = _table(rng)
    rows = rows[rows["species"] == "human"]
    a = ns.bootstrap(rows, resamples=40, seed=7)
    b = ns.bootstrap(rows, resamples=40, seed=7)
    assert a == b
    groups = ns.well_arrays(rows)
    assert len(groups) == 12 and all(g.shape == (6, 4) for g in groups)
    drawn = ns.draw(groups, np.array([0, 0, 3]))
    assert drawn.shape == (18, 4)
    assert np.array_equal(drawn[:6], groups[0]) and np.array_equal(drawn[6:12], groups[0])


def test_predictions_pass_when_human_bursting_predicts_and_rat_does_not():
    # Rat with a positive relation: "no negative relation" holds whatever the draw.
    payload = ns.analyse(_table(np.random.default_rng(5), rat_nb=0.5), None, resamples=200)
    preds = payload["predictions"]
    assert [preds[p]["outcome"] for p in ns.PREDICTIONS] == ["pass"] * 3
    assert all(preds[p]["pass"] is True for p in ns.PREDICTIONS)
    assert preds[ns.PREDICTIONS[0]]["ci95"][1] < 0


def test_n2_fails_when_rat_shows_the_same_relation_and_n3_when_rate_predicts_too():
    table = _table(np.random.default_rng(6), rat_nb=-1.0, human_mfr=-3.0)
    preds = ns.analyse(table, None, resamples=200)["predictions"]
    assert preds[ns.PREDICTIONS[1]]["outcome"] == "fail"
    assert preds[ns.PREDICTIONS[2]]["outcome"] == "fail"
    assert preds[ns.PREDICTIONS[2]]["pass"] is False


def test_n1_fails_without_a_relation():
    preds = ns.analyse(_table(np.random.default_rng(7), human_nb=0.0), None, 200)["predictions"]
    assert preds[ns.PREDICTIONS[0]]["outcome"] == "fail"


def test_too_few_recordings_or_wells_is_uninterpretable():
    few_wells = ns.analyse(_table(np.random.default_rng(8), n_wells=7, per_well=6), None, 50)
    assert {p["outcome"] for p in few_wells["predictions"].values()} == {"uninterpretable"}
    few_recs = ns.analyse(_table(np.random.default_rng(9), n_wells=9, per_well=3), None, 50)
    assert {p["outcome"] for p in few_recs["predictions"].values()} == {"uninterpretable"}
    assert all(p["pass"] is False for p in few_recs["predictions"].values())


def test_mostly_zero_human_bursting_makes_n1_and_n3_uninterpretable_only():
    table = _table(np.random.default_rng(10), zero=0.95)
    preds = ns.analyse(table, None, resamples=50)["predictions"]
    assert preds[ns.PREDICTIONS[0]]["outcome"] == "uninterpretable"
    assert preds[ns.PREDICTIONS[2]]["outcome"] == "uninterpretable"
    assert preds[ns.PREDICTIONS[1]]["outcome"] in {"pass", "fail"}
    assert preds[ns.PREDICTIONS[0]]["nb0_zero_share"] > 0.8


def test_missing_replication_pool_is_uninterpretable_with_a_status():
    payload = ns.analyse(None, None, resamples=10)
    for p in payload["predictions"].values():
        assert p["outcome"] == "uninterpretable" and p["pass"] is False
        assert "status" in p


def test_age_check_flags_an_age_effect_when_the_partial_rho_vanishes():
    rng = np.random.default_rng(11)
    rows = []
    for w in range(12):
        for j in range(8):
            div = 28 + 3 * j
            rows.append(
                {
                    "species": "human",
                    "unit": f"h:{w}",
                    "div": div,
                    "nb0": div + rng.normal(0, 0.5),
                    "late_b": -div + rng.normal(0, 0.5),
                }
            )
    got = ns.age_check(pd.DataFrame(rows))
    assert got["partial_rho"] > ns.AGE_PARTIAL
    assert got["per_div"]["n_divs"] == 8 and abs(got["per_div"]["median_rho"]) < 0.6


def test_unscored_recordings_are_counted():
    table = _table(np.random.default_rng(12))
    table.loc[:3, ["nb0", "sync0", "mfr0"]] = None
    payload = ns.analyse(table, None, resamples=20)
    human = payload["replication"]["human"]
    assert human["unscored"]["nb0"] == 4 and human["n_recordings"] == len(table) // 2


# ------------------------------------------------------------------ inputs and post hoc


def test_usable_recordings_come_from_the_split_half_replication_rows(tmp_path):
    path = tmp_path / "hpsc_split_half_REAL.json"
    wells = [
        {"pool": "primary", "plate": "P", "div": 29, "well": "A1", "usable": True},
        {"pool": "replication", "plate": "Q", "div": 28, "well": "A1", "usable": True},
        {"pool": "replication", "plate": "Q", "div": 31, "well": "A1", "usable": False},
    ]
    path.write_text(json.dumps({"wells": wells}))
    assert ns.usable_recordings(path) == {("Q", 28, "A1")}
    path.write_text(json.dumps({"wells": wells[:1]}))
    assert ns.usable_recordings(path) == set()


def test_post_hoc_pool_is_labelled_and_reports_all_three_features():
    rng = np.random.default_rng(13)
    table = _table(rng, n_wells=24, per_well=1)
    table["pool"] = "primary"
    payload = ns.analyse(None, table, resamples=50)
    post = payload["primary_pool_post_hoc"]
    assert post["post_hoc"] is True and "post hoc" in post["label"]
    assert set(post["human"]["rho"]) == set(ns.EARLY)
    assert post["human"]["n_wells"] == 24


def test_plot_writes_a_real_data_figure(tmp_path):
    rng = np.random.default_rng(14)
    rep = _table(rng)
    prim = _table(rng, n_wells=24, per_well=1)
    prim["pool"] = "primary"
    payload = ns.analyse(rep, prim, resamples=20)
    title = ns.plot(payload, rep, prim, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").exists()
    assert "REAL" in title and "Kapucu 2022" in title
    ns.plot(ns.analyse(None, prim, resamples=10), None, prim, tmp_path / "quick.png")
    assert (tmp_path / "quick.png").exists()
