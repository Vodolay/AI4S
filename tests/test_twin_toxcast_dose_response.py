"""Tests for the twin dose-response check against ToxCast curves (synthetic inputs only)."""

import numpy as np
import pandas as pd
import pytest

from neuromps.models import twin_toxcast_dose_response as dr


def test_concentration_bins_merge_half_decades_and_drop_zero():
    got = dr.concentration_bins(np.array([0.0, -1.0, 0.03, 0.1, 0.3, 1.0, 30.0, 40.0]))
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert got[2:].tolist() == [-3.0, -2.0, -1.0, 0.0, 3.0, 3.0]


def test_matched_positions_span_the_grid():
    assert dr.matched_positions(7) == [0, 1, 2, 3, 4, 5, 6]
    assert dr.matched_positions(4) == [0.0, 2.0, 4.0, 6.0]
    assert dr.matched_positions(13)[0] == 0 and dr.matched_positions(13)[-1] == 6


def _summaries(medians, moves=None):
    moves = [True] * len(medians) if moves is None else moves
    return [
        {"median": m, "n_defined": 12, "moves": mv} for m, mv in zip(medians, moves, strict=True)
    ]


def test_at_position_interpolates_and_brackets_moves():
    summaries = _summaries([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [False] * 6 + [True])
    assert dr.at_position(summaries, 2.0)["median"] == 2.0
    assert dr.at_position(summaries, 2.5)["median"] == 2.5
    assert dr.at_position(summaries, 2.5)["moves"] is False
    assert dr.at_position(summaries, 5.5)["moves"] is True  # brackets the moving level
    assert dr.at_position(summaries, 99)["median"] == 6.0  # clipped to the grid


def test_at_position_keeps_an_undefined_level_undefined():
    summaries = _summaries([0.0, None, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert dr.at_position(summaries, 1.5)["median"] is None


def test_matched_curve_uses_a_shorter_grid_when_asked():
    summaries = _summaries([0.0, 1.0, 2.0, 3.0, 4.0, 9.0, 9.0])
    full = dr.matched_curve(summaries, 4)
    short = dr.matched_curve(summaries, 4, levels=5)
    assert [p["median"] for p in full] == [0.0, 2.0, 4.0, 9.0]
    assert [p["median"] for p in short] == [0.0, 1.3333, 2.6667, 4.0]


def _vehicle_wells(dates=12, per_date=10, acid=2448, rval=-50.0):
    rows = []
    for d in range(dates):
        for _ in range(per_date):
            rows.append(
                {
                    "acid": acid,
                    "wllt": "n",
                    "spid": "DMSO",
                    "apid": f"d{d}",
                    "conc": 0.0,
                    "rval": rval,
                }
            )
    return rows


def test_vehicle_groups_are_per_date():
    wells = pd.DataFrame(_vehicle_wells())
    groups = dr.vehicle_groups(wells, 2448)
    assert len(groups) == 12 and all(g.size == 10 for g in groups)
    assert np.allclose(groups[0], -1.0)  # log2(1 - 0.5)


def test_null_draws_need_enough_vehicle_wells_per_date():
    rng = np.random.default_rng(0)
    groups = [np.random.default_rng(i).normal(0, 1, 10) for i in range(6)]
    assert dr.null_draws(groups, 3, 500, rng).size == 500
    assert dr.null_draws(groups, 8, 500, rng).size == 0  # 8 + 3 > 10


def test_null_threshold_is_wider_for_a_larger_family():
    groups = [np.random.default_rng(i).normal(0, 1, 12) for i in range(8)]
    narrow = dr.null_threshold(groups, 3, 1, seed=1)["threshold"]
    wide = dr.null_threshold(groups, 3, 7, seed=1)["threshold"]
    assert 0 < narrow < wide
    lower = dr.null_threshold(groups, 3, 7, seed=1, lower=True)["threshold"]
    assert lower < 0


def _curve(medians, wells=3, bins=None):
    bins = list(range(len(medians))) if bins is None else bins
    return [{"bin": b, "wells": wells, "median": m} for b, m in zip(bins, medians, strict=True)]


def test_flag_moving_marks_only_bins_above_the_vehicle_null():
    groups = [np.random.default_rng(i).normal(0, 0.1, 12) for i in range(10)]
    got = dr.flag_moving(_curve([0.0, 0.01, 2.0, -3.0]), groups, k=4, seed=5)
    assert [p["moves"] for p in got] == [False, False, True, True]
    assert all(p["threshold"] > 0 for p in got)


def test_flag_moving_leaves_an_empty_bin_alone():
    groups = [np.random.default_rng(i).normal(0, 0.1, 12) for i in range(10)]
    curve = [{"bin": 0, "wells": 0, "median": None}]
    assert dr.flag_moving(curve, groups, k=4, seed=5)[0]["moves"] is False


def test_excluded_bins_takes_the_lowest_flag_and_everything_above():
    groups = [np.random.default_rng(i).normal(0, 0.1, 12) for i in range(10)]
    curve = _curve([0.0, -2.0, 0.0, -3.0], bins=[0, 1, 2, 3])
    got = dr.excluded_bins(curve, groups, k=4)
    assert got["first_flagged_bin"] == 1 and got["excluded"] == [1, 2, 3]


def test_excluded_bins_excludes_nothing_when_viability_holds():
    groups = [np.random.default_rng(i).normal(0, 0.1, 12) for i in range(10)]
    got = dr.excluded_bins(_curve([0.0, 0.05, -0.02, 0.01]), groups, k=4)
    assert got["first_flagged_bin"] is None and got["excluded"] == []


def test_excluded_bins_reports_a_bin_whose_null_is_undefined_instead_of_passing_it():
    # Ten wells per date cannot judge a bin pooling nine wells (nine + three > ten),
    # so the drop is recorded as untestable rather than silently treated as fine.
    groups = [np.random.default_rng(i).normal(0, 0.1, 10) for i in range(10)]
    curve = _curve([0.0, 0.0, 0.0, -3.0], wells=9)
    got = dr.excluded_bins(curve, groups, k=4)
    assert got["first_flagged_bin"] is None and got["excluded"] == []
    assert got["untestable_bins"] == [0, 1, 2, 3]


def test_the_post_hoc_rescore_drops_the_bin_it_is_asked_to():
    bins = [0, 1, 2, 3, 4]
    real = {f: _curve([0.1, 0.4, 0.9, 1.6, 9.0], bins=bins) for f in dr.MAPPING}
    for curve in real.values():
        for point in curve:
            point["moves"] = True
    results = {
        "Lindane": {"scored": True, "knob": "gabaa_block", "retained_bins": bins, "real": real}
    }
    kept = {
        which: dr.rescore_dropping(results, _synthetic_twin(), which)["Lindane"]["features"][
            "synchrony_index"
        ]["bins"]
        for which in ("top", "bottom", "middle")
    }
    assert kept == {"top": 4, "bottom": 4, "middle": 4}


def test_the_post_hoc_rescore_skips_a_curve_left_too_short():
    real = {
        f: dr.flag_moving(
            _curve([0.1, 0.4, 0.9, 1.6, 9.0]),
            [np.random.default_rng(i).normal(0, 0.05, 12) for i in range(10)],
            k=5,
            seed=3,
        )
        for f in dr.MAPPING
    }
    twin = _synthetic_twin()
    results = {
        "Lindane": {
            "scored": True,
            "knob": "gabaa_block",
            "retained_bins": [0, 1, 2, 3, 4],
            "real": real,
        },
        "Endrin": {
            "scored": True,
            "knob": "gabaa_block",
            "retained_bins": [0, 1, 2, 3],
            "real": real,
        },
        "Aldrin": {"scored": False},
    }
    got = dr.rescore_dropping(results, twin, "top")
    assert set(got) == {"Lindane"}  # Endrin would be left with three bins
    assert got["Lindane"]["features"]["synchrony_index"]["bins"] == 4


def test_treated_effects_subtract_the_same_date_vehicle_median():
    rows = _vehicle_wells(dates=3, per_date=4)
    for d in range(3):
        for conc in (0.03, 0.3, 3.0):
            rows.append(
                {
                    "acid": 2448,
                    "wllt": "t",
                    "spid": "X",
                    "apid": f"d{d}",
                    "conc": conc,
                    "rval": 0.0,
                }
            )
    rows.append(
        {"acid": 2448, "wllt": "t", "spid": "X", "apid": "lonely", "conc": 1.0, "rval": 0.0}
    )
    got = dr.treated_effects(pd.DataFrame(rows), 2448, ["X"])
    assert len(got) == 9  # the well on a date without vehicle wells is dropped
    assert np.allclose(got["effect"], 1.0)
    assert sorted(got["bin"].unique()) == [-3.0, -1.0, 1.0]


def test_real_curve_reports_empty_bins():
    effects = pd.DataFrame({"apid": ["d0", "d0"], "bin": [0.0, 0.0], "effect": [1.0, 3.0]})
    got = dr.real_curve(effects, [0, 1])
    assert got[0] == {"bin": 0, "wells": 2, "median": 2.0}
    assert got[1] == {"bin": 1, "wells": 0, "median": None}


def test_pair_score_is_the_rank_correlation_and_needs_four_bins():
    real = _curve([0.1, 0.4, 0.9, 1.6])
    twin = _curve([0.2, 0.5, 0.8, 2.0])
    assert dr.pair_score(real, twin, [0, 1, 2, 3])["spearman"] == 1.0
    assert dr.pair_score(real, twin, [0, 1, 2, 3])["hit"] is True
    flipped = _curve([2.0, 0.8, 0.5, 0.2])
    assert dr.pair_score(real, flipped, [0, 1, 2, 3])["spearman"] == -1.0
    assert dr.pair_score(real, twin, [0, 1, 2])["spearman"] is None  # fewer than four bins


def test_pair_score_refuses_a_flat_curve():
    real = _curve([1.0, 1.0, 1.0, 1.0])
    twin = _curve([0.1, 0.4, 0.9, 1.6])
    assert dr.pair_score(real, twin, [0, 1, 2, 3])["spearman"] is None


def test_first_movers_keeps_ties_and_respects_retained_bins():
    curves = {
        "a": _curve([0.0, 1.0, 1.0, 1.0]),
        "b": _curve([0.0, 0.0, 1.0, 1.0]),
        "c": _curve([0.0, 1.0, 0.0, 0.0]),
    }
    for curve in curves.values():
        for point in curve:
            point["moves"] = point["median"] > 0
    assert dr.first_movers(curves, [0, 1, 2, 3]) == ["a", "c"]
    assert dr.first_movers(curves, [2, 3]) == ["a", "b"]  # bin 1 excluded
    assert dr.first_movers({"a": _curve([0.0])}, []) == []


def test_summarise_counts_hits_and_means():
    got = dr.summarise([{"spearman": 1.0, "hit": True}, {"spearman": 0.0, "hit": False}])
    assert got == {"pairs": 2, "hits": 1, "hit_rate": 0.5, "mean_spearman": 0.5}
    assert dr.summarise([])["mean_spearman"] is None


def test_evaluate_applies_the_registered_thresholds():
    twin = {"pairs": 10, "hits": 6, "hit_rate": 0.6, "mean_spearman": 0.5}
    nulls = {
        "follow_firing_rate": {"pairs": 10, "mean_spearman": 0.2},
        "shuffled_mechanism": {"pairs": 30, "mean_spearman": 0.45},
    }
    movers = {
        "a": {"match": True},
        "b": {"match": True},
        "c": {"match": False},
        "d": {"match": None},
    }
    got = dr.evaluate(twin, nulls, movers)
    assert got["P1_shape"]["pass"] is True
    assert got["P2_beats_follow_firing_rate"]["pass"] is True
    assert got["P3_beats_shuffled_mechanism"]["pass"] is False  # margin 0.05 < 0.1
    assert got["P4_first_mover"]["compounds"] == 3 and got["P4_first_mover"]["pass"] is False
    thin = dr.evaluate(
        {"pairs": 4, "hits": 4, "hit_rate": 1.0, "mean_spearman": 0.9}, nulls, movers
    )
    assert thin["P1_shape"]["pass"] is False  # fewer than eight pairs


def test_each_prediction_records_an_outcome_the_audit_counts():
    twin = {"pairs": 10, "hits": 6, "hit_rate": 0.6, "mean_spearman": 0.5}
    nulls = {
        "follow_firing_rate": {"pairs": 10, "mean_spearman": 0.2},
        "shuffled_mechanism": {"pairs": 30, "mean_spearman": 0.45},
    }
    movers = {k: {"match": k != "c"} for k in "abcd"}
    got = dr.evaluate(twin, nulls, movers)
    assert {k: v["outcome"] for k, v in got.items()} == {
        "P1_shape": "pass",
        "P2_beats_follow_firing_rate": "pass",
        "P3_beats_shuffled_mechanism": "fail",
        "P4_first_mover": "pass",
    }
    few = dr.evaluate(
        {"pairs": 4, "hits": 4, "hit_rate": 1.0, "mean_spearman": 0.9},
        nulls,
        {"a": {"match": True}, "b": {"match": True}},
    )
    assert few["P1_shape"]["outcome"] == "uninterpretable"  # under the eight-pair floor
    assert few["P4_first_mover"]["outcome"] == "uninterpretable"  # descriptive below four


def test_level_summary_follows_the_registered_move_rule():
    assert dr.level_summary(np.array([1.0] * 12))["moves"] is True
    assert dr.level_summary(np.array([0.05] * 12))["moves"] is False  # too small
    assert dr.level_summary(np.array([1.0] * 9 + [-1.0] * 3))["moves"] is False  # chips disagree
    assert dr.level_summary(np.array([1.0] * 7 + [np.nan] * 5))["moves"] is False  # too few chips
    assert dr.level_summary(np.array([np.nan] * 12)) == {
        "median": None,
        "n_defined": 0,
        "moves": False,
    }


def test_twin_curves_are_one_entry_per_level():
    rows = [
        {"knob": "gabaa_block", "level": level, "build_seed": seed, **{f: 0.5 for f in dr.MAPPING}}
        for level in range(dr.LEVELS)
        for seed in dr.BUILD_SEEDS
    ]
    got = dr.twin_curves(pd.DataFrame(rows))
    assert set(got) == {"gabaa_block"} and len(got["gabaa_block"]["mfr_hz"]) == dr.LEVELS
    assert got["gabaa_block"]["mfr_hz"][0]["moves"] is True


def test_registration_lists_only_knobs_the_twin_has():
    labels = set(dr.compounds().values())
    assert labels <= set(dr.LABEL_KNOB)
    assert set(dr.LABEL_KNOB.values()) <= set(dr.KNOB_GRIDS)
    assert all(len(grid) == dr.LEVELS for _, grid in dr.KNOB_GRIDS.values())


def _synthetic_screen(top_viability_drop: float = 0.0):
    """Wells for one compound: a monotone rise with concentration, flat vehicle wells."""
    concs = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    ephys, viability = [], []
    for date in range(6):
        for acid in dr.MAPPING.values():
            for _ in range(10):
                ephys.append(
                    {
                        "acid": acid,
                        "wllt": "n",
                        "spid": "DMSO",
                        "apid": f"d{date}",
                        "conc": 0.0,
                        "rval": 0.0,
                    }
                )
            for step, conc in enumerate(concs):
                ephys.append(
                    {
                        "acid": acid,
                        "wllt": "t",
                        "spid": "SP1",
                        "apid": f"d{date}",
                        "conc": conc,
                        "rval": 100.0 * (2**step - 1),  # log2 change = step
                    }
                )
        for _ in range(10):
            viability.append(
                {"wllt": "n", "spid": "DMSO", "apid": f"d{date}", "conc": 0.0, "rval": 100.0}
            )
        for step, conc in enumerate(concs):
            drop = top_viability_drop if step == len(concs) - 1 else 0.0
            viability.append(
                {
                    "wllt": "t",
                    "spid": "SP1",
                    "apid": f"d{date}",
                    "conc": conc,
                    "rval": 100.0 * 2**-drop,
                }
            )
    return pd.DataFrame(ephys), pd.DataFrame(viability)


def _synthetic_twin(medians=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
    return {
        knob: {feature: _summaries(list(medians)) for feature in dr.MAPPING}
        for knob in dr.KNOB_GRIDS
    }


def test_analyse_scores_a_synthetic_compound_end_to_end(tmp_path):
    wells, viability = _synthetic_screen()
    results = dr.analyse(wells, viability, _synthetic_twin(), {"Lindane": ["SP1"]})
    lindane = results["Lindane"]
    assert lindane["scored"] and lindane["bins"] == [-3, -2, -1, 0, 1, 2, 3]
    assert lindane["exclusion"]["excluded"] == []
    assert results["Fipronil"]["dropped"] == "not in the screen"
    scored = dr.collect(results)
    assert scored["twin"]["pairs"] == len(dr.MAPPING) - 1  # every unseen feature
    assert scored["twin"]["hit_rate"] == 1.0  # both curves rise monotonically
    # The follow-firing-rate null does as well here: the synthetic features move together.
    assert scored["nulls"]["follow_firing_rate"]["mean_spearman"] == 1.0
    for result in results.values():
        if result.get("scored"):
            result["twin_levels"] = {
                f: _summaries([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) for f in dr.MAPPING
            }
    dr.plot(scored, results, tmp_path / "figure.png")
    assert (tmp_path / "figure.png").stat().st_size > 0


def test_a_viability_drop_at_the_top_concentration_excludes_that_bin():
    wells, viability = _synthetic_screen(top_viability_drop=2.0)
    results = dr.analyse(wells, viability, _synthetic_twin(), {"Lindane": ["SP1"]})
    lindane = results["Lindane"]
    assert lindane["exclusion"]["first_flagged_bin"] == 3
    assert lindane["retained_bins"] == [-3, -2, -1, 0, 1, 2]
    assert lindane["scored"]


def test_score_compound_skips_a_feature_the_twin_does_not_move():
    real = {
        f: dr.flag_moving(
            _curve([0.1, 0.4, 0.9, 1.6]),
            [np.random.default_rng(i).normal(0, 0.05, 12) for i in range(10)],
            k=4,
            seed=3,
        )
        for f in dr.MAPPING
    }
    quiet = _summaries([0.0] * 7, [False] * 7)
    loud = _summaries([0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0])
    twin = {
        knob: {f: (loud if f == "synchrony_index" else quiet) for f in dr.MAPPING}
        for knob in dr.KNOB_GRIDS
    }
    got = dr.score_compound(real, twin, "gabaa_block", [0, 1, 2, 3], dr.LEVELS)
    assert got["features"]["synchrony_index"]["spearman"] == pytest.approx(1.0)
    assert got["features"]["burst_rate_per_min"]["scored"] is False
    assert len(got["shuffled_mechanism"]["synchrony_index"]) == len(dr.KNOB_GRIDS) - 1
