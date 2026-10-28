"""Tests for the registered human-vs-rat split-half noise comparison (synthetic inputs only)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from neuromps.features.mea import extract_features
from neuromps.models import hpsc_split_half as sh

# ------------------------------------------------------------------ registration


def test_constants_match_the_registration():
    reg = json.loads(sh.REGISTRATION.read_text())
    method = reg["method"]
    assert list(sh.FEATURES) == method["features"]
    assert str(sh.SEED) in method["ratio"] and f"{sh.RESAMPLES:,}" in method["ratio"]
    assert str(sh.FLOOR) in method["change"]
    assert f"{sh.MAD_SCALE}" in method["noise"]
    assert f"{sh.Z_ALPHA}" in reg["reported_not_tested"]["wells_needed"]
    assert f"{sh.Z_POWER}" in reg["reported_not_tested"]["wells_needed"]
    assert reg["outputs"]["metrics_key"] == sh.ROOT
    assert reg["outputs"]["results_file"].startswith(f"outputs/{sh.RESULTS_NAME}")
    assert reg["outputs"]["figure"] == f"outputs/figures/{sh.FIGURE_NAME}"


# ------------------------------------------------------------------ the split


def test_split_cuts_at_half_and_shifts_the_second_half_to_zero():
    channels = {"A1_11": np.array([0.5, 2.0, 4.99, 5.0, 7.5, 10.0])}
    first, second, half = sh.split_halves(channels, 10.0)
    assert half == 5.0
    np.testing.assert_allclose(first["A1_11"], [0.5, 2.0, 4.99])
    # A spike exactly at the cut starts the second half; the last spike ends it.
    np.testing.assert_allclose(second["A1_11"], [0.0, 2.5, 5.0])


def test_split_keeps_every_channel_even_when_one_half_is_empty():
    channels = {"A1_11": np.array([1.0, 2.0]), "A1_12": np.array([8.0])}
    first, second, _ = sh.split_halves(channels, 10.0)
    assert set(first) == set(second) == {"A1_11", "A1_12"}
    assert first["A1_12"].size == 0 and second["A1_11"].size == 0


def test_half_features_come_from_the_shared_extractor():
    rng = np.random.default_rng(0)
    channels = {f"A1_{i}": np.sort(rng.uniform(0, 100, 400)) for i in range(12)}
    row = sh.well_features(channels, 100.0)
    first, second, half = sh.split_halves(channels, 100.0)
    assert row["full"] == extract_features(channels, 100.0)
    assert row["first"] == extract_features(first, half)
    assert row["second"] == extract_features(second, half)


# ------------------------------------------------------------------ usable wells


@pytest.mark.parametrize(
    ("treatment", "a1", "a2", "untreated_only", "usable"),
    [
        ("notreatment", 10, 10, False, True),
        ("notreatment", 9, 16, False, False),
        ("notreatment", 16, 9, False, False),
        ("excludedwell", 16, 16, False, False),
        (None, 16, 16, False, True),  # absent from the log: not ExcludedWell
        (None, 16, 16, True, False),  # replication: NoTreatment wells only
        ("gabazine", 16, 16, True, False),
        ("notreatment", 16, 16, True, True),
    ],
)
def test_usable_well_rule(treatment, a1, a2, untreated_only, usable):
    assert sh.is_usable(treatment, a1, a2, untreated_only) is usable


# ------------------------------------------------------------------ change transform


def test_change_is_log2_of_the_ratio():
    assert sh.change(2.0, 4.0) == pytest.approx(1.0)
    assert sh.change(4.0, 2.0) == pytest.approx(-1.0)
    assert sh.change(3.0, 3.0) == pytest.approx(0.0)


def test_change_floors_the_ratio_at_five_percent():
    assert sh.change(10.0, 0.0) == pytest.approx(math.log2(0.05))
    assert sh.change(10.0, 0.1) == pytest.approx(math.log2(0.05))
    assert sh.change(1.0, -0.2) == pytest.approx(math.log2(0.05))  # STTC can go negative
    assert sh.change(10.0, 0.6) == pytest.approx(math.log2(0.06))


@pytest.mark.parametrize(("f1", "f2"), [(None, 1.0), (1.0, None), (0.0, 1.0), (-0.1, 1.0)])
def test_change_is_unscored_when_a_half_is_undefined_or_the_first_is_not_positive(f1, f2):
    assert sh.change(f1, f2) is None


# ------------------------------------------------------------------ noise


def test_sd_and_robust_sd():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert sh.sd(x) == pytest.approx(np.std(x, ddof=1))
    assert sh.robust_sd(x) == pytest.approx(1.4826 * 1.0)
    assert math.isnan(sh.sd(np.array([1.0])))


def test_robust_sd_matches_sd_on_a_large_normal_sample():
    x = np.random.default_rng(1).normal(0, 2.0, 200_000)
    assert sh.robust_sd(x) == pytest.approx(2.0, rel=0.02)


# ------------------------------------------------------------------ ratio bootstrap


def _singletons(x: np.ndarray) -> list[np.ndarray]:
    return [np.array([v]) for v in x]


def test_ratio_interval_brackets_a_known_ratio():
    rng = np.random.default_rng(2)
    human = _singletons(rng.normal(0, 2.0, 200))
    rat = _singletons(rng.normal(0, 1.0, 200))
    got = sh.ratio_interval(human, rat, 2000, 20261028)
    assert got["R"] == pytest.approx(2.0, rel=0.15)
    low, high = got["ci95"]
    assert low < got["R"] < high and low > 1.0 and high < 3.0
    assert got["resamples"] == 2000 and got["nonfinite"] == 0


def test_ratio_interval_is_reproducible_and_seeded():
    rng = np.random.default_rng(3)
    human, rat = _singletons(rng.normal(size=15)), _singletons(rng.normal(size=15))
    a = sh.ratio_interval(human, rat, 500, 20261028)
    b = sh.ratio_interval(human, rat, 500, 20261028)
    c = sh.ratio_interval(human, rat, 500, 1)
    assert a == b and a["ci95"] != c["ci95"]


def test_ratio_interval_resamples_whole_wells():
    # Each well's recordings move together: resampling recordings would narrow the interval.
    rng = np.random.default_rng(4)
    human = [np.full(10, v) for v in rng.normal(0, 1.0, 8)]
    rat = [np.full(10, v) for v in rng.normal(0, 1.0, 8)]
    flat = sh.ratio_interval(
        _singletons(np.concatenate(human)), _singletons(np.concatenate(rat)), 1000, 5
    )
    grouped = sh.ratio_interval(human, rat, 1000, 5)
    assert grouped["R"] == pytest.approx(flat["R"])
    width = grouped["ci95"][1] - grouped["ci95"][0]
    assert width > 2 * (flat["ci95"][1] - flat["ci95"][0])


# ------------------------------------------------------------------ wells needed


def test_wells_needed_formula():
    sd = 0.3
    want = math.ceil(2 * (1.959964 + 0.841621) ** 2 * sd**2 / math.log2(0.8) ** 2)
    assert sh.wells_needed(sd, 0.2) == want
    assert sh.wells_needed(sd, 0.5) == math.ceil(2 * (1.959964 + 0.841621) ** 2 * 0.09)
    assert sh.wells_needed(2 * sd, 0.2) >= 4 * want - 3


# ------------------------------------------------------------------ predictions


def _feature(sd_h: float, sd_r: float, scored: bool = True) -> dict:
    return {
        "scored": scored,
        "human": {"sd": sd_h},
        "rat": {"sd": sd_r},
        "R": sd_h / sd_r if scored else None,
    }


@pytest.mark.parametrize(
    ("ci", "wells", "outcome"),
    [
        ([1.1, 2.0], (10, 10), "pass"),
        ([0.9, 2.0], (10, 10), "fail"),
        ([0.5, 0.9], (30, 30), "fail"),
        ([1.1, 2.0], (9, 30), "uninterpretable"),
        ([1.1, 2.0], (30, 9), "uninterpretable"),
    ],
)
def test_p1_logic(ci, wells, outcome):
    pool = {
        "usable_wells": {"human": wells[0], "rat": wells[1]},
        "features": {"mfr_hz": {"R": 1.5, "ci95": ci, "scored": True}},
    }
    got = sh.predict_p1(pool)
    assert got["outcome"] == outcome
    assert got["pass"] is (outcome == "pass")
    assert got["ci95"] == ci and got["R"] == 1.5


def test_p1_boundary_interval_touching_one_fails():
    pool = {
        "usable_wells": {"human": 20, "rat": 20},
        "features": {"mfr_hz": {"R": 1.5, "ci95": [1.0, 2.0], "scored": True}},
    }
    assert sh.predict_p1(pool)["outcome"] == "fail"


def test_p2_logic():
    noisier = {f"f{i}": _feature(2.0, 1.0) for i in range(5)}
    quieter = {f"g{i}": _feature(1.0, 2.0) for i in range(3)}
    got = sh.predict_p2({"features": {**noisier, **quieter}})
    assert got["outcome"] == "pass" and got["pass"] is True
    assert got["features_scored"] == 8 and got["human_noisier"] == 5
    four = {**{f"f{i}": _feature(2.0, 1.0) for i in range(4)}, **quieter}
    assert sh.predict_p2({"features": four})["outcome"] == "fail"
    few = {
        **{f"f{i}": _feature(2.0, 1.0) for i in range(4)},
        **{f"u{i}": _feature(2.0, 1.0, scored=False) for i in range(4)},
    }
    got = sh.predict_p2({"features": few})
    assert got["outcome"] == "uninterpretable" and got["pass"] is False
    assert got["features_scored"] == 4  # unscored features are not counted as noisier


@pytest.mark.parametrize(
    ("R", "wells", "outcome"),
    [
        (1.2, (6, 6), "pass"),
        (0.8, (6, 6), "fail"),
        (1.0, (6, 6), "fail"),
        (1.2, (5, 12), "uninterpretable"),
    ],
)
def test_p3_logic(R, wells, outcome):
    pool = {
        "usable_wells": {"human": wells[0], "rat": wells[1]},
        "features": {"mfr_hz": {"R": R, "ci95": [0.5, 2.0], "scored": True}},
    }
    got = sh.predict_p3(pool)
    assert got["outcome"] == outcome and got["R"] == R


def test_p3_without_the_replication_plates_is_uninterpretable():
    got = sh.predict_p3(None)
    assert got["outcome"] == "uninterpretable" and got["pass"] is False


# ------------------------------------------------------------------ overlap check


def test_overlap_selection_keeps_wells_inside_both_ranges():
    human = np.linspace(4.0, 14.0, 11)  # p10 = 5, p90 = 13
    rat = np.linspace(1.0, 11.0, 11)  # p10 = 2, p90 = 10
    low, high = sh.overlap_range(human, rat)
    assert (low, high) == pytest.approx((5.0, 10.0))
    keep_h, keep_r = sh.in_range(human, low, high), sh.in_range(rat, low, high)
    assert human[keep_h].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert rat[keep_r].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def test_overlap_range_is_none_when_the_ranges_do_not_meet():
    assert sh.overlap_range(np.linspace(10, 20, 11), np.linspace(1, 5, 11)) is None


# ------------------------------------------------------------------ end to end


def _rows(species: str, n: int, noise: float, rng, pool: str = "primary", reps: int = 1):
    rows = []
    for w in range(n):
        level = rng.uniform(2.0, 8.0)
        for r in range(reps):
            row = {
                "pool": pool,
                "species": species,
                "plate": f"{species}_{pool}",
                "well": f"W{w}",
                "unit": f"{species}_{pool}:W{w}",
                "div": 30 + r,
                "treatment": "notreatment",
                "n_noisy": 1,
                "active_full": 16,
                "active_1": 16,
                "active_2": 16,
                "usable": True,
                "mfr_full": level,
            }
            for f in sh.FEATURES:
                row[f"d_{f}"] = float(rng.normal(0, noise))
            rows.append(row)
    return rows


def test_end_to_end_finds_the_noisier_human_set():
    rng = np.random.default_rng(7)
    rows = (
        _rows("human", 30, 0.6, rng)
        + _rows("rat", 30, 0.2, rng)
        + _rows("human", 8, 0.6, rng, "replication", reps=4)
        + _rows("rat", 8, 0.2, rng, "replication", reps=4)
    )
    # Two unusable wells, which must not count.
    bad = _rows("rat", 2, 5.0, rng)
    for row in bad:
        row["usable"], row["unit"] = False, row["unit"] + "x"
    payload = sh.analyse(pd.DataFrame(rows + bad), resamples=500)
    preds = payload["predictions"]
    assert {p["outcome"] for p in preds.values()} == {"pass"}
    assert payload["primary"]["usable_wells"] == {"human": 30, "rat": 30}
    assert payload["replication"]["usable_wells"] == {"human": 8, "rat": 8}
    assert payload["replication"]["recording_files"] == {"human": 4, "rat": 4}
    assert payload["primary"]["median_active_electrodes_usable"] == {"human": 16, "rat": 16}
    assert abs(payload["primary"]["features"]["isi_cv"]["rat"]["mean"]) < 0.2
    head = payload["primary"]["features"]["mfr_hz"]
    assert head["R"] > 1.5 and head["ci95"][0] > 1.0  # constructed ratio: 3
    assert head["human"]["sd"] == pytest.approx(
        np.std([r["d_mfr_hz"] for r in rows[:30]], ddof=1), abs=1e-4
    )
    needed = payload["wells_needed"]
    assert needed["human"]["down_20pct"] > needed["rat"]["down_20pct"]
    assert "LOWER BOUND" in needed["note"]
    check = payload["assumption_check"]
    assert check["kept"]["human"] > 0 and check["kept"]["rat"] > 0
    assert check["other_side_of_one"] is False


def test_end_to_end_unscored_changes_are_counted():
    rng = np.random.default_rng(8)
    rows = _rows("human", 12, 0.5, rng) + _rows("rat", 12, 0.5, rng)
    for row in rows[:5]:
        row["d_burst_duration_s"] = None
    payload = sh.analyse(pd.DataFrame(rows), resamples=200)
    feature = payload["primary"]["features"]["burst_duration_s"]
    assert feature["human"]["unscored"] == 5 and feature["human"]["n_wells"] == 7
    assert feature["scored"] is False and feature["R"] is None
    assert payload["predictions"]["P3_replicates_on_developmental_plates"]["outcome"] == (
        "uninterpretable"
    )


def test_recording_rows_read_a_synthetic_plate(tmp_path):
    rng = np.random.default_rng(9)
    rows = [
        (f"{well}_{e}", t)
        for well in ("A1", "A2")
        for e in range(11, 25)
        for t in np.sort(rng.uniform(0, 120, 300))
    ]
    pd.DataFrame(rows, columns=["Channel", "Time"]).to_csv(tmp_path / "x_spikes.csv", index=False)
    (tmp_path / "x_expLog.csv").write_text(
        "Project,Experiment.Date,Plate.SN,DIV,Well,Treatment,Size,Dose,Units\n"
        "h,1,M,DIV1,A1,NoTreatment,,,\nh,1,M,DIV1,A2,ExcludedWell,,,\nh,1,M,DIV1,A3,ExcludedWell,,,\n"
    )
    (tmp_path / "noisy_electrodes_x.csv").write_text("DIV1,DIV30\nA1_11,A1_12\n,A1_13\n")
    rec = sh.Recording(
        "replication",
        "human",
        "X",
        30,
        tmp_path / "x_spikes.csv",
        tmp_path / "x_expLog.csv",
        tmp_path / "noisy_electrodes_x.csv",
    )
    got = {r["well"]: r for r in sh.recording_rows(rec)}
    assert set(got) == {"A1", "A2", "A3"}
    assert got["A1"]["usable"] is True and got["A1"]["n_noisy"] == 2
    assert got["A2"]["usable"] is False and got["A3"]["active_1"] == 0
    assert got["A1"]["duration_s"] == pytest.approx(max(t for _, t in rows))
    assert got["A1"]["d_mfr_hz"] is not None


def test_plot_writes_a_real_data_figure(tmp_path):
    rng = np.random.default_rng(10)
    table = pd.DataFrame(_rows("human", 12, 0.5, rng) + _rows("rat", 12, 0.3, rng))
    payload = sh.analyse(table, resamples=100)
    path = tmp_path / "fig.png"
    title = sh.plot(payload, table, path)
    assert path.exists() and title.startswith("REAL data")
