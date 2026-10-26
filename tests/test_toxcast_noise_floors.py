"""Tests for the ToxCast vehicle-well noise floors (synthetic inputs only)."""

import numpy as np
import pandas as pd
import pytest

from neuromps.models import toxcast_noise_floors as nf


def _nested(n_groups: int, per_group: int, sigma_b: float, sigma_w: float, seed: int = 1):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    values = rng.normal(0, sigma_b, n_groups)[groups] + rng.normal(0, sigma_w, groups.size)
    return values, groups


def test_log2_change_from_percent():
    got = nf.log2_change(np.array([0.0, 100.0, -50.0, -100.0, -99.0, np.nan]))
    assert got[:3] == pytest.approx([0.0, 1.0, -1.0])
    assert got[3] == pytest.approx(np.log2(0.05))
    assert got[4] == pytest.approx(np.log2(0.05))  # below the floor is floored too
    assert np.isnan(got[5])


def test_variance_components_match_the_textbook_formula():
    values = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 10.0, 11.0])
    groups = np.array([0, 0, 0, 1, 1, 2, 2])
    got = nf.variance_components(values, groups)
    sizes = np.array([3, 2, 2])
    means = np.array([2.0, 5.5, 10.5])
    grand = values.mean()
    msb = (sizes * (means - grand) ** 2).sum() / 2
    msw = (((values - means[groups]) ** 2).sum()) / (7 - 3)
    n0 = (7 - (sizes**2).sum() / 7) / 2
    assert got["sigma_w2"] == pytest.approx(msw)
    assert got["sigma_b2"] == pytest.approx((msb - msw) / n0)
    assert got["share"] == pytest.approx(got["sigma_b2"] / (got["sigma_b2"] + msw))


def test_variance_components_recover_a_known_share():
    values, groups = _nested(400, 8, sigma_b=1.0, sigma_w=1.0)
    assert nf.variance_components(values, groups)["share"] == pytest.approx(0.5, abs=0.06)


def test_share_is_truncated_at_zero():
    values, groups = _nested(50, 8, sigma_b=0.0, sigma_w=1.0, seed=3)
    got = nf.variance_components(values, groups)
    assert got["sigma_b2"] >= 0 and got["share"] >= 0


def test_detection_of_a_gaussian_null():
    null = np.random.default_rng(0).normal(0, 1.0, 200_000)
    got = nf.detection(null)
    assert got["threshold"] == pytest.approx(1.96, abs=0.03)
    assert got["detectable_down"] == pytest.approx(1.96 + 0.84, abs=0.04)
    assert got["detectable_up"] == pytest.approx(1.96 + 0.84, abs=0.04)


def test_independent_design_averages_down_and_one_date_does_not():
    values, groups = _nested(300, 8, sigma_b=0.5, sigma_w=1.0)
    model = nf.null_model(values, groups)
    rng = np.random.default_rng(0)
    ind = [
        nf.detection(nf.null_draws(model, n, "independent", 20_000, rng))["threshold"]
        for n in (1, 4, 64)
    ]
    one = nf.detection(nf.null_draws(model, 64, "one_date", 20_000, rng))["threshold"]
    total = np.sqrt(0.25 + 1.0)
    assert ind[0] == pytest.approx(1.96 * total, rel=0.08)
    assert ind[1] == pytest.approx(1.96 * total / 2, rel=0.08)
    # one date: the date's effect stays, so 64 wells cannot get below ~1.96 * sigma_b
    assert one == pytest.approx(1.96 * np.sqrt(0.25 + 1.0 / 64), rel=0.1)
    assert one > 3 * ind[2]


def test_same_date_vehicles_cancel_the_date_effect():
    values, groups = _nested(300, 8, sigma_b=2.0, sigma_w=1.0)
    model = nf.null_model(values, groups)
    rng = np.random.default_rng(0)
    got = nf.detection(nf.null_draws(model, 16, "same_date_vehicle", 20_000, rng))["threshold"]
    assert got == pytest.approx(1.96 * np.sqrt(2 / 16), rel=0.1)


def test_wells_needed_picks_the_smallest_grid_n():
    curve = pd.DataFrame({"n": [1, 2, 4, 8], "detectable_down": [2.0, 1.4, 1.0, 0.7]})
    assert nf.wells_needed(curve, 1.0, "detectable_down") == 4
    assert nf.wells_needed(curve, 0.5, "detectable_down") is None


def test_resampled_one_date_uses_only_dates_with_enough_wells():
    values = np.array([0.0, 0.0, 0.0, 10.0])
    groups = np.array([0, 0, 0, 1])
    draws = nf.resampled_one_date(values, groups, 2, 100, np.random.default_rng(0))
    assert np.all(draws == 0.0)  # date 1 has one well and cannot supply two


def _wells(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for date in range(30):
        shift = rng.normal(0, 0.2)
        for well in range(8):
            change = 2 ** (shift + rng.normal(0, 0.3)) - 1
            rows.append(
                {
                    "acid": 2448,
                    "wllt": "n",
                    "spid": "DMSO" if well else "Water",
                    "apid": f"d{date}",
                    "plate": f"d{date}_p{well % 3}",
                    "rval": 100 * change,
                }
            )
    rows.append(
        {"acid": 2448, "wllt": "n", "spid": "DMSO", "apid": "d0", "plate": "d0_p0", "rval": np.nan}
    )
    rows.append(
        {
            "acid": 2448,
            "wllt": "p",
            "spid": "Bicuculline",
            "apid": "d0",
            "plate": "d0_p0",
            "rval": 500.0,
        }
    )
    return pd.DataFrame(rows)


def test_analyse_component_end_to_end():
    record, by_design = nf.analyse_component(_wells(), 2448, quick=True)
    assert record["wells"] == 240 and record["rval_missing"] == 1 and record["dates"] == 30
    assert record["dmso_only"]["wells"] == 210
    assert 0 < record["date_share"] < 1
    one, ind = by_design["one_date"], by_design["independent"]
    assert one["threshold"].iloc[-1] > ind["threshold"].iloc[-1]
    needed = record["wells_needed"]
    assert needed["independent"]["down_50pct"] <= needed["independent"]["down_20pct"]
    table = nf.markdown_table({"firing_rate_mean": record})
    assert "toxcast_noise_floors.components.firing_rate_mean.date_share" in table
