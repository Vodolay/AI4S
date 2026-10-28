"""The post hoc settings ladder: our extractor under the depositors' stated settings."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuromps.features.mea import FeatureParams
from neuromps.models.extractor_agreement import (
    DEFAULT_DEFINITION,
    Definition,
    noisy_lists,
    paired_table,
)
from neuromps.models.extractor_ladder import (
    ALL_DEFINITIONS,
    LADDER,
    ONE_AT_A_TIME,
    bursting_fraction,
    changes_from,
    gap_closed,
    ladder_stats,
    plot_ladder,
    s0_reproduces,
)

PREFIX = "X_MEA1"


def _bursty(n: int, period: float, offset: float = 0.0) -> np.ndarray:
    return np.concatenate([offset + b * period + np.arange(6) * 0.01 for b in range(n)])


def _write_plate(root: Path, plate: str = "P1") -> Path:
    """A two-well, two-DIV plate in the deposit's layout, with one noisy electrode at DIV7."""
    spikes_dir = root / plate / f"{plate}_spikes_noise_explogs"
    out_dir = root / plate / f"{plate}_meaRtools_output"
    spikes_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    channels = {
        "_11": _bursty(10, 6.0),  # bursting, 60 spikes/min
        "_12": np.arange(0.5, 60.0, 7.5),  # 8 spikes/min: active at >=5, not at >10
        "_13": np.arange(1.0, 60.0, 1.0),  # tonic, 59 spikes/min, never bursts
        "_14": np.arange(0.2, 60.0, 0.5),  # tonic, 120 spikes/min, the noisy one at DIV7
    }
    for div in (7, 14):
        rows = [
            (f"{well}{suffix}", float(t))
            for well in ("A1", "A2")
            for suffix, times in channels.items()
            for t in times
        ]
        frame = pd.DataFrame(rows, columns=["Channel", "Time"]).sort_values("Time")
        frame.to_csv(spikes_dir / f"{PREFIX}_DIV{div}_spikes.csv", index=False)
    (spikes_dir / f"noisy_electrodes_{plate}.csv").write_text("DIV7,DIV14\nA1_14,\n")
    for stem in ("nae", "bursts_per_min", "meanfiringrate_by_active_electordes"):
        (out_dir / f"{PREFIX}_{stem}.csv").write_text(
            "well,treatment,divDIV7,divDIV14\nA1,NoTreatment,2.0,3.0\nA2,NoTreatment,2.5,3.5\n"
        )
    return root


def _value(table: pd.DataFrame, variant: str, well: str, div: int, feature: str) -> float:
    rows = table[
        (table["variant"] == variant)
        & (table["well"] == well)
        & (table["div"] == div)
        & (table["feature"] == feature)
    ]
    assert len(rows) == 1
    return float(rows["ours"].iloc[0])


@pytest.fixture(scope="module")
def plate_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_plate(tmp_path_factory.mktemp("kapucu"))


@pytest.fixture(scope="module")
def ladder_table(plate_root: Path) -> pd.DataFrame:
    return paired_table(("P1",), root=plate_root, definitions=ALL_DEFINITIONS, processes=2)


def test_s0_rows_of_the_ladder_are_the_published_pairing_exactly(
    plate_root: Path, ladder_table: pd.DataFrame
) -> None:
    """The ladder's S0 must be the existing agreement's table, row for row."""
    published = paired_table(("P1",), root=plate_root, processes=2)
    assert "variant" not in published.columns
    s0 = ladder_table[ladder_table["variant"] == DEFAULT_DEFINITION.name]
    s0 = s0.drop(columns="variant").reset_index(drop=True)
    pd.testing.assert_frame_equal(s0, published.reset_index(drop=True))


def test_noisy_electrodes_are_dropped_only_for_their_own_div(ladder_table: pd.DataFrame) -> None:
    s0 = _value(ladder_table, "S0_ours", "A1", 7, "active_electrodes")
    dropped = _value(ladder_table, "only_noisy_dropped", "A1", 7, "active_electrodes")
    assert s0 == 4 and dropped == 3
    # A1_14 is listed under DIV7 only; at DIV14 it stays, and well A2 is never touched.
    assert _value(ladder_table, "only_noisy_dropped", "A1", 14, "active_electrodes") == 4
    assert _value(ladder_table, "only_noisy_dropped", "A2", 7, "active_electrodes") == 4


def test_the_all_divs_reading_drops_a_listed_electrode_everywhere(
    ladder_table: pd.DataFrame,
) -> None:
    """The assumption check: the union of every DIV column, at S1's threshold."""
    assert _value(ladder_table, "check_noisy_dropped_all_divs", "A1", 14, "active_electrodes") == 2
    assert _value(ladder_table, "S2_noisy_dropped", "A1", 14, "active_electrodes") == 3


def test_the_strict_threshold_and_the_denominator_move_what_they_should(
    ladder_table: pd.DataFrame,
) -> None:
    # A1_12 fires 8 spikes/min: active for S0, not for S1.
    assert _value(ladder_table, "S1_active_gt10", "A2", 7, "active_electrodes") == 3
    # Ten bursts on the one bursting electrode; the file lasts until its last spike, 59.7 s.
    minutes = 59.7 / 60.0
    s2 = _value(ladder_table, "S2_noisy_dropped", "A1", 7, "burst_rate_per_min")
    s3 = _value(ladder_table, "S3_rate_per_bursting_electrode", "A1", 7, "burst_rate_per_min")
    assert s3 == pytest.approx(10.0 / minutes)  # per bursting electrode: A1_11 only
    assert s2 == pytest.approx(10.0 / 2 / minutes)  # per active: A1_11 and A1_13


def test_noisy_lists_reads_one_column_and_the_union(tmp_path: Path) -> None:
    path = tmp_path / "noisy.csv"
    path.write_text("DIV3,DIV7\nB2_32,B2_32\nA4_75, C4_86\n")
    lists = noisy_lists(path, 7)
    assert lists["dropped_per_div"] == frozenset({"B2_32", "C4_86"})
    assert lists["dropped_all_divs"] == frozenset({"B2_32", "A4_75", "C4_86"})
    # A DIV the file has no column for drops nothing rather than borrowing another DIV's list.
    assert noisy_lists(path, 99)["dropped_per_div"] == frozenset()
    assert noisy_lists(None, 7) == {"dropped_per_div": frozenset(), "dropped_all_divs": frozenset()}


def test_definitions_change_one_thing_at_a_time() -> None:
    names = [d.name for d in LADDER]
    assert names == [
        "S0_ours",
        "S1_active_gt10",
        "S2_noisy_dropped",
        "S3_rate_per_bursting_electrode",
    ]
    assert changes_from(LADDER[0], LADDER[1]) == ["active_min_spikes_per_min", "active_strict"]
    assert changes_from(LADDER[1], LADDER[2]) == ["noisy"]
    assert changes_from(LADDER[2], LADDER[3]) == ["burst_rate_denominator"]
    for one in ONE_AT_A_TIME:
        changed = changes_from(LADDER[0], one)
        assert changed in (
            ["active_min_spikes_per_min", "active_strict"],
            ["noisy"],
            ["burst_rate_denominator"],
        ), one.name
    # The burst detector itself is never changed: that difference is not tested.
    for d in ALL_DEFINITIONS:
        assert d.params.burst_max_isi_s == FeatureParams().burst_max_isi_s
        assert d.params.burst_min_spikes == FeatureParams().burst_min_spikes
        assert d.params.sttc_dt_s == FeatureParams().sttc_dt_s


def test_changes_from_covers_every_field() -> None:
    a = Definition("a")
    b = Definition("b", FeatureParams(sttc_dt_s=0.01), noisy="dropped_per_div")
    assert changes_from(a, b) == ["sttc_dt_s", "noisy"]
    assert len([f.name for f in fields(FeatureParams)]) >= 10


def _ladder_frame(spec: dict[tuple[str, str], tuple[list[float], list[float]]]) -> pd.DataFrame:
    rows = []
    for (variant, plate), (ours, theirs) in spec.items():
        for i, (o, t) in enumerate(zip(ours, theirs, strict=True)):
            rows.append(
                {
                    "variant": variant,
                    "plate": plate,
                    "well": f"W{i}",
                    "div": i,
                    "feature": "burst_rate_per_min",
                    "ours": o,
                    "theirs": t,
                }
            )
    return pd.DataFrame(rows)


def test_ladder_stats_per_plate_and_pooled() -> None:
    theirs = [1.0, 2.0, 3.0, 4.0]
    table = _ladder_frame(
        {
            ("S0_ours", "P1"): ([0.5 * v for v in theirs], theirs),
            ("S0_ours", "P2"): (list(theirs), theirs),
        }
    )
    stats = ladder_stats(table)["S0_ours"]
    assert stats["P1"]["burst_rate_per_min"]["median_ratio"] == 0.5
    assert stats["P2"]["burst_rate_per_min"]["median_ratio"] == 1.0
    assert stats["P2"]["burst_rate_per_min"]["fraction_within_10pct"] == 1.0
    pooled = stats["pooled"]["burst_rate_per_min"]
    assert pooled["n"] == 8
    assert pooled["median_ratio"] == 0.75
    assert pooled["fraction_within_10pct"] == 0.5
    assert stats["P1"]["burst_rate_per_min"]["spearman_rho"] == 1.0


def test_gap_closed_is_the_share_of_the_log_ratio_removed() -> None:
    theirs = [1.0, 2.0, 3.0, 4.0]
    table = _ladder_frame(
        {
            ("S0_ours", "P1"): ([0.5 * v for v in theirs], theirs),
            ("S3_rate_per_bursting_electrode", "P1"): (list(theirs), theirs),
            ("S0_ours", "P2"): ([0.5 * v for v in theirs], theirs),
            ("S3_rate_per_bursting_electrode", "P2"): ([0.25 * v for v in theirs], theirs),
            ("S0_ours", "P3"): (list(theirs), theirs),
            ("S3_rate_per_bursting_electrode", "P3"): ([0.5 * v for v in theirs], theirs),
        }
    )
    gap = gap_closed(table)["burst_rate_per_min"]
    assert gap["P1"] == pytest.approx(1.0)
    assert gap["P2"] == pytest.approx(-1.0)  # the gap doubled in log terms
    assert gap["P3"] is None  # nothing to close
    assert "pooled" in gap


def test_bursting_fraction_is_the_s2_over_s3_rate() -> None:
    table = _ladder_frame(
        {
            ("S2_noisy_dropped", "P1"): ([1.0, 2.0, 3.0, 0.0], [1.0] * 4),
            ("S3_rate_per_bursting_electrode", "P1"): ([2.0, 8.0, 3.0, 0.0], [1.0] * 4),
        }
    )
    # Wells with no bursts carry no bursting fraction; the rest are 0.5, 0.25 and 1.0.
    assert bursting_fraction(table)["P1"] == pytest.approx(0.5)


def test_s0_check_compares_every_published_statistic() -> None:
    published = {
        "mfr_hz": {
            "n_paired": 4,
            "spearman_rho": 1.0,
            "median_ratio_ours_over_theirs": 0.5,
            "fraction_within_10pct": 0.0,
        },
        "isi_cv_like": {"n": 2, "note": "too few paired points"},
    }
    ladder = {
        "mfr_hz": {"n": 4, "spearman_rho": 1.0, "median_ratio": 0.5, "fraction_within_10pct": 0.0}
    }
    assert s0_reproduces(ladder, published) is True
    ladder["mfr_hz"]["median_ratio"] = 0.5001
    assert s0_reproduces(ladder, published) is False
    assert s0_reproduces({}, published) is False


def test_the_ladder_plot_creates_its_directory(tmp_path: Path) -> None:
    theirs = [1.0, 2.0, 3.0, 4.0]
    spec = {}
    for d in LADDER:
        for plate in ("hPSC_MEA1", "Rat_MEA1"):
            spec[(d.name, plate)] = ([0.5 * v for v in theirs], theirs)
    table = _ladder_frame(spec)
    for feature in ("active_electrodes", "mfr_hz", "synchrony_index"):
        extra = table.copy()
        extra["feature"] = feature
        table = pd.concat([table, extra])
    target = tmp_path / "quick" / "figures" / "extractor_settings_ladder_real.png"
    assert plot_ladder(ladder_stats(table), target).exists()
