"""The translation table between definition sets, and its check against the published ladder."""

import numpy as np
import pandas as pd

from neuromps.models.extractor_ladder import ladder_stats
from neuromps.models.feature_definitions import (
    NOT_IN_TIER,
    PAIRS,
    agreement_definitions,
    ladder_check,
    markdown_table,
    pair_stats,
    translation,
    translation_record,
    verdict,
    verdicts,
)


def test_verdicts_follow_the_agreement_thresholds() -> None:
    assert verdict(1.04, 0.96) == "same number"
    assert verdict(0.951, 0.95) == "same number"
    assert verdict(1.2, 0.99) == "scale factor"
    assert verdict(1.0, 0.94) == "no conversion"
    assert verdict(1.0, None) == "no conversion"


def test_pair_stats_divide_target_by_reference_and_skip_zero_references() -> None:
    reference = np.array([1.0, 2.0, 4.0, 0.0])
    target = 2 * reference + np.array([0.0, 0.0, 0.0, 0.5])  # a zero reference: no ratio
    stats = pair_stats(reference, target)
    assert stats["n"] == 4
    assert stats["median_ratio"] == 2.0
    assert stats["fraction_within_10pct"] == 0.0
    assert stats["verdict"] == "scale factor"


def test_agreement_definitions_mirror_the_registry() -> None:
    d = {x.name: x for x in agreement_definitions()}
    assert d["neuromps"].noisy == "kept"
    assert d["kapucu2022"].noisy == d["kapucu2022_logisi"].noisy == "dropped_per_div"


def _table(seed: int = 0) -> pd.DataFrame:
    """Recordings on two plates: kapucu2022 = 2x ours on burst rate, equal on firing rate."""
    rng = np.random.default_rng(seed)
    rows = []
    for plate in ("hPSC_MEA1", "Rat_MEA1"):
        for i in range(12):
            base = {"plate": plate, "well": f"A{i}", "div": 7}
            ours = {"burst_rate_per_min": 1.0 + i, "mfr_hz": 0.5 + i}
            theirs = {"burst_rate_per_min": 2.0 + 2 * i, "mfr_hz": 0.5 + i}
            for feature in ours:
                values = {
                    "neuromps": ours[feature],
                    "kapucu2022": 2 * ours[feature]
                    if feature == "burst_rate_per_min"
                    else ours[feature],
                    "kapucu2022_logisi": rng.uniform(1, 30),
                }
                rows += [
                    {
                        **base,
                        "feature": feature,
                        "ours": v,
                        "theirs": theirs[feature],
                        "variant": name,
                    }
                    for name, v in values.items()
                ]
    return pd.DataFrame(rows)


def test_translation_reads_each_pair_and_the_target_against_the_depositors() -> None:
    pairs = translation(_table())
    assert set(pairs) == {f"{a}_to_{b}" for a, b in PAIRS}
    k = pairs["neuromps_to_kapucu2022"]
    assert k["burst_rate_per_min"]["median_ratio"] == 2.0
    assert k["burst_rate_per_min"]["verdict"] == "scale factor"
    assert k["mfr_hz"]["verdict"] == "same number"
    assert k["burst_rate_per_min"]["target_vs_depositors"]["median_ratio"] == 1.0
    assert pairs["neuromps_to_kapucu2022_logisi"]["mfr_hz"]["verdict"] == "no conversion"
    assert k["burst_rate_per_min"]["median_ratio_by_plate"] == {"Rat_MEA1": 2.0, "hPSC_MEA1": 2.0}
    summary = verdicts(pairs)["neuromps_to_kapucu2022"]
    assert summary == {
        "same number": ["mfr_hz"],
        "scale factor": ["burst_rate_per_min"],
        "no conversion": [],
    }


def test_ladder_check_passes_only_on_identical_published_ratios() -> None:
    table = _table()
    stats = ladder_stats(table)
    published = {
        "settings_ladder": {
            "by_variant": {
                "S0_ours": stats["neuromps"],
                "S3_rate_per_bursting_electrode": stats["kapucu2022"],
            }
        }
    }
    check = ladder_check(table, published)
    assert check["neuromps"]["pass"] and check["kapucu2022"]["pass"]
    assert check["kapucu2022_logisi"]["skipped"]
    published["settings_ladder"]["by_variant"]["S3_rate_per_bursting_electrode"]["pooled"][
        "mfr_hz"
    ]["median_ratio"] = 0.5
    check = ladder_check(table, published)
    assert not check["kapucu2022"]["pass"]
    assert check["kapucu2022"]["n_equal"] == check["kapucu2022"]["n_compared"] - 1


def test_an_absent_deposit_is_reported_not_raised() -> None:
    assert translation_record(pd.DataFrame(), {}) == {"status": NOT_IN_TIER}


def test_the_markdown_table_names_the_scope_and_every_verdict() -> None:
    record = translation_record(_table(), {})
    text = markdown_table(record["pairs"], record["n_recordings"])
    assert "24 recordings" in text
    assert "## neuromps -> kapucu2022" in text
    assert "| `burst_rate_per_min` | 24 | 2.0 | 2.0-2.0 |" in text
    assert "scale factor" in text and "same number" in text and "no conversion" in text
    assert "experimental" in text and "REAL recordings" in text


def test_the_agreement_path_and_the_registry_extract_agree(tmp_path) -> None:
    from neuromps.features.definitions import REGISTRY, extract
    from neuromps.features.mea import extract_features
    from neuromps.models.extractor_agreement import _channels_for

    channels = {f"A1_{i}": np.arange(0.0, 60.0, 0.2 * i) for i in range(1, 6)}
    noisy = {"dropped_per_div": frozenset({"A1_1"}), "dropped_all_divs": frozenset()}
    for definition in agreement_definitions():
        kept = _channels_for(channels, noisy, definition)
        expected = extract_features(kept, 60.0, definition.params)
        assert extract(REGISTRY[definition.name], channels, 60.0, ("A1_1",)) == expected


def test_only_plates_whose_spike_files_the_tier_lists_are_read(tmp_path) -> None:
    from neuromps.models.feature_definitions import plates_in_tier

    for plate in ("hPSC_MEA1", "Rat_MEA1"):
        folder = tmp_path / plate / f"{plate}_spikes_noise_explogs"
        folder.mkdir(parents=True)
        (folder / f"{plate}_DIV7_spikes.csv").write_text("Channel,Time\n")
        (folder / f"{plate}_DIV14_spikes.csv").write_text("Channel,Time\n")
    hashes = {"hPSC_MEA1_DIV7_spikes.csv": "a", "hPSC_MEA1_DIV14_spikes.csv": "b"}
    assert plates_in_tier(hashes, tmp_path) == ("hPSC_MEA1",)
    assert plates_in_tier({"hPSC_MEA1_DIV7_spikes.csv": "a"}, tmp_path) == ()


def test_checks_need_the_asset_match_coverage_on_a_full_run_and_the_ladder() -> None:
    from neuromps.models.feature_definitions import NOT_IN_TIER, checks_pass

    sidecar = {
        "check_reference_equals_asset": {"pass": True},
        "asset_spike_time_records_without_rows": 3,
    }
    assert checks_pass(sidecar, {"status": NOT_IN_TIER}, quick=True)
    assert not checks_pass(sidecar, {"status": NOT_IN_TIER}, quick=False)
    sidecar["asset_spike_time_records_without_rows"] = 0
    ladder = {"neuromps": {"pass": True}, "kapucu2022": {"pass": False}}
    assert not checks_pass(sidecar, {"ladder_check": ladder}, quick=False)
    ladder["kapucu2022"]["pass"] = True
    assert checks_pass(sidecar, {"ladder_check": ladder}, quick=False)
