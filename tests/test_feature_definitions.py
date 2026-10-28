"""Named definition sets: a registry of `FeatureParams` bundles plus an electrode policy."""

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from neuromps.features.definitions import (
    REFERENCE,
    REGISTRY,
    STATUSES,
    DefinitionSet,
    extract,
    get,
    registry_summary,
)
from neuromps.features.mea import FeatureParams, extract_features


def _bursty(n: int, period: float, offset: float = 0.0) -> np.ndarray:
    return np.concatenate([offset + b * period + np.arange(6) * 0.01 for b in range(n)])


CHANNELS = {
    "A1_11": _bursty(10, 6.0),
    "A1_12": np.arange(0.5, 60.0, 7.5),  # 8 spikes/min: active at >=5, not at >10
    "A1_13": np.arange(1.0, 60.0, 1.0),
    "A1_14": np.arange(0.2, 60.0, 0.5),
    "A1_21": _bursty(8, 7.0, 0.3),
}


def test_the_registry_holds_exactly_the_three_sets() -> None:
    assert list(REGISTRY) == ["neuromps", "kapucu2022", "kapucu2022_logisi"]
    assert REFERENCE == "neuromps"


def test_each_set_has_a_known_status_and_the_expected_one() -> None:
    assert {d.status for d in REGISTRY.values()} <= set(STATUSES)
    assert get("neuromps").status == "reference"
    assert get("kapucu2022").status == "validated"
    assert get("kapucu2022_logisi").status == "experimental"


def test_the_reference_set_is_the_extractor_defaults_and_keeps_noisy_electrodes() -> None:
    reference = get("neuromps")
    assert reference.params == FeatureParams()
    assert reference.drop_noisy is False
    assert reference.evidence is None


def test_kapucu2022_is_the_ladders_s3() -> None:
    d = get("kapucu2022")
    assert d.params.active_min_spikes_per_min == 10.0
    assert d.params.active_strict is True
    assert d.params.burst_rate_denominator == "bursting"
    assert d.params.burst_detector == "fixed"
    assert d.drop_noisy is True
    assert d.evidence == "extractor_agreement.settings_ladder"


def test_the_logisi_set_differs_from_kapucu2022_only_in_the_detector() -> None:
    a, b = get("kapucu2022"), get("kapucu2022_logisi")
    changed = [
        f.name
        for f in fields(FeatureParams)
        if getattr(a.params, f.name) != getattr(b.params, f.name)
    ]
    assert changed == ["burst_detector"]
    assert b.params.burst_detector == "logisi"
    assert b.drop_noisy is a.drop_noisy
    assert b.evidence == "extractor_agreement.logisi"
    assert "1 of 4" in b.description and "open" in b.description


def test_every_choice_names_its_source() -> None:
    for d in REGISTRY.values():
        sources = d.choice_sources
        assert {
            "active_threshold",
            "noisy_electrodes",
            "burst_rate_denominator",
            "burst_detector",
        } <= set(sources), d.name
        assert all(isinstance(v, str) and v for v in sources.values()), d.name


def test_a_definition_set_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        get("neuromps").name = "other"  # type: ignore[misc]


def test_an_unknown_status_is_refused() -> None:
    with pytest.raises(ValueError, match="status"):
        DefinitionSet("x", "d", FeatureParams(), False, (), "draft", None)


def test_an_unknown_name_is_refused_with_the_known_ones() -> None:
    with pytest.raises(KeyError, match="neuromps"):
        get("meartools")


def test_extract_under_the_reference_is_the_extractor_itself() -> None:
    assert extract(get("neuromps"), CHANNELS, 60.0, noisy=("A1_14",)) == extract_features(
        CHANNELS, 60.0
    )


def test_extract_drops_listed_electrodes_only_when_the_set_says_so() -> None:
    d = get("kapucu2022")
    kept = {k: v for k, v in CHANNELS.items() if k != "A1_14"}
    assert extract(d, CHANNELS, 60.0, noisy=("A1_14",)) == extract_features(kept, 60.0, d.params)
    assert extract(d, CHANNELS, 60.0) == extract_features(CHANNELS, 60.0, d.params)


def test_the_noisy_list_may_name_electrodes_the_recording_lacks() -> None:
    d = get("kapucu2022")
    assert extract(d, CHANNELS, 60.0, noisy=("Z9_99",)) == extract_features(
        CHANNELS, 60.0, d.params
    )


def test_the_summary_names_status_evidence_and_what_changes_from_the_reference() -> None:
    summary = registry_summary()
    assert list(summary) == list(REGISTRY)
    assert summary["neuromps"]["changes_from_reference"] == []
    assert summary["kapucu2022"]["changes_from_reference"] == [
        "active_min_spikes_per_min",
        "active_strict",
        "burst_rate_denominator",
        "drop_noisy",
    ]
    assert summary["kapucu2022_logisi"]["evidence"] == "extractor_agreement.logisi"
    assert summary["kapucu2022"]["method_record"] == get("kapucu2022").params.as_method_record()


def test_the_ladder_and_logisi_study_build_their_s3_and_s4_from_the_registry() -> None:
    from neuromps.models.extractor_ladder import LADDER, S4_LOGISI

    assert LADDER[0].params == get("neuromps").params and LADDER[0].noisy == "kept"
    assert LADDER[3].name == "S3_rate_per_bursting_electrode"
    assert LADDER[3].params == get("kapucu2022").params
    assert LADDER[3].noisy == "dropped_per_div"
    assert S4_LOGISI.name == "S4_logisi"
    assert S4_LOGISI.params == get("kapucu2022_logisi").params
    assert S4_LOGISI.noisy == "dropped_per_div"
