"""The rate-tracking test on four-chamber cultures with instrumented tunnels."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neuromps.models import tunnels


def _record(device: str, compartment: str, rate: float, links: list[dict]) -> dict:
    return {
        "source": "brewer2021",
        "device.device_id": device,
        "device.compartment_id": compartment,
        "features.mfr_hz": rate,
        "coupling": json.dumps(links),
    }


def _link(to: str, asymmetry: float | None) -> dict:
    return {
        "to_compartment": to,
        "method": "cross-correlation",
        "asymmetry": asymmetry,
        "peak_correlation": 0.4,
        "lag_ms": 3.0,
    }


def test_a_pair_is_counted_once_not_in_both_orientations() -> None:
    """Both orientations are the same measurement; counting both halves every interval."""
    rows = [
        _record("d1", "A", 4.0, [_link("B", 0.3)]),
        _record("d1", "B", 2.0, [_link("A", -0.3)]),
    ]
    pairs = tunnels.chamber_pairs(pd.DataFrame(rows))
    assert len(pairs) == 1
    assert pairs.iloc[0]["pair"] == "A->B"
    assert pairs.iloc[0]["log2_rate_ratio"] == 1.0


def test_a_link_below_the_noise_floor_is_dropped() -> None:
    """`directed_coupling` returns a null asymmetry when there is none to apportion."""
    rows = [
        _record("d1", "A", 4.0, [_link("B", None)]),
        _record("d1", "B", 2.0, [_link("A", None)]),
    ]
    assert tunnels.chamber_pairs(pd.DataFrame(rows)).empty


def test_a_silent_chamber_cannot_contribute_a_rate_ratio() -> None:
    rows = [
        _record("d1", "A", 0.0, [_link("B", 0.3)]),
        _record("d1", "B", 2.0, [_link("A", -0.3)]),
    ]
    assert tunnels.chamber_pairs(pd.DataFrame(rows)).empty


def test_the_correlation_is_resampled_over_cultures_not_pairs() -> None:
    """Six pairs of one culture share four populations; they are not six observations."""
    rng = np.random.default_rng(0)
    rows = []
    for device in range(6):
        for pair in range(4):
            rows.append(
                {
                    "device_id": f"d{device}",
                    "pair": f"p{pair}",
                    "asymmetry": float(rng.normal()),
                    "log2_rate_ratio": float(rng.normal()),
                    "peak_correlation": 0.5,
                    "lag_ms": 1.0,
                }
            )
    result = tunnels.rate_tracking(pd.DataFrame(rows), seed=0)
    assert result["n_devices"] == 6
    assert result["n_pairs"] == 24
    low, high = result["spearman_rho_ci95"]
    assert low < high


def test_a_perfect_rate_relationship_is_detected() -> None:
    """The test must be able to see the effect 7.2 predicts, or a null means nothing."""
    rows = []
    for device in range(8):
        for i, ratio in enumerate((-2.0, -1.0, 1.0, 2.0)):
            rows.append(
                {
                    "device_id": f"d{device}",
                    "pair": f"p{i}",
                    "asymmetry": ratio / 4,
                    "log2_rate_ratio": ratio,
                    "peak_correlation": 0.5,
                    "lag_ms": 1.0,
                }
            )
    result = tunnels.rate_tracking(pd.DataFrame(rows), seed=0)
    assert result["spearman_rho"] > 0.9
    assert result["excludes_zero"]


def test_an_asset_without_the_source_is_skipped_quietly() -> None:
    assert tunnels.run(pd.DataFrame({"source": ["kapucu2022"]})) is None

