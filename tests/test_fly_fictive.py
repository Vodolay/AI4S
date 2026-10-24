"""Phase between two rhythmic signals, on sine waves with known offsets."""

from __future__ import annotations

import numpy as np
import pytest

from neuromps.fly import fictive

T = np.arange(1800) * 0.001


def _wave(offset_cycles: float, hz: float = 6.0) -> np.ndarray:
    return 1.0 + np.sin(2 * np.pi * (hz * T - offset_cycles))


@pytest.mark.parametrize("offset", [0.0, 0.25, 0.5, 0.75])
def test_phase_recovers_a_known_offset(offset: float) -> None:
    got = fictive.phase(_wave(0.0), _wave(offset))
    assert fictive.circular_distance(got, offset) < 0.03


def test_circular_distance_wraps_around() -> None:
    assert fictive.circular_distance(0.95, 0.05) == pytest.approx(0.1)
    assert fictive.circular_distance(0.5, 0.5) == 0.0


def test_a_flat_reference_has_no_phase() -> None:
    assert fictive.phase(np.ones(1800), _wave(0.3)) is None


def test_stage2a_scoring_reads_the_registered_rules() -> None:
    import json

    import pandas as pd

    spec = json.loads(fictive.REGISTRATION.read_text())
    rows = []
    for drive in fictive.DRIVES:
        for seed in range(8):
            walking = drive == "forward_walking"
            rows.append(
                {
                    "drive": drive,
                    "seed": seed,
                    "rhythmicity_L1_flexor": 0.6 if walking else 0.1,
                    "phase_L1flex_L1ext": 0.5,
                    "phase_L1_R1": 0.48,
                    "phase_L1_L2": 0.52,
                    "phase_L1_L3": 0.02,
                    "rate_L1_flexor": 0.0 if drive == "none" else 5.0,
                }
            )
    got = fictive.score(pd.DataFrame(rows), spec)
    assert all(v["outcome"] == "pass" for v in got.values())
