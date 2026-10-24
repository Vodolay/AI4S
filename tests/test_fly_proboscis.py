"""The flybody proboscis: command mapping, rest, extension, step-size agreement (SIMULATED)."""

from __future__ import annotations

import numpy as np
import pytest

from neuromps.fly import proboscis

needs_model = pytest.mark.skipif(not proboscis.model_ready(), reason="flybody not fetched")


def test_positive_net_drive_commands_extension() -> None:
    got = proboscis.command(np.array([1.0, 0.5, 0.0, -1.0, 2.0]), "rostrum")
    np.testing.assert_allclose(got, [-1.24, -0.62, 0.0, 0.183, -1.24])


@needs_model
def test_zero_commands_leave_the_proboscis_at_rest() -> None:
    ext = proboscis.extension_mm({j: np.zeros(100) for j in proboscis.JOINTS})
    assert np.abs(ext).max() < 1e-3


@needs_model
def test_extension_limits_extend_the_proboscis() -> None:
    held = {j: np.full(300, proboscis.EXTEND[j]) for j in proboscis.JOINTS}
    assert proboscis.extension_mm(held)[-1] > 0.2


@needs_model
def test_rostrum_alone_extends_less_than_both_joints() -> None:
    both = {j: np.full(300, proboscis.EXTEND[j]) for j in proboscis.JOINTS}
    rostrum = {"rostrum": np.full(300, proboscis.EXTEND["rostrum"]), "haustellum": np.zeros(300)}
    assert 0 < proboscis.extension_mm(rostrum)[-1] < proboscis.extension_mm(both)[-1]
