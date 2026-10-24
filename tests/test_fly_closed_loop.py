"""The closed loop: position tuning, and a tiny network that moves the leg and senses it."""

from __future__ import annotations

import numpy as np
import pytest

from neuromps.fly import body, closed_loop, connectome, model

needs_leg = pytest.mark.skipif(not body.model_ready(), reason="FlyMimic not fetched")


def test_tuning_is_zero_at_rest_and_full_at_the_joint_limit() -> None:
    t0 = 1.6
    assert closed_loop.claw_rate(t0, t0, "extension") == 0.0
    assert closed_loop.claw_rate(closed_loop.EXTENDED, t0, "extension") == closed_loop.RATE_MAX_HZ
    assert closed_loop.claw_rate(closed_loop.FLEXED, t0, "flexion") == closed_loop.RATE_MAX_HZ
    assert closed_loop.claw_rate(1.0, t0, "flexion") == 0.0


@needs_leg
@pytest.mark.slow
def test_a_driven_extensor_extends_the_tibia_and_the_extension_sensor_fires() -> None:
    """Neuron 0 is the tibia extensor's motor neuron (driven); neuron 2 a sensor."""
    con = connectome.Connectome(
        root_ids=np.arange(3),
        pre=np.array([0]),
        post=np.array([1]),
        synapses=np.array([1]),
        sign=np.array([1]),
        pre_class=np.array(["ach"], dtype=object),
        neuropil=np.array(["A"] * 3, dtype=object),
    )
    loop = closed_loop.Loop(
        neurons_by_muscle={"LFTibia_extensor_93932": [0]},
        sensors=np.array([2]),
        tunings=["extension"],
    )
    params = model.FlyParams(duration_s=0.3, stim_rate_hz=150.0, seed=1)
    spikes = model.simulate(con, params, excite=[0], extend=closed_loop.extender(loop, params))
    rec = loop.record
    assert len(rec["tibia"]) == 300
    assert rec["tibia"][-1] < rec["theta_0"] - 0.3
    assert rec["sensor_rates"][-1][0] > 50.0
    assert np.sum(spikes.neurons == 2) > 5
