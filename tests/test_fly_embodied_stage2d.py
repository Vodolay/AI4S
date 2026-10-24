"""Stage 2d: the two glutamate rules' networks and the registered selection rule."""

from __future__ import annotations

import numpy as np

from neuromps.fly import connectome
from neuromps.fly import embodied_stage2d as s2d


def _net() -> connectome.Connectome:
    """0 (glut) -| 2 (motor); 0 (glut) -| 1; 1 (gaba) -| 2."""
    return connectome.Connectome(
        root_ids=np.arange(3),
        pre=np.array([0, 0, 1]),
        post=np.array([2, 1, 2]),
        synapses=np.array([3, 4, 5]),
        sign=np.array([-1, -1, -1]),
        pre_class=np.array(["glut", "glut", "gaba"], dtype=object),
        neuropil=np.array(["A"] * 3, dtype=object),
    )


def test_the_motor_rule_flips_glutamate_onto_motor_neurons_only() -> None:
    nets = s2d.networks(_net(), motor=np.array([2]))
    np.testing.assert_array_equal(nets["glut_motor"].sign, [1, -1, -1])
    np.testing.assert_array_equal(nets["glut_all"].sign, [1, 1, -1])
    assert set(nets) == {"glut_motor", "glut_all", "glut_motor_rewired", "glut_all_rewired"}


def _cell(rule, b, tau, passed, tie):
    return {"rule": rule, "b_mv": b, "tau_ms": tau, "passed": passed, "tie_distance": tie}


def test_selection_prefers_the_smaller_change_on_a_tie() -> None:
    cells = [_cell("glut_all", 0.0, 100.0, 2, 0.6), _cell("glut_motor", 4.0, 200.0, 2, 0.6)]
    assert s2d.select(cells)["rule"] == "glut_motor"
    assert s2d.select([_cell("glut_all", 1.0, 25.0, 0, 2.0)]) is None
