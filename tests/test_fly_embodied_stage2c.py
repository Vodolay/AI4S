"""Stage 2c: the period measure, the registered selection rule and the stop rule."""

from __future__ import annotations

import json

import numpy as np

from neuromps.fly import embodied_stage2c as s2c


def test_period_of_a_sine_is_its_cycle() -> None:
    t = np.arange(0, 1.8, 0.001)
    assert abs(s2c.period_s(np.sin(2 * np.pi * 8 * t)) - 0.125) <= 0.002
    assert s2c.period_s(np.zeros(500)) is None


def _cell(b, tau, passed, tie):
    return {"b_mv": b, "tau_ms": tau, "passed": passed, "tie_distance": tie}


def test_selection_prefers_tests_passed_then_distance_then_smaller_parameters() -> None:
    cells = [_cell(4, 100, 2, 0.9), _cell(2, 200, 3, 1.1), _cell(1, 400, 3, 0.7)]
    assert s2c.select(cells)["b_mv"] == 1
    tied = [_cell(8, 50, 3, 0.7), _cell(2, 400, 3, 0.7), _cell(2, 100, 3, 0.7)]
    assert (s2c.select(tied)["b_mv"], s2c.select(tied)["tau_ms"]) == (2, 100)


def test_no_cell_passing_anything_stops_the_stage() -> None:
    assert s2c.select([_cell(1, 25, 0, 2.0), _cell(2, 50, 0, 2.0)]) is None
    spec = json.loads(s2c.REGISTRATION.read_text())
    got = s2c.stopped(spec)
    assert set(got) == set(spec["predictions"])
    assert all(v["outcome"] == "fail" for v in got.values())


def test_undefined_phases_count_half_a_cycle_in_the_tie_break() -> None:
    assert s2c._distance(None) == 0.5
    assert s2c._distance(0.12) == 0.12
