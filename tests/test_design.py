"""The experiment-design lookup over the §7.6 grid."""

import json
from pathlib import Path

import pytest

from neuromps.design import (
    CAVEAT,
    HEADLINE_ARM,
    HEADLINE_FEATURES,
    at_configuration,
    candidates,
    cheapest_for,
    main,
    sweep_rows,
    table,
)

METRICS = Path(__file__).resolve().parents[1] / "outputs" / "metrics.json"


def _rows() -> list[dict]:
    return [
        {
            "arm": "free_rate",
            "feature_set": "all_rate_residualized",
            "electrodes_per_compartment": 6,
            "duration_s": 10.0,
            "auc": 0.88,
            "auc_ci95": [0.82, 0.93],
        },
        {
            "arm": "free_rate",
            "feature_set": "all_rate_residualized",
            "electrodes_per_compartment": 48,
            "duration_s": 60.0,
            "auc": 0.97,
            "auc_ci95": [0.94, 0.99],
        },
        {
            "arm": "free_rate",
            "feature_set": "rate_only",
            "electrodes_per_compartment": 6,
            "duration_s": 10.0,
            "auc": 0.99,
            "auc_ci95": [0.98, 1.0],
        },
        {
            "arm": "rate_matched",
            "feature_set": "all_rate_residualized",
            "electrodes_per_compartment": 6,
            "duration_s": 10.0,
            "auc": 0.99,
            "auc_ci95": [0.98, 1.0],
        },
    ]


def test_candidates_keep_only_the_reported_arm_and_feature_set() -> None:
    got = candidates(_rows())
    assert len(got) == 2
    assert {r["arm"] for r in got} == {HEADLINE_ARM}
    assert {r["feature_set"] for r in got} == {HEADLINE_FEATURES}


def test_the_cheapest_design_is_chosen_on_the_interval_lower_bound() -> None:
    """0.88 has a lower bound of 0.82, so a 0.85 target must not accept it."""
    assert cheapest_for(_rows(), 0.80)["electrodes_per_compartment"] == 6
    assert cheapest_for(_rows(), 0.85)["electrodes_per_compartment"] == 48


def test_cheapest_prefers_the_smaller_experiment_not_the_higher_auc() -> None:
    answer = cheapest_for(_rows(), 0.80)
    assert answer["cost_electrode_seconds"] == 60.0


def test_an_unreachable_target_returns_nothing_rather_than_the_best_available() -> None:
    assert cheapest_for(_rows(), 0.999) is None


def test_a_configuration_outside_the_grid_is_not_interpolated() -> None:
    assert at_configuration(_rows(), 6, 10.0) is not None
    assert at_configuration(_rows(), 24, 10.0) is None


def test_every_answer_carries_the_simulation_caveat() -> None:
    for answer in (cheapest_for(_rows(), 0.80), at_configuration(_rows(), 6, 10.0)):
        assert answer["caveat"] == CAVEAT
        assert "SIMULATED" in answer["caveat"]
        assert "7.7" in answer["caveat"], "the null on real devices must travel with the number"


def test_the_table_is_sorted_and_carries_the_lower_bound() -> None:
    rows = table(_rows())
    assert [r["electrodes_per_compartment"] for r in rows] == [6, 48]
    assert rows[0]["auc_ci95_lower"] == 0.82


def test_sweep_rows_is_empty_rather_than_raising_when_the_twin_has_not_run() -> None:
    assert sweep_rows({}) == []
    assert sweep_rows({"twin_hd16_directionality": {}}) == []


@pytest.mark.skipif(not METRICS.exists(), reason="no full-run metrics")
def test_the_tool_reproduces_the_reports_own_headline_configuration() -> None:
    """The tool and §7.6 must not drift: same grid, same target, same answer."""
    metrics = json.loads(METRICS.read_text())
    block = metrics.get("twin_hd16_directionality", {})
    reported = block.get("cheapest_configuration", {}).get("free_rate_residualized")
    if not reported or not block.get("sweep"):
        pytest.skip("this metrics.json has no directionality sweep")
    answer = cheapest_for(sweep_rows(metrics), float(block["auc_target"]))
    assert answer is not None
    assert answer["electrodes_per_compartment"] == reported["electrodes_per_compartment"]
    assert answer["duration_s"] == reported["duration_s"]


@pytest.mark.skipif(not METRICS.exists(), reason="no full-run metrics")
def test_the_cli_prints_the_caveat_on_every_path(capsys) -> None:
    for argv in ([], ["--target-auc", "0.8"], ["--electrodes", "6", "--duration", "10"]):
        assert main(argv) == 0
        assert "SIMULATED" in capsys.readouterr().out
