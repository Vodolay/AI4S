"""Scoring rules of the depression-under-block test, on hand-made numbers."""

from __future__ import annotations

import pandas as pd
import pytest

from neuromps.twin import depression_drugs as dd


def _rows(mfr_at_block: float, burst_slope: float) -> pd.DataFrame:
    rows = []
    for seed in (0, 1, 2):
        for g in dd.DOSE_SERIES:
            block = 1 - g
            rows.append(
                {
                    "setting": "s",
                    "condition": f"gabaa_{g:g}",
                    "seed": seed,
                    "mfr_hz": 5.0 * (2 ** (mfr_at_block * block / 0.8)),
                    "burst_rate_per_min": 10.0 + burst_slope * block + 0.01 * seed,
                    "network_burst_rate_per_min": 8.0 + burst_slope * block,
                }
            )
    return pd.DataFrame(rows)


def test_rule_i_reads_the_firing_rate_change_at_the_strongest_block() -> None:
    assert dd.rule_i(_rows(-1.0, -5.0)) == pytest.approx(-1.0)
    assert dd.rule_i(_rows(+0.5, -5.0)) == pytest.approx(0.5)


def test_rule_ii_needs_both_burst_rates_to_fall_with_dose() -> None:
    falling = dd.rule_ii(_rows(-1.0, -5.0))
    assert falling["passes"] is True
    rising = dd.rule_ii(_rows(-1.0, +5.0))
    assert rising["passes"] is False


def test_the_setting_label_round_trips() -> None:
    assert dd.label(0.4, 600.0) == "u0.4_tau600"
    assert dd.label(0.0, 600.0) == "u0_tau600"
