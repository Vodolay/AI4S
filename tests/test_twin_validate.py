"""The confound summary that §7.2's robustness claim is read from."""

from neuromps.twin.validate import _span, confound_summary


def _rows(raw: dict, white: dict) -> list[dict]:
    """Sweep rows carrying the two measures, keyed by (ratio, drive)."""
    return [
        {
            "forward_ratio": ratio,
            "background_scale_target": drive,
            "asymmetry_mean": value,
            "asymmetry_prewhitened_mean": white[(ratio, drive)],
        }
        for (ratio, drive), value in raw.items()
    ]


def test_span_averages_the_range_over_the_held_factor() -> None:
    rows = [
        {"forward_ratio": 1.0, "background_scale_target": 1.0, "a": 0.0},
        {"forward_ratio": 1.0, "background_scale_target": 0.7, "a": 0.4},
        {"forward_ratio": 4.0, "background_scale_target": 1.0, "a": 0.0},
        {"forward_ratio": 4.0, "background_scale_target": 0.7, "a": 0.6},
    ]
    # Drive moves the measure by 0.4 at one ratio and 0.6 at the other.
    assert _span(rows, "a", "forward_ratio", "background_scale_target") == 0.5
    # Wiring moves it not at all at drive 1.0 and by 0.2 at drive 0.7.
    assert _span(rows, "a", "background_scale_target", "forward_ratio") == 0.1


def test_span_ignores_undefined_runs_and_single_point_slices() -> None:
    rows = [
        {"forward_ratio": 1.0, "background_scale_target": 1.0, "a": None},
        {"forward_ratio": 1.0, "background_scale_target": 0.7, "a": 0.4},
        {"forward_ratio": 4.0, "background_scale_target": 1.0, "a": 0.1},
        {"forward_ratio": 4.0, "background_scale_target": 0.7, "a": 0.9},
    ]
    # Ratio 1.0 has one usable point left, so it contributes no span at all.
    assert _span(rows, "a", "forward_ratio", "background_scale_target") == 0.8
    assert _span([rows[0]], "a", "forward_ratio", "background_scale_target") is None


def test_summary_reports_reduction_when_prewhitening_flattens_the_drive_effect() -> None:
    keys = [(1.0, 1.0), (1.0, 0.7), (4.0, 1.0), (4.0, 0.7)]
    raw = dict(zip(keys, [0.0, 0.8, 0.0, 0.8], strict=True))
    white = dict(zip(keys, [0.0, 0.2, 0.0, 0.2], strict=True))
    summary = confound_summary(_rows(raw, white))
    assert summary["raw"]["drive_span"] == 0.8
    assert summary["prewhitened"]["drive_span"] == 0.2
    assert summary["prewhitening_reduces_drive_dependence"] is True
    assert summary["raw"]["drive_moves_measure_more_than_wiring"] is True


def test_summary_reports_no_reduction_when_prewhitening_amplifies_it() -> None:
    keys = [(1.0, 1.0), (1.0, 0.7), (4.0, 1.0), (4.0, 0.7)]
    raw = dict(zip(keys, [0.1, 0.7, 0.0, 0.8], strict=True))
    white = dict(zip(keys, [0.2, 0.9, 0.0, 1.0], strict=True))
    summary = confound_summary(_rows(raw, white))
    assert summary["prewhitened"]["drive_span"] > summary["raw"]["drive_span"]
    assert summary["prewhitening_reduces_drive_dependence"] is False


def test_summary_is_undefined_rather_than_false_when_a_measure_is_missing() -> None:
    keys = [(1.0, 1.0), (1.0, 0.7)]
    raw = dict(zip(keys, [0.1, 0.7], strict=True))
    white = dict.fromkeys(keys)
    summary = confound_summary(_rows(raw, white))
    assert summary["prewhitened"]["drive_span"] is None
    assert summary["prewhitened"]["drive_moves_measure_more_than_wiring"] is None
    assert summary["prewhitening_reduces_drive_dependence"] is None
