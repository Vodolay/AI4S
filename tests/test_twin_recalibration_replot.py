"""Tests for redrawing the recalibration figure from regenerated ranks (D31, entry 52).

All inputs are synthetic: the redraw path must refuse to draw unless every value it
recomputes equals the recorded one, and must never write to metrics.json.
"""

import json

import numpy as np
import pytest

from neuromps.twin import recalibration_replot as rp
from neuromps.twin.inference import PARAMETER_NAMES

MODELS = ("M0", "M1", "M2", "M3")
N_SAMPLES = 200


def _cache(seed: int = 0, n: int = 1000) -> rp.RankCache:
    rng = np.random.default_rng(seed)
    k = len(PARAMETER_NAMES)
    ranks = {m: rng.integers(0, N_SAMPLES + 1, size=(n, k)) for m in MODELS}
    widths = {m: rng.uniform(0.01, 0.5, size=(n, k)) for m in MODELS}
    near = rng.permutation(n)[:300]
    meta = {
        "n_samples": N_SAMPLES,
        "near_real": {"n_real": 3, "n_selected": 300},
        "budget": {"pool_simulations": 15750, "held_out": n},
    }
    return rp.RankCache(ranks=ranks, widths=widths, near_index=near, meta=meta)


def _section(cache: rp.RankCache) -> dict:
    """A recorded section exactly as the run writes it (through a JSON round trip)."""
    section = {
        "results": rp.results_from_ranks(cache),
        "near_real": cache.meta["near_real"],
        "budget": cache.meta["budget"],
        "posterior_samples": N_SAMPLES,
        "amortized_inference": {"M2": {"median_seconds_per_posterior": 1.953}},
    }
    return json.loads(json.dumps(section))


# --- comparing with the record -----------------------------------------------------


def test_differences_is_empty_for_equal_records() -> None:
    record = {"a": {"b": 0.1234, "c": [1, 2]}, "d": True}
    assert rp.differences(record, json.loads(json.dumps(record))) == []


def test_differences_names_every_changed_leaf_by_its_path() -> None:
    record = {"a": {"b": 0.1234, "c": 3}, "d": True}
    other = {"a": {"b": 0.1235, "c": 3}, "d": False}
    found = rp.differences(record, other)
    assert [d[0] for d in found] == ["a.b", "d"]
    assert found[0][1:] == (0.1234, 0.1235)


def test_differences_reports_missing_and_extra_keys() -> None:
    found = rp.differences({"a": 1, "b": 2}, {"a": 1, "c": 2})
    assert {d[0] for d in found} == {"b", "c"}


def test_largest_numeric_difference_ignores_non_numbers() -> None:
    found = [("x", 0.5, 0.5003), ("y", True, False), ("z", 1, None)]
    assert rp.largest_difference(found) == pytest.approx(0.0003)


def test_count_values_counts_leaves() -> None:
    assert rp.count_values({"a": {"b": 1, "c": [1, 2]}, "d": "x"}) == 3


def test_check_passes_when_recomputed_from_the_same_ranks() -> None:
    cache = _cache()
    check = rp.check_against_record(_section(cache), cache)
    assert check["equal"], check["differences"][:3]
    assert check["n_values_checked"] > 4 * 3 * len(PARAMETER_NAMES)


def test_check_fails_and_names_the_value_when_one_rank_moves() -> None:
    cache = _cache()
    section = _section(cache)
    cache.ranks["M2"][0, 0] = (cache.ranks["M2"][0, 0] + 100) % (N_SAMPLES + 1)
    check = rp.check_against_record(section, cache)
    assert not check["equal"]
    paths = [d[0] for d in check["differences"]]
    assert paths
    assert all(p.startswith("results.M2.") for p in paths)


def test_check_covers_the_budget_and_the_near_real_distances() -> None:
    cache = _cache()
    section = _section(cache)
    section["budget"]["pool_simulations"] = 15749
    section["near_real"]["n_real"] = 4
    paths = [d[0] for d in rp.check_against_record(section, cache)["differences"]]
    assert paths == ["budget.pool_simulations", "near_real.n_real"]


def test_check_ignores_the_timing_it_does_not_recompute() -> None:
    cache = _cache()
    section = _section(cache)
    section["amortized_inference"]["M2"]["median_seconds_per_posterior"] = 9.9
    assert rp.check_against_record(section, cache)["equal"]


def test_results_use_the_first_300_all_1000_and_the_near_real_rows() -> None:
    from neuromps.twin.recalibration_run import summarize

    cache = _cache()
    out = rp.results_from_ranks(cache)["M1"]
    ranks, widths = cache.ranks["M1"], cache.widths["M1"]
    assert out["H300"] == summarize(ranks, widths, np.arange(300), N_SAMPLES)
    assert out["H1000"] == summarize(ranks, widths, np.arange(1000), N_SAMPLES)
    assert out["near_real"] == summarize(ranks, widths, cache.near_index, N_SAMPLES)


# --- persisting the ranks ---------------------------------------------------------


def test_ranks_round_trip_exactly(tmp_path) -> None:
    cache = _cache()
    path = tmp_path / "ranks_simulated.npz"
    rp.save_ranks(path, cache)
    back = rp.load_ranks(path)
    for m in MODELS:
        np.testing.assert_array_equal(back.ranks[m], cache.ranks[m])
        np.testing.assert_array_equal(back.widths[m], cache.widths[m])
        assert back.ranks[m].dtype.kind == "i"
    np.testing.assert_array_equal(back.near_index, cache.near_index)
    assert back.meta == cache.meta
    assert [p.name for p in tmp_path.iterdir()] == ["ranks_simulated.npz"]


def test_ranks_file_is_labelled_simulated() -> None:
    assert "simulated" in rp.ranks_path().name


def test_regeneration_refuses_without_a_recorded_chunk_count() -> None:
    with pytest.raises(ValueError, match="recorded"):
        rp.recorded_train_chunks({"budget": {}})
    with pytest.raises(ValueError, match="recorded"):
        rp.recorded_train_chunks({"budget": {"new_train_chunks_completed": 0}})
    assert rp.recorded_train_chunks({"budget": {"new_train_chunks_completed": 51}}) == 51


# --- the redraw itself -----------------------------------------------------------


def _write_record(tmp_path, section: dict):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"other": {"x": 1}, rp.rc.METRICS_KEY: section}, indent=2))
    return metrics


def test_replot_draws_from_cached_ranks_and_never_writes_metrics(tmp_path) -> None:
    cache = _cache()
    ranks = tmp_path / "ranks_simulated.npz"
    rp.save_ranks(ranks, cache)
    metrics = _write_record(tmp_path, _section(cache))
    before = metrics.read_bytes()
    figure = tmp_path / "figure_simulated.png"
    assert rp.replot(metrics=metrics, ranks=ranks, figure=figure) == 0
    assert figure.exists()
    assert metrics.read_bytes() == before


def test_replot_refuses_to_draw_when_a_value_differs(tmp_path) -> None:
    cache = _cache()
    ranks = tmp_path / "ranks_simulated.npz"
    rp.save_ranks(ranks, cache)
    section = _section(cache)
    section["results"]["M0"]["H300"]["ks"]["gabaa_scale"]["ks_p"] += 0.0001
    metrics = _write_record(tmp_path, section)
    before = metrics.read_bytes()
    figure = tmp_path / "figure_simulated.png"
    assert rp.replot(metrics=metrics, ranks=ranks, figure=figure) == 1
    assert not figure.exists()
    assert metrics.read_bytes() == before


def test_the_coverage_legend_overlaps_no_panel() -> None:
    """The defect this redraw fixes: the legend sat on the threshold_scale panel."""
    from neuromps.twin.recalibration_run import calibration_figure

    cache = _cache()
    results = rp.results_from_ranks(cache)
    fig = calibration_figure(cache.ranks, results, N_SAMPLES, 15750)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert len(fig.legends) == 1
    legend = fig.legends[0].get_window_extent(renderer)
    for ax in fig.axes:
        assert ax.get_legend() is None
        assert not legend.overlaps(ax.get_tightbbox(renderer)), ax.get_title()
    import matplotlib.pyplot as plt

    plt.close(fig)


# --- HD14's cache under a later summary-code fingerprint --------------------------


def _hd14_cache(fingerprint: str) -> dict:
    from neuromps.twin.inference import (
        SUMMARY_NAMES,
        calibration_fingerprint,
        default_layout,
    )

    rng = np.random.default_rng(1)
    return {
        "theta": rng.uniform(size=(6, len(PARAMETER_NAMES))),
        "x": rng.normal(size=(6, len(SUMMARY_NAMES))),
        "summary_names": list(SUMMARY_NAMES),
        "metadata": {
            "summary_pipeline_sha256": fingerprint,
            "calibration_sha256": calibration_fingerprint(),
            "electrodes_per_compartment": default_layout().electrodes_per_compartment,
        },
    }


def test_a_current_cache_is_accepted_without_resimulating() -> None:
    from neuromps.twin.inference import summary_pipeline_fingerprint

    cached = _hd14_cache(summary_pipeline_fingerprint())

    def never(_cached):
        raise AssertionError("must not resimulate")

    assert rp.accept_hd14_cache(cached, never) == {"summary_code": "current"}


def test_an_older_summary_fingerprint_is_accepted_only_if_rows_reproduce() -> None:
    cached = _hd14_cache("older0000000000")
    same = {"rows": [0, 1], "n_checked": 2, "n_identical": 2, "max_abs_difference": 0.0}
    out = rp.accept_hd14_cache(cached, lambda _c: same)
    assert out["summary_code"] == "older, reproduced"
    assert out["cached_fingerprint"] == "older0000000000"
    assert out["resimulated"] == same
    differ = {**same, "n_identical": 1, "max_abs_difference": 0.01}
    with pytest.raises(SystemExit, match="does not reproduce"):
        rp.accept_hd14_cache(cached, lambda _c: differ)


def test_other_cache_guards_stay_strict() -> None:
    cached = _hd14_cache("older0000000000")
    cached["metadata"]["calibration_sha256"] = "another calibration"
    same = {"rows": [0], "n_checked": 1, "n_identical": 1, "max_abs_difference": 0.0}
    with pytest.raises(SystemExit, match="calibration"):
        rp.accept_hd14_cache(cached, lambda _c: same)
