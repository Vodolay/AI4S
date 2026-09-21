"""Tests for turning the Mateus recordings into schema records."""

import numpy as np

from neuromps.etl.mateus2024_records import (
    CONDITION_GEOMETRY,
    chambers,
    compartment_of,
    record_id,
)


def test_electrodes_split_into_two_chambers_around_the_channel_band() -> None:
    labels = [f"F{row}" for row in range(1, 17)]
    above, band, below = chambers(labels, band=(7, 11))
    assert above == [f"F{r}" for r in range(1, 7)]
    assert band == [f"F{r}" for r in range(7, 12)]
    assert below == [f"F{r}" for r in range(12, 17)]


def test_an_electrode_is_assigned_by_its_row() -> None:
    assert compartment_of("F3", band=(7, 11)) == "source"
    assert compartment_of("F9", band=(7, 11)) == "microchannel"
    assert compartment_of("F14", band=(7, 11)) == "target"
    assert compartment_of("nonsense", band=(7, 11)) is None


def test_the_chamber_names_are_neutral_about_direction() -> None:
    """The deposit does not record how a device was seated, so names must not imply flow."""
    from neuromps.etl.mateus2024_records import COMPARTMENTS

    assert set(COMPARTMENTS) == {"source", "target"}
    labels = [f"F{r}" for r in range(1, 17)]
    above, _, below = chambers(labels, band=(7, 11))
    assert len(above) + len(below) < len(labels)  # the band belongs to neither chamber


def test_record_ids_are_unique_per_recording_and_compartment() -> None:
    first = record_id("EXP1", "chip_1", "rams", "DIV12", "source")
    second = record_id("EXP1", "chip_1", "rams", "DIV12", "target")
    assert first != second
    assert record_id("EXP2", "chip_1", "rams", "DIV12", "source") != first


def test_every_condition_maps_to_a_stated_geometry() -> None:
    for condition in ("control", "arrows", "rams", "tesla", "tesla_v2"):
        assert condition in CONDITION_GEOMETRY
    assert CONDITION_GEOMETRY["control"]["asymmetric"] is False
    assert all(CONDITION_GEOMETRY[c]["asymmetric"] for c in ("arrows", "rams", "tesla", "tesla_v2"))


def test_a_recording_without_a_detected_band_yields_no_records() -> None:
    from neuromps.etl.mateus2024_records import records_for_recording

    spikes = {f"F{r}": np.array([1.0, 2.0]) for r in range(1, 17)}
    out = records_for_recording(
        spikes, duration_s=600.0, band=None, meta={"condition": "control"}, sha="x"
    )
    assert out == []


def test_records_carry_provenance_and_the_shared_extractor(tmp_path) -> None:
    from neuromps.etl.mateus2024_records import records_for_recording

    rng = np.random.default_rng(0)
    spikes = {
        f"{col}{row}": np.sort(rng.uniform(0, 600, 400))
        for col in ("F", "G", "H")
        for row in range(1, 17)
    }
    meta = {
        "condition": "rams",
        "experiment": "EXP1",
        "chip": "chip_1",
        "div": 12,
        "file": "x.h5",
    }
    out = records_for_recording(spikes, 600.0, band=(7, 11), meta=meta, sha="abc123")
    assert len(out) == 2
    for record in out:
        assert record["source"] == "mateus2024"
        assert record["license"] == "CC-BY-4.0"
        assert record["data_level"] == "spike_times"
        assert record["feature_method"]["computed_by"] == "neuromps.features.mea"
        assert record["provenance"]["source_files"][0]["sha256"] == "abc123"
        assert record["device"]["compartment_id"] in {"source", "target"}
        assert record["features"]["mfr_hz"] is not None
    assert {r["device"]["compartment_id"] for r in out} == {"source", "target"}


def test_the_two_chambers_are_coupled_to_each_other() -> None:
    from neuromps.etl.mateus2024_records import records_for_recording

    rng = np.random.default_rng(1)
    shared = np.sort(rng.uniform(0, 600, 800))
    spikes = {}
    for row in range(1, 17):
        spikes[f"F{row}"] = np.sort(shared + rng.normal(0, 0.002, shared.size))
    meta = {
        "condition": "control",
        "experiment": "EXP9",
        "chip": "chip_2",
        "div": 17,
        "file": "y.h5",
    }
    out = records_for_recording(spikes, 600.0, band=(7, 11), meta=meta, sha="d")
    for record in out:
        assert len(record["coupling"]) == 1
        assert record["coupling"][0]["to_compartment"] in {"source", "target"}
        assert record["coupling"][0]["peak_correlation"] is not None


def test_the_geometry_is_recorded_on_the_device_block() -> None:
    from neuromps.etl.mateus2024_records import records_for_recording

    rng = np.random.default_rng(2)
    spikes = {f"F{r}": np.sort(rng.uniform(0, 600, 300)) for r in range(1, 17)}
    meta = {
        "condition": "tesla_v2",
        "experiment": "EXP5",
        "chip": "chip_1",
        "div": 21,
        "file": "z.h5",
    }
    out = records_for_recording(spikes, 600.0, band=(7, 11), meta=meta, sha="e")
    device = out[0]["device"]
    assert device["microchannel_geometry"] == "tesla_v2"
    assert device["asymmetric_microchannels"] is True
    assert device["n_compartments"] == 2


def test_culture_age_comes_from_the_filename() -> None:
    from neuromps.etl.mateus2024_records import records_for_recording

    rng = np.random.default_rng(3)
    spikes = {f"F{r}": np.sort(rng.uniform(0, 600, 300)) for r in range(1, 17)}
    meta = {
        "condition": "arrows",
        "experiment": "EXP2",
        "chip": "chip_3",
        "div": 17,
        "file": "w.h5",
    }
    out = records_for_recording(spikes, 600.0, band=(7, 11), meta=meta, sha="f")
    assert out[0]["culture_age_days"] == 17


def test_chambers_handle_a_band_at_the_edge() -> None:
    labels = [f"F{row}" for row in range(1, 17)]
    above, band, below = chambers(labels, band=(1, 5))
    assert above == []
    assert len(band) == 5
    assert len(below) == 11
