"""Tests for the Kapucu 2022 ETL: experiment logs, spike loading and record building."""

from pathlib import Path

import pytest

from neuromps.etl.kapucu import (
    PHARMACOLOGY_PLATES,
    build_plate_records,
    mechanism_class,
    parse_explog,
    parse_noisy_electrodes,
)
from neuromps.qc.schema import validate_record

EXPLOG = """Project,Experiment.Date,Plate.SN,DIV,Well,Treatment,Size,Dose,Units
Rat,50618,MEA2Pharma,DIV22,A1,ExcludedWell,,,
Rat,50618,MEA2Pharma,DIV22,A3,Control,,,
Rat,50618,MEA2Pharma,DIV22,A8,Gabazine,,30uM,
Rat,50618,MEA2Pharma,DIV22,B4,DAP5,,50uM,
Rat,50618,MEA2Pharma,DIV22,B5,D-AP5,,50uM,
Rat,50618,MEA2Pharma,DIV22,C2,KainicAcid,,5uM,
"""
SPIKES = "Channel,Time\n" + "".join(
    f"{well}_{el},{t:.4f}\n"
    for well in ("A3", "A8")
    for el in (11, 12)
    for t in [i * 0.05 for i in range(400)]
)


@pytest.fixture()
def plate_dir(tmp_path: Path) -> Path:
    d = tmp_path / "Rat_MEA2_Pharmacology" / "Rat_MEA2Pharma_DIV22"
    sub = d / "Rat_MEA2Pharma_DIV22_spikes_noise_explogs"
    sub.mkdir(parents=True)
    (sub / "Rat_50618_MEA2Pharma_DIV22_expLog.csv").write_text(EXPLOG)
    (sub / "Rat_50618_MEA2Pharma_DIV22_spikes.csv").write_text(SPIKES)
    (sub / "noisy_electrodes_Rat_MEA2Pharma_DIV22.csv").write_text("DIV22\nA8_12\n")
    return d


def test_parse_explog_normalizes_compound_names_and_dose() -> None:
    df = parse_explog(EXPLOG.splitlines(keepends=True))
    row = df.set_index("well").loc["A8"]
    assert row["compound"] == "gabazine"
    assert row["dose_value"] == 30.0
    assert row["dose_unit"] == "uM"


def test_parse_explog_unifies_the_two_spellings_of_dap5() -> None:
    df = parse_explog(EXPLOG.splitlines(keepends=True)).set_index("well")
    assert df.loc["B4", "compound"] == df.loc["B5", "compound"] == "d-ap5"


def test_parse_explog_marks_excluded_wells() -> None:
    df = parse_explog(EXPLOG.splitlines(keepends=True)).set_index("well")
    assert bool(df.loc["A1", "excluded"]) is True
    assert bool(df.loc["A3", "excluded"]) is False


def test_mechanism_class_covers_every_compound_in_the_plates() -> None:
    assert mechanism_class("gabazine") == "gabaa_antagonist"
    assert mechanism_class("kainicacid") == "glutamatergic_agonist"
    assert mechanism_class("cnqx") == "glutamatergic_antagonist"
    assert mechanism_class("d-ap5") == "glutamatergic_antagonist"
    assert mechanism_class("gaba") == "gabaa_agonist_or_pam"
    assert mechanism_class("ttx") == "na_channel_blocker"
    assert mechanism_class("control") is None


def test_parse_noisy_electrodes_skips_the_div_header() -> None:
    assert parse_noisy_electrodes("DIV22\nA8_12\nF2_31\n") == {"A8": ["A8_12"], "F2": ["F2_31"]}


def test_built_records_validate_against_the_schema(plate_dir: Path) -> None:
    records = build_plate_records(plate_dir, PHARMACOLOGY_PLATES[0], hashes={})
    assert records
    for rec in records:
        assert validate_record(rec) == [], rec["record_id"]


def test_records_carry_treatment_phase_and_pairable_unit(plate_dir: Path) -> None:
    records = {r["unit_id"]: r for r in build_plate_records(plate_dir, PHARMACOLOGY_PLATES[0], {})}
    assert records["A8"]["compound"]["name"] == "gabazine"
    assert records["A8"]["compound"]["mechanism_class"] == "gabaa_antagonist"
    assert records["A8"]["recording"]["phase"] == "exposure"
    assert records["A3"]["compound"]["role"] == "untreated_control"


def test_noisy_electrodes_become_a_qc_flag(plate_dir: Path) -> None:
    records = {r["unit_id"]: r for r in build_plate_records(plate_dir, PHARMACOLOGY_PLATES[0], {})}
    assert records["A8"]["qc"]["n_noisy_electrodes"] == 1
    assert "noisy_electrodes_present" in records["A8"]["qc"]["flags"]
    assert records["A3"]["qc"]["n_noisy_electrodes"] == 0


def test_excluded_wells_are_kept_but_marked_unusable(plate_dir: Path) -> None:
    """Upstream exclusions are preserved, not silently dropped (STANDARD.md 5).

    A1 is ExcludedWell in the log and has no spikes at all, which is how the real
    plates record exclusions: it must still appear, flagged, with null features.
    """
    records = {r["unit_id"]: r for r in build_plate_records(plate_dir, PHARMACOLOGY_PLATES[0], {})}
    assert "A1" in records
    assert records["A1"]["qc"]["usable"] is False
    assert "excluded_well_upstream" in records["A1"]["qc"]["flags"]
    assert records["A1"]["features"]["mfr_hz"] is None
    assert records["A1"]["features"]["active_electrodes"] == 0
    assert validate_record(records["A1"]) == []
    assert "excluded_well_upstream" not in records["A3"]["qc"]["flags"]


def test_wells_in_the_log_without_spikes_are_still_recorded(plate_dir: Path) -> None:
    records = {r["unit_id"] for r in build_plate_records(plate_dir, PHARMACOLOGY_PLATES[0], {})}
    assert {"B4", "B5", "C2"} <= records  # in the log, no spikes in this fixture


def test_features_are_computed_by_the_shared_extractor(plate_dir: Path) -> None:
    records = {r["unit_id"]: r for r in build_plate_records(plate_dir, PHARMACOLOGY_PLATES[0], {})}
    rec = records["A3"]
    assert rec["feature_method"]["computed_by"] == "neuromps.features.mea"
    assert rec["features"]["mfr_hz"] == pytest.approx(20.0, rel=0.05)


def test_provenance_hashes_are_taken_from_the_manifest(plate_dir: Path) -> None:
    fake = {p.name: "a" * 64 for p in plate_dir.rglob("*.csv")}
    rec = build_plate_records(plate_dir, PHARMACOLOGY_PLATES[0], hashes=fake)[0]
    assert rec["provenance"]["source_files"]
    assert {f["sha256"] for f in rec["provenance"]["source_files"]} == {"a" * 64}
    assert rec["provenance"]["generated_by"] == "neuromps.etl.kapucu"


def test_provenance_hashes_files_absent_from_the_manifest(plate_dir: Path) -> None:
    """A file the manifest does not list is hashed here; provenance is never blank."""
    rec = build_plate_records(plate_dir, PHARMACOLOGY_PLATES[0], hashes={})[0]
    hashes = {f["sha256"] for f in rec["provenance"]["source_files"]}
    assert all(len(h) == 64 and h.strip("0123456789abcdef") == "" for h in hashes)


def test_millimolar_doses_are_rescaled_to_micromolar() -> None:
    """A mM dose must become uM x1000, not a relabelled uM (regression)."""
    log = EXPLOG + "Rat,50618,MEA2Pharma,DIV22,C7,Gabazine,,5mM,\n"
    row = parse_explog(log.splitlines(keepends=True)).set_index("well").loc["C7"]
    assert row["dose_value"] == 5000.0
    assert row["dose_unit"] == "uM"
