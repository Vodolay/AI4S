"""The definition sidecar: spike trains re-read per record, every set, checked against the asset."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from neuromps.etl import kapucu, screenneuropharm, trujillo
from neuromps.etl.build import flatten_records
from neuromps.etl.definition_sidecar import (
    COLUMNS,
    QUICK_RECORDS,
    SLOW_FIRST,
    _job,
    build_table,
    check_reference,
    counts,
    first_unit_per_source,
    smallest,
    sort_table,
)
from neuromps.etl.spike_trains import LOADERS, SpikeRecord, Unit, load, units
from neuromps.features.definitions import REGISTRY, extract, get
from neuromps.features.mea import FEATURE_NAMES

EXPLOG = """Project,Experiment.Date,Plate.SN,DIV,Well,Treatment,Size,Dose,Units
Rat,50618,MEA2Pharma,DIV22,A1,ExcludedWell,,,
Rat,50618,MEA2Pharma,DIV22,A3,Control,,,
Rat,50618,MEA2Pharma,DIV22,A8,Gabazine,,30uM,
"""


def _kapucu_spikes() -> str:
    rows = []
    for well in ("A3", "A8"):
        for el, step in ((11, 0.05), (12, 0.5), (13, 2.0)):
            rows += [f"{well}_{el},{t:.4f}\n" for t in np.arange(0.0, 20.0, step)]
    return "Channel,Time\n" + "".join(rows)


@pytest.fixture()
def raw_root(tmp_path: Path) -> Path:
    """A raw tree: one Kapucu pharmacology plate, a ScreenNeuroPharm archive, a Trujillo session."""
    plate = kapucu.PHARMACOLOGY_PLATES[0]
    sub = tmp_path / "kapucu2022" / "Data" / plate.directory / "x_spikes_noise_explogs"
    sub.mkdir(parents=True)
    (sub / "Rat_50618_MEA2Pharma_DIV22_expLog.csv").write_text(EXPLOG)
    (sub / "Rat_50618_MEA2Pharma_DIV22_spikes.csv").write_text(_kapucu_spikes())
    (sub / "noisy_electrodes_Rat_MEA2Pharma_DIV22.csv").write_text("DIV22\nA8_11\n")
    _write_snp(tmp_path / screenneuropharm.SOURCE / screenneuropharm.ARCHIVE_NAME)
    _write_trujillo(tmp_path / trujillo.SOURCE / "LFP_Sp_160810.spikes.parquet")
    return tmp_path


def _ptrain(step: int, total: int = 200_001) -> str:
    return f"{total},0\n" + "".join(f"{s},30.5\n" for s in range(1000, total, step))


def _write_snp(path: Path) -> None:
    path.parent.mkdir(parents=True)
    root = "ScreenNeuroPharm-Bicuculline-6a1949e/Data"
    single = f"{root}/Br11/Cx/18331/20092022_PeakDetectionMAT_PLP2ms_RP1ms_Cut_TXT"
    joint = f"{root}/Br18/CxCx/24544/20210329_PeakDetectionMAT_PLP2ms_RP1ms_Phases_Joint_TXT"
    with zipfile.ZipFile(path, "w") as z:
        for condition in ("01_nbasal", "02_BIC100nM"):
            for electrode in range(4):
                folder = f"{single}/ptrain_20092022_01_{condition}"
                name = f"{folder}/ptrain_20092022_01_{condition}_{electrode:03d}.txt"
                z.writestr(name, _ptrain(300 + 100 * electrode))
        for colour, electrodes in (("Red", ("A04", "A05")), ("Green", ("A08", "A09"))):
            for el in electrodes:
                folder = f"{joint}/ptrain_20210329_02_01_nbasal_{colour}"
                z.writestr(f"{folder}/ptrain_20210329_02_01_nbasal_Joint_{el}.txt", _ptrain(700))


def _write_trujillo(path: Path) -> None:
    path.parent.mkdir(parents=True)
    rows = [
        (well, electrode, float(t))
        for well in (1, 2)
        for electrode in (11, 12)
        for t in np.arange(0.0, 30.0, 0.1 * electrode / 10)
    ]
    frame = pd.DataFrame(rows, columns=["well", "electrode", "time_s"])
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table.replace_schema_metadata({b"duration_s": b"30.0"}), path)


HASHES = {screenneuropharm.ARCHIVE_NAME: "x", "LFP_Sp_160810.spikes.parquet": "y"}


def _asset(raw_root: Path) -> pd.DataFrame:
    """The records the ETL builders make from the same raw tree, flattened as the asset is."""
    plate = kapucu.PHARMACOLOGY_PLATES[0]
    archive = raw_root / screenneuropharm.SOURCE / screenneuropharm.ARCHIVE_NAME
    records = kapucu.build_plate_records(
        raw_root / "kapucu2022" / "Data" / plate.directory, plate, {}
    )
    records += screenneuropharm.build_records(archive, {})
    records += screenneuropharm.build_compartment_records(archive, {})
    records += trujillo.build_records(raw_root / trujillo.SOURCE, HASHES)
    return flatten_records(records)


def test_units_cover_every_present_source_in_the_build_order(raw_root: Path) -> None:
    found = units(raw_root, HASHES, full=True)
    assert [u.loader for u in found] == ["kapucu", "snp_single", "snp_joint", "trujillo"]


def test_a_source_absent_from_the_tiers_manifest_has_no_units(raw_root: Path) -> None:
    found = units(raw_root, {}, full=True)
    assert {u.source for u in found} == {"kapucu2022"}


def test_record_ids_are_the_builders_own(raw_root: Path) -> None:
    ids = {r.record_id for u in units(raw_root, HASHES, True) for r in load(u, raw_root)}
    assert ids == set(_asset(raw_root)["record_id"])


def test_kapucu_records_carry_the_deposits_noisy_list_and_empty_logged_wells(
    raw_root: Path,
) -> None:
    (unit,) = [u for u in units(raw_root, HASHES, True) if u.loader == "kapucu"]
    records = {r.record_id.rsplit(":", 1)[-1]: r for r in load(unit, raw_root)}
    assert records["a8"].noisy == ("A8_11",)
    assert records["a3"].noisy == ()
    assert records["a1"].channels == {}  # excluded upstream, in the log, no spikes


def test_reference_rows_equal_the_asset_features_exactly(raw_root: Path) -> None:
    table = build_table(raw_root, HASHES, full=True, processes=2)
    check = check_reference(table, _asset(raw_root))
    assert check["pass"], check
    assert check["n_values_mismatched"] == 0
    assert check["n_records"] == len(_asset(raw_root))


def test_check_fails_on_one_changed_value_or_an_unknown_record(raw_root: Path) -> None:
    table = build_table(raw_root, HASHES, full=True, processes=2)
    asset = _asset(raw_root)
    changed = asset.copy()
    changed.loc[changed["features.mfr_hz"].notna().idxmax(), "features.mfr_hz"] += 1e-12
    assert check_reference(table, changed)["n_values_mismatched"] == 1
    fewer = asset.iloc[1:]
    assert check_reference(table, fewer)["n_records_missing_from_asset"] == 1


def test_every_record_appears_under_every_set_and_feature(raw_root: Path) -> None:
    table = build_table(raw_root, HASHES, full=True, processes=2)
    assert list(table.columns) == COLUMNS
    per_record = table.groupby("record_id").size()
    assert (per_record == len(REGISTRY) * len(FEATURE_NAMES)).all()


def test_the_kapucu2022_rows_drop_the_noisy_electrode(raw_root: Path) -> None:
    (unit,) = [u for u in units(raw_root, HASHES, True) if u.loader == "kapucu"]
    a8 = next(r for r in load(unit, raw_root) if r.record_id.endswith(":a8"))
    rows = pd.DataFrame(_job((unit, raw_root, None)), columns=COLUMNS)
    got = rows[(rows["record_id"] == a8.record_id) & (rows["definition"] == "kapucu2022")]
    expected = extract(get("kapucu2022"), a8.channels, a8.duration_s, ("A8_11",))
    assert dict(zip(got["feature"], got["value"], strict=True))["active_electrodes"] == float(
        expected["active_electrodes"]
    )
    reference = extract(get("neuromps"), a8.channels, a8.duration_s)
    assert expected["active_electrodes"] < reference["active_electrodes"]


def test_quick_takes_the_first_unit_per_source_and_a_few_records(raw_root: Path) -> None:
    found = units(raw_root, HASHES, True)
    assert [u.loader for u in first_unit_per_source(found)] == ["kapucu", "snp_single", "trujillo"]
    table = build_table(raw_root, HASHES, full=True, quick=True, processes=2)
    per_source = table.groupby("source")["record_id"].nunique()
    assert (per_source <= QUICK_RECORDS).all()


def test_an_unreadable_unit_yields_no_rows(raw_root: Path) -> None:
    assert _job((Unit("kapucu2022", "kapucu", (5,)), raw_root, None)) == []


def test_counts_report_rows_per_definition_against_the_asset(raw_root: Path) -> None:
    table = build_table(raw_root, HASHES, full=True, processes=2)
    out = counts(table, _asset(raw_root))
    assert out["kapucu2022"]["records"] == out["kapucu2022"]["asset_spike_time_records"] == 3
    assert out["kapucu2022"]["rows_by_definition"]["kapucu2022_logisi"] == 3 * len(FEATURE_NAMES)


def test_sorting_puts_definitions_in_registry_order() -> None:
    rows = [("r", "s", d, f, 0.0) for d in reversed(list(REGISTRY)) for f in FEATURE_NAMES]
    table = sort_table(pd.DataFrame(rows, columns=COLUMNS))
    assert list(dict.fromkeys(table["definition"])) == list(REGISTRY)
    assert list(table["feature"][: len(FEATURE_NAMES)]) == list(FEATURE_NAMES)


def test_quick_picks_the_records_with_the_fewest_spikes() -> None:
    records = [
        SpikeRecord("b", "s", {"x": np.arange(3.0)}, 1.0),
        SpikeRecord("a", "s", {"x": np.arange(5.0)}, 1.0),
        SpikeRecord("c", "s", {}, 1.0),
    ]
    assert [r.record_id for r in smallest(records, 0) + smallest(records, 1)] == ["c", "b"]
    assert smallest(records, 3) == []


def test_every_loader_has_a_place_in_the_job_order() -> None:
    assert set(SLOW_FIRST) == set(LOADERS)
