"""End-to-end extraction from a small synthetic mysqldump (gzip), schema as in invitrodb v4.3."""

import gzip
import json

import pandas as pd
import pytest

from neuromps.etl import toxcast_wells as tw
from neuromps.etl.toxcast_wells_summary import (
    component_summary,
    physical_plate,
    physical_plate_counts,
)

SCHEMA = {
    "assay": "`aid` int,\n  `asid` int,\n  `assay_name` varchar(255)",
    "assay_component": "`acid` int,\n  `aid` int,\n  `assay_component_name` varchar(255)",
    "assay_component_endpoint": (
        "`aeid` int,\n  `acid` int,\n  `assay_component_endpoint_name` varchar(255),\n"
        "  `normalized_data_type` varchar(255)"
    ),
    "chemical": "`chid` int,\n  `casn` varchar(45),\n  `chnm` varchar(255)",
    "mc0": (
        "`m0id` bigint,\n  `acid` bigint unsigned,\n  `spid` varchar(50),\n  `apid` varchar(100),\n"
        "  `wllt` varchar(1),\n  `wllq` smallint,\n  `conc` double,\n  `rval` double"
    ),
    "mc1": "`m1id` bigint,\n  `m0id` bigint,\n  `acid` bigint,\n  `cndx` int",
    "mc3": "`m3id` bigint,\n  `aeid` bigint,\n  `m0id` bigint,\n  `acid` bigint,\n  `resp` double",
    "mc3_aeid": "`mc3_mthd_id` int,\n  `aeid` int,\n  `exec_ordr` int",
    "mc3_methods": "`mc3_mthd_id` int,\n  `mc3_mthd` varchar(50)",
    "sample": "`spid` varchar(50),\n  `chid` int",
}
INSERTS = {
    "assay": ["(1,9,'CCTE_Shafer_MEA_acute'),(2,9,'OTHER')"],
    "assay_component": [
        "(10,1,'CCTE_Shafer_MEA_acute_firing_rate_mean'),(11,1,'CCTE_Shafer_MEA_acute_AB')",
        "(12,2,'CCTE_Shafer_MEA_dev_firing_rate_mean'),(13,2,'other, with (parens)')",
    ],
    "assay_component_endpoint": [
        "(100,10,'CCTE_Shafer_MEA_acute_firing_rate_mean','percent_activity'),"
        "(101,11,'CCTE_Shafer_MEA_acute_AB','percent_activity'),(102,12,'dev',NULL)"
    ],
    "chemical": ["(1,'67-68-5','Dimethyl sulfoxide'),(2,'1-1-1','Chem \\'A\\', salt')"],
    "mc0": [
        "(1,10,'DMSO','P1','n',1,0.1,-3.5),(2,10,'DMSO','P1','n',0,0.1,NULL),"
        "(3,10,'S1','P1','t',1,3,-50),(4,12,'DMSO','P1','n',1,0.1,2)",
        "(5,10,'DMSO','P2','n',1,0.1,4.25),(6,13,'X','P9','t',1,1,10)",
        "(7,11,'S1','P2','t',1,10,99),(8,10,'S1','P2','n',1,0,1.5)",
    ],
    "mc1": ["(1,1,10,1),(2,4,12,1),(3,7,11,2)"],
    "mc3": ["(1,100,1,10,3.5),(2,102,4,12,0),(3,101,99,11,1)"],
    "mc3_aeid": ["(5,100,1),(17,100,2),(5,102,1)"],
    "mc3_methods": ["(5,'resp.pc'),(17,'bval.apid.nwllslowconc.med')"],
    "sample": ["('DMSO',1),('S1',2),('X',NULL)"],
}


def _dump_text() -> str:
    parts = ["-- MySQL dump 10.13\n", "/*!40101 SET NAMES utf8mb4 */;\n"]
    for t in sorted(SCHEMA):
        parts.append(f"DROP TABLE IF EXISTS `{t}`;\n")
        parts.append(f"CREATE TABLE `{t}` (\n  {SCHEMA[t]},\n  PRIMARY KEY (`x`)\n) ENGINE=x;\n")
        parts += [f"INSERT INTO `{t}` VALUES {v};\n" for v in INSERTS.get(t, [])]
    return "".join(parts)


@pytest.fixture
def dump(tmp_path):
    p = tmp_path / "raw" / "fake.sql.gz"
    p.parent.mkdir()
    with gzip.open(p, "wt") as fh:
        fh.write(_dump_text())
    return p


def test_extract_selects_acute_components_and_their_rows(dump, tmp_path):
    ex = tw.extract(tw.stream_lines(dump))
    assert ex.keys == {"acid": {10, 11}, "aeid": {100, 101}}
    assert set(ex.schemas) == set(SCHEMA)
    tables = tw.filter_tables(ex.frames(), ex.keys["acid"])
    assert sorted(tables["mc0"]["m0id"]) == [1, 2, 3, 5, 7, 8]
    assert sorted(tables["mc1"]["m1id"]) == [1, 3]
    assert list(tables["mc3"]["m3id"]) == [1, 3]
    assert sorted(tables["sample"]["spid"]) == ["DMSO", "S1"]
    assert sorted(tables["chemical"]["chid"]) == [1, 2]
    assert list(tables["assay"]["aid"]) == [1]
    assert tables["mc0"]["rval"].dtype == float
    assert tables["mc0"]["rval"].isna().sum() == 1


def test_write_outputs(dump, tmp_path):
    ex = tw.extract(tw.stream_lines(dump))
    out = tmp_path / "out"
    s = tw.write_outputs(ex, out, {"dump": dump.name})
    assert "CREATE TABLE `mc0`" in (out / "schema.sql.txt").read_text()
    assert pd.read_parquet(out / "mc0_mea_acute.parquet").shape[0] == 6
    assert set(pd.read_parquet(out / "components.parquet")["aeid"]) == {100, 101}
    assert "Chem 'A', salt" in set(pd.read_parquet(out / "samples.parquet")["chnm"])
    on_disk = json.loads((out / "summary.json").read_text())
    assert on_disk["row_counts"] == s["row_counts"]
    assert on_disk["insert_lines"]["mc0"] == 3
    fr = next(c for c in on_disk["components"] if c["acid"] == 10)
    assert fr["n_vehicle_wells"] == 4 and fr["n_vehicle_wells_wllq1"] == 3
    assert fr["n_plates"] == 2 and fr["n_plates_with_vehicle"] == 2
    assert fr["vehicle_wells_per_plate"] == {"median": 2.0, "min": 2, "max": 2, "counts": {"2": 2}}
    # one vehicle well holds a test sample: both identifications are reported
    assert fr["vehicle_well_labels"] == {"Dimethyl sulfoxide": 3, "Chem 'A', salt": 1}
    assert fr["wllt_of_vehicle_labels"]["Chem 'A', salt"] == {"t": 1, "n": 1}
    ab = next(c for c in on_disk["components"] if c["acid"] == 11)
    assert fr["n_mc3_rows_without_mc0"] == 0 and ab["n_mc3_rows_without_mc0"] == 1
    assert on_disk["components_without_mc0"] == []
    assert on_disk["mc3_methods_by_aeid"] == {"100": ["resp.pc", "bval.apid.nwllslowconc.med"]}


def test_physical_plates_from_srcf():
    f = "TC_MW 1073-16_20150729_20150811_13_00(000)_Neural Statistics Compiler(000).csv"
    g = "TC_20150811_MW69-0102_13_00(000)(000).csv;TC_20150811_MW69-0102_13_01(000)(000).csv"
    mc0 = pd.DataFrame(
        {
            "apid": ["20150811"] * 4 + ["20150812", "20150812"],
            "wllt": ["n", "n", "n", "t", "n", "n"],
            "srcf": [f, f, g, g, f, "Calculations_Group_2.xlsx"],
        }
    )
    assert physical_plate(mc0).tolist()[:5] == [
        "20150811_MW1073-16",
        "20150811_MW1073-16",
        "20150811_MW69-0102",
        "20150811_MW69-0102",
        "20150812_MW1073-16",  # same plate name on another date is another plate
    ]
    c = physical_plate_counts(mc0)
    assert c["n_srcf_plates"] == 3 and c["n_wells_srcf_unparsed"] == 1
    assert c["srcf_plates_per_apid"] == {"2": 1, "1": 1}
    assert c["vehicle_wells_per_srcf_plate"]["counts"] == {"1": 2, "2": 1}


def test_component_summary_without_vehicle_wells():
    mc0 = pd.DataFrame(
        {"spid": ["S"], "apid": ["P"], "wllt": ["t"], "wllq": [1], "conc": [1.0], "rval": [0.0]}
    )
    s = component_summary(1, mc0, pd.Series(dtype=object))
    assert s["n_vehicle_wells"] == 0 and s["n_plates_without_vehicle"] == 1
    assert s["vehicle_wells_per_plate"]["median"] is None


def test_well_rows_before_components_raise(tmp_path):
    p = tmp_path / "bad.sql.gz"
    text = (
        "CREATE TABLE `assay_component` (\n  `acid` int,\n  `assay_component_name` text\n);\n"
        "CREATE TABLE `mc0` (\n  `m0id` int,\n  `acid` int\n);\n"
        "INSERT INTO `mc0` VALUES (1,10);\n"
    )
    with gzip.open(p, "wt") as fh:
        fh.write(text)
    with pytest.raises(ValueError, match="before mc0"):
        tw.extract(tw.stream_lines(p))


def test_truncated_gzip_raises(dump, tmp_path):
    cut = tmp_path / "cut.sql.gz"
    cut.write_bytes(dump.read_bytes()[:-20])
    with pytest.raises(RuntimeError, match="zcat exited"):
        list(tw.stream_lines(cut))
