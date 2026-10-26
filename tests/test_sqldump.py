"""The mysqldump text parser: CREATE TABLE columns and extended INSERT rows.

The dump is read as text, never executed; these snippets pin the quoting rules
the invitrodb dump relies on.
"""

import pytest

from neuromps.fetch.sqldump import (
    TableSchema,
    insert_table,
    parse_create_table,
    parse_insert,
    split_rows,
    unescape,
)

CREATE = """CREATE TABLE `mc0` (
  `m0id` bigint NOT NULL AUTO_INCREMENT,
  `acid` bigint unsigned DEFAULT '0',
  `spid` varchar(50) CHARACTER SET utf8mb4 DEFAULT NULL,
  `wllt` varchar(1) DEFAULT NULL COMMENT 'well type, e.g. ''n''',
  `conc` double DEFAULT NULL,
  `created_date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`m0id`),
  KEY `SPID/ACID` (`acid`,`spid`)
) ENGINE=InnoDB AUTO_INCREMENT=818999611 DEFAULT CHARSET=utf8mb4
/*!50100 PARTITION BY LINEAR HASH (`m0id`)
PARTITIONS 10 */;"""


def test_create_table_columns_and_types():
    s = parse_create_table(CREATE)
    assert s.name == "mc0"
    assert s.columns == ("m0id", "acid", "spid", "wllt", "conc", "created_date")
    assert s.types[:2] == ("bigint", "bigint")
    assert s.numeric == frozenset({"m0id", "acid", "conc"})


def test_insert_table_name():
    assert insert_table(b"INSERT INTO `mc0` VALUES (1,2);\n") == "mc0"
    assert insert_table(b"INSERT INTO `mc0` (`a`,`b`) VALUES (1,2);") == "mc0"
    assert insert_table(b"-- INSERT INTO `mc0` VALUES") is None
    assert insert_table(b"CREATE TABLE `mc0` (") is None


def test_multi_row_values_with_commas_quotes_null():
    line = (
        "INSERT INTO `mc0` VALUES (1,2454,'DMSO, 0.1%','n',NULL,'2024-01-01 00:00:00'),"
        "(2,2454,'it\\'s (odd),(x)','t',0.3,'2024-01-01'),"
        "(3,2455,'a''b','n',-1.5e-3,'');\n"
    )
    cols, rows = parse_insert(line)
    assert cols is None
    assert rows == [
        (1, 2454, "DMSO, 0.1%", "n", None, "2024-01-01 00:00:00"),
        (2, 2454, "it's (odd),(x)", "t", 0.3, "2024-01-01"),
        (3, 2455, "a'b", "n", -0.0015, ""),
    ]


def test_insert_with_column_list():
    cols, rows = parse_insert("INSERT INTO `t` (`a`,`b`) VALUES (1,'x'),(2,NULL);")
    assert cols == ("a", "b")
    assert rows == [(1, "x"), (2, None)]


def test_quoted_numbers_stay_strings():
    _, rows = parse_insert("INSERT INTO `t` VALUES ('0042',42,'NULL');")
    assert rows == [("0042", 42, "NULL")]


def test_escapes():
    assert unescape(r"a\\b\"c\nd\te\0f\Z") == 'a\\b"c\nd\te\x00f\x1a'
    assert unescape("x''y") == "x'y"
    # an escaped backslash before a quote does not escape the quote
    _, rows = parse_insert("INSERT INTO `t` VALUES ('ends with \\\\',1);")
    assert rows == [("ends with \\", 1)]


def test_split_rows_keeps_parentheses_inside_strings():
    rows = split_rows("(1,'a),(b'),(2,'c')")
    assert rows == ["1,'a),(b'", "2,'c'"]


def test_malformed_values_raise():
    with pytest.raises(ValueError):
        parse_insert("INSERT INTO `t` VALUES (1,'unterminated);")
    with pytest.raises(ValueError):
        parse_insert("INSERT INTO `t` VALUES 1,2;")


def test_row_width_checked_against_schema():
    s = TableSchema("t", ("a", "b"), ("int", "int"))
    _, rows = parse_insert("INSERT INTO `t` VALUES (1,2),(3,4);")
    assert s.check_rows(rows) == 2
    with pytest.raises(ValueError):
        s.check_rows([(1, 2, 3)])
