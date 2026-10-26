"""Read a mysqldump file as text: CREATE TABLE columns and extended INSERT rows.

The dump is untrusted data. Nothing here executes SQL: statements are matched
and tokenised with regular expressions, and values come back as Python
``None`` / ``int`` / ``float`` / ``str``. Quoted values always stay strings, so
``'0042'`` is not turned into ``42``.

Covered syntax (what mysqldump writes): one statement per line, multi-row
``VALUES (..),(..);``, optional column list, backslash escapes
(``\\0 \\' \\" \\b \\n \\r \\t \\Z \\\\``) and doubled quotes (``''``).
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

_INSERT = re.compile(rb"^INSERT INTO `([^`]+)`")
_INSERT_HEAD = re.compile(r"^INSERT INTO `[^`]+`\s*(?:\(([^)]*)\))?\s*VALUES\s*")
_CREATE_NAME = re.compile(r"CREATE TABLE `([^`]+)`")
_COLUMN = re.compile(r"^\s*`([^`]+)`\s+([a-z]+)", re.IGNORECASE)
# Possessive quantifiers (Python 3.11+) so a malformed line fails fast instead of
# backtracking through a megabyte-long statement.
_QUOTED = r"'(?:[^'\\]++|\\.|'')*+'"
# One row: "(" then quoted strings or anything without quotes/parentheses, then ")".
_ROW = re.compile(r"\(((?:" + _QUOTED + r"|[^'()]++)*+)\)\s*(,|;?\s*$)", re.DOTALL)
_FIELD = re.compile(r"\s*((?:_binary\s*)?" + _QUOTED + r"|[^,']++)\s*(?:,|$)", re.DOTALL)
_ESCAPES = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}
_ESCAPE = re.compile(r"\\(.)|''", re.DOTALL)
_INT = re.compile(r"^-?\d+$")
NUMERIC_TYPES = frozenset(
    {
        "tinyint",
        "smallint",
        "mediumint",
        "int",
        "integer",
        "bigint",
        "double",
        "float",
        "decimal",
        "real",
        "numeric",
    }
)


@dataclass(frozen=True)
class TableSchema:
    """Column names and base SQL types of one table, in declaration order."""

    name: str
    columns: tuple[str, ...]
    types: tuple[str, ...]

    @property
    def numeric(self) -> frozenset[str]:
        """Columns whose SQL type is numeric."""
        return frozenset(
            c for c, t in zip(self.columns, self.types, strict=True) if t in NUMERIC_TYPES
        )

    def check_rows(self, rows: list[tuple]) -> int:
        """Raise ValueError unless every row has one value per column; return the count."""
        for r in rows:
            if len(r) != len(self.columns):
                n = len(self.columns)
                raise ValueError(f"{self.name}: row has {len(r)} values, schema has {n}")
        return len(rows)


def parse_create_table(statement: str) -> TableSchema:
    """Columns of a ``CREATE TABLE`` statement (keys and options are ignored)."""
    m = _CREATE_NAME.search(statement)
    if not m:
        raise ValueError("not a CREATE TABLE statement")
    cols, types = [], []
    for line in statement.splitlines()[1:]:
        c = _COLUMN.match(line)
        if c:
            cols.append(c.group(1))
            types.append(c.group(2).lower())
    return TableSchema(m.group(1), tuple(cols), tuple(types))


def insert_table(line: bytes) -> str | None:
    """Table name if ``line`` is an ``INSERT INTO `t``` statement, else None."""
    m = _INSERT.match(line)
    return m.group(1).decode() if m else None


def unescape(body: str) -> str:
    """Undo mysqldump string escaping on the text between the quotes."""
    return _ESCAPE.sub(
        lambda m: "'" if m.group(1) is None else _ESCAPES.get(m.group(1), m.group(1)), body
    )


def _value(token: str):
    if token.startswith("_binary"):
        token = token[len("_binary") :].lstrip()
    if token.startswith("'"):
        return unescape(token[1:-1])
    if token == "NULL":
        return None
    if _INT.match(token):
        return int(token)
    try:
        return float(token)
    except ValueError:
        return token


def split_rows(values: str) -> list[str]:
    """Split ``(..),(..)`` into the text inside each pair of parentheses."""
    out, pos, values = [], 0, values.rstrip()
    while pos < len(values):
        m = _ROW.match(values, pos)
        if not m:
            raise ValueError(f"cannot parse VALUES near offset {pos}: {values[pos : pos + 60]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


def parse_fields(row: str) -> tuple:
    """Values of one row's text (no surrounding parentheses)."""
    vals, pos = [], 0
    while pos < len(row):
        m = _FIELD.match(row, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot parse row near offset {pos}: {row[pos : pos + 60]!r}")
        vals.append(_value(m.group(1).strip()))
        pos = m.end()
    return tuple(vals)


def parse_insert(line: str) -> tuple[tuple[str, ...] | None, list[tuple]]:
    """(column list or None, rows) of one extended INSERT statement."""
    m = _INSERT_HEAD.match(line)
    if not m:
        raise ValueError(f"not an INSERT statement: {line[:60]!r}")
    cols = None
    if m.group(1) is not None:
        cols = tuple(c.strip().strip("`") for c in m.group(1).split(","))
    body = line[m.end() :]
    if not body.lstrip().startswith("("):
        raise ValueError(f"VALUES must start with '(': {body[:60]!r}")
    return cols, [parse_fields(r) for r in split_rows(body)]


def iter_create_tables(lines: Iterable[str]):
    """Yield each complete ``CREATE TABLE`` statement found in ``lines``."""
    buf: list[str] = []
    for line in lines:
        if not buf and not line.startswith("CREATE TABLE"):
            continue
        buf.append(line.rstrip("\n"))
        if line.rstrip().endswith(";"):
            yield "\n".join(buf)
            buf = []
