"""Check that every number cited in a report matches `outputs/metrics.json`.

The project rule is that any number which could appear in the report is produced
by a script and written to `outputs/metrics.json`. Prose cannot enforce that by
itself: a metric regenerates, the document keeps the old figure, and nothing
complains. Reports therefore cite their source inline,

    0.537 <!-- metrics.json: mechanism_baselines...balanced_accuracy_mean -->

and this module re-reads every citation and compares it with the live value, at
the precision the document displays. It is run as a test, so stale numbers fail
the build rather than reaching a reader.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger("neuromps.report")
REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS = REPO_ROOT / "outputs" / "metrics.json"
REPORT_DIR = REPO_ROOT / "report"

# The comment naming the source. The number it refers to is whatever the
# document shows just before it, which may be separated by a few words.
COMMENT = re.compile(r"<!--\s*metrics\.json:\s*(?P<key>[\w.\[\]-]+)\s*-->")
MINUS = {"\u2212": "-", "\u2013": "-"}  # typographic minus signs
WINDOW = 160  # characters before the comment that may hold the cited value
NUMBER = re.compile(r"[-+]?[\d,]*\.?\d+\s*%?")
NAME = re.compile(r"`([^`]+)`\s*$|\"([^\"]+)\"\s*$")
TIMESTAMP = re.compile(r"\d{4}-\d\d-\d\dT[\d:+-]+\s*$")


def lookup(metrics: dict, key: str):
    """Value at a dotted path, or None if the path does not exist."""
    node = metrics
    for part in key.split("."):
        if isinstance(node, list):
            try:
                node = node[int(part)]
                continue
            except (ValueError, IndexError):
                return None
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _to_float(token: str) -> float:
    """A displayed number as a float, honouring typographic minus signs."""
    for character, replacement in MINUS.items():
        token = token.replace(character, replacement)
    return float(token.replace(",", "").replace("%", "").strip())


def _trailing_numbers(window: str, count: int) -> list[str] | None:
    """The last `count` numbers shown before a citation, in order."""
    for character, replacement in MINUS.items():
        window = window.replace(character, replacement)
    found = NUMBER.findall(window)
    return found[-count:] if len(found) >= count else None


UNCHECKED = "pointer"


def compare(window: str, value) -> tuple[bool | str, str]:
    """Whether the value shown before a citation agrees with the metric.

    Returns the verdict and what the document appeared to say, so a mismatch can
    be reported in the document's own terms. Rounding, thousands separators and
    writing a fraction as a percentage are not discrepancies; a different number
    or a different sign is.
    """
    window = window.rstrip()
    if isinstance(value, bool) or value is None:
        return window.lower().endswith(str(value).lower()), window[-30:]
    if isinstance(value, str):
        match = TIMESTAMP.search(window) or NAME.search(window)
        if match is None:
            # Prose naming a text field as its source without quoting a value.
            return UNCHECKED, window[-40:].strip()
        return match.group(0).strip().strip('`"') == value.strip(), match.group(0).strip()
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        match = NAME.search(window)
        if match is not None:
            shown = match.group(1) or match.group(2)
            return shown in value, shown
        tokens = _trailing_numbers(window, 1)
        if tokens is not None and abs(_to_float(tokens[0]) - len(value)) < 1e-9:
            return True, f"{tokens[0].strip()} (list length)"
        return UNCHECKED, window[-40:].strip()
    if isinstance(value, dict) or (
        isinstance(value, list) and any(isinstance(v, (bool, dict, list)) for v in value)
    ):
        # A structured field cited as the source of a qualitative claim. Booleans
        # are not numbers: comparing them as 1.0 and 0.0 would be meaningless.
        return UNCHECKED, window[-40:].strip()
    try:
        expected = [float(v) for v in value] if isinstance(value, list) else [float(value)]
    except (TypeError, ValueError):
        return UNCHECKED, window[-40:].strip()
    tokens = _trailing_numbers(window, len(expected))
    if tokens is None:
        return False, "no number before the citation"
    shown = ", ".join(t.strip() for t in tokens)
    percent = "%" in "".join(tokens)
    scale = 100.0 if percent else 1.0
    for token, want in zip(tokens, expected, strict=True):
        got = _to_float(token)
        text = token.strip().rstrip("%")
        decimals = len(text.split(".")[1]) if "." in text else 0
        tolerance = max(0.5 * 10 ** (-decimals) * scale, 1e-9)
        if abs(got - want * scale) > tolerance + 1e-12:
            return False, shown
    return True, shown


def check_text(text: str, metrics: dict) -> list[dict]:
    """Problems found in one document: unknown keys and numbers that disagree."""
    problems = []
    for found in COMMENT.finditer(text):
        key = found.group("key")
        line_number = text.count("\n", 0, found.start()) + 1
        value = lookup(metrics, key)
        if value is None:
            problems.append({"line": line_number, "key": key, "problem": "key not in metrics"})
            continue
        window = text[max(0, found.start() - WINDOW) : found.start()]
        agrees, shown = compare(window, value)
        if agrees is UNCHECKED:
            problems.append(
                {"line": line_number, "key": key, "problem": "pointer, no value quoted"}
            )
            continue
        if not agrees:
            problems.append(
                {
                    "line": line_number,
                    "key": key,
                    "problem": "does not match metrics",
                    "cited": shown,
                    "actual": value,
                }
            )
    return problems


def check_documents(paths: list[Path], metrics_path: Path = METRICS) -> dict[str, list[dict]]:
    """Check several documents, returning problems per document."""
    metrics = json.loads(Path(metrics_path).read_text())
    return {str(p): check_text(p.read_text(), metrics) for p in paths}


def citation_count(text: str) -> int:
    """How many claims in a document are backed by a metrics key."""
    return len(COMMENT.findall(text))


def main(argv: list[str] | None = None) -> int:
    """Check every report document; non-zero exit if any number is stale."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="accepted for run_all; no effect")
    p.add_argument("paths", nargs="*", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = args.paths or sorted(REPORT_DIR.glob("*.md"))
    found = check_documents(paths)
    total = 0
    for document, problems in found.items():
        name = Path(document).name
        cited = citation_count(Path(document).read_text())
        failures = [p for p in problems if p["problem"] != "pointer, no value quoted"]
        pointers = len(problems) - len(failures)
        verified = cited - len(problems)
        print(
            f"{name}: {cited} citations - {verified} verified, {pointers} pointers, "
            f"{len(failures)} failing"
        )
        for problem in failures:
            print(f"   line {problem['line']}: {problem['problem']} - {problem['key']}")
            if "cited" in problem:
                print(f"      document says {problem['cited']}, metrics say {problem['actual']}")
        total += len(failures)
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
