"""The architecture figure: generated, and naming modules that exist."""

import importlib
import re
from pathlib import Path

from neuromps.report.architecture import ARROWS, BOXES, draw

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_box_that_names_a_path_names_one_that_exists() -> None:
    """A diagram that drifts from the code is worse than no diagram."""
    for *_, title, _body in BOXES:
        # A box may name more than one, as "etl/ + qc/" does.
        for part in (p.strip() for p in title.split("+")):
            if not part.endswith("/") and "." not in part:
                continue  # prose boxes like "the data asset"
            name = part.rstrip("/")
            candidates = [
                REPO_ROOT / "src" / "neuromps" / name,
                (REPO_ROOT / "src" / "neuromps" / name).with_suffix(".py"),
                REPO_ROOT / name,  # repository directories such as outputs/
            ]
            assert any(c.exists() for c in candidates), part


def test_the_feature_extractor_box_names_the_importable_module() -> None:
    assert importlib.import_module("neuromps.features.mea") is not None


def test_the_boxes_do_not_overlap() -> None:
    for i, (x0, y0, w0, h0, a, _) in enumerate(BOXES):
        for x1, y1, w1, h1, b, _ in BOXES[i + 1 :]:
            apart = x0 + w0 <= x1 or x1 + w1 <= x0 or y0 + h0 <= y1 or y1 + h1 <= y0
            assert apart, f"{a} overlaps {b}"


def test_every_arrow_starts_and_ends_somewhere_on_the_canvas() -> None:
    for (x0, y0), (x1, y1), _ in ARROWS:
        for value in (x0, y0, x1, y1):
            assert 0.0 <= value <= 1.0
        assert (x0, y0) != (x1, y1)


def test_the_record_count_in_the_diagram_matches_the_report() -> None:
    """The one number on the figure is the one the report leads with."""
    counts = {body for *_, body in BOXES if "records" in body}
    assert counts, "the asset box should state a record count"
    shown = re.search(r"([\d,]+) records", " ".join(counts)).group(1)
    report = (REPO_ROOT / "report" / "TECHNICAL_REPORT.md").read_text()
    assert f"{shown} records" in report


def test_draw_writes_a_figure(tmp_path: Path) -> None:
    path = draw(tmp_path / "architecture.png")
    assert path.exists() and path.stat().st_size > 10_000
