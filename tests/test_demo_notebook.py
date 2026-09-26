"""The generated Kaggle notebook: valid, self-contained, and honest about its inputs."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from neuromps.demo.notebook import (
    DATA_FILES,
    NOTEBOOK_NAME,
    build_notebook,
    code_cells,
    write_notebook,
)


def test_the_notebook_is_valid_nbformat_4() -> None:
    nb = build_notebook()
    assert nb["nbformat"] == 4
    assert nb["cells"], "a notebook with no cells is not a deliverable"
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_every_code_cell_is_syntactically_valid_python() -> None:
    """A syntax error in a generated cell is invisible until someone runs it."""
    for i, source in enumerate(code_cells()):
        try:
            ast.parse(source)
        except SyntaxError as error:  # noqa: PERF203 - the index is the useful part
            raise AssertionError(f"code cell {i} does not parse: {error}") from error


def test_the_notebook_carries_the_project_disclaimer() -> None:
    text = json.dumps(build_notebook())
    assert "decision-support" in text
    assert "efficacy" in text or "clinical" in text


def test_the_notebook_only_reads_files_it_declares() -> None:
    """The Kaggle dataset is built from DATA_FILES, so a file read but not declared
    would make the notebook work locally and fail for everyone else."""
    joined = "\n".join(code_cells())
    for name in ("grid.parquet", "meta.json", "real_records.parquet", "real_posteriors.npz"):
        if name in joined:
            assert any(name in str(f) for f in DATA_FILES), f"{name} is read but not declared"


def test_it_resolves_its_data_directory_on_kaggle_and_locally() -> None:
    """A notebook that only works in the repository is not a Kaggle notebook."""
    joined = "\n".join(code_cells())
    assert "/kaggle/input" in joined
    assert "outputs/demo" in joined


def test_writing_produces_a_file_that_reloads_as_json(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "demo.ipynb")
    assert path.exists()
    reloaded = json.loads(path.read_text())
    assert reloaded["nbformat"] == 4


DEMO_PRESENT = (
    Path("outputs/demo/grid.parquet").exists() and Path("data/processed/neuromps.parquet").exists()
)


@pytest.mark.skipif(not DEMO_PRESENT, reason="needs outputs/demo and the asset from a full run")
def test_every_cell_runs_and_every_displayed_chart_serialises() -> None:
    """Executing the notebook is the only check that finds most of its bugs.

    Syntax checking passes a cell that reads a column which does not exist, or an
    npz key that was never written. Both happened. A chart that fails to
    serialise renders as nothing in the saved notebook, which is exactly the
    failure a reader cannot diagnose.
    """
    import matplotlib

    matplotlib.use("Agg")
    import altair as alt

    namespace: dict = {"__name__": "__main__"}
    for i, source in enumerate(code_cells()):
        tree = ast.parse(source)
        last = tree.body[-1]
        if isinstance(last, ast.Expr):
            body = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(body, f"<cell {i}>", "exec"), namespace)  # noqa: S102
            shown = eval(compile(ast.Expression(last.value), f"<cell {i}>", "eval"), namespace)  # noqa: S307
        else:
            exec(compile(tree, f"<cell {i}>", "exec"), namespace)  # noqa: S102
            shown = None
        if isinstance(shown, alt.TopLevelMixin):
            spec = shown.to_dict()
            assert spec.get("$schema"), f"cell {i} produced a chart without a schema"


@pytest.mark.skipif(not DEMO_PRESENT, reason="needs outputs/demo and the asset from a full run")
def test_the_twin_chart_is_self_contained_and_stays_interactive() -> None:
    """The whole reason for Vega-Lite here: it works in saved output, no kernel."""
    import json as _json

    import matplotlib

    matplotlib.use("Agg")

    namespace: dict = {"__name__": "__main__"}
    shown = None
    for i, source in enumerate(code_cells()[:5]):
        tree = ast.parse(source)
        last = tree.body[-1]
        if isinstance(last, ast.Expr):
            body = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(body, f"<cell {i}>", "exec"), namespace)  # noqa: S102
            shown = eval(compile(ast.Expression(last.value), f"<cell {i}>", "eval"), namespace)  # noqa: S307
        else:
            exec(compile(tree, f"<cell {i}>", "exec"), namespace)  # noqa: S102
    doc = shown.to_dict()
    # Altair emits rows either inline as `values` or in a named `datasets` block;
    # both are self-contained, which is the property that matters here.
    embedded = sum(len(rows) for rows in (doc.get("datasets") or {}).values())
    assert embedded or '"values"' in _json.dumps(doc), (
        "the chart must carry its own data, not a reference to a kernel"
    )
    spec = _json.dumps(doc)
    assert '"bind"' in spec and '"select"' in spec, "the control must be client-side"


def test_the_demo_simulates_at_the_calibrated_density() -> None:
    """The demo used to hardcode 150 neurons while the calibration fitted 967.

    Cell density is what sets the twin's synchrony regime (§7.3), so a demo at a
    different density shows a different model from the one the report describes -
    and says so in a `meta.json` field nobody reads.
    """
    import inspect

    from neuromps.demo import precompute

    source = inspect.getsource(precompute.main)
    assert "calibrated_n_per_compartment()" in source
    assert "else 150" not in source, "the hardcoded density is back"


def test_the_calibrated_density_falls_back_rather_than_failing() -> None:
    """A clone with no calibration on disk must still be able to build the demo."""
    from neuromps.demo.precompute import calibrated_n_per_compartment

    assert calibrated_n_per_compartment(default=321) in {321, calibrated_n_per_compartment()}
    assert calibrated_n_per_compartment() > 0


def test_short_posterior_draws_are_padded_rather_than_crashing() -> None:
    """A recording the posterior finds unlikely comes back with fewer draws.

    `sample_posterior` asks for partial results on timeout, and stacking ragged
    arrays raises - which is exactly how the demo stage failed once the twin
    moved to its calibrated density. The shortfall must survive as data.
    """
    import numpy as np

    from neuromps.demo.precompute import pad_ragged_samples

    full = np.ones((200, 5))
    short = np.ones((193, 5)) * 2
    samples, counts = pad_ragged_samples([full, short], 200)
    assert samples.shape == (2, 200, 5)
    assert list(counts) == [200, 193]
    assert not np.isnan(samples[0]).any()
    assert np.isnan(samples[1, 193:]).all(), "missing draws must be NaN, not zero"
    assert np.nanmedian(samples[1, :, 0]) == 2.0, "the draws that were made must survive"


def test_a_recording_that_cannot_be_sampled_does_not_kill_the_build() -> None:
    """sbi raises, rather than returning partial, when nothing is accepted.

    One impossible recording should not destroy the whole demo. That it could not
    be sampled is the misspecification signal, not a crash to propagate.
    """
    import numpy as np

    from neuromps.demo.precompute import draw_or_empty, pad_ragged_samples

    class _Refuses:
        def sample(self, *a, **k):
            raise RuntimeError("rejection sampling exceeded max_sampling_time")

    empty = draw_or_empty(_Refuses(), np.zeros(19), 200)
    assert empty.shape[0] == 0
    samples, counts = pad_ragged_samples([np.ones((200, 5)), empty], 200)
    assert list(counts) == [200, 0]
    assert np.isnan(samples[1]).all(), "an unsampled recording must be NaN throughout"


def test_the_draw_shortfall_is_summarised_for_the_report() -> None:
    """The report cites how often the posterior could not sample a real recording,
    so that count has to be produced by the pipeline rather than read off a log."""
    import pandas as pd

    from neuromps.demo.precompute import posterior_draw_summary

    records = pd.DataFrame({"n_posterior_draws": [200, 200, 193, 1, 0]})
    out = posterior_draw_summary({"records": records}, requested=200)
    assert out["n_recordings"] == 5
    assert out["n_short_of_requested"] == 3
    assert out["n_unsampled"] == 1
    assert out["fewest_draws"] == 0
    assert out["median_draws"] == 193


def test_the_summary_survives_an_asset_with_no_two_chamber_recordings() -> None:
    """A --quick asset has no two-chamber source, so there is nothing to sample."""
    from neuromps.demo.precompute import posterior_draw_summary

    out = posterior_draw_summary({"records": None}, requested=200)
    assert out["n_recordings"] == 0


def test_the_generated_notebook_passes_the_linter() -> None:
    """CI lints the whole repository, and the generated notebook is part of it.

    A stray blank line in a string literal in the generator failed CI while every
    local check passed, because the notebook is linted like any other file and
    nothing in the test suite looked at it. The generator formats its cells now;
    this makes the failure visible from `pytest` rather than only from `ruff`.
    """
    path = Path("outputs/demo") / NOTEBOOK_NAME
    if not path.exists():
        pytest.skip("notebook not generated in this checkout")
    ruff = [sys.executable, "-m", "ruff"]
    for command in ([*ruff, "check", str(path)], [*ruff, "format", "--check", str(path)]):
        done = subprocess.run(command, capture_output=True, text=True)
        assert done.returncode == 0, f"{' '.join(command)}:\n{done.stdout}{done.stderr}"
