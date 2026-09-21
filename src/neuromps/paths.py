"""Where a run reads and writes, so a quick run cannot overwrite a full one.

There are two run roots. The default one - ``outputs/`` and
``data/processed/`` - holds the full run whose numbers the report cites. The
quick root, ``outputs/quick/`` and ``data/processed/quick/``, holds the CI-path
run: the same pipeline on the small-tier sources only, with fewer draws. Both
are complete runs of the same code; they differ in what went in.

``run_all.py --quick`` selects the quick root by setting ``NEUROMPS_RUN_MODE``,
and every writer resolves its path through this module **when it runs**, not
when it is imported, so one environment variable redirects the whole pipeline.
``run_all.py`` is the only caller of :func:`set_run_mode`: a module that set it
from its own ``--quick`` flag would change where every later import writes, and
in a test session that outlives the call.

Reading is separate from writing. A quick stage that needs an artifact an
earlier stage produces - the trained posterior, for ``--stage demo --quick`` -
finds it in the quick root if that run made it, and otherwise falls back to the
full run's copy and says so in the log. Mixing the two is sometimes what you
want and never what you should do silently.
"""

import logging
import os
from pathlib import Path

log = logging.getLogger("neuromps.paths")
REPO_ROOT = Path(__file__).resolve().parents[2]
MODE_ENV = "NEUROMPS_RUN_MODE"
QUICK = "quick"
FULL = "full"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
RAW_ROOT = REPO_ROOT / "data" / "raw"


def run_mode() -> str:
    """``"quick"`` when this process is a quick run, else ``"full"``."""
    return QUICK if os.environ.get(MODE_ENV, FULL).lower() == QUICK else FULL


def is_quick() -> bool:
    """Whether writes go to the quick root."""
    return run_mode() == QUICK


def set_run_mode(mode: str) -> None:
    """Select the run root for this process and anything it imports later."""
    if mode not in (QUICK, FULL):
        raise ValueError(f"run mode must be {QUICK!r} or {FULL!r}, not {mode!r}")
    os.environ[MODE_ENV] = mode


def outputs_dir() -> Path:
    """Root for figures, metrics and twin artifacts of the active run."""
    return OUTPUTS_ROOT / QUICK if is_quick() else OUTPUTS_ROOT


def processed_dir() -> Path:
    """Root for the built data asset of the active run."""
    return PROCESSED_ROOT / QUICK if is_quick() else PROCESSED_ROOT


def interim_dir() -> Path:
    """Root for cached intermediates of the active run.

    Split like the other roots: a quick run's partial cache must not be picked
    up by a full one, which would silently mix tiers.
    """
    return INTERIM_ROOT / QUICK if is_quick() else INTERIM_ROOT


def figures_dir() -> Path:
    """``outputs/figures`` of the active run."""
    return outputs_dir() / "figures"


def twin_dir() -> Path:
    """``outputs/twin`` of the active run."""
    return outputs_dir() / "twin"


def demo_dir() -> Path:
    """``outputs/demo`` of the active run."""
    return outputs_dir() / "demo"


def metrics_path() -> Path:
    """``outputs/metrics.json`` of the active run."""
    return outputs_dir() / "metrics.json"


def data_card_path() -> Path:
    """``outputs/DATA_CARD.md`` of the active run."""
    return outputs_dir() / "DATA_CARD.md"


def asset_path() -> Path:
    """The built parquet asset of the active run."""
    return processed_dir() / "neuromps.parquet"


def manifest_path(raw_root: Path | None = None) -> Path:
    """The provenance manifest of the active run.

    Raw downloads are shared: a quick fetch pulls a subset of the same files
    into the same ``data/raw/`` tree. The manifest is not shared, because it
    records what this run verified. A quick run writing the committed manifest
    would drop the full-tier entries from it - deleting the full run's
    provenance record to describe a machine that holds fewer files.
    """
    root = Path(raw_root) if raw_root is not None else RAW_ROOT
    return root / ("MANIFEST.quick.json" if is_quick() else "MANIFEST.json")


def simulation_cache() -> Path:
    """The sbi simulation cache of the active run.

    Per mode, and deliberately outside the :func:`for_reading` fallback: the
    quick run draws 150 simulations where the full run draws 3000, so a shared
    cache would let a quick run leave 150 draws behind for the next full run to
    train its posterior on, with nothing in the output saying so.
    """
    root = INTERIM_ROOT / QUICK if is_quick() else INTERIM_ROOT
    return root / "sbi_simulations.npz"


def full_run_equivalent(path: Path) -> Path:
    """The same artifact in the full run's root; ``path`` itself in a full run."""
    for quick_root, full_root in (
        (OUTPUTS_ROOT / QUICK, OUTPUTS_ROOT),
        (PROCESSED_ROOT / QUICK, PROCESSED_ROOT),
    ):
        try:
            return full_root / Path(path).relative_to(quick_root)
        except ValueError:
            continue
    return Path(path)


def for_reading(path: Path) -> Path:
    """``path`` if this run produced it, else the full run's copy, logged.

    Returns ``path`` unchanged when neither exists, so the caller's own
    missing-file handling - a warning and a skip, usually - still applies.
    """
    path = Path(path)
    if path.exists():
        return path
    fallback = full_run_equivalent(path)
    if fallback != path and fallback.exists():
        log.warning(
            "quick run: %s was not produced by this run; reading the full run's %s instead",
            path,
            fallback,
        )
        return fallback
    return path
