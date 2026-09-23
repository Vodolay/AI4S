"""Single writer for ``metrics.json``: every reported number lands here.

Which metrics.json depends on the run: a quick run has its own, so it cannot
overwrite the numbers the report cites. See :mod:`neuromps.paths`.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from neuromps.paths import is_quick, metrics_path

log = logging.getLogger("neuromps.metrics")


def warn_if_quick_run_writes_to_the_full_root() -> bool:
    """Say so loudly when a `--quick` module is about to overwrite full results.

    `run_all.py --quick` sets ``NEUROMPS_RUN_MODE``; a stage module run on its
    own does not, by design, because deriving the root from a module's own flag
    would change where every later import writes. The cost of that design is
    this trap: ``python -m neuromps.models.x --quick`` writes reduced numbers
    into the root the report cites, and nothing used to notice until the
    citation checker failed. Returns whether it warned, so a test can see it.
    """
    if "--quick" in sys.argv[1:] and not is_quick():
        log.warning(
            "writing to the FULL metrics root while --quick was passed. Stage modules follow "
            "NEUROMPS_RUN_MODE, not their own flag: prefix with NEUROMPS_RUN_MODE=quick, or "
            "re-run this stage without --quick afterwards to restore the numbers the report cites."
        )
        return True
    return False


def update_metrics(section: str, payload: dict[str, Any], path: Path | None = None) -> dict:
    """Merge ``payload`` into ``section`` of metrics.json and return the whole document.

    Writes to the active run's metrics.json unless ``path`` says otherwise.
    """
    if path is None:
        warn_if_quick_run_writes_to_the_full_root()
    path = Path(path) if path is not None else metrics_path()
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc[section] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return doc
