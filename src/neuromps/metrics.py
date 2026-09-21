"""Single writer for ``metrics.json``: every reported number lands here.

Which metrics.json depends on the run: a quick run has its own, so it cannot
overwrite the numbers the report cites. See :mod:`neuromps.paths`.
"""

import json
from pathlib import Path
from typing import Any

from neuromps.paths import metrics_path


def update_metrics(section: str, payload: dict[str, Any], path: Path | None = None) -> dict:
    """Merge ``payload`` into ``section`` of metrics.json and return the whole document.

    Writes to the active run's metrics.json unless ``path`` says otherwise.
    """
    path = Path(path) if path is not None else metrics_path()
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc[section] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return doc
