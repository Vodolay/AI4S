"""Single writer for ``outputs/metrics.json``: every reported number lands here."""

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = REPO_ROOT / "outputs" / "metrics.json"


def update_metrics(section: str, payload: dict[str, Any], path: Path = METRICS_PATH) -> dict:
    """Merge ``payload`` into ``section`` of metrics.json and return the whole document."""
    path = Path(path)
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc[section] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return doc
