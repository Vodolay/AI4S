"""Count the outcomes of every registered prediction, for the paper's audit table.

``python -m neuromps.report.audit`` reads `outputs/metrics.json`, finds every study that
records registered predictions (a ``predictions`` mapping whose entries carry an
``outcome``, at the top level or one level down, as for the size rules of stages 2f-2h),
and writes the counts of pass, fail and uninterpretable outcomes per study, and in total,
under ``registration_audit``. Keys one level down are joined with a double underscore,
because a dot would break the citation lookup.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from neuromps.metrics import update_metrics
from neuromps.paths import outputs_dir

OUTCOMES = ("pass", "fail", "uninterpretable")


def _registered(entry) -> bool:
    preds = entry.get("predictions") if isinstance(entry, dict) else None
    return (
        isinstance(preds, dict)
        and bool(preds)
        and all(isinstance(p, dict) and "outcome" in p for p in preds.values())
    )


def _counts(preds: dict) -> dict:
    got = Counter(p["outcome"] for p in preds.values())
    out = {o: int(got.get(o, 0)) for o in OUTCOMES}
    out["other"] = int(sum(n for o, n in got.items() if o not in OUTCOMES))
    out["total"] = int(sum(got.values()))
    return out


def audit(metrics: dict) -> dict:
    """Per study (and per sub-design), the outcome counts; plus the totals."""
    studies = {}
    for key, entry in metrics.items():
        if key == "registration_audit":
            continue
        if _registered(entry):
            studies[key] = _counts(entry["predictions"])
        elif isinstance(entry, dict):
            for sub, inner in entry.items():
                if _registered(inner):
                    studies[f"{key}__{sub}"] = _counts(inner["predictions"])
    total = {o: sum(s[o] for s in studies.values()) for o in (*OUTCOMES, "other", "total")}
    return {"studies": studies, "total": total, "n_studies": len(studies)}


def main(argv: list[str] | None = None) -> int:
    """CLI: write the audit counts into metrics.json."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = json.loads((outputs_dir() / "metrics.json").read_text())
    got = audit(metrics)
    update_metrics("registration_audit", got)
    print(json.dumps(got["total"]), got["n_studies"], "studies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
