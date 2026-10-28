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
from pathlib import Path

from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

OUTCOMES = ("pass", "fail", "uninterpretable")
# Registrations of the manuscripts in preparation, one file name per line. The file exists only
# in the private manuscript repository; without it the submission is every study.
MANUSCRIPT_ONLY = "MANUSCRIPT_ONLY.txt"
EXCLUDED = {
    "registration_audit": "this audit's own output",
    "rhythm_audit": "re-reads the predictions of stages 1d-2e, counted under their own keys",
}


def _predictions(entry):
    """The registered predictions, also when nested under ``by_prediction``."""
    preds = entry.get("predictions") if isinstance(entry, dict) else None
    if isinstance(preds, dict) and isinstance(preds.get("by_prediction"), dict):
        return preds["by_prediction"]
    return preds


def _registered(entry) -> bool:
    preds = _predictions(entry)
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


def _unreadable(entry) -> bool:
    preds = _predictions(entry)
    return isinstance(preds, dict) and bool(preds) and not _registered(entry)


def _registration_file(*entries) -> str | None:
    """The file name of the first registration the entries name, innermost first."""
    for entry in entries:
        reg = entry.get("registration")
        reg = reg.get("file") if isinstance(reg, dict) else reg
        if isinstance(reg, str):
            return Path(reg).name
    return None


def _totals(studies: dict) -> dict:
    return {o: sum(s[o] for s in studies.values()) for o in (*OUTCOMES, "other", "total")}


def manuscript_only(folder: Path = REPO_ROOT / "preregistration") -> set[str]:
    """The registration file names listed as manuscript-only (empty without the list)."""
    path = folder / MANUSCRIPT_ONLY
    if not path.exists():
        return set()
    lines = (line.strip() for line in path.read_text().splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def audit(metrics: dict, manuscript_only: set[str] = frozenset()) -> dict:
    """Per study (and per sub-design), the outcome counts; plus the totals.

    ``unrecognised`` names every study with a ``predictions`` mapping that carries no
    ``outcome``, so that a study the counts leave out is listed rather than silently missed.
    ``submission`` repeats the totals without the studies registered in a manuscript-only file.
    """
    studies, unrecognised, files = {}, [], {}
    for key, entry in metrics.items():
        if key in EXCLUDED:
            continue
        if _registered(entry):
            studies[key], files[key] = _counts(_predictions(entry)), _registration_file(entry)
        elif _unreadable(entry):
            unrecognised.append(key)
        elif isinstance(entry, dict):
            for sub, inner in entry.items():
                name = f"{key}__{sub}"
                if _registered(inner):
                    studies[name] = _counts(_predictions(inner))
                    files[name] = _registration_file(inner, entry)
                elif _unreadable(inner):
                    unrecognised.append(name)
    shown = {k: v for k, v in studies.items() if files[k] not in manuscript_only}
    return {
        "studies": studies,
        "total": _totals(studies),
        "n_studies": len(studies),
        "submission": {"n_studies": len(shown), "total": _totals(shown)},
        "unrecognised": unrecognised,
        "excluded": {k: v for k, v in EXCLUDED.items() if k in metrics},
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: write the audit counts into metrics.json."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = outputs_dir() / "metrics.json"
    if not path.exists():
        logging.warning("%s not written yet; nothing to audit", path)
        return 0
    metrics = json.loads(path.read_text())
    got = audit(metrics, manuscript_only())
    update_metrics("registration_audit", got)
    print(json.dumps(got["total"]), got["n_studies"], "studies")
    if got["unrecognised"]:
        logging.warning("predictions without an outcome: %s", ", ".join(got["unrecognised"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
