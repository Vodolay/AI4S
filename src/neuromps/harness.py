"""What the pipeline run itself cost, recorded where the report can cite it.

The reproducibility appendix quotes a wall clock and the versions it ran under.
Neither had a metrics key, which made them the only numbers in a document whose
opening paragraph promises every number has one that nothing could check, and
the one place a stale figure could outlive a rebuild.

This lives in the package rather than in `run_all.py` so that it can be tested
without importing the entry point. `import run_all` worked locally only because
`python -m pytest` puts the working directory on the path, and CI calls `pytest`
directly, where it does not.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from importlib.metadata import PackageNotFoundError, version

log = logging.getLogger("neuromps.harness")

# The pins the reproducibility appendix names. Read from what imported, not from
# pyproject.toml, so the appendix describes the environment that produced the
# numbers beside it rather than the one that was asked for.
PINNED = ("numpy", "brian2", "torch", "sbi", "scikit-learn")


def pinned_versions() -> dict[str, str]:
    """The versions this run actually imported."""
    versions = {"python": platform.python_version()}
    for package in PINNED:
        try:
            versions[package.replace("-", "_")] = version(package)
        except PackageNotFoundError:  # an environment without the optional stack
            continue
    return versions


def harness_record(
    timings: dict[str, float],
    elapsed: float,
    quick: bool,
    stages: list[str],
    all_stages: list[str],
    previous: dict | None = None,
) -> tuple[str, dict]:
    """The key and payload to write for one run.

    ``total_s`` is only replaced by a run that executed every stage. A partial
    run - ``--stage models`` while debugging - updates its own stage time and
    leaves the end-to-end figure the report cites alone, rather than quietly
    reducing it to the length of one stage.
    """
    previous = previous or {}
    merged = {**previous.get("by_stage_s", {}), **timings}
    complete = set(stages) >= set(all_stages)
    key = "harness_quick" if quick else "harness_full"
    payload = {
        "label": (
            f"wall clock and environment for the most recent complete "
            f"{'--quick' if quick else 'full'} run"
        ),
        "total_s": elapsed if complete else previous.get("total_s"),
        "total_is_complete_pass": complete or bool(previous.get("total_is_complete_pass")),
        "by_stage_s": merged,
        "stages_run": list(stages),
        "cpu_count": os.cpu_count(),
        **pinned_versions(),
    }
    return key, payload


def record(
    timings: dict[str, float], elapsed: float, quick: bool, stages: list[str], all_stages: list[str]
) -> None:
    """Write the harness record into the metrics root the report cites.

    This writes to the FULL metrics.json from either tier. That is the single
    deliberate exception to the quick run's isolation, and a narrow one: the key
    names its tier, only that key is touched, and a test holds it to that. How
    long the harness takes is a property of the harness, not a result of the
    data tier, so a quick run reporting its own wall clock overwrites nothing.
    """
    from neuromps.metrics import update_metrics
    from neuromps.paths import full_run_equivalent, metrics_path

    canonical = full_run_equivalent(metrics_path())
    document = json.loads(canonical.read_text()) if canonical.exists() else {}
    key, payload = harness_record(
        timings, elapsed, quick, stages, all_stages, previous=document.get(key_for(quick), {})
    )
    update_metrics(key, payload, path=canonical)
    log.info("recorded %s in %s", key, canonical)


def key_for(quick: bool) -> str:
    """The metrics key a run of this tier writes."""
    return "harness_quick" if quick else "harness_full"
