"""What the pipeline run itself cost, and how that number reaches the report.

The reproducibility appendix quotes a wall clock and the versions it ran under.
Neither had a metrics key, which made them the only numbers in a document whose
opening paragraph promises every number has one that nothing could check.

A run records its own cost into its own metrics root, like every other stage.
Publishing it into the root the report cites is a separate, explicit step:

    python run_all.py --quick          # records harness_quick in outputs/quick/
    python -m neuromps.harness --from quick   # copies it to outputs/metrics.json

That split is not ceremony. `--quick` writes only under `outputs/quick/` so that
running it cannot alter the numbers the report cites, and CI enforces exactly
that with `git diff --exit-code` after a smoke run. An earlier version of this
module wrote across the boundary and was caught by that check. The published
figure should also come from a known machine rather than from whichever runner
happened to go last, which the explicit step makes true by construction.

This lives in the package rather than in `run_all.py` so it can be tested
without importing the entry point: `import run_all` resolves only because
`python -m pytest` puts the working directory on the path, and CI calls `pytest`
directly, where it does not.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

log = logging.getLogger("neuromps.harness")

# The pins the reproducibility appendix names. Read from what imported, not from
# pyproject.toml, so the appendix describes the environment that produced the
# numbers beside it rather than the one that was asked for.
PINNED = ("numpy", "brian2", "torch", "sbi", "scikit-learn")


def key_for(quick: bool) -> str:
    """The metrics key a run of this tier writes."""
    return "harness_quick" if quick else "harness_full"


def pinned_versions() -> dict[str, str]:
    """The versions this run actually imported."""
    versions = {"python": platform.python_version()}
    for package in PINNED:
        try:
            versions[package.replace("-", "_")] = version(package)
        except PackageNotFoundError:  # an environment without the optional stack
            continue
    return versions


def payload(
    timings: dict[str, float],
    elapsed: float,
    quick: bool,
    stages: list[str],
    all_stages: list[str],
    previous: dict | None = None,
) -> dict:
    """The record to write for one run.

    ``total_s`` is only replaced by a run that executed every stage. A partial
    run - ``--stage models`` while debugging - updates its own stage time and
    leaves the end-to-end figure alone, rather than quietly reducing it to the
    length of one stage.
    """
    previous = previous or {}
    complete = set(stages) >= set(all_stages)
    return {
        "label": (
            f"wall clock and environment for the most recent complete "
            f"{'--quick' if quick else 'full'} run"
        ),
        "total_s": elapsed if complete else previous.get("total_s"),
        "total_is_complete_pass": complete or bool(previous.get("total_is_complete_pass")),
        "by_stage_s": {**previous.get("by_stage_s", {}), **timings},
        "stages_run": list(stages),
        "cpu_count": os.cpu_count(),
        **pinned_versions(),
    }


def record(
    timings: dict[str, float], elapsed: float, quick: bool, stages: list[str], all_stages: list[str]
) -> None:
    """Write this run's cost into this run's own metrics root, and nowhere else."""
    from neuromps.metrics import update_metrics
    from neuromps.paths import metrics_path

    destination = metrics_path()
    key = key_for(quick)
    existing = json.loads(destination.read_text()) if destination.exists() else {}
    update_metrics(key, payload(timings, elapsed, quick, stages, all_stages, existing.get(key)))
    log.info("recorded %s in %s", key, destination)


def publish(quick: bool, source: Path | None = None, destination: Path | None = None) -> Path:
    """Copy one tier's harness record into the metrics root the report cites.

    Deliberate and separate, because this is the one write that crosses from a
    quick run's root into the full one, and because the number the report quotes
    should come from a machine someone chose.
    """
    from neuromps.metrics import update_metrics
    from neuromps.paths import OUTPUTS_ROOT, QUICK

    source = Path(source) if source else (OUTPUTS_ROOT / QUICK if quick else OUTPUTS_ROOT)
    source = source if source.name.endswith(".json") else source / "metrics.json"
    destination = Path(destination) if destination else OUTPUTS_ROOT / "metrics.json"
    key = key_for(quick)
    recorded = json.loads(source.read_text()).get(key)
    if recorded is None:
        raise KeyError(f"{source} has no {key}; run the pipeline on that tier first")
    if not recorded.get("total_is_complete_pass"):
        raise ValueError(
            f"{key} in {source} is from a partial run; the report should not quote it as "
            "an end-to-end time"
        )
    update_metrics(key, recorded, path=destination)
    log.info("published %s from %s to %s", key, source, destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    """CLI: publish a tier's harness record into the cited metrics root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from",
        dest="tier",
        choices=("quick", "full"),
        default="quick",
        help="which tier's record to publish (default: quick)",
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    publish(quick=arguments.tier == "quick")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
