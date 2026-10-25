"""Every fly study after the chip-to-animal study, in dependency order.

``python -m neuromps.fly.pipeline`` reruns, one after another, the registered fly studies
that each have their own command: the male-CNS replication and its follow-ups, the
output-tract electrode, the motor-level readouts, the diagnostics, and every embodied
stage with its exploratory follow-ups and the rhythm-measure audit. ``run_all.py``
calls it as the ``embodied`` stage of a full run, after the ``fly`` stage (the
chip-to-animal study, `fly.study`), whose outputs several steps read.

``--list`` prints the steps; ``--from NAME`` starts at a step; ``--only NAME`` runs one.
Every step writes its own outputs and metrics; all of it is **simulated**.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass

log = logging.getLogger("neuromps.fly")


@dataclass(frozen=True)
class Step:
    """One registered study (or follow-up): `neuromps.fly.<module>.<function>(*args)`."""

    name: str
    module: str
    function: str = "main"
    needs: tuple[str, ...] = ()

    def command(self) -> list[str]:
        """The step as its own process, exactly as each study was run when it was recorded."""
        flag = FLAGS.get(self.function)
        return [sys.executable, "-m", f"neuromps.fly.{self.module}", *([flag] if flag else [])]

    def run(self) -> int:
        # A process per step: several studies hold whole connectomes in memory, which would
        # accumulate if every step ran in this one.
        return subprocess.run(self.command(), check=False).returncode


# The entry points other than main() are reached through these command-line flags.
FLAGS = {
    "run_robustness": "--robustness",
    "diagnose": "--diagnose",
    "check": "--check",
    "precision": "--precision",
}


STEPS: tuple[Step, ...] = (
    Step("replication", "replication"),
    Step("confirm", "confirm", needs=("replication",)),
    Step("screen_cutoffs", "screen_cutoffs", needs=("confirm",)),
    Step("checks", "checks"),
    Step("tract", "tract"),
    Step("tract_robustness", "tract", "run_robustness", needs=("tract",)),
    Step("motor", "motor", needs=("replication",)),
    Step("tie_break", "tie_break", needs=("motor",)),
    Step("spread", "spread"),
    Step("excitability", "excitability"),
    Step("roadmap_inventory", "roadmap_inventory"),
    Step("stage1", "embodied_stage1"),
    Step("stage1b", "embodied_stage1b"),
    Step("stage1c", "embodied_stage1c"),
    Step("stage1d", "embodied_stage1d"),
    Step("stage1e", "relay"),
    Step("stage1f", "embodied_stage1f"),
    Step("stage2a", "fictive"),
    Step("stage2b", "embodied_stage2b"),
    Step("stage2c", "embodied_stage2c"),
    Step("stage2c_diagnostic", "flexor_diagnostic"),
    Step("stage2d", "embodied_stage2d"),
    Step("stage2d_confirm", "embodied_stage2d_confirm"),
    Step("stage2e", "embodied_stage2e"),
    Step("stage2f", "embodied_stage2f"),
    Step("stage2f_diagnostic", "embodied_stage2f", "diagnose"),
    Step("stage2g", "embodied_stage2g"),
    Step("stage2h", "embodied_stage2h"),
    Step("stage2h_check", "embodied_stage2h", "check", needs=("stage2h",)),
    Step("stage3a", "embodied_stage3a"),
    Step("stage3a_check", "embodied_stage3a", "check", needs=("stage3a",)),
    Step("stage3b", "embodied_stage3b"),
    Step("stage3b_check", "embodied_stage3b", "check"),
    Step("stage3c", "embodied_stage3c"),
    Step("stage3d", "embodied_stage3d"),
    Step("stage4a", "embodied_stage4a", needs=("tract",)),
    Step("stage4a_precision", "embodied_stage4a", "precision", needs=("stage4a",)),
    Step("stage4b", "embodied_stage4b", needs=("confirm", "stage4a")),
    Step("stage4c", "embodied_stage4c", needs=("stage4b",)),
    Step("stage5", "embodied_stage5", needs=("stage4a",)),
    Step("stage5b", "embodied_stage5b"),
    Step("stage5c", "embodied_stage5c"),
    Step(
        "rhythm_audit",
        "rhythm_audit",
        needs=("stage1d", "stage2a", "stage2c", "stage2d", "stage2e"),
    ),
)


def select(
    steps: tuple[Step, ...], start: str | None = None, only: str | None = None
) -> list[Step]:
    """The steps to run: all, from `start` on, or just `only`."""
    names = [s.name for s in steps]
    for name in (start, only):
        if name is not None and name not in names:
            raise SystemExit(f"unknown step {name!r}; known: {', '.join(names)}")
    if only is not None:
        return [s for s in steps if s.name == only]
    return list(steps[names.index(start) :]) if start is not None else list(steps)


def run(steps: list[Step]) -> dict[str, float]:
    """Run steps in order; stop at the first that fails. Returns seconds per step."""
    timings = {}
    for step in steps:
        started = time.time()
        log.info("fly pipeline: %s (%s.%s)", step.name, step.module, step.function)
        code = step.run()
        timings[step.name] = round(time.time() - started, 1)
        if code:
            raise SystemExit(f"fly pipeline: step {step.name} returned {code}")
    return timings


def main(argv: list[str] | None = None) -> int:
    """CLI: run the pipeline, or part of it."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="print the steps and exit")
    p.add_argument("--from", dest="start", help="start at this step")
    p.add_argument("--only", help="run only this step")
    args = p.parse_args(argv)
    if args.list:
        for s in STEPS:
            after = f"  (after {', '.join(s.needs)})" if s.needs else ""
            print(f"{s.name:20s} neuromps.fly.{s.module}.{s.function}{after}")
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    timings = run(select(STEPS, args.start, args.only))
    log.info("fly pipeline finished: %s", timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
