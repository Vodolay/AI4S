"""How many worker processes an embarrassingly parallel stage should use.

One helper, because the answer was copied into three modules and was wrong in
all of them on a small machine. ``cpu_count() - 2`` leaves the OS room to
breathe on a workstation; on a two- or four-core CI runner it throws away half
the machine, which is what turned the quick run from a few minutes into a
timeout. Reserve cores only where there are cores to spare.

Affinity, not the raw CPU count: a runner, a container or ``taskset`` restricts
which CPUs this process may use, and asking the kernel what we are allowed to
run on is the only way to see that restriction.
"""

from __future__ import annotations

import os

RESERVE_ABOVE = 4
DEFAULT_CAP = 16


def available_cpus() -> int:
    """CPUs this process may actually run on."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def worker_processes(cap: int = DEFAULT_CAP, available: int | None = None) -> int:
    """Processes to run in parallel: at least one, never more than `cap`.

    Two cores are held back for the OS and the parent process, but only on a
    machine with more than `RESERVE_ABOVE` of them.
    """
    available = available_cpus() if available is None else available
    available = max(1, available)
    reserved = 2 if available > RESERVE_ABOVE else 0
    return max(1, min(available - reserved, cap))
