"""How many processes a parallel stage asks for.

The rule this pins down is the one that broke CI: reserving two cores is
sensible on a workstation and ruinous on a two-core runner, where it halves an
already small machine.
"""

from __future__ import annotations

import pytest

from neuromps.parallel import available_cpus, worker_processes


@pytest.mark.parametrize(
    ("available", "expected"),
    [(1, 1), (2, 2), (3, 3), (4, 4), (5, 3), (8, 6), (32, 16)],
)
def test_small_machines_keep_every_core(available: int, expected: int) -> None:
    assert worker_processes(available=available) == expected


def test_the_cap_is_respected() -> None:
    assert worker_processes(cap=4, available=64) == 4


def test_a_nonsense_count_still_yields_a_usable_process_count() -> None:
    assert worker_processes(available=0) == 1
    assert worker_processes(available=-3) == 1


def test_available_cpus_is_positive_and_respects_affinity() -> None:
    import os

    assert available_cpus() >= 1
    if hasattr(os, "sched_getaffinity"):
        assert available_cpus() == len(os.sched_getaffinity(0))
