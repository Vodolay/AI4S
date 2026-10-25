"""The fly pipeline: every metrics producer present, steps after what they need."""

from __future__ import annotations

import importlib

import pytest

from neuromps.fly import pipeline


def test_every_step_follows_the_steps_it_needs() -> None:
    order = [s.name for s in pipeline.STEPS]
    assert len(order) == len(set(order))
    for k, step in enumerate(pipeline.STEPS):
        for need in step.needs:
            assert need in order[:k], (step.name, need)


def test_every_step_names_a_real_entry_point() -> None:
    for step in pipeline.STEPS:
        module = importlib.import_module(f"neuromps.fly.{step.module}")
        assert callable(getattr(module, step.function)), step.name


def test_every_fly_study_module_with_a_command_is_a_step() -> None:
    modules = {s.module for s in pipeline.STEPS}
    expected = {
        "replication",
        "confirm",
        "screen_cutoffs",
        "checks",
        "tract",
        "motor",
        "tie_break",
        "spread",
        "excitability",
        "roadmap_inventory",
        "embodied_stage1",
        "embodied_stage1b",
        "embodied_stage1c",
        "embodied_stage1d",
        "relay",
        "embodied_stage1f",
        "fictive",
        "embodied_stage2b",
        "embodied_stage2c",
        "flexor_diagnostic",
        "embodied_stage2d",
        "embodied_stage2e",
        "embodied_stage2f",
        "embodied_stage2g",
        "embodied_stage2h",
        "embodied_stage3a",
        "embodied_stage3b",
        "embodied_stage4a",
        "embodied_stage5",
        "rhythm_audit",
    }
    assert expected <= modules


def test_steps_run_as_their_own_commands_with_their_flags() -> None:
    by_name = {s.name: s for s in pipeline.STEPS}
    assert by_name["tract_robustness"].command()[-2:] == ["neuromps.fly.tract", "--robustness"]
    assert by_name["stage2f_diagnostic"].command()[-1] == "--diagnose"
    assert by_name["stage5"].command()[-1] == "neuromps.fly.embodied_stage5"
    assert all(s.function == "main" or s.function in pipeline.FLAGS for s in pipeline.STEPS)


def test_selection_by_start_and_only() -> None:
    assert [s.name for s in pipeline.select(pipeline.STEPS, only="stage5")] == ["stage5"]
    tail = pipeline.select(pipeline.STEPS, start="stage4a")
    assert [s.name for s in tail] == [
        "stage4a",
        "stage4a_precision",
        "stage5",
        "stage5b",
        "rhythm_audit",
    ]
    with pytest.raises(SystemExit):
        pipeline.select(pipeline.STEPS, only="nope")
