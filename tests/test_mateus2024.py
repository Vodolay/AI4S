

def test_runs_can_be_traced_along_rows_as_well_as_columns() -> None:
    """The deposit never records how a device was seated, so the axis is an assumption."""
    from neuromps.etl.mateus2024 import channel_runs

    column_labels = [f"A{row}" for row in range(1, 6)]
    row_labels = [f"{col}1" for col in ("A", "B", "C", "D", "E")]
    assert len(channel_runs(column_labels, "column")) == 1
    assert channel_runs(column_labels, "row") == []
    assert len(channel_runs(row_labels, "row")) == 1
    assert channel_runs(row_labels, "column") == []


def test_an_unknown_orientation_is_refused() -> None:
    import pytest

    from neuromps.etl.mateus2024 import channel_runs

    with pytest.raises(ValueError, match="orientation"):
        channel_runs(["A1"], "diagonal")


def test_one_missing_electrode_recovers_events_a_strict_chain_loses() -> None:
    """A dead electrode mid-channel breaks a strict chain and loses every crossing."""
    import numpy as np

    from neuromps.etl.mateus2024 import propagation

    step = 0.0004
    starts = np.arange(0.0, 1.0, 0.05)
    trains = [starts + i * step for i in range(5)]
    trains[2] = np.array([])  # the middle electrode records nothing
    assert propagation(trains)[0] == 0
    forward, _ = propagation(trains, min_electrodes=4)
    assert forward == len(starts)
