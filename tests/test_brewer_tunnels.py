"""Tunnel geometry, which the direction of every axonal measurement rests on."""

import io
import zipfile

import numpy as np
import pytest
import scipy.io as sio

from neuromps.etl.brewer_tunnels import (
    LOOP,
    Tunnel,
    build_tunnels,
    coordinate_disagreements,
    corner_subregions,
    electrode_xy,
    read_tunnels,
    strip_of,
)
from neuromps.models.axonal import ARCHIVE, crossing_counts, tunnel_flow

# Orientation A: the EC-DG tunnels lie on the left of the array.
STRIPS_A = {"left": "EC-DG", "right": "CA3-CA1", "top": "CA1-EC", "bottom": "DG-CA3"}
# Orientation B: the same device mounted a quarter turn round.
STRIPS_B = {"left": "CA1-EC", "right": "DG-CA3", "top": "EC-DG", "bottom": "CA3-CA1"}


def test_electrode_xy_follows_the_mcs_naming_convention() -> None:
    assert electrode_xy("A1") == (1, 12)
    assert electrode_xy("A12") == (1, 1)
    # The letter I is skipped so that no channel name reads as a number.
    assert electrode_xy("H6") == (8, 7)
    assert electrode_xy("J6") == (9, 7)
    assert electrode_xy("M12") == (12, 1)


def test_electrode_xy_rejects_names_it_cannot_place() -> None:
    assert electrode_xy("I6") is None
    assert electrode_xy("A13") is None
    assert electrode_xy("A0") is None
    assert electrode_xy("") is None
    assert electrode_xy("ref") is None


def test_strip_of_separates_the_four_tunnel_bands_from_the_centre() -> None:
    assert strip_of(electrode_xy("A6")) == "left"
    assert strip_of(electrode_xy("M7")) == "right"
    assert strip_of(electrode_xy("F1")) == "top"
    assert strip_of(electrode_xy("G12")) == "bottom"
    # The four electrodes in the middle of the array are in no tunnel.
    assert strip_of(electrode_xy("F6")) is None
    assert strip_of(electrode_xy("G7")) is None
    # Nor is a chamber electrode.
    assert strip_of(electrode_xy("A4")) is None


@pytest.mark.parametrize(
    ("strips", "expected"),
    [
        (
            STRIPS_A,
            {"top_left": "EC", "top_right": "CA1", "bottom_left": "DG", "bottom_right": "CA3"},
        ),
        (
            STRIPS_B,
            {"top_left": "EC", "top_right": "DG", "bottom_left": "CA1", "bottom_right": "CA3"},
        ),
    ],
)
def test_corner_subregions_recovers_the_mounting_orientation(strips, expected) -> None:
    assert corner_subregions(strips) == expected


def test_corner_subregions_leaves_a_corner_out_when_its_strips_are_missing() -> None:
    assert corner_subregions({"left": "EC-DG"}) == {}


def _layout(strips: dict[str, str]) -> tuple[list[str], list[str]]:
    """Channel names and labels for a full 44-electrode tunnel file."""
    names, labels = [], []
    for column in "ABCDE":  # left strip, five tunnels of two electrodes
        names += [f"{column}6", f"{column}7"]
        labels += [strips["left"]] * 2
    for column in "HJKLM":  # right strip
        names += [f"{column}6", f"{column}7"]
        labels += [strips["right"]] * 2
    for number in range(1, 6):  # top strip
        names += [f"F{number}", f"G{number}"]
        labels += [strips["top"]] * 2
    for number in range(8, 13):  # bottom strip
        names += [f"F{number}", f"G{number}"]
        labels += [strips["bottom"]] * 2
    names += ["F6", "F7", "G6", "G7"]
    labels += ["NONE"] * 4
    return names, labels


@pytest.mark.parametrize("strips", [STRIPS_A, STRIPS_B])
def test_build_tunnels_finds_twenty_tunnels_in_either_orientation(strips) -> None:
    tunnels = build_tunnels(*_layout(strips))
    assert len(tunnels) == 20
    assert {t.pair for t in tunnels} == {"CA1->CA3", "CA1->EC", "CA3->DG", "DG->EC"}
    # Every tunnel of a pair must be oriented the same way, or their crossings
    # would cancel when summed.
    for pair in {t.pair for t in tunnels}:
        directions = {(t.from_chamber, t.to_chamber) for t in tunnels if t.pair == pair}
        assert len(directions) == 1


def test_build_tunnels_orients_every_tunnel_along_the_depositors_loop() -> None:
    for tunnel in build_tunnels(*_layout(STRIPS_A)):
        step = (LOOP.index(tunnel.to_chamber) - LOOP.index(tunnel.from_chamber)) % len(LOOP)
        assert step == 1


def test_the_two_orientations_describe_the_same_device() -> None:
    """A tunnel's chambers must not depend on how the culture was mounted."""
    a = {(t.from_chamber, t.to_chamber) for t in build_tunnels(*_layout(STRIPS_A))}
    b = {(t.from_chamber, t.to_chamber) for t in build_tunnels(*_layout(STRIPS_B))}
    assert a == b


def test_build_tunnels_skips_a_strip_position_with_only_one_electrode() -> None:
    names, labels = _layout(STRIPS_A)
    keep = [i for i, n in enumerate(names) if n != "A7"]
    tunnels = build_tunnels([names[i] for i in keep], [labels[i] for i in keep])
    assert len(tunnels) == 19


def test_pair_naming_matches_the_chamber_analysis_and_flags_reversed_tunnels() -> None:
    forward = Tunnel("CA1", "EC", "a", "b")
    assert forward.pair == "CA1->EC"
    assert forward.forward_is_pair_order is True
    reversed_ = Tunnel("EC", "DG", "a", "b")
    assert reversed_.pair == "DG->EC"
    assert reversed_.forward_is_pair_order is False


def test_crossing_counts_separates_the_two_directions() -> None:
    source = np.array([1.0, 2.0])
    # 1 ms after each source spike, and 1 ms before the second only.
    target = np.array([1.001, 1.999, 2.001])
    after, before = crossing_counts(source, target, 0.002)
    assert (after, before) == (2, 1)


def test_crossing_counts_excludes_the_window_edge_consistently() -> None:
    source = np.array([0.0])
    assert crossing_counts(source, np.array([0.002]), 0.002) == (1, 0)
    assert crossing_counts(source, np.array([0.0021]), 0.002) == (0, 0)
    assert crossing_counts(source, np.array([]), 0.002) == (0, 0)
    assert crossing_counts(np.array([]), np.array([1.0]), 0.002) == (0, 0)


def test_a_simultaneous_spike_counts_for_neither_direction() -> None:
    assert crossing_counts(np.array([1.0]), np.array([1.0]), 0.002) == (0, 0)


def test_tunnel_flow_reports_the_index_in_the_pairs_own_orientation() -> None:
    """A tunnel stored EC->DG must report as DG->EC with the sign flipped."""
    names, labels = _layout(STRIPS_A)
    source = np.arange(1.0, 100.0, 0.5)
    reading = {
        "culture": "1ECDGCA3CA1 x",
        "names": names,
        "subregions": labels,
        "duration_s": 300.0,
        # Every EC-side electrode fires, every DG-side one fires 1 ms later, so
        # the physical traffic runs EC -> DG in all five left-strip tunnels.
        "spikes": {"A6": source, "A7": source + 0.001},
    }
    rows = [r for r in tunnel_flow(reading, 0.002) if r["pair"] == "DG->EC"]
    assert len(rows) == 1
    assert rows[0]["device_id"] == "1ECDGCA3CA1"
    # EC is second in "DG->EC", so flow toward EC is negative.
    assert rows[0]["flow_index"] < 0
    assert rows[0]["forward"] < rows[0]["backward"]


@pytest.mark.skipif(not ARCHIVE.exists(), reason="Brewer archive not downloaded")
def test_derived_orientation_matches_the_chamber_files_electrode_positions() -> None:
    """The geometry check the whole analysis rests on, against the real files.

    `corner_subregions` reads the mounting orientation off the tunnel labels.
    The chamber files place their own electrodes independently, so the two must
    agree - in every culture whose channel names scipy can read.
    """
    with zipfile.ZipFile(ARCHIVE) as archive:
        from neuromps.etl.brewer import cultures

        checked = 0
        for culture in cultures(archive):
            reading = read_tunnels(archive, culture)
            if reading["names_recovered_from_coordinates"]:
                continue
            strips = {}
            for name, label in zip(reading["names"], reading["subregions"], strict=True):
                position = electrode_xy(name)
                strip = strip_of(position) if position else None
                if strip and label != "NONE":
                    strips[strip] = label
            corners = corner_subregions(strips)
            for region in LOOP:
                member = f"spike_trains_data/{culture}/spikes/mea_{region}.mat"
                mea = sio.loadmat(io.BytesIO(archive.read(member)))["mea"][0, 0]
                positions = [electrode_xy(str(x[0])) for x in mea["channel_names"].ravel()]
                positions = [p for p in positions if p is not None]
                if not positions:
                    continue
                mean_x = float(np.mean([p[0] for p in positions]))
                mean_y = float(np.mean([p[1] for p in positions]))
                corner = ("top" if mean_y > 6.5 else "bottom") + (
                    "_left" if mean_x < 6.5 else "_right"
                )
                assert corners[corner] == region, (culture, region, corners)
                checked += 1
        assert checked >= 32


@pytest.mark.skipif(not ARCHIVE.exists(), reason="Brewer archive not downloaded")
def test_stored_coordinates_really_do_disagree_with_the_names() -> None:
    """The defect that makes this module read positions from channel names."""
    with zipfile.ZipFile(ARCHIVE) as archive:
        from neuromps.etl.brewer import cultures

        disagreeing = 0
        for culture in cultures(archive):
            for entry in coordinate_disagreements(archive, culture).values():
                if entry and entry["n_disagree"]:
                    disagreeing += 1
        assert disagreeing > 0, "if this ever passes at 0 the workaround can go"


def _flow_and_chamber(signs: list[int], informative: bool, cultures: int = 4):
    """Paired tables where the chamber readouts either carry the sign or do not."""
    import pandas as pd

    rows_flow, rows_chamber = [], []
    per_culture = max(1, len(signs) // cultures)
    for i, sign in enumerate(signs):
        # Consecutive pairs share a culture, so each culture carries both signs -
        # as 7 of the 9 real cultures do. A culture of one sign contributes
        # nothing to a within-culture permutation null.
        device = f"D{min(i // per_culture, cultures - 1)}"
        pair = f"P{i}"
        rows_flow.append(
            {
                "device_id": device,
                "pair": pair,
                "window_ms": 1.0,
                "flow_index": 0.5 * sign,
                "n_crossings": 500,
            }
        )
        # When informative, asymmetry carries the sign and the rest are noise.
        rows_chamber.append(
            {
                "device_id": device,
                "pair": pair,
                "asymmetry": (0.4 * sign if informative else 0.1),
                "peak_correlation": 0.2,
                "lag_ms": 1.0 + 0.01 * i,
                "log2_rate_ratio": 0.05 * i,
            }
        )
    return pd.DataFrame(rows_flow), pd.DataFrame(rows_chamber)


def test_the_chamber_readouts_recover_a_direction_they_do_carry() -> None:
    from neuromps.models.axonal import predicts_axonal_direction

    signs = [1, -1] * 8
    flow, chamber = _flow_and_chamber(signs, informative=True)
    out = predicts_axonal_direction(flow, chamber)
    assert out["n_pairs"] == 16
    assert out["balanced_accuracy"] > 0.9
    assert out["beats_chance"] is True


def test_the_chamber_readouts_do_not_recover_a_direction_they_do_not_carry() -> None:
    from neuromps.models.axonal import predicts_axonal_direction

    signs = [1, -1] * 8
    flow, chamber = _flow_and_chamber(signs, informative=False)
    out = predicts_axonal_direction(flow, chamber)
    assert out["beats_chance"] is False
    assert out["permutation_p"] > 0.05
    # The power statement has to be present whichever way the answer goes. It
    # can equal chance exactly when the readouts are constant, as here, because
    # then every permutation scores the same; it must never sit below it.
    assert out["detectable_balanced_accuracy"] >= out["chance"]


def test_a_pair_with_no_measurable_flow_is_not_given_a_sign() -> None:
    from neuromps.models.axonal import predicts_axonal_direction

    flow, chamber = _flow_and_chamber([1, -1] * 8, informative=True)
    flow.loc[0, "flow_index"] = 0.0
    assert predicts_axonal_direction(flow, chamber)["n_pairs"] == 15


def test_too_few_pairs_is_declined_rather_than_fitted() -> None:
    from neuromps.models.axonal import predicts_axonal_direction

    flow, chamber = _flow_and_chamber([1, -1, 1], informative=True, cultures=3)
    assert "verdict" in predicts_axonal_direction(flow, chamber)
