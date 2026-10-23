"""Tab 4: the chip's drug knobs on a whole fly brain, read as a chip and as behaviour."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app import loaders

FAMILIES = {
    "gaba_block": "GABA block",
    "glucl_block": "GluCl block",
    "gaba_potentiate": "GABA potentiation",
    "ach_block": "ACh block",
    "excitability_block": "Threshold raised",
}
READOUTS = (
    ("mn9_sugar", "MN9 | sugar", "Hz", "feeding motor neuron under sugar drive"),
    ("mn9_sugar_bitter", "MN9 | sugar + bitter", "Hz", "the same, with bitter drive added"),
    ("abn1_jon_ce", "aBN1 | JO-CE", "Hz", "antennal grooming interneuron"),
    ("dn_recruited", "Descending neurons firing", "", "under sugar drive, at 10 Hz or more"),
)


def render(stamps: dict[str, float]) -> None:
    """Dose slider over the registered sweep, then the scorecard and directionality."""
    frame = loaders.load_fly_conditions(stamps["fly"])
    metrics = loaders.load_metrics(stamps["metrics"])
    if frame.empty:
        st.info(loaders.MISSING_FLY)
        return
    size = metrics.get("fly_models", {}).get("flywire783", {}).get("neurons")
    neurons = f"{size:,} neurons, " if size else ""
    st.markdown(
        "The digital twin's drug knobs, applied to a **simulated** whole adult fly brain "
        f"({neurons}FlyWire 783, the Shiu et al. 2024 model). The same brain is read two "
        "ways: by a virtual electrode array scored with the chip's own feature extractor, and "
        "by the behaviour neurons the published model validated. Which chip changes come with "
        "behavioural ones?"
    )
    family, row, vehicle, rows = _controls(frame)
    _readouts(row, vehicle)
    _dose_chart(rows, vehicle, row)
    _ladder(metrics)
    st.divider()
    _scorecard(metrics)
    st.divider()
    _direction(metrics)


def _controls(frame: pd.DataFrame):
    vehicle = frame[frame["family"] == "vehicle"].iloc[0]
    family = (
        st.segmented_control(
            "Mechanism",
            list(FAMILIES),
            format_func=FAMILIES.get,
            default="gaba_block",
            key="fly_family",
        )
        or "gaba_block"
    )
    rows = frame[frame["family"] == family].sort_values("dose")
    levels = ["untreated", *[f"{v:g}" for v in rows["level"]]]
    pick = st.select_slider(
        f"{FAMILIES[family]}: knob level (1 = untreated)",
        options=levels,
        value=levels[-1],
        key=f"fly_level_{family}",
    )
    row = (
        vehicle
        if pick == "untreated"
        else rows[rows["level"].map(lambda v: f"{v:g}") == pick].iloc[0]
    )
    return family, row, vehicle, rows


def _readouts(row: pd.Series, vehicle: pd.Series) -> None:
    with st.container(border=True):
        st.caption("Chip readout (virtual array, change from untreated, log2)")
        cols = st.columns(3)
        for col, (key, label) in zip(
            cols,
            (
                ("delta.mfr_hz", "Firing rate"),
                ("delta.synchrony_index", "Synchrony"),
                ("delta.isi_cv", "ISI CV"),
            ),
            strict=True,
        ):
            value = row.get(key)
            col.metric(label, "n/a" if pd.isna(value) else f"{value:+.2f}")
    with st.container(border=True):
        st.caption("Behaviour readouts (model neurons; rates averaged over 8 one-second trials)")
        cols = st.columns(len(READOUTS))
        for col, (key, label, unit, help_text) in zip(cols, READOUTS, strict=True):
            col.metric(
                label,
                f"{row[key]:.0f} {unit}".strip(),
                delta=f"{row[key] - vehicle[key]:+.0f}",
                delta_color="off",
                help=help_text,
            )


def _dose_chart(rows: pd.DataFrame, vehicle: pd.Series, row: pd.Series) -> None:
    data = pd.concat([vehicle.to_frame().T, rows])
    long = pd.DataFrame(
        {
            "knob change from untreated": data["dose"].astype(float).tolist() * 2,
            "value": data["abs_delta_mfr"].astype(float).tolist()
            + data["disruption"].astype(float).tolist(),
            "readout": ["chip: |log2 change in firing rate|"] * len(data)
            + ["behaviour: disruption index"] * len(data),
        }
    )
    chart = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x="knob change from untreated:Q",
            y="value:Q",
            color=alt.Color(
                "readout:N",
                scale=alt.Scale(range=["#0072B2", "#D55E00"]),
                legend=alt.Legend(orient="bottom", labelLimit=400, title=None),
            ),
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"x": [float(row["dose"])]}))
        .mark_rule(strokeDash=[4, 4], color="grey")
        .encode(x="x:Q")
    )
    st.altair_chart((chart + marker).properties(height=260), width="stretch")
    st.caption(
        "Both on one scale (log2 units). Disruption is the mean absolute log2 change of the four "
        "behaviour readouts. SIMULATED."
    )


def _scorecard(metrics: dict) -> None:
    st.subheader("Registered predictions")
    rows = []
    for section, label in (
        ("fly_chip_to_animal", "FlyWire"),
        ("fly_malecns_replication", "MaleCNS"),
        ("fly_confirm_network_bursts", "Confirmation"),
    ):
        for name, entry in metrics.get(section, {}).get("predictions", {}).items():
            rows.append(
                {
                    "brain": label,
                    "prediction": name,
                    "test": entry["test"],
                    "outcome": entry["outcome"],
                    "value": str(entry["value"]),
                }
            )
    if not rows:
        st.info("The scorecard appears once the full fly stage has run.")
        return
    reg = metrics.get("fly_chip_to_animal", {}).get("registration", {})
    st.caption(
        f"Registered in commit `{str(reg.get('registered_in_commit'))[:12]}` "
        f"({reg.get('committed_at')}), before the sweep was run; "
        f"`{reg.get('file')}`, SHA-256 `{str(reg.get('sha256'))[:16]}…`. "
        "Each prediction is shown whichever way it fell."
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _direction(metrics: dict) -> None:
    st.subheader("Can activity recover wiring direction between regions?")
    figure = loaders.FIGURES / "fly_direction_SIMULATED.png"
    if figure.exists():
        st.image(str(figure), width="stretch")
    d = metrics.get("fly_chip_to_animal", {}).get("direction", {})
    if d:
        st.caption(
            f"{d.get('n_pairs')} region pairs joined by 2,000 or more synapses; the connectome "
            "gives the true direction. The chip's cross-correlation measure on spontaneous "
            "activity against pulsed stimulation read 5-30 ms after each pulse. SIMULATED."
        )


def _ladder(metrics: dict) -> None:
    figure = loaders.FIGURES / "fly_sentinel_ladder_SIMULATED.png"
    if not figure.exists():
        return
    st.subheader("Which readout raises the alarm first?")
    st.image(str(figure), width="stretch")
    confirm = metrics.get("fly_confirm_network_bursts", {})
    alarms = metrics.get("fly_checks", {}).get("false_alarm_rate_untreated", {})
    pooled = confirm.get("pooled_out_of_sample", {})
    if pooled and alarms:
        st.caption(
            "First dose at which behaviour is impaired (disruption >= 0.5) and at which each chip "
            "feature changes by 0.5 log2. Network-burst rate looks early "
            f"({pooled['network_bursts_at_or_before_behaviour']} of {pooled['tests']} "
            "out-of-sample tests), but it is noise: two groups of untreated recordings already "
            f"differ by 0.5 log2 in {alarms['by_feature']['network_burst_rate_per_min']:.0%} of "
            "random splits. Firing rate never does, and it is late. SIMULATED."
        )
    _precision(confirm.get("precision", {}))


def _precision(precision: dict) -> None:
    needed = precision.get("abs_log2_mfr_change_at_behavioural_onset", {})
    if not needed:
        return
    st.subheader("How precise must a chip be to flag each mechanism in time?")
    table = pd.DataFrame(needed).rename(index=FAMILIES)
    table.columns = [c.replace("_", " ") for c in table.columns]
    st.dataframe(table, width="stretch")
    noise = precision.get("real_chip_noise", {})
    floors = ", ".join(f"{k.replace('_', ' ')}: {v['q95_abs_log2']}" for k, v in noise.items())
    st.caption(
        "The |log2| change in the array's firing rate at the first dose that impairs behaviour: a "
        "chip must resolve a change this small to flag the drug no later than the behaviour "
        f"readouts fail. Untreated variation measured on real chips (95th percentile): {floors}. "
        "SIMULATED."
    )
