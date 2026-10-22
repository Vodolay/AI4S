"""Tab 2: the digital twin, its posterior, and what it does on real recordings."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from app import loaders
from neuromps.demo import plots

KNOB_EXPECTATION = {
    "gabaa_scale": "a GABA-A antagonist should push this below 1",
    "ampa_scale": "a glutamate antagonist should push this below 1",
    "threshold_scale": "a sodium-channel blocker should push this above 1",
}


def render(stamps: dict[str, float]) -> None:
    """Sliders over the precomputed grid, then the real-data and HD16 results."""
    meta = loaders.load_demo_meta(stamps["demo"])
    grid = loaders.load_grid(stamps["demo"])
    if grid.empty or not meta:
        st.warning(loaders.MISSING_DEMO)
        return
    _architecture_diagram()
    point, run = _slider_panel(grid, meta)
    _simulation_panel(point, run, meta, stamps)
    st.divider()
    _inversion_diagram()
    _real_recordings_panel(stamps)
    st.divider()
    _directionality_panel(stamps)


def _architecture_diagram() -> None:
    """What the twin is, before the sliders that move it."""
    figure = loaders.diagram("twin_architecture_simulated.png")
    if figure is None:
        return
    with st.container(border=True):
        st.image(str(figure), width="stretch")
        st.caption(
            "The sliders below move the two knobs marked inferred, plus the spike threshold. "
            "Every other value shown is the calibrated draw, read from `outputs/metrics.json`."
        )


def _inversion_diagram() -> None:
    """What inference does, before the posteriors it produces."""
    figure = loaders.diagram("inference_network_simulated.png")
    if figure is None:
        return
    with st.container(border=True):
        st.image(str(figure), width="stretch")
        st.caption(
            "The same twin, run backwards. Two of the five parameters fail the rank-uniformity "
            "check, so their intervals below are shown but are not claimed to be calibrated."
        )


def _slider_panel(grid: pd.DataFrame, meta: dict) -> tuple[pd.Series, dict]:
    """Three sliders that snap to the precomputed grid, so nothing is simulated here."""
    st.subheader("Simulated chip")
    axes = meta["grid"]
    columns = st.columns(3)
    ratios = axes["forward_backward_ratio"]
    gabaa_values = axes["gabaa_scale"]
    ratio = columns[0].select_slider("forward / backward ratio", options=ratios, value=ratios[0])
    gabaa = columns[1].select_slider(
        "GABA-A scale", options=gabaa_values, value=gabaa_values[len(gabaa_values) // 2]
    )
    ampa_values = axes["ampa_scale"]
    # Mid-range, not the prior's maximum: the top of the AMPA prior drives the
    # network into one saturated synchronous burst, which is a real part of the
    # grid but a misleading first impression.
    ampa = columns[2].select_slider(
        "AMPA scale", options=ampa_values, value=ampa_values[len(ampa_values) // 2]
    )
    match = grid[
        np.isclose(grid["forward_backward_ratio"], ratio)
        & np.isclose(grid["gabaa_scale"], gabaa)
        & np.isclose(grid["ampa_scale"], ampa)
    ]
    if match.empty:
        st.error("No precomputed run at these values.")
        st.stop()
    point = match.iloc[0]
    return point, loaders.load_run(point["run_file"])


def _simulation_panel(point: pd.Series, run: dict, meta: dict, stamps: dict[str, float]) -> None:
    """Raster, features against the real baseline, and the posterior for this run."""
    left, right = st.columns([1.15, 1])
    with left:
        st.pyplot(
            plots.raster(
                {k: run[k] for k in run if k.endswith(("_times", "_neurons"))},
                2.0,
                "SIMULATED spike raster, first 2 s of the precomputed run",
            )
        )
        st.caption(
            f"Simulated {meta['simulated_duration_s']:.0f} s at "
            f"{meta['n_per_compartment']} neurons per compartment; "
            f"inter-compartment in-degree and threshold are held at "
            f"{meta['fixed_parameters']['inter_in_degree']} and "
            f"{meta['fixed_parameters']['threshold_scale']}."
        )
        _live_run(point)
    with right:
        real = loaders.real_baseline_features(stamps["asset"])
        simulated = run["features"][0]
        shown = {k: v for k, v in simulated.items() if v is not None and k in real}
        if shown:
            st.pyplot(
                plots.features_against_real(
                    shown, real, "SIMULATED features against the REAL baseline distribution"
                )
            )
        _posterior_panel(run, meta, stamps)


def _identifiability_note(metrics: dict) -> str:
    """What the posterior may be read as, written from the metrics not from memory.

    This caption used to say "only threshold_scale is well identified
    (contraction 0.8)". The density correction changed both halves of that: the
    contraction is no longer 0.8 and three parameters now pass, not one. A
    sentence the checker cannot see is a sentence that outlives its result.
    """
    inference = metrics.get("twin_hd14_inference", {})
    contraction = inference.get("contraction", {})
    ks_p = inference.get("rank_uniformity_ks_p", {})
    if not contraction or not ks_p:
        return "Posterior diagnostics are not in this run's metrics."
    passing = sorted(name for name, value in ks_p.items() if value["ks_p"] > 0.05)
    failing = sorted(name for name, value in ks_p.items() if value["ks_p"] <= 0.05)
    best = max(contraction, key=lambda name: contraction[name]["contraction"])
    return (
        f"{len(passing)} of {len(ks_p)} parameters pass the rank-uniformity check; "
        f"`{'`, `'.join(failing)}` do not, so their intervals are shown but are not "
        f"calibrated. The tightest posterior is `{best}` at contraction "
        f"{contraction[best]['contraction']}. The twin supports a contrast, not a measurement."
    )


def _posterior_panel(run: dict, meta: dict, stamps: dict[str, float]) -> None:
    """Posterior marginals for this run, with the generating values marked."""
    names = meta["parameter_names"]
    st.pyplot(
        plots.posterior_marginals(
            run["posterior_samples"],
            names,
            run["theta"],
            "SIMULATED: posterior given this run (dashed = value that generated it)",
        )
    )
    st.caption(_identifiability_note(loaders.load_metrics(stamps["metrics"])))


# Opening a page must never cost a visitor four seconds of CPU, so `?live=1` is
# inert unless the server was started with this set. The capture script sets it
# on the server it starts itself; a deployed app does not, and the query
# parameter then does nothing at all.
LIVE_ON_LOAD_ENV = "NEUROMPS_ALLOW_LIVE_ON_LOAD"


def _live_requested() -> bool:
    """Whether `?live=1` asks for the simulation to run without a click.

    A screenshot tool cannot press a button, and the demo video wants a frame of
    this panel. The run is cached on its parameter tuple, so a capture script can
    load the page twice: the first load pays for Brian2, the second renders from
    the cache fast enough to be photographed. Moving a slider still runs it fresh.

    This is gated on the environment rather than on the URL alone, because a URL
    is exactly how a visitor would trip it by accident.
    """
    if os.environ.get(LIVE_ON_LOAD_ENV, "") != "1":
        return False
    value = st.query_params.get("live")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return str(value).strip().lower() in {"1", "true", "yes"}


def _live_run(point: pd.Series) -> None:
    """One real Brian2 run at the current values, so the demo is not canned."""
    if not loaders.live_simulation_available():
        st.info(
            "The live Brian2 run needs the full local install (see Setup in the README). "
            "This copy shows the precomputed simulations above, which are the same model."
        )
        return
    clicked = st.button("Run one live simulation (Brian2)", type="primary")
    if not clicked and not _live_requested():
        return
    with st.spinner("Running Brian2..."):
        result = loaders.live_simulation(
            float(point["forward_backward_ratio"]),
            float(point["gabaa_scale"]),
            float(point["ampa_scale"]),
        )
    if "error" in result:
        st.error(f"The live run failed: {result['error']}")
        return
    st.success(f"Simulated {result['duration_s']:.0f} s in {result['elapsed_s']} s of wall clock.")
    st.pyplot(
        plots.raster(result["raster"], 2.0, "SIMULATED live run, first 2 s (not precomputed)")
    )
    live = {k: round(v, 3) for k, v in result["features"][0].items() if v is not None}
    st.caption(f"Compartment 0 features from this live run: {live}")


def _real_recordings_panel(stamps: dict[str, float]) -> None:
    """The HD15 result: the posterior applied to real two-chamber recordings."""
    st.subheader("Applied to real recordings (HD15)")
    samples, records = loaders.load_real_posteriors(stamps["demo"])
    if records.empty:
        st.warning(loaders.MISSING_DEMO)
        return
    meta = loaders.load_demo_meta(stamps["demo"])
    names = meta["parameter_names"]
    labels = _recording_labels(records)
    columns = st.columns([1, 1])
    choice = columns[0].selectbox("Recording", labels)
    knob = columns[1].selectbox("Drug knob", [n for n in names if n in KNOB_EXPECTATION])
    index = labels.index(choice)
    st.pyplot(
        plots.drug_knob_posterior(
            samples[index][:, names.index(knob)],
            knob,
            KNOB_EXPECTATION[knob],
            f"REAL recording, posterior trained on SIMULATIONS: {choice}",
        )
    )
    _hd15_summary(stamps)


def _recording_labels(records: pd.DataFrame) -> list[str]:
    """One readable label per real recording: device and dose."""
    labels = []
    for row in records.itertuples():
        dose = 0.0 if pd.isna(row.dose_value) else float(row.dose_value)
        unit = row.dose_unit if isinstance(row.dose_unit, str) else "uM"
        labels.append(f"{row.device_id} - " + ("baseline" if dose == 0 else f"{dose:g} {unit}"))
    return labels


def _hd15_summary(stamps: dict[str, float]) -> None:
    """The measured result, quoted from metrics rather than recomputed."""
    metrics = loaders.load_metrics(stamps["metrics"]).get("twin_hd15_real_inference", {})
    trend = metrics.get("gabaa_scale_dose_trend", {})
    baseline = metrics.get("gabaa_scale_at_baseline", {})
    if not trend:
        return
    left, right = st.columns(2)
    left.metric(
        "GABA-A scale vs bicuculline dose",
        f"rho = {trend.get('spearman_rho')}",
        help="Spearman correlation, clustered over devices.",
    )
    left.caption(f"95% CI {trend.get('spearman_rho_ci95')} over {trend.get('n_devices')} devices.")
    right.metric("Untreated recordings infer", f"{baseline.get('median')}")
    right.caption(f"95% CI {baseline.get('median_ci95')}, where a neutral value would be 1.0.")
    st.error(
        "**The honest reading.** There is no dose trend, and untreated chips already infer a "
        "strongly reduced GABA-A scale, so the knob is absorbing model mismatch rather than "
        "tracking the compound. Only two-chamber recordings can be shown here; the gabazine, "
        "NBQX and TTX arms are single-chamber and cannot be fed to this posterior at all."
    )


def _directionality_panel(stamps: dict[str, float]) -> None:
    """HD16: what it takes to tell symmetric from asymmetric wiring."""
    st.subheader("Directionality: electrodes and recording length (HD16)")
    metrics = loaders.load_metrics(stamps["metrics"]).get("twin_hd16_directionality", {})
    cheapest = (metrics.get("cheapest_configuration") or {}).get("rate_matched_residualized")
    figure = loaders.TWIN_FIGURES / "hd16_directionality_simulated.png"
    left, right = st.columns([1.4, 1])
    if figure.exists():
        left.image(str(figure), caption="SIMULATED: AUC by recording length and electrode count.")
    if cheapest:
        right.metric("Cheapest configuration", f"{cheapest['duration_s']:.0f} s")
        right.caption(
            f"{cheapest['electrodes_per_compartment']} electrodes per compartment reach "
            f"AUC {cheapest['auc']} (95% CI {cheapest['auc_ci95']}) once features are "
            "residualised against firing rate."
        )
    summary = metrics.get("summary_by_feature_set", {}).get("free_rate", {})
    if summary:
        right.markdown(
            f"Without that control, rate features alone reach AUC "
            f"{summary['rate_only']['min']}-{summary['rate_only']['max']}: the naive version of "
            "this experiment measures firing rate, not direction."
        )
