"""Tab 3: the digital twin, its posterior, and what it does on real recordings."""

from __future__ import annotations

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
    point, run = _slider_panel(grid, meta)
    _simulation_panel(point, run, meta, stamps)
    st.divider()
    _real_recordings_panel(stamps)
    st.divider()
    _directionality_panel(stamps)


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
        _posterior_panel(run, meta)


def _posterior_panel(run: dict, meta: dict) -> None:
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
    st.caption(
        "HD14: only `threshold_scale` is well identified (contraction 0.8). The connectivity "
        "parameters stay wide, which is why the twin supports a contrast, not a measurement."
    )


def _live_run(point: pd.Series) -> None:
    """One real Brian2 run at the current values, so the demo is not canned."""
    if not st.button("Run one live simulation (Brian2)", type="primary"):
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
