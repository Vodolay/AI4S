"""Tab 1: mechanism prediction with a conformal set, and the transfer results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import loaders
from neuromps.demo import plots

EXPECTED_CI_EXCLUDING_ZERO = "network_burst_rate_per_min"


def render(stamps: dict[str, float]) -> None:
    """Left: predict a mechanism. Right: what transfers between species."""
    metrics = loaders.load_metrics(stamps["metrics"])
    fitted = loaders.load_mechanism_model()
    if not fitted:
        st.warning(loaders.MISSING_DEMO)
        return
    _model_diagram()
    left, right = st.columns([1, 1])
    with left:
        _prediction_panel(fitted, stamps)
    with right:
        _transfer_panel(metrics)


def _model_diagram() -> None:
    """What the classifier is, before the panel that runs it."""
    figure = loaders.diagram("mechanism_classifier_real.png")
    if figure is None:
        return
    with st.container(border=True):
        st.image(str(figure), width="stretch")
        st.caption(
            "One linear model over nine paired log-ratios. Cross-validation groups by culture, "
            "which is why the accuracy below is lower than most published numbers and why it is "
            "the honest one. Drawn by `neuromps.report.diagrams` from `outputs/metrics.json`."
        )


def _prediction_panel(fitted: dict, stamps: dict[str, float]) -> None:
    """Predict from a real pair, or from a delta vector the user dials in."""
    st.subheader("Mechanism prediction")
    pairs = loaders.load_pairs(stamps["asset"])
    mode = st.radio("Input", ["A real record", "Dial a feature vector"], horizontal=True)
    features = list(fitted["feature_ranges"])
    if mode == "A real record" and not pairs.empty:
        vector, truth = _vector_from_record(pairs, features)
    else:
        vector, truth = _vector_from_sliders(fitted, features), None

    probabilities = fitted["model"].predict_proba(np.asarray(vector).reshape(1, -1))[0]
    labels = list(fitted["model"].classes_)
    predicted = labels[int(np.argmax(probabilities))]
    chosen = [
        label for label, p in zip(labels, probabilities, strict=True) if p >= fitted["threshold"]
    ]

    st.metric("Predicted mechanism", predicted, help="Highest-probability class.")
    if truth is not None:
        st.caption(f"Recorded label for this pair: **{truth}**")
    table = pd.DataFrame({"class": labels, "probability": np.round(probabilities, 3)})
    st.dataframe(
        table.sort_values("probability", ascending=False), hide_index=True, width="stretch"
    )
    _conformal_note(chosen, fitted)


def _vector_from_record(pairs: pd.DataFrame, features: list[str]) -> tuple[list[float], str]:
    """A delta vector taken from one real treated/baseline pair."""
    compounds = sorted(pairs["compound"].dropna().unique())
    compound = st.selectbox("Compound", compounds)
    subset = pairs[pairs["compound"] == compound]
    label = st.selectbox("Pair", subset["pair_id"].tolist())
    row = subset[subset["pair_id"] == label].iloc[0]
    st.caption(
        f"{row['source']} - {row['species']} - dose {row['dose_value']} {row['dose_unit'] or ''}"
    )
    return [float(row[name]) for name in features], str(row["label"])


def _vector_from_sliders(fitted: dict, features: list[str]) -> list[float]:
    """A delta vector dialled in, bounded by the range the real pairs span."""
    st.caption("Each value is log2(exposure / baseline); 0 means unchanged.")
    vector = []
    for name in features:
        low, high = _slider_bounds(fitted["feature_ranges"][name])
        vector.append(
            st.slider(
                name.removeprefix("delta."),
                low,
                high,
                0.0,
                step=0.05,
                key=f"slider_{name}",
            )
        )
    return vector


def _slider_bounds(bounds: list[float]) -> tuple[float, float]:
    """Usable slider ends, whatever the stored range looks like.

    A feature the sources do not all report can arrive with a missing or
    collapsed range; a slider whose ends are equal raises rather than rendering.
    """
    low, high = (float(b) for b in bounds)
    if not (np.isfinite(low) and np.isfinite(high)) or high - low < 0.1:
        return -1.0, 1.0
    return round(low, 2), round(high, 2)


def _conformal_note(chosen: list[str], fitted: dict) -> None:
    """The conformal set, and what its size means."""
    alpha = fitted["alpha"]
    st.markdown(f"**Conformal prediction set (α = {alpha})**")
    if not chosen:
        st.error(
            "Empty set: no class reached the calibrated threshold. The recording looks unlike "
            "anything in the calibration set."
        )
        return
    st.info(" · ".join(f"`{c}`" for c in chosen))
    if len(chosen) == 1:
        st.caption(
            "A single-class set: at this confidence level the evidence is consistent with only "
            "this mechanism."
        )
    else:
        st.caption(
            f"A set of {len(chosen)} classes means the data cannot separate them at "
            f"{int((1 - alpha) * 100)}% confidence - the set, not the top class, is the honest "
            "answer. With 73 calibration pairs, sets this wide are expected."
        )


def _transfer_panel(metrics: dict) -> None:
    """Rat to human transfer, the species shift, and calibration - all from metrics."""
    st.subheader("Transfer, shift and calibration")
    transfer = metrics.get("species_transfer", {}).get("transfer", {})
    if transfer:
        none = transfer.get("none", {})
        rows = [
            {
                "label": "human, within",
                "value": none["human_within"]["balanced_accuracy"],
                "ci95": [
                    none["human_within"]["balanced_accuracy"],
                    none["human_within"]["balanced_accuracy"],
                ],
            },
            *[
                {
                    "label": f"rat to human, {name}",
                    "value": entry["rat_to_human"]["balanced_accuracy"],
                    "ci95": entry["rat_to_human"]["balanced_accuracy_ci95"],
                }
                for name, entry in sorted(transfer.items())
            ],
        ]
        st.pyplot(
            plots.transfer_bars(
                rows,
                none.get("chance_balanced_accuracy", 0.2),
                "REAL data: rat-trained model applied to human recordings",
            )
        )
        st.caption(
            "Both domain-adaptation methods made transfer worse, not better: with 37 training "
            "records, CORAL is estimating a covariance it cannot support."
        )
    _shift_and_reliability(metrics)


def _shift_and_reliability(metrics: dict) -> None:
    """The figures the report uses, shown as the report produced them."""
    shift = metrics.get("species_transfer", {}).get("species_shift", {}).get("by_feature", {})
    excluding = [
        name
        for name, entry in shift.items()
        if entry.get("difference_ci95")
        and not entry["difference_ci95"][0] < 0 < entry["difference_ci95"][1]
    ]
    if excluding:
        st.markdown(
            "**Only one feature's interval excludes zero:** "
            + ", ".join(f"`{name}`" for name in excluding)
            + ". Every other feature's species difference is compatible with no shift at all."
        )
    for figure, caption in (
        (
            "species_shift.png",
            "REAL data: per-feature species shift with cluster-bootstrap intervals.",
        ),
        ("task1_reliability_real.png", "REAL data: reliability of the mechanism classifier."),
        ("conformal_set_size.png", "REAL data: conformal set size against target coverage."),
    ):
        path = loaders.FIGURES / figure
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
