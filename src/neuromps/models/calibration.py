"""Calibration of the mechanism classifier: is a stated probability trustworthy?

Balanced accuracy says how often the top label is right; it says nothing about
whether a 0.8 means 0.8. Expected calibration error compares confidence with
accuracy over out-of-fold predictions, binned by confidence, and the reliability
diagram shows where the model is over- or under-confident.

Confidence here is the probability of the predicted (top) class, the standard
multiclass formulation. Predictions come from the same grouped cross-validation
used everywhere else, so no culture is scored by a model that saw it.
"""

import logging

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from neuromps.models.dataset import delta_columns
from neuromps.models.mechanism import MODELS

log = logging.getLogger("neuromps.models")
DEFAULT_BINS = 10


def out_of_fold_probabilities(
    pairs, model_name: str = "logistic_regression", n_splits: int = 4, seed: int = 0
):
    """Out-of-fold class probabilities, the truth, the class order and the groups."""
    counts = pairs["label"].value_counts()
    n_splits = max(2, min(n_splits, int(counts.min())))
    x = pairs[delta_columns()].to_numpy()
    y = pairs["label"].to_numpy()
    groups = pairs["group"].to_numpy()
    classes = np.array(sorted(set(y)))
    probabilities = np.zeros((len(y), classes.size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(x, y, groups):
        fitted = MODELS[model_name].fit(x[train_idx], y[train_idx])
        fold = fitted.predict_proba(x[test_idx])
        for column, label in enumerate(fitted.classes_):
            probabilities[test_idx, int(np.flatnonzero(classes == label)[0])] = fold[:, column]
    return y, probabilities, classes, groups


def reliability_bins(truth, probabilities, classes, n_bins: int = DEFAULT_BINS) -> list[dict]:
    """Per-confidence-bin counts, mean confidence and accuracy."""
    truth = np.asarray(truth)
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    correct = predicted == truth
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (
            (confidence > lower) & (confidence <= upper) if lower > 0 else (confidence <= upper)
        )
        bins.append(
            {
                "lower": round(float(lower), 3),
                "upper": round(float(upper), 3),
                "count": int(in_bin.sum()),
                "confidence": round(float(confidence[in_bin].mean()), 4) if in_bin.any() else None,
                "accuracy": round(float(correct[in_bin].mean()), 4) if in_bin.any() else None,
            }
        )
    return bins


def expected_calibration_error(truth, probabilities, classes, n_bins: int = DEFAULT_BINS) -> dict:
    """ECE plus the bins it was computed from."""
    bins = reliability_bins(truth, probabilities, classes, n_bins)
    total = sum(b["count"] for b in bins)
    ece = sum(
        b["count"] / total * abs(b["accuracy"] - b["confidence"])
        for b in bins
        if b["count"] and b["accuracy"] is not None
    )
    return {"ece": round(float(ece), 4), "n_bins": n_bins, "n": int(total), "bins": bins}


def plot_reliability(bins: list[dict], ece: float, path, title: str) -> None:
    """Reliability diagram: accuracy against confidence, with per-bin counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centres = [(b["lower"] + b["upper"]) / 2 for b in bins]
    accuracy = [b["accuracy"] if b["count"] else np.nan for b in bins]
    counts = [b["count"] for b in bins]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(5.6, 6.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot([0, 1], [0, 1], "--", color="black", lw=1, label="perfect calibration")
    top.plot(centres, accuracy, "o-", color="#D55E00", label=f"observed (ECE = {ece:.3f})")
    top.set_ylabel("accuracy")
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.legend(fontsize=8)
    top.grid(alpha=0.3)
    top.set_title(title, fontsize=10)
    bottom.bar(centres, counts, width=0.09, color="#0072B2")
    bottom.set_xlabel("confidence (probability of the predicted class)")
    bottom.set_ylabel("predictions")
    bottom.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibration_report(
    pairs,
    model_name: str = "logistic_regression",
    n_splits: int = 4,
    seed: int = 0,
    n_bins: int = DEFAULT_BINS,
    n_resamples: int = 1000,
) -> dict:
    """ECE with a cluster-bootstrap interval, plus the bins for the diagram."""
    from neuromps.models.bootstrap import group_bootstrap_metric

    truth, probabilities, classes, groups = out_of_fold_probabilities(
        pairs, model_name, n_splits, seed
    )
    report = expected_calibration_error(truth, probabilities, classes, n_bins)

    def metric(idx: np.ndarray) -> float:
        return expected_calibration_error(truth[idx], probabilities[idx], classes, n_bins)["ece"]

    interval = group_bootstrap_metric(metric, groups, n_resamples=n_resamples, seed=seed)
    return {
        **report,
        "model": model_name,
        "ece_ci95": interval["ci95"],
        "bootstrap_resamples": interval["n_resamples"],
    }
