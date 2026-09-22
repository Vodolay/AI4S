"""Conformal prediction sets for the mechanism classifier.

A balanced accuracy of 0.55 on six classes is not something to hand a
pharmacologist as a single label. Split conformal prediction turns the same
model into sets with a stated coverage level: at alpha = 0.1, the set contains
the true mechanism about 90% of the time, and the *size* of the set is the
honest signal of how much the features actually pin down.

Calibration splits are grouped by culture, like every other split in this
project, so the calibration scores are not drawn from wells the model trained on.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from neuromps.models.dataset import delta_columns
from neuromps.models.mechanism import MODELS

log = logging.getLogger("neuromps.models")


def quantile_threshold(scores: np.ndarray, alpha: float) -> float:
    """The conformal quantile: the (1-alpha)(n+1)/n empirical quantile of calibration scores.

    Scores here are 1 - probability of the true class, so the threshold is a
    probability floor: labels at or above it enter the prediction set.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(1.0 - np.quantile(scores, level, method="higher"))


def prediction_sets(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> list[list[str]]:
    """Labels whose predicted probability reaches the threshold, per sample."""
    return [
        [str(label) for label, p in zip(labels, row, strict=True) if p >= threshold]
        for row in probabilities
    ]


def coverage_report(truth: np.ndarray, sets: list[list[str]], labels: list[str]) -> dict:
    """Empirical coverage, set sizes and per-class coverage."""
    covered = np.array([t in s for t, s in zip(truth, sets, strict=True)])
    sizes = np.array([len(s) for s in sets])
    by_class = {}
    for label in labels:
        mask = truth == label
        by_class[str(label)] = round(float(covered[mask].mean()), 3) if mask.any() else None
    return {
        "n_test": int(len(truth)),
        "empirical_coverage": round(float(covered.mean()), 3),
        "mean_set_size": round(float(sizes.mean()), 3),
        "fraction_singleton": round(float((sizes == 1).mean()), 3),
        "fraction_empty": round(float((sizes == 0).mean()), 3),
        "coverage_by_class": by_class,
    }


def conformal_sets(
    pairs: pd.DataFrame, alpha: float = 0.1, seed: int = 0, n_splits: int = 4
) -> dict:
    """Cross-conformal prediction: every pair is scored by a model that never saw its culture.

    Each fold's training half is split again, grouped, into a proper training set
    and a calibration set; the calibration scores set the threshold applied to
    that fold's test pairs.
    """
    counts = pairs["label"].value_counts()
    n_splits = max(2, min(n_splits, int(counts.min())))
    x = pairs[delta_columns()].to_numpy()
    y = pairs["label"].to_numpy()
    groups = pairs["group"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_truth, all_sets, thresholds = [], [], []
    for train_idx, test_idx in outer.split(x, y, groups):
        threshold, model = _fit_and_calibrate(
            x[train_idx], y[train_idx], groups[train_idx], alpha, seed
        )
        probabilities = model.predict_proba(x[test_idx])
        all_sets += prediction_sets(probabilities, model.classes_, threshold)
        all_truth.append(y[test_idx])
        thresholds.append(threshold)
    report = coverage_report(np.concatenate(all_truth), all_sets, sorted(set(y)))
    return {
        "alpha": alpha,
        "target_coverage": round(1 - alpha, 3),
        "n_splits": n_splits,
        "mean_threshold": round(float(np.mean(thresholds)), 3),
        **report,
    }


def _fit_and_calibrate(x, y, groups, alpha: float, seed: int):
    """Fit on a proper-training part and take the conformal threshold from the rest."""
    inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    fit_idx, cal_idx = next(inner.split(x, y, groups))
    model = MODELS["logistic_regression"].fit(x[fit_idx], y[fit_idx])
    probabilities = model.predict_proba(x[cal_idx])
    class_index = {label: i for i, label in enumerate(model.classes_)}
    scores = np.array(
        [1.0 - probabilities[i, class_index[label]] for i, label in enumerate(y[cal_idx])]
    )
    return quantile_threshold(scores, alpha), model


def plot_set_size(sweep: list[dict], path, title: str) -> None:
    """Coverage and mean set size against alpha: the honest picture of what is known."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = [row["alpha"] for row in sweep]
    coverage = [row["empirical_coverage"] for row in sweep]
    target = [row["target_coverage"] for row in sweep]
    sizes = [row["mean_set_size"] for row in sweep]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
    top.plot(alphas, coverage, "o-", color="#0072B2", label="empirical coverage")
    top.plot(alphas, target, "--", color="black", label="target (1 - alpha)")
    top.set_ylabel("coverage")
    top.legend(fontsize=8)
    top.grid(alpha=0.3)
    bottom.plot(alphas, sizes, "s-", color="#D55E00")
    bottom.set_ylabel("mean labels per set")
    bottom.set_xlabel("alpha")
    bottom.grid(alpha=0.3)
    top.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Conformal sweep plus feature ablations, to metrics and a figure."""
    import argparse
    import json

    from neuromps.metrics import update_metrics
    from neuromps.models.ablation import ablate, leave_one_out
    from neuromps.models.dataset import load_pairs
    from neuromps.paths import figures_dir

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="fewer bootstrap resamples")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    figures = figures_dir()
    figures.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs()
    if pairs.empty:
        log.warning(
            "no exposure/baseline pairs in the asset; skipping conformal sets and ablations. "
            "This is what a blocked or unfetched source looks like downstream - "
            "check the fetch summary for a source with status=blocked."
        )
        return 0
    kapucu = pairs[pairs["source"] == "kapucu2022"]
    sweep = [conformal_sets(kapucu, alpha=a, seed=args.seed) for a in (0.05, 0.1, 0.2, 0.3)]
    results = {
        "n_pairs": int(len(kapucu)),
        "n_classes": int(kapucu["label"].nunique()),
        "conformal_sweep": sweep,
        "feature_subsets": ablate(kapucu, seed=args.seed, quick=args.quick),
        "leave_one_out": leave_one_out(kapucu, seed=args.seed, quick=args.quick),
        "note": (
            "with 73 pairs the calibration set is small, so neighbouring alphas can share a "
            "threshold; set size, not coverage, is the informative quantity here"
        ),
    }
    plot_set_size(
        sweep,
        figures / "conformal_set_size.png",
        "Conformal prediction sets for drug mechanism (real data, Kapucu 2022)\n"
        "6 classes: 90% coverage needs most of them,\n"
        "so the features do not pin a mechanism down",
    )
    update_metrics("uncertainty_and_ablations", results)
    log.info("wrote metrics and %s", figures / "conformal_set_size.png")
    print(
        json.dumps(
            {"conformal_sweep": sweep, "feature_subsets": results["feature_subsets"]}, indent=2
        )[:1200]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
