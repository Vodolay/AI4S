"""Baselines for classifying a drug's mechanism from MEA feature deltas.

Two tasks, kept separate on purpose:

1. **Within-platform multiclass** (Kapucu 2022): every class is present on one
   platform, in one lab, so the model cannot win by recognising the platform.
   Cross-validation is grouped by culture unit, because a well's drug and TTX
   recordings share a baseline and a culture.
2. **External check**: fit on that platform and apply to other labs' data. The
   external sets have no vehicle controls, so only a detection rate is defined;
   a different mechanism (diazepam, a GABA-A *positive* modulator) serves as the
   negative control instead of a fabricated specificity.

Numbers land in outputs/metrics.json; figures in outputs/figures/.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neuromps.models.dataset import delta_columns

log = logging.getLogger("neuromps.models")


def _pipeline(estimator) -> Pipeline:
    """Impute then scale, so every model sees the same preprocessing."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


MODELS: dict[str, Pipeline] = {
    "dummy_most_frequent": _pipeline(DummyClassifier(strategy="most_frequent")),
    "logistic_regression": _pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    ),
    "gradient_boosting": _pipeline(
        HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    ),
}


def grouped_scores(pairs: pd.DataFrame, model: Pipeline, n_splits: int = 5, seed: int = 0) -> dict:
    """Cross-validate with cultures kept whole, returning per-fold scores."""
    x = pairs[delta_columns()].to_numpy()
    y = pairs["label"].to_numpy()
    groups = pairs["group"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    balanced, macro_f1, fold_groups, predictions = [], [], [], []
    for train_idx, test_idx in splitter.split(x, y, groups):
        fitted = model.fit(x[train_idx], y[train_idx])
        predicted = fitted.predict(x[test_idx])
        balanced.append(float(balanced_accuracy_score(y[test_idx], predicted)))
        macro_f1.append(float(f1_score(y[test_idx], predicted, average="macro")))
        fold_groups.append((set(groups[train_idx]), set(groups[test_idx])))
        predictions.append((y[test_idx], predicted))
    return {
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "fold_groups": fold_groups,
        "predictions": predictions,
    }


def summarize(pairs: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> dict:
    """Run every baseline model and collect comparable metrics."""
    counts = pairs["label"].value_counts().to_dict()
    n_splits = min(n_splits, int(min(counts.values())))
    out = {
        "n_pairs": int(len(pairs)),
        "n_groups": int(pairs["group"].nunique()),
        "n_splits": n_splits,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "chance_balanced_accuracy": round(1 / len(counts), 3),
        "models": {},
    }
    for name, model in MODELS.items():
        scores = grouped_scores(pairs, model, n_splits=n_splits, seed=seed)
        out["models"][name] = {
            "balanced_accuracy_mean": round(float(np.mean(scores["balanced_accuracy"])), 3),
            "balanced_accuracy_sd": round(float(np.std(scores["balanced_accuracy"])), 3),
            "macro_f1_mean": round(float(np.mean(scores["macro_f1"])), 3),
            "per_fold_balanced_accuracy": [round(s, 3) for s in scores["balanced_accuracy"]],
        }
    return out


def confusion(pairs: pd.DataFrame, model_name: str, n_splits: int = 5, seed: int = 0):
    """Pooled out-of-fold confusion matrix and its labels."""
    scores = grouped_scores(pairs, MODELS[model_name], n_splits=n_splits, seed=seed)
    truth = np.concatenate([t for t, _ in scores["predictions"]])
    predicted = np.concatenate([p for _, p in scores["predictions"]])
    labels = sorted(set(pairs["label"]))
    return confusion_matrix(truth, predicted, labels=labels, normalize="true"), labels


def external_check(train_pairs: pd.DataFrame, external_pairs: pd.DataFrame, positive: str) -> dict:
    """Fit on one platform, then measure how often another platform's pairs are detected.

    Only a detection rate is reported: the external sets contain no vehicle
    controls, so specificity is undefined and is not invented. Where doses vary,
    the Spearman correlation between dose and predicted probability is reported
    as a dose-response check.
    """
    model = MODELS["logistic_regression"]
    binary = np.where(train_pairs["label"] == positive, positive, "other")
    fitted = model.fit(train_pairs[delta_columns()].to_numpy(), binary)
    x = external_pairs[delta_columns()].to_numpy()
    predicted = fitted.predict(x)
    index = list(fitted.classes_).index(positive)
    probability = fitted.predict_proba(x)[:, index]
    out = {
        "n": int(len(external_pairs)),
        "positive_class": positive,
        "detection_rate": round(float(np.mean(predicted == positive)), 3),
        "mean_probability": round(float(np.mean(probability)), 3),
    }
    doses = pd.to_numeric(external_pairs.get("dose_value"), errors="coerce")
    if doses is not None and doses.notna().sum() > 3 and doses.nunique() > 2:
        rho, p_value = spearmanr(doses, probability, nan_policy="omit")
        out["dose_spearman"] = round(float(rho), 3)
        out["dose_spearman_p"] = float(f"{p_value:.3g}")
    return out


def dose_in_um(pairs: pd.DataFrame) -> pd.Series:
    """Doses on one scale, so nM and uM conditions can be ordered together."""
    value = pd.to_numeric(pairs["dose_value"], errors="coerce")
    return np.where(pairs["dose_unit"] == "nM", value / 1000.0, value)


def detection_by_dose(
    train_pairs: pd.DataFrame, external_pairs: pd.DataFrame, positive: str
) -> list[dict]:
    """Detection rate per dose: a flat overall rate can hide a real dose trend."""
    model = MODELS["logistic_regression"]
    binary = np.where(train_pairs["label"] == positive, positive, "other")
    fitted = model.fit(train_pairs[delta_columns()].to_numpy(), binary)
    index = list(fitted.classes_).index(positive)
    external = external_pairs.copy()
    external["_p"] = fitted.predict_proba(external[delta_columns()].to_numpy())[:, index]
    external["_dose"] = dose_in_um(external)
    rows = []
    for dose, part in external.groupby("_dose"):
        rows.append(
            {
                "dose_uM": float(dose),
                "n": int(len(part)),
                "detection_rate": round(float((part["_p"] > 0.5).mean()), 3),
                "mean_probability": round(float(part["_p"].mean()), 3),
            }
        )
    return rows


def platform_delta_contrast(pairs: pd.DataFrame, mechanism: str, min_dose_um: float = 30.0) -> dict:
    """Median feature delta for one mechanism, per source.

    Asks the harmonization question directly: does the same mechanism move the
    same features in the same direction on different platforms?
    """
    selected = pairs[pairs["label"] == mechanism].copy()
    selected["_dose"] = dose_in_um(selected)
    selected = selected[selected["_dose"].isna() | (selected["_dose"] >= min_dose_um)]
    out = {"mechanism": mechanism, "min_dose_uM": min_dose_um, "by_source": {}}
    for source, part in selected.groupby("source"):
        out["by_source"][str(source)] = {
            "n": int(len(part)),
            "compounds": sorted({str(c) for c in part["compound"].dropna()}),
            "median_delta": {
                c.removeprefix("delta."): round(float(part[c].median()), 3) for c in delta_columns()
            },
        }
    sources = list(out["by_source"])
    if len(sources) == 2:
        a, b = (out["by_source"][s]["median_delta"] for s in sources)
        agree = [f for f in a if np.sign(a[f]) == np.sign(b[f]) and a[f] != 0]
        out["features_agreeing_in_sign"] = sorted(agree)
        out["features_disagreeing_in_sign"] = sorted(set(a) - set(agree))
    return out


def plot_platform_contrast(contrast: dict, path, title: str) -> None:
    """Grouped bars of median delta per platform for one mechanism."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = list(contrast["by_source"])
    features = list(contrast["by_source"][sources[0]]["median_delta"])
    width, positions = 0.38, np.arange(len(features))
    colours = ["#0072B2", "#E69F00"]  # colourblind-safe
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, source in enumerate(sources):
        values = [contrast["by_source"][source]["median_delta"][f] for f in features]
        label = f"{source} ({', '.join(contrast['by_source'][source]['compounds'])})"
        ax.bar(positions + (i - 0.5) * width, values, width, label=label, color=colours[i % 2])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, features, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("median log2(exposure / baseline)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, labels: list[str], path, title: str) -> None:
    """Confusion heatmap, colorblind-safe, titled with what it shows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="cividis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[i, j] < 0.6 else "black",
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run both tasks, write metrics and the confusion figure."""
    import argparse
    from pathlib import Path

    from neuromps.metrics import update_metrics
    from neuromps.models.dataset import load_pairs

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parents[3]
    figures = repo_root / "outputs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs()
    kapucu = pairs[pairs["source"] == "kapucu2022"]
    results = {"task1_within_platform": summarize(kapucu, seed=args.seed)}
    matrix, labels = confusion(
        kapucu, "logistic_regression", results["task1_within_platform"]["n_splits"], args.seed
    )
    plot_confusion(
        matrix,
        labels,
        figures / "mechanism_confusion.png",
        "Mechanism classification from MEA feature deltas (real data, Kapucu 2022)\n"
        "logistic regression, cross-validation grouped by culture",
    )
    external = {}
    bicuculline = pairs[pairs["source"] == "screenneuropharm_bic"]
    if len(bicuculline):
        external["screenneuropharm_bicuculline"] = external_check(
            kapucu, bicuculline, positive="gabaa_antagonist"
        )
    diazepam = pairs[pairs["source"] == "sharf2022"]
    if len(diazepam):
        external["sharf_diazepam_negative_control"] = external_check(
            kapucu, diazepam, positive="gabaa_antagonist"
        )
    if len(bicuculline):
        external["screenneuropharm_bicuculline"]["by_dose"] = detection_by_dose(
            kapucu, bicuculline, positive="gabaa_antagonist"
        )
    results["task2_external_detection"] = external
    contrast = platform_delta_contrast(pairs, "gabaa_antagonist")
    results["task3_platform_contrast"] = contrast
    if len(contrast["by_source"]) == 2:
        plot_platform_contrast(
            contrast,
            figures / "platform_contrast_gabaa_antagonist.png",
            "Same mechanism, two platforms (real data): median feature delta for\n"
            "GABA-A antagonists at >=30 uM, Kapucu 2022 vs ScreenNeuroPharm",
        )
    update_metrics("mechanism_baselines", results)
    log.info("wrote metrics and %s", figures / "mechanism_confusion.png")
    import json

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
