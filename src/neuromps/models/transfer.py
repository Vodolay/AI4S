"""Does a mechanism classifier trained on rat cultures work on human cells?

The question the asset was built for. Kapucu 2022 is the only source with both
species on one platform, in one lab, with the same compounds, so the comparison
isolates species rather than platform.

Reported honestly: transfer score, the within-species baseline it should be
compared against, a bootstrap interval (n is small), the per-feature species
shift, and whether two standard domain adaptations help or not.
"""

import logging

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from neuromps.models.bootstrap import group_bootstrap_metric, resample_count
from neuromps.models.dataset import delta_columns
from neuromps.models.mechanism import MODELS

log = logging.getLogger("neuromps.models")
BOOTSTRAP = 2000


def per_species_standardize(pairs: pd.DataFrame) -> pd.DataFrame:
    """Z-score each feature within each species: removes species offset and scale."""
    out = pairs.copy()
    columns = delta_columns()
    for _, index in out.groupby("species").groups.items():
        block = out.loc[index, columns]
        centred = (block - block.mean()) / block.std(ddof=0).replace(0, np.nan)
        out.loc[index, columns] = centred.fillna(0.0)
    return out


def coral(source: np.ndarray, target: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """CORAL: recolour the source features to the target's covariance (Sun et al. 2016)."""
    source = np.nan_to_num(source)
    target = np.nan_to_num(target)
    cs = np.cov(source, rowvar=False) + epsilon * np.eye(source.shape[1])
    ct = np.cov(target, rowvar=False) + epsilon * np.eye(target.shape[1])
    whitened = (source - source.mean(0)) @ linalg.inv(linalg.sqrtm(cs)).real
    return (whitened @ linalg.sqrtm(ct).real) + target.mean(0)


def species_shift(pairs: pd.DataFrame, quick: bool = False, seed: int = 0) -> dict:
    """Per-feature difference between species, and which features shift least.

    Each difference carries a cluster-bootstrap interval: cultures are resampled
    within each species, so the interval reflects how few cultures there are.
    """
    rat = pairs[pairs["species"] == "rat"]
    human = pairs[pairs["species"] == "human"]
    n_resamples = resample_count(quick)
    by_feature = {}
    for column in delta_columns():
        name = column.removeprefix("delta.")
        a, b = rat[column].dropna(), human[column].dropna()
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else 0.0
        difference = float(human[column].median() - rat[column].median())
        interval = _difference_ci(pairs, column, n_resamples, seed)
        by_feature[name] = {
            "rat_median": round(float(a.median()), 3) if len(a) else None,
            "human_median": round(float(b.median()), 3) if len(b) else None,
            "difference": round(difference, 3),
            "difference_ci95": interval["ci95"],
            "bootstrap_resamples": interval["n_resamples"],
            "standardized_difference": round(float(difference / pooled), 3) if pooled else None,
        }
    ranked = sorted(
        (f for f in by_feature if by_feature[f]["standardized_difference"] is not None),
        key=lambda f: abs(by_feature[f]["standardized_difference"]),
    )
    return {
        "n_rat": int(len(rat)),
        "n_human": int(len(human)),
        "bootstrap_resamples": n_resamples,
        "by_feature": by_feature,
        "most_invariant_features": ranked[:3],
        "most_shifted_features": list(reversed(ranked))[:3],
    }


def _difference_ci(pairs: pd.DataFrame, column: str, n_resamples: int, seed: int) -> dict:
    """Interval for the human-minus-rat median difference, resampling cultures."""
    values = pairs[column].to_numpy(dtype=float)
    species = pairs["species"].to_numpy()
    groups = pairs["group"].to_numpy()

    def metric(idx: np.ndarray) -> float:
        drawn_species, drawn_values = species[idx], values[idx]
        rat = drawn_values[drawn_species == "rat"]
        human = drawn_values[drawn_species == "human"]
        rat, human = rat[np.isfinite(rat)], human[np.isfinite(human)]
        if not rat.size or not human.size:
            return float("nan")
        return float(np.median(human) - np.median(rat))

    return group_bootstrap_metric(metric, groups, n_resamples=n_resamples, seed=seed)


def _bootstrap_ci(truth: np.ndarray, predicted: np.ndarray, seed: int) -> list[float]:
    """Percentile bootstrap interval for balanced accuracy."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(truth), len(truth))
        if len(set(truth[idx])) < 2:
            continue
        scores.append(balanced_accuracy_score(truth[idx], predicted[idx]))
    if not scores:
        return [float("nan"), float("nan")]
    return [
        round(float(np.percentile(scores, 2.5)), 3),
        round(float(np.percentile(scores, 97.5)), 3),
    ]


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, adapt: str | None, seed: int):
    """Train on one species, predict the other, applying the chosen adaptation."""
    columns = delta_columns()
    if adapt == "standardize":
        combined = per_species_standardize(pd.concat([train, test], ignore_index=True))
        train = combined.iloc[: len(train)]
        test = combined.iloc[len(train) :]
    x_train, x_test = train[columns].to_numpy(), test[columns].to_numpy()
    if adapt == "coral":
        x_train = coral(x_train, x_test)
    model = MODELS["logistic_regression"]
    fitted = model.fit(x_train, train["label"].to_numpy())
    return test["label"].to_numpy(), fitted.predict(x_test)


def _within_species(pairs: pd.DataFrame, seed: int, n_splits: int = 4, repeats: int = 5) -> dict:
    """Grouped cross-validation inside one species: the baseline transfer must beat.

    Repeated with different split seeds, because the smallest class here has a
    handful of members and a single split assignment is mostly noise.
    """
    counts = pairs["label"].value_counts()
    n_splits = max(2, min(n_splits, int(counts.min())))
    x = pairs[delta_columns()].to_numpy()
    y = pairs["label"].to_numpy()
    groups = pairs["group"].to_numpy()
    scores = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for train_idx, test_idx in splitter.split(x, y, groups):
            fitted = MODELS["logistic_regression"].fit(x[train_idx], y[train_idx])
            scores.append(balanced_accuracy_score(y[test_idx], fitted.predict(x[test_idx])))
    return {
        "balanced_accuracy": round(float(np.mean(scores)), 3),
        "balanced_accuracy_sd": round(float(np.std(scores)), 3),
        "n": int(len(pairs)),
        "n_splits": n_splits,
        "repeats": repeats,
    }


def _permutation_p(truth: np.ndarray, predicted: np.ndarray, seed: int, n: int = 5000) -> float:
    """Probability of this balanced accuracy if the target labels were arbitrary."""
    observed = balanced_accuracy_score(truth, predicted)
    rng = np.random.default_rng(seed)
    shuffled = truth.copy()
    hits = 0
    for _ in range(n):
        rng.shuffle(shuffled)
        hits += balanced_accuracy_score(shuffled, predicted) >= observed
    return float((hits + 1) / (n + 1))


def transfer_scores(pairs: pd.DataFrame, adapt: str | None = None, seed: int = 0) -> dict:
    """Train on rat, test on human, against the within-species baselines.

    Only labels present in both species are scored: a class the target species
    never received cannot be predicted or credited.
    """
    shared = sorted(
        set(pairs.loc[pairs["species"] == "rat", "label"])
        & set(pairs.loc[pairs["species"] == "human", "label"])
    )
    data = pairs[pairs["label"].isin(shared)]
    rat = data[data["species"] == "rat"]
    human = data[data["species"] == "human"]
    if len(shared) < 2:
        return {
            "adaptation": adapt or "none",
            "shared_labels": shared,
            "chance_balanced_accuracy": None,
            "rat_to_human": None,
            "note": (
                "fewer than two mechanism classes are shared by both species, so transfer "
                "cannot be scored"
            ),
        }
    truth, predicted = _fit_predict(rat, human, adapt, seed)
    chance = 1 / len(shared) if shared else float("nan")
    out = {
        "adaptation": adapt or "none",
        "shared_labels": shared,
        "chance_balanced_accuracy": round(chance, 3),
        "rat_to_human": {
            "balanced_accuracy": round(float(balanced_accuracy_score(truth, predicted)), 3),
            "balanced_accuracy_ci95": _bootstrap_ci(truth, predicted, seed),
            "permutation_p": round(_permutation_p(truth, predicted, seed), 4),
            "n_train": int(len(rat)),
            "n_test": int(len(human)),
        },
    }
    if len(human) >= 8:
        out["human_within"] = _within_species(human, seed)
    if len(rat) >= 8:
        out["rat_within"] = _within_species(rat, seed)
    return out


def plot_species_shift(shift: dict, path, title: str) -> None:
    """Per-feature median delta by species, so the shift is visible feature by feature."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = list(shift["by_feature"])
    rat = [shift["by_feature"][f]["rat_median"] or 0 for f in features]
    human = [shift["by_feature"][f]["human_median"] or 0 for f in features]
    positions, width = np.arange(len(features)), 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(positions - width / 2, rat, width, label=f"rat (n={shift['n_rat']})", color="#0072B2")
    ax.bar(
        positions + width / 2, human, width, label=f"human (n={shift['n_human']})", color="#D55E00"
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, features, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("median log2(exposure / baseline)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run the transfer study on the one source with both species on one platform."""
    import argparse
    import json
    from pathlib import Path

    from neuromps.metrics import update_metrics
    from neuromps.models.dataset import load_pairs

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="fewer bootstrap resamples")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parents[3]
    figures = repo_root / "outputs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs()
    kapucu = pairs[pairs["source"] == "kapucu2022"]
    shift = species_shift(kapucu, quick=args.quick, seed=args.seed)
    results = {
        "source": "kapucu2022",
        "species_shift": shift,
        "transfer": {
            (adapt or "none"): transfer_scores(kapucu, adapt=adapt, seed=args.seed)
            for adapt in (None, "standardize", "coral")
        },
    }
    plot_species_shift(
        shift,
        figures / "species_shift.png",
        "Species shift in MEA feature deltas (real data, Kapucu 2022)\n"
        "median log2 change under drug, rat vs hPSC-derived cultures, same platform",
    )
    update_metrics("species_transfer", results)
    log.info("wrote metrics and %s", figures / "species_shift.png")
    print(json.dumps(results["transfer"], indent=2))
    print(
        json.dumps(
            {k: shift[k] for k in ("most_invariant_features", "most_shifted_features")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
