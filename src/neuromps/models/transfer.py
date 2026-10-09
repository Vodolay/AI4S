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


def _bootstrap_ci(
    truth: np.ndarray, predicted: np.ndarray, seed: int, groups: np.ndarray
) -> list[float]:
    """Percentile interval for balanced accuracy, resampling cultures not rows.

    §5.1 of the report states this rule for every interval in the project, and
    this function used to break it: it drew rows with replacement, which treats
    two recordings from one culture as two independent observations and reports
    an interval far too narrow. The test frame carries its culture labels, so
    there was never a reason to.
    """

    def metric(idx: np.ndarray) -> float:
        drawn_truth, drawn_predicted = truth[idx], predicted[idx]
        if len(set(drawn_truth.tolist())) < 2:
            return float("nan")
        return float(balanced_accuracy_score(drawn_truth, drawn_predicted))

    interval = group_bootstrap_metric(metric, groups, n_resamples=BOOTSTRAP, seed=seed)
    low, high = interval.get("ci95", [float("nan"), float("nan")])
    return [round(float(low), 3), round(float(high), 3)]


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
    truth, predicted, culture = [], [], []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for train_idx, test_idx in splitter.split(x, y, groups):
            fitted = MODELS["logistic_regression"].fit(x[train_idx], y[train_idx])
            fold_prediction = fitted.predict(x[test_idx])
            scores.append(balanced_accuracy_score(y[test_idx], fold_prediction))
            truth.extend(y[test_idx])
            predicted.extend(fold_prediction)
            culture.extend(groups[test_idx])
    # The interval is on the out-of-fold predictions pooled over every repeat,
    # resampling cultures: a drawn culture brings all of its rows, from every
    # repeat, so repeating the split never makes one culture count as several.
    # These two rows of the species table had no interval at all, beside rows
    # that did - which read as missing data rather than as a deliberate choice.
    interval = _bootstrap_ci(np.asarray(truth), np.asarray(predicted), seed, np.asarray(culture))
    return {
        "balanced_accuracy": round(float(np.mean(scores)), 3),
        "balanced_accuracy_sd": round(float(np.std(scores)), 3),
        "balanced_accuracy_ci95": interval,
        "pooled_out_of_fold_balanced_accuracy": round(
            float(balanced_accuracy_score(truth, predicted)), 3
        ),
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
            "balanced_accuracy_ci95": _bootstrap_ci(
                truth, predicted, seed, human["group"].to_numpy()
            ),
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
    """Per-feature medians by species, and the difference with its interval.

    The second panel exists because the first cannot carry uncertainty: the
    bootstrap intervals are on the rat-human *difference*, not on either median,
    and several of them reach zero. Drawing only the bars invites the reader to
    take an ordering as an effect.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = list(shift["by_feature"])
    rat = [shift["by_feature"][f]["rat_median"] or 0 for f in features]
    human = [shift["by_feature"][f]["human_median"] or 0 for f in features]
    positions, width = np.arange(len(features)), 0.38
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.0), sharex=True)
    axes[0].bar(
        positions - width / 2, rat, width, label=f"rat (n={shift['n_rat']})", color="#0072B2"
    )
    axes[0].bar(
        positions + width / 2, human, width, label=f"human (n={shift['n_human']})", color="#D55E00"
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("median log2(exposure / baseline)", fontsize=9)
    axes[0].legend(fontsize=8)
    _difference_panel(axes[1], shift, features, positions)
    axes[1].set_xticks(positions, features, rotation=40, ha="right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _difference_panel(axis, shift: dict, features: list[str], positions) -> None:
    """Rat minus human per feature, with the 95% cluster-bootstrap interval."""
    differences, lower, upper, colours = [], [], [], []
    for name in features:
        entry = shift["by_feature"][name]
        difference = entry.get("difference") or 0.0
        low, high = entry.get("difference_ci95") or [difference, difference]
        differences.append(difference)
        lower.append(difference - low)
        upper.append(high - difference)
        colours.append("#009E73" if (low > 0 or high < 0) else "#999999")
    axis.errorbar(
        positions,
        differences,
        yerr=[lower, upper],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=3,
        markersize=0,
    )
    axis.scatter(positions, differences, color=colours, zorder=3, s=36)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("rat - human, log2 units", fontsize=9)
    axis.set_title("green = interval excludes zero; grey = compatible with no shift", fontsize=8)


def main(argv: list[str] | None = None) -> int:
    """Run the transfer study on the one source with both species on one platform."""
    import argparse
    import json

    from neuromps.metrics import update_metrics
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
            "no exposure/baseline pairs in the asset; skipping the transfer study. "
            "This is what a blocked or unfetched source looks like downstream - "
            "check the fetch summary for a source with status=blocked."
        )
        return 0
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
