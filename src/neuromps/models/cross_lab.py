"""Does mechanism signal survive a change of laboratory? An asymmetric test.

§6.1 does not pool laboratories, because across the whole asset a pair's source
is very nearly determined by its label. The structural reason is sharper than
that number: **no two laboratories share more than one mechanism class**, and a
classifier needs two to discriminate between. Leave-one-laboratory-out
cross-validation cannot be constructed here at all - not "would be underpowered",
but no valid split exists.

What can be built is a deliberately confounded test. Three classes appear in more
than one source, all three of them in `kapucu2022`; train there and test on the
other laboratories' versions of the same three. In that test set each class comes
from exactly one laboratory, so a model that read laboratory identity alone would
score perfectly.

That asymmetry is the point:

- a **high** score is uninterpretable, because the confound alone produces it;
- a **low** score is strong evidence, because the confound was maximally helpful
  and the model failed anyway.

The module refuses to interpret a high score, rather than leaving that to a
reader who may only see the number.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("neuromps.models")

# Above this the confound alone explains the result, so the run reports the
# number and declines to draw a conclusion from it.
CHANCE_IS_MEANINGFUL_BELOW = 0.60
TRAIN_SOURCE = "kapucu2022"


def shared_classes(pairs: pd.DataFrame) -> list[str]:
    """Mechanism classes measured in more than one laboratory."""
    by_label = pairs.groupby("label")["source"].nunique()
    return sorted(by_label[by_label > 1].index)


def train_test_split_by_source(pairs: pd.DataFrame, train_source: str = TRAIN_SOURCE):
    """Train on one laboratory, test on every other, over the shared classes only."""
    shared = shared_classes(pairs)
    usable = pairs[pairs["label"].isin(shared)]
    train = usable[usable["source"] == train_source]
    test = usable[usable["source"] != train_source]
    # Both sides must carry the same classes, or the comparison is not a comparison.
    common = sorted(set(train["label"]) & set(test["label"]))
    return (
        train[train["label"].isin(common)].reset_index(drop=True),
        test[test["label"].isin(common)].reset_index(drop=True),
    )


def label_source_correspondence(test: pd.DataFrame) -> float:
    """Share of test pairs whose label is determined by their source.

    1.0 means laboratory identity is a perfect stand-in for the answer.
    """
    if test.empty:
        return 0.0
    per_label = test.groupby("label")["source"].agg(lambda s: s.value_counts().iloc[0])
    return float(per_label.sum() / len(test))


def summarise_cross_lab(train, test, balanced_accuracy: float, ci95: list) -> dict:
    """The result, with the rule for reading it attached."""
    correspondence = label_source_correspondence(test)
    interpretable = balanced_accuracy < CHANCE_IS_MEANINGFUL_BELOW
    return {
        "label": "REAL pairs, trained on one laboratory and tested on the others",
        "train_source": TRAIN_SOURCE,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "classes": sorted(set(train["label"])),
        "chance": round(1.0 / max(len(set(train["label"])), 1), 4),
        "balanced_accuracy": round(float(balanced_accuracy), 4),
        "balanced_accuracy_ci95": [round(float(v), 4) for v in ci95],
        "label_source_correspondence": round(correspondence, 4),
        "confound_ceiling": 1.0 if correspondence >= 1.0 else round(correspondence, 4),
        "interpretable": bool(interpretable),
        "reads_as": (
            "Each test class comes from one laboratory, so laboratory identity alone would score "
            "perfectly and only a low score carries information. "
            + (
                "This score is low: the model failed although the confound was helping it."
                if interpretable
                else "This score is not low, so the confound explains it and nothing is concluded."
            )
        ),
    }


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> dict:
    """Fit on the training laboratory and score on the others, per class and overall."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import balanced_accuracy_score, recall_score

    from neuromps.models.bootstrap import group_bootstrap_metric
    from neuromps.models.dataset import delta_columns
    from neuromps.models.mechanism import MODELS

    columns = delta_columns()
    model = clone(MODELS["logistic_regression"])
    model.fit(train[columns].to_numpy(), train["label"].to_numpy())
    truth = test["label"].to_numpy()
    predicted = model.predict(test[columns].to_numpy())
    classes = sorted(set(train["label"]))

    def metric(index: np.ndarray) -> float:
        return float(balanced_accuracy_score(truth[index], predicted[index]))

    interval = group_bootstrap_metric(metric, test["group"].to_numpy(), n_resamples=1000, seed=seed)
    per_class = recall_score(truth, predicted, labels=classes, average=None, zero_division=0)
    return {
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "ci95": interval["ci95"],
        "recall_by_class": {c: round(float(r), 4) for c, r in zip(classes, per_class, strict=True)},
        "predicted_counts": {c: int((predicted == c).sum()) for c in classes},
    }


def within_source_control(train: pd.DataFrame, seed: int = 0) -> dict:
    """The same classes, the same tiny sample, without crossing a laboratory.

    The cross-laboratory test trains on 20 pairs, so a failure there could be the
    sample size rather than the change of laboratory. This scores the same
    classes inside the training laboratory by grouped cross-validation: if it
    also fails, the test says nothing about laboratories, and if it does not, the
    difference between the two is what crossing a laboratory costs.
    """
    from sklearn.base import clone
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    from neuromps.models.dataset import delta_columns
    from neuromps.models.mechanism import MODELS

    counts = train["label"].value_counts()
    splits = int(min(3, counts.min()))
    if splits < 2:
        return {"status": "too few pairs in the smallest class for a within-source split"}
    x, y = train[delta_columns()].to_numpy(), train["label"].to_numpy()
    predicted = cross_val_predict(
        clone(MODELS["logistic_regression"]),
        x,
        y,
        groups=train["group"].to_numpy(),
        cv=StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed),
    )
    return {
        "n_pairs": int(len(train)),
        "n_splits": splits,
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predicted)), 4),
        "reads_as": (
            "same classes and sample size, no laboratory crossed; compare with the "
            "cross-laboratory score to separate small-sample failure from laboratory shift"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: run the asymmetric cross-laboratory test and record it."""
    import argparse
    import json

    from neuromps.metrics import update_metrics
    from neuromps.models.dataset import load_pairs

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="accepted for pipeline symmetry")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pairs = load_pairs()
    train, test = train_test_split_by_source(pairs)
    if train.empty or test.empty or len(set(train["label"])) < 2:
        log.warning("not enough shared classes across laboratories for this test; skipping")
        update_metrics("cross_lab_transfer", {"status": "not constructible in this asset"})
        return 0
    scored = fit_and_score(train, test, seed=args.seed)
    result = {
        **summarise_cross_lab(train, test, scored["balanced_accuracy"], scored["ci95"]),
        "recall_by_class": scored["recall_by_class"],
        "predicted_counts": scored["predicted_counts"],
        "within_source_control": within_source_control(train, seed=args.seed),
        "max_classes_shared_by_any_two_laboratories": max_shared_between_pairs(pairs),
    }
    update_metrics("cross_lab_transfer", result)
    print(json.dumps(result, indent=2))
    return 0


def max_shared_between_pairs(pairs: pd.DataFrame) -> int:
    """The largest number of classes any two laboratories have in common.

    Below 2, no cross-laboratory classifier can be built at all, which is the
    reason §6.1 does not pool rather than a preference about sample size.
    """
    import itertools

    by_source = {s: set(g["label"]) for s, g in pairs.groupby("source")}
    return max(
        (len(a & b) for a, b in itertools.combinations(by_source.values(), 2)),
        default=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
