"""A rate-only arm: how far does one firing-rate change carry, on its own?

Section 6.6 reports that two rate features score as well as all nine on Task 1.
That invites a sharper question than the ablation can answer, because the
ablation varies the feature set on one source: **does a rate change carry
mechanism on a platform, preparation and species the classifier has never
seen?**

`fofie2025` answers it. hiPSC-derived sensory neurons on Axion plates, spike
counts per well at a baseline and after a compound, published without spike
times - so the only feature it can supply is the rate change, and that is
exactly the feature under test. Each well is its own control, so the delta is
free of the plate effects that the rest of the asset has to model around.

Two analyses, in increasing difficulty:

- **Within source.** Grouped cross-validation over condition blocks, predicting
  mechanism class from one number. This bounds what a rate change carries when
  everything else is held constant.
- **Across sources.** A model fitted on Kapucu's cortical `delta.mfr_hz` and
  applied to Fofie's sensory-neuron rate change, on the classes they share. This
  is the claim of section 6.6 put to a test it could fail, and the one worth
  quoting.

The scope limit is real and is stated wherever the numbers are: these are
**sensory neurons**, and a sodium-channel blocker silencing a nociceptor is not
evidence about a cortical network. What transfers here is the mechanism, not the
preparation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("neuromps.models")

RESAMPLES = 1000
MIN_GROUPS_PER_CLASS = 2
# Classes Kapucu's cortical pairs and Fofie's sensory pairs both contain.
SHARED_WITH_KAPUCU = (
    "na_channel_blocker",
    "glutamatergic_antagonist",
    "gabaa_agonist_or_pam",
    "control",
)


def fofie_pairs() -> pd.DataFrame:
    """Fofie's rate deltas, restricted to classes with enough blocks to split."""
    from neuromps.etl.fofie import build_pairs

    pairs = build_pairs()
    if pairs.empty:
        return pairs
    groups_per_class = pairs.groupby("mechanism_class")["group"].nunique()
    keep = groups_per_class[groups_per_class >= MIN_GROUPS_PER_CLASS].index
    dropped = sorted(set(groups_per_class.index) - set(keep))
    if dropped:
        log.info("dropping classes with one block only: %s", dropped)
    out = pairs[pairs["mechanism_class"].isin(keep)].copy()
    out.attrs["dropped_single_block_classes"] = dropped
    return out


def _grouped_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    """Balanced accuracy from grouped cross-validation on one feature."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_splits = min(4, len(np.unique(groups)))
    if n_splits < 2:
        return {"note": "too few groups to split"}
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predicted = np.empty(len(y), dtype=object)
    baseline = np.empty(len(y), dtype=object)
    for train, test in splitter.split(x, y, groups):
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
        ).fit(x[train], y[train])
        predicted[test] = model.predict(x[test])
        baseline[test] = (
            DummyClassifier(strategy="most_frequent").fit(x[train], y[train]).predict(x[test])
        )
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predicted)), 4),
        "dummy_balanced_accuracy": round(float(balanced_accuracy_score(y, baseline)), 4),
        "chance": round(1.0 / len(np.unique(y)), 4),
        "n_splits": int(n_splits),
        "predicted": predicted,
    }


def within_source(pairs: pd.DataFrame, seed: int = 0) -> dict:
    """Mechanism from one rate change, inside Fofie, grouped over blocks."""
    from sklearn.metrics import balanced_accuracy_score

    from neuromps.models.bootstrap import group_bootstrap_metric

    if pairs.empty or pairs["mechanism_class"].nunique() < 2:
        return {"note": "not enough classes"}
    x = pairs[["delta_log2_rate"]].to_numpy(dtype=float)
    y = pairs["mechanism_class"].to_numpy()
    groups = pairs["group"].to_numpy()
    scored = _grouped_scores(x, y, groups, seed)
    if "note" in scored:
        return scored
    predicted = scored.pop("predicted")

    def metric(idx: np.ndarray) -> float:
        if len(set(y[idx])) < 2:
            return float("nan")
        return float(balanced_accuracy_score(y[idx], predicted[idx]))

    boot = group_bootstrap_metric(metric, groups, RESAMPLES, seed)
    interval = boot.get("ci95")
    return {
        **scored,
        "balanced_accuracy_ci95": interval,
        "beats_chance": None
        if not interval or interval[0] is None
        else bool(interval[0] > scored["chance"]),
        "n_pairs": int(len(pairs)),
        "n_groups": int(len(set(groups))),
        "n_classes": int(len(set(y))),
        "feature": "delta_log2_rate",
        "bootstrap_resamples": RESAMPLES,
    }


def kapucu_rate_deltas() -> pd.DataFrame:
    """Kapucu's cortical pairs, reduced to the one feature Fofie can supply."""
    from neuromps.models.dataset import load_pairs

    pairs = load_pairs()
    if pairs.empty or "delta.mfr_hz" not in pairs.columns:
        return pd.DataFrame()
    kapucu = pairs[pairs["source"] == "kapucu2022"].copy()
    kapucu = kapucu[kapucu["delta.mfr_hz"].notna() & np.isfinite(kapucu["delta.mfr_hz"])]
    return pd.DataFrame(
        {
            "delta_log2_rate": kapucu["delta.mfr_hz"].to_numpy(dtype=float),
            "mechanism_class": kapucu["label"].to_numpy(),
            "group": kapucu["group"].to_numpy()
            if "group" in kapucu.columns
            else kapucu.index.astype(str),
        }
    )


def across_sources(fofie: pd.DataFrame, seed: int = 0) -> dict:
    """Fit on Kapucu's cortical rate change, test on Fofie's sensory one.

    The honest version of section 6.6's claim. Both sides are the same quantity -
    log2 of treated over that culture's own baseline firing rate - so a model
    fitted on one can be asked about the other without rescaling.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from neuromps.models.bootstrap import group_bootstrap_metric

    kapucu = kapucu_rate_deltas()
    if kapucu.empty or fofie.empty:
        return {"note": "one of the two sources is unavailable"}
    shared = sorted(
        set(kapucu["mechanism_class"]) & set(fofie["mechanism_class"]) & set(SHARED_WITH_KAPUCU)
    )
    train = kapucu[kapucu["mechanism_class"].isin(shared)]
    test = fofie[fofie["mechanism_class"].isin(shared)]
    if len(shared) < 2 or train.empty or test.empty:
        return {"shared_classes": shared, "note": "too few shared classes"}

    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    ).fit(train[["delta_log2_rate"]].to_numpy(dtype=float), train["mechanism_class"].to_numpy())
    truth = test["mechanism_class"].to_numpy()
    predicted = model.predict(test[["delta_log2_rate"]].to_numpy(dtype=float))
    groups = test["group"].to_numpy()

    def metric(idx: np.ndarray) -> float:
        if len(set(truth[idx])) < 2:
            return float("nan")
        return float(balanced_accuracy_score(truth[idx], predicted[idx]))

    observed = float(balanced_accuracy_score(truth, predicted))
    boot = group_bootstrap_metric(metric, groups, RESAMPLES, seed)
    interval = boot.get("ci95")
    return {
        "shared_classes": shared,
        "n_train_pairs": int(len(train)),
        "n_test_pairs": int(len(test)),
        "n_test_groups": int(len(set(groups))),
        "balanced_accuracy": round(observed, 4),
        "balanced_accuracy_ci95": interval,
        "chance": round(1.0 / len(shared), 4),
        "beats_chance": None
        if not interval or interval[0] is None
        else bool(interval[0] > 1.0 / len(shared)),
        "bootstrap_resamples": RESAMPLES,
        "note": (
            "trained on kapucu2022 cortical delta.mfr_hz, tested on fofie2025 sensory "
            "neuron rate change; one feature, and a different preparation"
        ),
    }


def direction_agreement(fofie: pd.DataFrame) -> dict:
    """Does each class move the same way here as the twin's knobs predict?

    A cheap, assumption-light check that the parsed deltas are the right sign:
    an untreated well should not move, and a channel blocker should silence.
    """
    out = {}
    for mechanism, rows in fofie.groupby("mechanism_class"):
        values = rows["delta_log2_rate"].to_numpy(dtype=float)
        out[str(mechanism)] = {
            "n_pairs": int(len(values)),
            "n_groups": int(rows["group"].nunique()),
            "median_log2_change": round(float(np.median(values)), 4),
            "fraction_suppressed": round(float(np.mean(values < 0)), 4),
        }
    controls = out.get("control", {}).get("median_log2_change")
    blockers = out.get("na_channel_blocker", {}).get("median_log2_change")
    out["control_is_flat_and_blocker_is_not"] = bool(
        controls is not None and blockers is not None and abs(controls) < 0.5 and blockers < -1.0
    )
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: how far one rate change carries, within a source and across two."""
    import argparse
    import json

    from neuromps.etl.fofie import WORKBOOK
    from neuromps.metrics import update_metrics

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="fewer resamples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not WORKBOOK.exists():
        log.warning("fofie2025 workbook not fetched; skipping")
        update_metrics("rate_only_arm", {"status": "fofie2025 not fetched"})
        return 0
    pairs = fofie_pairs()
    if pairs.empty:
        update_metrics("rate_only_arm", {"status": "no usable pairs"})
        return 0
    result = {
        "label": "REAL hiPSC sensory-neuron wells, rate change only",
        "question": (
            "does a firing-rate change alone carry mechanism, and does it carry it "
            "to a preparation the model has not seen"
        ),
        "source": "fofie2025",
        "n_pairs": int(len(pairs)),
        "n_groups": int(pairs["group"].nunique()),
        "dropped_single_block_classes": list(pairs.attrs.get("dropped_single_block_classes", [])),
        "within_source": within_source(pairs, args.seed),
        "across_sources": across_sources(pairs, args.seed),
        "by_mechanism": direction_agreement(pairs),
        "caveat": (
            "sensory neurons, not cortical: what is shared with the rest of the asset is the "
            "compound mechanism, not the preparation or the network"
        ),
    }
    update_metrics("rate_only_arm", result)
    print(json.dumps({k: v for k, v in result.items() if k != "by_mechanism"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
