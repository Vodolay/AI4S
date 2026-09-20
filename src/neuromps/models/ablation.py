"""Which features carry the mechanism signal?

Two views, both cross-validated with cultures kept whole:

- named subsets, including the `species_invariant` set that the transfer study
  singled out, so the claim "these features are the portable ones" is tested
  rather than asserted;
- leave-one-out, which says what each feature adds to the full vector.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from neuromps.models.dataset import DELTA_FEATURES
from neuromps.models.mechanism import MODELS

log = logging.getLogger("neuromps.models")

FEATURE_SUBSETS: dict[str, tuple[str, ...]] = {
    "all": DELTA_FEATURES,
    "rate_only": ("mfr_hz", "active_electrodes"),
    "burst_only": (
        "burst_rate_per_min",
        "burst_duration_s",
        "spikes_per_burst",
        "pct_spikes_in_bursts",
    ),
    "network_only": ("network_burst_rate_per_min", "synchrony_index"),
    "synchrony_only": ("synchrony_index",),
    "mfr_only": ("mfr_hz",),
    # The features the rat -> human study found least shifted between species.
    "species_invariant": ("mfr_hz", "pct_spikes_in_bursts", "burst_rate_per_min"),
}


def _score(pairs: pd.DataFrame, features: tuple[str, ...], seed: int, n_splits: int = 4) -> dict:
    """Grouped cross-validated balanced accuracy for one feature subset."""
    counts = pairs["label"].value_counts()
    n_splits = max(2, min(n_splits, int(counts.min())))
    columns = [f"delta.{f}" for f in features]
    x = np.nan_to_num(pairs[columns].to_numpy(), nan=0.0)
    y = pairs["label"].to_numpy()
    groups = pairs["group"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in splitter.split(x, y, groups):
        fitted = MODELS["logistic_regression"].fit(x[train_idx], y[train_idx])
        scores.append(balanced_accuracy_score(y[test_idx], fitted.predict(x[test_idx])))
    return {
        "n_features": len(features),
        "features": list(features),
        "balanced_accuracy_mean": round(float(np.mean(scores)), 3),
        "balanced_accuracy_sd": round(float(np.std(scores)), 3),
    }


def ablate(pairs: pd.DataFrame, seed: int = 0) -> dict:
    """Score every named feature subset."""
    return {name: _score(pairs, features, seed) for name, features in FEATURE_SUBSETS.items()}


def leave_one_out(pairs: pd.DataFrame, seed: int = 0) -> dict:
    """Score the full vector minus each feature, against the full vector."""
    full = _score(pairs, DELTA_FEATURES, seed)["balanced_accuracy_mean"]
    out = {}
    for feature in DELTA_FEATURES:
        remaining = tuple(f for f in DELTA_FEATURES if f != feature)
        result = _score(pairs, remaining, seed)
        result["delta_vs_all"] = round(result["balanced_accuracy_mean"] - full, 3)
        out[feature] = result
    return out
