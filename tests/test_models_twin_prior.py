"""The prior-centred logistic regression: zero prior is the baseline, a strong prior wins."""

from __future__ import annotations

import numpy as np
import pytest

from neuromps.models import twin_prior


def _data(seed: int = 0, n: int = 60):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.array(["a", "b", "c"]), n // 3)
    centres = {"a": [2, 0, 0], "b": [0, 2, 0], "c": [0, 0, 2]}
    x = np.array([centres[label] for label in y], dtype=float) + rng.normal(0, 1.0, (n, 3))
    return x, y


def test_with_no_synthetic_data_it_is_the_baseline_classifier() -> None:
    from neuromps.models.mechanism import MODELS

    x, y = _data()
    ours = twin_prior.PriorLogisticRegression(x_prior=None, y_prior=None).fit(x, y)
    base = MODELS["logistic_regression"].fit(x, y)
    assert list(ours.classes_) == list(base.classes_)
    assert np.allclose(ours.predict_proba(x), base.predict_proba(x), atol=2e-3)


def test_a_prior_pulls_the_weights_toward_the_synthetic_fit() -> None:
    x, y = _data(n=12)
    xs, ys = _data(seed=5, n=300)
    weak = twin_prior.PriorLogisticRegression(x_prior=None, y_prior=None).fit(x, y)
    pulled = twin_prior.PriorLogisticRegression(x_prior=xs, y_prior=ys).fit(x, y)
    assert np.linalg.norm(pulled.coef_ - pulled.prior_coef_) < np.linalg.norm(
        weak.coef_ - pulled.prior_coef_
    )


def test_a_class_without_synthetic_pairs_is_shrunk_toward_zero() -> None:
    x, y = _data()
    xs, ys = _data(seed=5, n=300)
    keep = ys != "c"
    model = twin_prior.PriorLogisticRegression(x_prior=xs[keep], y_prior=ys[keep]).fit(x, y)
    row = list(model.classes_).index("c")
    assert np.allclose(model.prior_coef_[row], 0.0)


def test_synthetic_pairs_use_the_real_log_ratio_rule() -> None:
    base = {"mfr_hz": 4.0, "burst_rate_per_min": 0.0}
    treated = {"mfr_hz": 8.0, "burst_rate_per_min": 0.0}
    got = twin_prior.pair_deltas(treated, base)
    assert got["delta.mfr_hz"] == pytest.approx(1.0)
