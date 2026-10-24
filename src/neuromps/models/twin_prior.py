"""A twin-trained prior for the mechanism classifier (registered test).

The baseline classifier shrinks its weights toward zero, which with 73 real
pairs is most of what it knows. Here the shrinkage target is instead the
weights the same model learns from pairs simulated by the twin, one chip
recorded twice, once with a drug knob moved. Everything else - imputation,
scaling on the real training fold, balanced class weights, C = 1, the grouped
splits and the conformal sets - is the baseline's, so any difference is the
prior's. See `preregistration/twin_prior_classifier.json`. The synthetic pairs
are **simulated**; every score is on **real** pairs.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from neuromps.models.dataset import delta_columns, log_ratio

C = 1.0


def pair_deltas(treated: dict, baseline: dict) -> dict:
    """The nine delta features of one simulated pair, by the real pairs' rule."""
    out = {}
    for column in delta_columns():
        feature = column.removeprefix("delta.")
        out[column] = log_ratio(treated.get(feature), baseline.get(feature))
    return out


class PriorLogisticRegression:
    """Multinomial logistic regression with an L2 penalty centred on a synthetic fit."""

    def __init__(self, x_prior: np.ndarray | None, y_prior: np.ndarray | None, c: float = C):
        self.x_prior, self.y_prior, self.c = x_prior, y_prior, c

    def _prior(self, d: int) -> np.ndarray:
        prior = np.zeros((len(self.classes_), d))
        if self.x_prior is None or len(self.x_prior) == 0:
            return prior
        xs = self.scaler_.transform(self.imputer_.transform(self.x_prior))
        fit = LogisticRegression(max_iter=2000, class_weight="balanced", C=self.c).fit(
            xs, self.y_prior
        )
        rows = {label: i for i, label in enumerate(fit.classes_)}
        coef = fit.coef_ if len(fit.classes_) > 2 else np.vstack([-fit.coef_, fit.coef_]) / 2
        for k, label in enumerate(self.classes_):
            if label in rows:
                prior[k] = coef[rows[label]]
        return prior

    def fit(self, x: np.ndarray, y: np.ndarray) -> PriorLogisticRegression:
        self.imputer_ = SimpleImputer(strategy="median").fit(x)
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(x))
        z = self.scaler_.transform(self.imputer_.transform(x))
        self.classes_ = np.unique(y)
        k, d = len(self.classes_), z.shape[1]
        onehot = (y[:, None] == self.classes_[None, :]).astype(float)
        weights = len(y) / (k * onehot.sum(axis=0))
        sample_w = onehot @ weights
        self.prior_coef_ = self._prior(d)
        result = minimize(
            self._objective,
            np.zeros(k * d + k),
            jac=True,
            method="L-BFGS-B",
            args=(z, onehot, sample_w, k, d),
            options={"maxiter": 5000},
        )
        self.coef_ = result.x[: k * d].reshape(k, d)
        self.intercept_ = result.x[k * d :]
        return self

    def _objective(self, params, z, onehot, sample_w, k, d):
        w, b = params[: k * d].reshape(k, d), params[k * d :]
        logits = z @ w.T + b
        log_p = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -np.sum(sample_w * np.sum(onehot * log_p, axis=1))
        shift = w - self.prior_coef_
        loss += 0.5 / self.c * np.sum(shift**2)
        grad_logits = (np.exp(log_p) - onehot) * sample_w[:, None]
        grad_w = grad_logits.T @ z + shift / self.c
        return loss, np.concatenate([grad_w.ravel(), grad_logits.sum(axis=0)])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler_.transform(self.imputer_.transform(x))
        logits = z @ self.coef_.T + self.intercept_
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


REGISTRATION = "preregistration/twin_prior_classifier.json"
KNOB_RANGES = {
    "control": None,
    "gabaa_antagonist": ("gabaa_scale", 0.1, 0.6),
    "gabaa_agonist_or_pam": ("gabaa_scale", 1.5, 3.0),
    "glutamatergic_antagonist": ("ampa_scale", 0.2, 0.7),
    "na_channel_blocker": ("threshold_scale", 1.2, 2.0),
}
PAIRS_PER_CLASS = 30
DURATION_S = 40.0


def _features(knobs: dict, build_seed: int, noise_seed: int) -> dict:
    from neuromps.twin.calibrate import calibrated_params
    from neuromps.twin.mea import ElectrodeLayout
    from neuromps.twin.network import NetworkParams
    from neuromps.twin.perturbation import _features_for

    tuned = calibrated_params()
    network_keys = set(NetworkParams().as_record())
    layout = ElectrodeLayout(
        **{k: v for k, v in tuned.items() if k in set(ElectrodeLayout().as_record())}
    )
    kwargs = {
        "n_per_compartment": 250,
        "duration_s": DURATION_S,
        "seed": build_seed,
        **{k: v for k, v in tuned.items() if k in network_keys},
        "noise_seed": noise_seed,
        **knobs,
    }
    return _features_for(NetworkParams(**kwargs), layout, build_seed)


def _pair(job) -> dict:
    label, knobs, k = job
    baseline = _features({}, k, 1000 + k)
    treated = _features(knobs, k, 2000 + k)
    return {"label": label, "build_seed": k, **knobs, **pair_deltas(treated, baseline)}


def synthetic_jobs(n_per_class: int = PAIRS_PER_CLASS, seed: int = 0) -> list:
    """(label, knob overrides, build seed) for every synthetic pair, knobs drawn once."""
    rng = np.random.default_rng(seed)
    out = []
    for label, knob in KNOB_RANGES.items():
        for k in range(n_per_class):
            knobs = {} if knob is None else {knob[0]: float(rng.uniform(knob[1], knob[2]))}
            out.append((label, knobs, k))
    return out


def simulate_pairs(processes: int):
    """The registered synthetic training pairs, simulated in parallel."""
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    with ProcessPoolExecutor(processes) as pool:
        return pd.DataFrame(list(pool.map(_pair, synthetic_jobs())))


def evaluate(pairs, synthetic, n_splits: int, seed: int = 0) -> dict:
    """Baseline and twin-prior classifiers on the same grouped splits and conformal sets."""
    from sklearn.metrics import balanced_accuracy_score

    from neuromps.models.conformal import conformal_sets
    from neuromps.models.mechanism import MODELS, grouped_scores

    xs = synthetic[delta_columns()].to_numpy(dtype=float)
    ys = synthetic["label"].to_numpy()
    prior = _Factory(xs, ys)
    out = {}
    for name, model, factory in (
        ("baseline", MODELS["logistic_regression"], None),
        ("twin_prior", prior(), prior),
    ):
        scores = grouped_scores(pairs, model, n_splits=n_splits, seed=seed)
        sets = conformal_sets(pairs, alpha=0.1, seed=seed, make_model=factory)
        out[name] = {
            "balanced_accuracy_mean": round(float(np.mean(scores["balanced_accuracy"])), 3),
            "per_fold": [round(s, 3) for s in scores["balanced_accuracy"]],
            "conformal_alpha_0.1": {
                k: sets[k] for k in ("mean_set_size", "empirical_coverage", "fraction_singleton")
            },
        }
    alone = PriorLogisticRegression(None, None).fit(xs, ys)
    real_x = pairs[delta_columns()].to_numpy(dtype=float)
    out["twin_alone"] = {
        "balanced_accuracy": round(
            float(balanced_accuracy_score(pairs["label"].to_numpy(), alone.predict(real_x))), 3
        )
    }
    return out


class _Factory:
    """A picklable maker of fresh prior classifiers, for the conformal splits."""

    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys

    def __call__(self) -> PriorLogisticRegression:
        return PriorLogisticRegression(self.xs, self.ys)


def main(argv: list[str] | None = None) -> int:
    """CLI: simulate the synthetic pairs, evaluate, score the registered predictions."""
    import json
    import logging

    import pandas as pd

    from neuromps.metrics import update_metrics
    from neuromps.models.dataset import load_pairs
    from neuromps.parallel import worker_processes
    from neuromps.paths import REPO_ROOT, twin_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cache = twin_dir() / "twin_prior_pairs_simulated.csv"
    synthetic = pd.read_csv(cache) if cache.exists() else simulate_pairs(worker_processes(cap=18))
    synthetic.to_csv(cache, index=False)
    pairs = load_pairs()
    pairs = pairs[pairs["source"] == "kapucu2022"].reset_index(drop=True)
    metrics = json.loads((REPO_ROOT / "outputs" / "metrics.json").read_text())
    n_splits = metrics["mechanism_baselines"]["task1_within_platform"]["n_splits"]
    got = evaluate(pairs, synthetic, n_splits)
    got["twin_alone"].update(twin_alone_detail(pairs, synthetic))
    spec = json.loads((REPO_ROOT / REGISTRATION).read_text())["predictions"]
    base, prior = got["baseline"], got["twin_prior"]
    checks = {
        "W1_accuracy_improves": (
            {
                "twin_prior": prior["balanced_accuracy_mean"],
                "baseline": base["balanced_accuracy_mean"],
            },
            prior["balanced_accuracy_mean"] >= base["balanced_accuracy_mean"] + 0.05,
        ),
        "W2_sets_shrink": (
            {"twin_prior": prior["conformal_alpha_0.1"], "baseline": base["conformal_alpha_0.1"]},
            prior["conformal_alpha_0.1"]["mean_set_size"]
            < base["conformal_alpha_0.1"]["mean_set_size"]
            and prior["conformal_alpha_0.1"]["empirical_coverage"] >= 0.85,
        ),
        "W3_twin_alone_transfers": (
            got["twin_alone"]["balanced_accuracy"],
            got["twin_alone"]["balanced_accuracy"] > 0.25,
        ),
    }
    result = {
        "label": "SIMULATED twin prior, scored on REAL Kapucu pairs",
        **got,
        "n_synthetic_pairs": int(len(synthetic)),
        "n_real_pairs": int(len(pairs)),
        "predictions": {
            k: {"test": spec[k]["test"], "value": v, "outcome": "pass" if ok else "fail"}
            for k, (v, ok) in checks.items()
        },
    }
    update_metrics("twin_prior_classifier", result)
    for name, entry in result["predictions"].items():
        print(f"{name:28s} {entry['outcome']:>5}  {entry['value']}")
    return 0


def twin_alone_detail(pairs, synthetic, n_resamples: int = 1000, seed: int = 0) -> dict:
    """Exploratory: per-class recall of the twin-only classifier, with a culture-bootstrap CI."""
    from neuromps.models.mechanism import balanced_accuracy_ci

    xs = synthetic[delta_columns()].to_numpy(dtype=float)
    model = PriorLogisticRegression(None, None).fit(xs, synthetic["label"].to_numpy())
    truth = pairs["label"].to_numpy()
    predicted = model.predict(pairs[delta_columns()].to_numpy(dtype=float)).astype(str)
    interval = balanced_accuracy_ci(truth, predicted, pairs["group"].to_numpy(), n_resamples, seed)
    recall = {
        str(c): round(float(np.mean(predicted[truth == c] == c)), 3) for c in np.unique(truth)
    }
    return {
        "balanced_accuracy_ci95": interval["ci95"],
        "recall_by_class": recall,
        "bootstrap_resamples": interval["n_resamples"],
        "classes_without_twin_knob": ["glutamatergic_agonist"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
