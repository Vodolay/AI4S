"""Can region-to-region direction be read off activity, where the wiring is known?

The chip's directionality measure is the cross-correlation asymmetry of
`features/coupling.py`. Here it is scored against the connectome itself: for
every pair of brain regions joined by enough synapses, the true asymmetry is
(A->B - B->A) / (A->B + B->A) in synapse counts. Two ways of measuring are
compared on the same pairs: spontaneous cross-correlation, as the chip assay
does, and pulsed stimulation of one region with the response read in an early
window after each pulse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neuromps.features.coupling import directed_coupling


def pair_truth(flow: pd.Series, regions: list[str], min_synapses: int) -> pd.DataFrame:
    """Unordered region pairs (a < b) with their synapse-count asymmetry."""
    rows = []
    names = sorted(regions)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ab, ba = int(flow.get((a, b), 0)), int(flow.get((b, a), 0))
            if ab + ba >= min_synapses:
                rows.append(
                    {"a": a, "b": b, "ab": ab, "ba": ba, "asymmetry": (ab - ba) / (ab + ba)}
                )
    return pd.DataFrame(rows, columns=["a", "b", "ab", "ba", "asymmetry"])


def score(pairs: pd.DataFrame, measured: np.ndarray, strong: float) -> dict:
    """Rank correlation and sign agreement of measured against true asymmetry."""
    truth = pairs["asymmetry"].to_numpy(dtype=float)
    measured = np.asarray(measured, dtype=float)
    ok = np.isfinite(measured)
    truth, measured = truth[ok], measured[ok]
    agree = np.sign(truth) == np.sign(measured)
    big = np.abs(truth) >= strong
    rho, p = spearmanr(truth, measured) if truth.size > 2 else (np.nan, np.nan)
    return {
        "n_pairs": int(truth.size),
        "n_strong": int(big.sum()),
        "spearman": None if not np.isfinite(rho) else round(float(rho), 4),
        "spearman_p": None if not np.isfinite(p) else float(p),
        "sign_agreement": round(float(agree.mean()), 4) if truth.size else None,
        "sign_agreement_strong": round(float(agree[big].mean()), 4) if big.any() else None,
    }


def spontaneous_asymmetry(trains: dict[str, dict], pairs: pd.DataFrame, duration_s: float):
    """Cross-correlation asymmetry a -> b for each pair (NaN where it is undefined)."""
    out = []
    for a, b in pairs[["a", "b"]].itertuples(index=False):
        got = directed_coupling(trains.get(a, {}), trains.get(b, {}), duration_s)["asymmetry"]
        out.append(np.nan if got is None else got)
    return np.asarray(out, dtype=float)


def evoked_log_ratio(
    times: np.ndarray,
    regions: np.ndarray,
    n_regions: int,
    duration_s: float,
    period_ms: float,
    window_ms: tuple[float, float],
    reference_ms: tuple[float, float],
) -> np.ndarray:
    """Per region: log2 of spikes in the post-pulse window over the pre-pulse window."""
    period = period_ms / 1000
    onsets = np.arange(period, duration_s - period / 2, period)  # skip the first pulse
    post = np.zeros(n_regions)
    pre = np.zeros(n_regions)
    for counts, (lo, hi) in ((post, window_ms), (pre, reference_ms)):
        for t0 in onsets:
            sel = (times >= t0 + lo / 1000) & (times < t0 + hi / 1000)
            counts += np.bincount(regions[sel], minlength=n_regions)
    return np.log2((post + 1.0) / (pre + 1.0))


def evoked_asymmetry(response: dict[str, dict[str, float]], pairs: pd.DataFrame) -> np.ndarray:
    """Driving a moves b more than driving b moves a: response[a][b] - response[b][a]."""
    return np.array(
        [
            response.get(a, {}).get(b, np.nan) - response.get(b, {}).get(a, np.nan)
            for a, b in pairs[["a", "b"]].itertuples(index=False)
        ]
    )
