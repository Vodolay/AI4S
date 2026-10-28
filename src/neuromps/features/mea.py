"""The single MEA feature extractor: real recordings and the digital twin both call this.

Inputs are spike times per electrode, in seconds. Outputs are exactly the
`features` fields of `schema/neuromps_schema.json`. A feature that is undefined
for the input (no active electrodes, no bursts) is ``None``, never 0.0.

Definitions are chosen to be stated, not standard: burst detection and synchrony
differ between tools, so `FeatureParams.as_method_record()` emits the method
names that the schema requires alongside the values.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

SECONDS_PER_MINUTE = 60.0
BURST_RATE_DENOMINATORS = ("active", "bursting")
BURST_DETECTORS = ("fixed", "logisi")

# The feature vector, in one place. Everything downstream - the schema, the
# twin's summary vector, the report's "nine features" - is counting this tuple,
# and `extract_features` builds its result from it so the two cannot drift.
FEATURE_NAMES = (
    "mfr_hz",
    "burst_rate_per_min",
    "burst_duration_s",
    "spikes_per_burst",
    "pct_spikes_in_bursts",
    "network_burst_rate_per_min",
    "synchrony_index",
    "isi_cv",
    "active_electrodes",
)


@dataclass(frozen=True)
class FeatureParams:
    """Feature definitions. Defaults follow the EPA acute MEA assay where it defines one."""

    active_min_spikes_per_min: float = 5.0
    burst_max_isi_s: float = 0.1
    burst_min_spikes: int = 5
    network_burst_bin_s: float = 0.025
    network_burst_min_electrodes: int = 3
    network_burst_min_fraction: float = 0.25
    network_burst_false_positives: float = 0.05
    sttc_dt_s: float = 0.05
    # Alternative definitions, off by default (D33). They exist so that one
    # extractor can be run under another tool's stated settings - the post hoc
    # comparison with the Kapucu 2022 meaRtools output - without a fork.
    # `active_strict`: active means a spike count above the threshold, not at or
    # above it. `burst_rate_denominator`: bursts per minute per active electrode
    # ("active") or per electrode with at least one burst ("bursting").
    active_strict: bool = False
    burst_rate_denominator: str = "active"
    # The burst detector (D34, registered in preregistration/extractor_logisi.json):
    # "fixed" is the rule above (ISI <= burst_max_isi_s, at least burst_min_spikes);
    # "logisi" is the adaptive detector of Pasquale et al. 2010 with the depositors'
    # adjustments (see `logisi_threshold`). The logisi_* fields are read only by it.
    burst_detector: str = "fixed"
    logisi_cutoff_s: float = 0.1
    logisi_void_threshold: float = 0.7
    logisi_bin_log10: float = 0.1
    logisi_smooth_bins: int = 5
    logisi_max_threshold_s: float = 1.0
    logisi_merge_s: float = 0.1

    def __post_init__(self) -> None:
        if self.burst_rate_denominator not in BURST_RATE_DENOMINATORS:
            raise ValueError(
                f"burst_rate_denominator must be one of {BURST_RATE_DENOMINATORS}, "
                f"got {self.burst_rate_denominator!r}"
            )
        if self.burst_detector not in BURST_DETECTORS:
            raise ValueError(
                f"burst_detector must be one of {BURST_DETECTORS}, got {self.burst_detector!r}"
            )

    def as_method_record(self) -> dict[str, object]:
        """Method fields for a schema record (see STANDARD.md section 4.2).

        The alternative definitions appear only when they are not the default, so
        every record built with the defaults is unchanged.
        """
        record = self._default_method_record()
        if self.active_strict:
            record["active_electrode_threshold_strict"] = True
        if self.burst_rate_denominator != "active":
            record["burst_rate_denominator"] = self.burst_rate_denominator
        if self.burst_detector != "fixed":
            record["burst_detector"] = self.burst_detector
            record["burst_method"] = (
                f"logisi_pasquale2010(cutoff={self.logisi_cutoff_s}s,"
                f"void>{self.logisi_void_threshold},bin={self.logisi_bin_log10}log10,"
                f"smooth={self.logisi_smooth_bins}bins,ceiling={self.logisi_max_threshold_s}s,"
                f"merge<{self.logisi_merge_s}s,>={self.burst_min_spikes}spikes)"
            )
        return record

    def _default_method_record(self) -> dict[str, object]:
        """The method fields every record has carried since Phase 1."""
        return {
            "computed_by": "neuromps.features.mea",
            "burst_method": (
                f"maxinterval_simple(isi<={self.burst_max_isi_s}s,>={self.burst_min_spikes}spikes)"
            ),
            "network_burst_method": (
                f"binned_participation(bin={self.network_burst_bin_s}s,"
                f">=max({self.network_burst_min_electrodes},"
                f"{self.network_burst_min_fraction}*active,chance)electrodes,"
                f"chance_fp<{self.network_burst_false_positives}/recording)"
            ),
            "synchrony_method": "sttc",
            "active_electrode_threshold_spikes_per_min": self.active_min_spikes_per_min,
        }


def _bursts(times: np.ndarray, params: FeatureParams) -> list[tuple[int, int]]:
    """Index ranges [start, end] of bursts under ``params.burst_detector``."""
    if params.burst_detector == "logisi":
        return _logisi_bursts(times, params)
    return runs_below(times, params.burst_max_isi_s, params.burst_min_spikes)


def runs_below(times: np.ndarray, max_isi_s: float, min_spikes: int) -> list[tuple[int, int]]:
    """Index ranges [start, end] of runs of at least ``min_spikes`` spikes with ISI <= max."""
    if times.size < min_spikes:
        return []
    close = np.diff(times) <= max_isi_s
    out, start = [], None
    for i, is_close in enumerate(close):
        if is_close and start is None:
            start = i
        elif not is_close and start is not None:
            if i - start + 1 >= min_spikes:
                out.append((start, i))
            start = None
    if start is not None and times.size - start >= min_spikes:
        out.append((start, times.size - 1))
    return out


def _sttc_pair(a: np.ndarray, b: np.ndarray, duration_s: float, dt: float) -> float | None:
    """Spike time tiling coefficient (Cutts & Eglen 2014) for one electrode pair."""
    if a.size == 0 or b.size == 0:
        return None
    ta = _tiled_fraction(a, duration_s, dt)
    tb = _tiled_fraction(b, duration_s, dt)
    pa = _proportion_within(a, b, dt)
    pb = _proportion_within(b, a, dt)
    if any(np.isclose(x, 1.0) for x in (pa * tb, pb * ta)):
        return None
    return 0.5 * ((pa - tb) / (1 - pa * tb) + (pb - ta) / (1 - pb * ta))


def _tiled_fraction(times: np.ndarray, duration_s: float, dt: float) -> float:
    """Fraction of the recording covered by +/- dt tiles around spikes."""
    starts = np.maximum(times - dt, 0.0)
    ends = np.minimum(times + dt, duration_s)
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    covered, current_end = 0.0, -np.inf
    for s, e in zip(starts, ends, strict=True):
        s = max(s, current_end)
        if e > s:
            covered += e - s
            current_end = e
    return covered / duration_s


def _proportion_within(a: np.ndarray, b: np.ndarray, dt: float) -> float:
    """Fraction of spikes in ``a`` that fall within dt of any spike in ``b``."""
    idx = np.searchsorted(b, a)
    left = np.clip(idx - 1, 0, b.size - 1)
    right = np.clip(idx, 0, b.size - 1)
    nearest = np.minimum(np.abs(a - b[left]), np.abs(a - b[right]))
    return float(np.mean(nearest <= dt))


def _network_burst_rate(
    trains: list[np.ndarray], duration_s: float, params: FeatureParams
) -> float:
    """Network events per minute: contiguous bins where enough electrodes fire together.

    This is coincidence-based, in the spirit of Axion's "network spike", not a
    rate-elevation burst detector: electrodes firing synchronously but tonically
    still count. Recorded as ``binned_participation`` in the method record.
    """
    n_bins = max(int(np.ceil(duration_s / params.network_burst_bin_s)), 1)
    threshold = max(
        params.network_burst_min_electrodes,
        int(np.ceil(params.network_burst_min_fraction * len(trains))),
        _chance_threshold(trains, duration_s, n_bins, params),
    )
    if len(trains) < threshold:
        return 0.0
    participants = np.zeros(n_bins, dtype=int)
    for t in trains:
        bins = np.unique(np.clip((t / params.network_burst_bin_s).astype(int), 0, n_bins - 1))
        participants[bins] += 1
    active_bins = participants >= threshold
    starts = np.flatnonzero(active_bins & ~np.r_[False, active_bins[:-1]])
    return len(starts) / (duration_s / SECONDS_PER_MINUTE)


def _chance_threshold(
    trains: list[np.ndarray], duration_s: float, n_bins: int, params: FeatureParams
) -> int:
    """Smallest participation count whose expected chance occurrences stay below the budget.

    Without this, independent electrodes firing at a few Hz coincide often enough
    in a 25 ms bin to produce tens of spurious "network bursts" per minute.
    Electrodes are modelled as independent Poisson processes at their own rates.
    """
    n = len(trains)
    rates = np.array([t.size / duration_s for t in trains])
    p_fire = float(np.mean(1.0 - np.exp(-rates * params.network_burst_bin_s)))
    for threshold in range(2, n + 1):
        expected = n_bins * float(binom.sf(threshold - 1, n, p_fire))
        if expected < params.network_burst_false_positives:
            return threshold
    return n + 1  # not reachable by chance control alone


def extract_features(
    spikes: Mapping[str, np.ndarray], duration_s: float, params: FeatureParams | None = None
) -> dict[str, float | int | None]:
    """Compute the core feature vector from per-electrode spike times (seconds).

    Args:
        spikes: electrode id -> spike times in seconds; order is irrelevant.
        duration_s: recording duration, used for rates and tiling.
        params: feature definitions; defaults to :class:`FeatureParams`.

    Returns:
        The `features` block of the schema. Undefined features are ``None``.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    p = params or FeatureParams()
    trains = {k: np.sort(np.asarray(v, dtype=float)) for k, v in spikes.items()}
    active = _active_trains(trains.values(), duration_s, p)
    out: dict[str, float | int | None] = dict.fromkeys(
        n for n in FEATURE_NAMES if n != "active_electrodes"
    )
    out["active_electrodes"] = len(active)
    if not active:
        return out
    out["mfr_hz"] = float(np.mean([t.size / duration_s for t in active]))
    out["isi_cv"] = _mean_isi_cv(active)
    out.update(_burst_features(active, duration_s, p))
    out["network_burst_rate_per_min"] = _network_burst_rate(active, duration_s, p)
    out["synchrony_index"] = _mean_sttc(active, duration_s, p)
    return out


def _active_trains(
    trains: Iterable[np.ndarray], duration_s: float, p: FeatureParams
) -> list[np.ndarray]:
    """Electrodes at (or, if ``active_strict``, above) the active threshold, with a spike."""
    return [t for t in trains if _is_active(t, duration_s, p)]


def _is_active(t: np.ndarray, duration_s: float, p: FeatureParams) -> bool:
    """One electrode at (or, if ``active_strict``, above) the active threshold, with a spike."""
    min_spikes = p.active_min_spikes_per_min * duration_s / SECONDS_PER_MINUTE
    if p.active_strict:
        return t.size > min_spikes and t.size > 0
    return t.size >= min_spikes and t.size > 0


def logisi_thresholds(
    spikes: Mapping[str, np.ndarray], duration_s: float, params: FeatureParams
) -> list[dict[str, object]]:
    """The logISI threshold of every active electrode, as `extract_features` selects them.

    A diagnostic for the registered logISI study (D34): which electrodes found a valid
    threshold, its value, and why the others fell back to the cutoff.
    """
    trains = {k: np.sort(np.asarray(v, dtype=float)) for k, v in spikes.items()}
    return [
        {"electrode": name, "n_spikes": int(t.size), **logisi_threshold(t, params)}
        for name, t in trains.items()
        if _is_active(t, duration_s, params)
    ]


def _mean_isi_cv(active: list[np.ndarray]) -> float | None:
    """Mean over electrodes of the coefficient of variation of inter-spike intervals."""
    cvs = []
    for t in active:
        isi = np.diff(t)
        if isi.size >= 2 and isi.mean() > 0:
            cvs.append(isi.std(ddof=1) / isi.mean())
    return float(np.mean(cvs)) if cvs else None


def _burst_features(
    active: list[np.ndarray], duration_s: float, p: FeatureParams
) -> dict[str, float | None]:
    """Burst rate, duration, size and the share of spikes inside bursts."""
    n_bursts, durations, sizes, spikes_in_bursts, total, bursting = 0, [], [], 0, 0, 0
    for t in active:
        total += t.size
        found = _bursts(t, p)
        bursting += bool(found)
        for start, end in found:
            n_bursts += 1
            durations.append(t[end] - t[start])
            sizes.append(end - start + 1)
            spikes_in_bursts += end - start + 1
    denominator = len(active) if p.burst_rate_denominator == "active" else bursting
    return {
        "burst_rate_per_min": (
            n_bursts / denominator / (duration_s / SECONDS_PER_MINUTE) if denominator else None
        ),
        "burst_duration_s": float(np.mean(durations)) if durations else None,
        "spikes_per_burst": float(np.mean(sizes)) if sizes else None,
        "pct_spikes_in_bursts": 100.0 * spikes_in_bursts / total if total else None,
    }


def _mean_sttc(active: list[np.ndarray], duration_s: float, p: FeatureParams) -> float | None:
    """Mean pairwise STTC over electrode pairs."""
    values = [
        v
        for i in range(len(active))
        for j in range(i + 1, len(active))
        if (v := _sttc_pair(active[i], active[j], duration_s, p.sttc_dt_s)) is not None
    ]
    return float(np.mean(values)) if values else None


# ------------------------------------------------------------------ logISI (D34)
# The adaptive detector of Pasquale, Martinoia & Chiappalone 2010 (J Comput Neurosci
# 29:213) as the literature describes it, with the adjustments the Kapucu 2022
# depositors state. Every choice and its source is written out in
# preregistration/extractor_logisi.json (method.implementation).

LOGISI_VALID = ("valid_below_cutoff", "valid_above_cutoff")


def _log_isi_histogram(times: np.ndarray, bin_log10: float) -> tuple[np.ndarray, np.ndarray]:
    """Counts of log10(ISI) in bins on a fixed grid of multiples of ``bin_log10``, and centres.

    Zero ISIs (two spikes at one time stamp) have no logarithm and are left out.
    """
    isi = np.diff(times)
    logs = np.log10(isi[isi > 0])
    if logs.size == 0:
        return np.zeros(0), np.zeros(0)
    k = np.floor(logs / bin_log10 + 1e-9).astype(int)
    first = int(k.min())
    counts = np.bincount(k - first).astype(float)
    centres = (np.arange(counts.size) + first + 0.5) * bin_log10
    return counts, centres


def _smooth(counts: np.ndarray, bins: int) -> np.ndarray:
    """Local linear regression with tricube weights over ``bins`` neighbouring bins.

    Lowess in one pass without robustness iterations; the window shrinks at the edges.
    Negative fitted values are set to 0.
    """
    half = bins // 2
    x_all = np.arange(counts.size, dtype=float)
    out = np.empty_like(counts)
    for i in range(counts.size):
        lo, hi = max(0, i - half), min(counts.size, i + half + 1)
        x, y = x_all[lo:hi] - i, counts[lo:hi]
        w = (1.0 - (np.abs(x) / (half + 1)) ** 3) ** 3
        xm = np.sum(w * x) / np.sum(w)
        ym = np.sum(w * y) / np.sum(w)
        sxx = np.sum(w * (x - xm) ** 2)
        slope = np.sum(w * (x - xm) * (y - ym)) / sxx if sxx > 0 else 0.0
        out[i] = ym - slope * xm
    return np.clip(out, 0.0, None)


def _peaks(h: np.ndarray) -> list[int]:
    """Bins above their left neighbour and at least their right one (edges: one neighbour)."""
    found = []
    for i in range(h.size):
        left = h[i - 1] if i > 0 else -np.inf
        right = h[i + 1] if i < h.size - 1 else -np.inf
        if h[i] > 0 and h[i] > left and h[i] >= right:
            found.append(i)
    return found


def _void_minimum(h: np.ndarray, ibp: int, later: list[int], void: float) -> int | None:
    """The first minimum between the intraburst peak and a later peak with void above ``void``."""
    for p in later:
        m = ibp + 1 + int(np.argmin(h[ibp + 1 : p]))
        scale = np.sqrt(h[ibp] * h[p])
        if scale > 0 and 1.0 - h[m] / scale > void:
            return m
    return None


def _fallback(reason: str, cutoff_s: float, threshold_s: float | None = None) -> dict:
    return {
        "threshold_s": threshold_s,
        "core_s": cutoff_s,
        "extend_s": None,
        "reason": reason,
        "fallback": True,
    }


def logisi_threshold(times: np.ndarray, params: FeatureParams) -> dict[str, object]:
    """The logISI threshold of one spike train, and how bursts are then detected.

    Returns ``threshold_s`` (the ISI at the void minimum, or None), ``core_s`` (the ISI
    threshold that detects bursts or burst cores), ``extend_s`` (the ISI within which cores
    are extended, or None), ``reason`` and ``fallback`` (no valid threshold: the cutoff
    alone, which is the fixed rule).
    """
    p = params
    counts, centres = _log_isi_histogram(
        np.sort(np.asarray(times, dtype=float)), p.logisi_bin_log10
    )
    if counts.size == 0:
        return _fallback("too_few_isis", p.logisi_cutoff_s)
    h = _smooth(counts, p.logisi_smooth_bins)
    peaks = _peaks(h)
    below = [i for i in peaks if centres[i] <= np.log10(p.logisi_cutoff_s) + 1e-9]
    if not below:
        return _fallback("no_intraburst_peak", p.logisi_cutoff_s)
    ibp = max(below, key=lambda i: (h[i], -i))
    m = _void_minimum(h, ibp, [i for i in peaks if i > ibp], p.logisi_void_threshold)
    if m is None:
        return _fallback("no_void_minimum", p.logisi_cutoff_s)
    threshold = float(10.0 ** centres[m])
    if threshold > p.logisi_max_threshold_s:
        return _fallback("threshold_above_ceiling", p.logisi_cutoff_s, threshold)
    if threshold <= p.logisi_cutoff_s:
        return {
            "threshold_s": threshold,
            "core_s": threshold,
            "extend_s": None,
            "reason": "valid_below_cutoff",
            "fallback": False,
        }
    return {
        "threshold_s": threshold,
        "core_s": p.logisi_cutoff_s,
        "extend_s": threshold,
        "reason": "valid_above_cutoff",
        "fallback": False,
    }


def _extend(times: np.ndarray, bursts: list[tuple[int, int]], within_s: float) -> list:
    """Grow each burst at both edges while the next spike out is within ``within_s``."""
    out = []
    for start, end in bursts:
        while start > 0 and times[start] - times[start - 1] <= within_s:
            start -= 1
        while end < times.size - 1 and times[end + 1] - times[end] <= within_s:
            end += 1
        out.append((start, end))
    return out


def _merge(times: np.ndarray, bursts: list[tuple[int, int]], gap_s: float) -> list:
    """Join bursts that overlap or whose gap (last spike to next first spike) is under ``gap_s``."""
    out: list[tuple[int, int]] = []
    for start, end in sorted(bursts):
        if out and (start <= out[-1][1] or times[start] - times[out[-1][1]] < gap_s):
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _logisi_bursts(times: np.ndarray, p: FeatureParams) -> list[tuple[int, int]]:
    """Bursts under logISI: cores at ``core_s`` with the minimum spikes, extended, merged."""
    if times.size < p.burst_min_spikes:
        return []
    rule = logisi_threshold(times, p)
    bursts = runs_below(times, float(rule["core_s"]), p.burst_min_spikes)
    if rule["extend_s"] is not None:
        bursts = _extend(times, bursts, float(rule["extend_s"]))
    return _merge(times, bursts, p.logisi_merge_s)
