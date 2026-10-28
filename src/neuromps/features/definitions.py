"""Named definition sets: one MEA feature, more than one stated definition of it.

A feature name does not fix a number. "Burst rate" from two laboratories can differ
twofold from the definitions alone (report §3). This module names each set of
definitions we can compute, so a record's features can be read under ours and, beside
them, under another laboratory's - and a reader can see where a translation between
the two is possible.

A definition set is **not** a second extractor. It is a named bundle of the options
`features/mea.py` already has (`FeatureParams`) plus one electrode policy: whether the
electrodes a deposit lists as noisy are dropped before extraction. `extract` calls the
one extractor, so the twin and the asset keep a single code path, and the reference set
is exactly the defaults every record in the asset was built with (decision D37).

Statuses: ``reference`` is what the asset uses; ``validated`` reproduces another
implementation's published output on real recordings, with the evidence named by its
metrics key; ``experimental`` is computed and shown but used for no claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType

import numpy as np

from neuromps.features.mea import FeatureParams, extract_features

STATUSES = ("reference", "validated", "experimental")
REFERENCE = "neuromps"
KAPUCU_METHODS = "Kapucu et al. 2022, Methods (data/raw/kapucu2022/paper/PMC8969177.xml)"


@dataclass(frozen=True)
class DefinitionSet:
    """One named set of feature definitions.

    Attributes:
        name: the registry key, quoted in tables and in the sidecar's ``definition`` column.
        description: what the set is, in a sentence or two, including its limits.
        params: the shared extractor's options, passed to it unchanged.
        drop_noisy: drop a deposit's listed noisy electrodes before extraction.
        sources: (choice, where the choice comes from) pairs, one per stated choice.
        status: one of :data:`STATUSES`.
        evidence: the ``outputs/metrics.json`` key of the validation evidence, if any.
    """

    name: str
    description: str
    params: FeatureParams
    drop_noisy: bool
    sources: tuple[tuple[str, str], ...]
    status: str
    evidence: str | None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")

    @property
    def choice_sources(self) -> dict[str, str]:
        """Choice -> where it comes from."""
        return dict(self.sources)


_KAPUCU_PARAMS = FeatureParams(
    active_min_spikes_per_min=10.0, active_strict=True, burst_rate_denominator="bursting"
)
_KAPUCU_SOURCES = (
    ("active_threshold", f"more than 10 spikes/min; {KAPUCU_METHODS}"),
    ("noisy_electrodes", f"the deposit's noisy-electrode lists are discarded; {KAPUCU_METHODS}"),
    ("burst_rate_denominator", f"electrodes exhibiting bursts; {KAPUCU_METHODS}"),
)

_SETS = (
    DefinitionSet(
        name="neuromps",
        description=(
            "The asset's own definitions, the FeatureParams defaults every record and the twin "
            "use: active at 5 or more spikes/min, noisy electrodes flagged and kept, fixed burst "
            "rule (ISI <= 100 ms, at least 5 spikes), burst rate per active electrode."
        ),
        params=FeatureParams(),
        drop_noisy=False,
        sources=(
            ("active_threshold", "EPA acute MEA assay documents, section 2.5 (decision D33)"),
            ("noisy_electrodes", "flagged, not dropped: schema/STANDARD.md section 5 (D33)"),
            ("burst_rate_denominator", "every active electrode (D33)"),
            ("burst_detector", "fixed rule, ISI <= 100 ms and at least 5 spikes (D33)"),
        ),
        status="reference",
        evidence=None,
    ),
    DefinitionSet(
        name="kapucu2022",
        description=(
            "The Kapucu 2022 depositors' stated settings that change our numbers, with our fixed "
            "burst rule: the settings ladder's S3. Active above 10 spikes/min, the deposit's noisy "
            "electrodes dropped where a list exists, burst rate per bursting electrode. On their "
            "developmental plates it reproduces their firing rate, active-electrode count and STTC "
            "and most of their burst rate; burst duration stays short of theirs (post hoc)."
        ),
        params=_KAPUCU_PARAMS,
        drop_noisy=True,
        sources=(
            *_KAPUCU_SOURCES,
            ("burst_detector", "ours, not theirs: fixed rule, ISI <= 100 ms (D33)"),
        ),
        status="validated",
        evidence="extractor_agreement.settings_ladder",
    ),
    DefinitionSet(
        name="kapucu2022_logisi",
        description=(
            "kapucu2022 with the depositors' adaptive logISI burst detector (the ladder's S4, "
            "registered as D34). 1 of 4 registered predictions passed: burst duration and spikes "
            "per burst reach their scale but not their ordering, and the hPSC burst rate "
            "overshoots. Whether the detector or our implementation of it is the cause is open, "
            "so this set is used for no claim."
        ),
        params=replace(_KAPUCU_PARAMS, burst_detector="logisi"),
        drop_noisy=True,
        sources=(
            *_KAPUCU_SOURCES,
            (
                "burst_detector",
                f"logISI, Pasquale et al. 2010, as the depositors state; {KAPUCU_METHODS}",
            ),
        ),
        status="experimental",
        evidence="extractor_agreement.logisi",
    ),
)
REGISTRY: Mapping[str, DefinitionSet] = MappingProxyType({d.name: d for d in _SETS})


def get(name: str) -> DefinitionSet:
    """The definition set called ``name``."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown definition set {name!r}; known: {sorted(REGISTRY)}") from None


def extract(
    definition: DefinitionSet,
    spikes: Mapping[str, np.ndarray],
    duration_s: float,
    noisy: Iterable[str] = (),
) -> dict[str, float | int | None]:
    """The feature block of one recording under one definition set.

    Args:
        definition: the set to apply.
        spikes: electrode id -> spike times in seconds.
        duration_s: recording duration in seconds.
        noisy: the deposit's noisy electrodes for this recording; dropped before extraction
            only when ``definition.drop_noisy``. Ids absent from ``spikes`` are ignored.

    Returns:
        What `features.mea.extract_features` returns for the kept electrodes.
    """
    if definition.drop_noisy:
        drop = set(noisy)
        spikes = {name: times for name, times in spikes.items() if name not in drop}
    return extract_features(spikes, duration_s, definition.params)


def changes_from_reference(definition: DefinitionSet) -> list[str]:
    """The extractor options, then the electrode policy, that differ from the reference set."""
    base = REGISTRY[REFERENCE]
    changed = [
        f.name
        for f in fields(FeatureParams)
        if getattr(base.params, f.name) != getattr(definition.params, f.name)
    ]
    return changed + (["drop_noisy"] if base.drop_noisy != definition.drop_noisy else [])


def registry_summary() -> dict[str, dict]:
    """Each set's status, evidence, sources and settings, for metrics.json."""
    return {
        d.name: {
            "status": d.status,
            "description": d.description,
            "evidence": d.evidence,
            "drop_noisy": d.drop_noisy,
            "changes_from_reference": changes_from_reference(d),
            "choice_sources": d.choice_sources,
            "method_record": d.params.as_method_record(),
        }
        for d in REGISTRY.values()
    }
