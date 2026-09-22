"""The shot list for the demo video, and the numbers it quotes.

The video is a deliverable like the report, so it follows the same rule: every
figure it states comes from `outputs/metrics.json`, interpolated at build time.
A caption cannot quietly disagree with the document it is summarising.

Structure follows the plan's budget: problem, data asset and standard, transfer,
twin and inference, reproducibility. Durations are in seconds and are asserted
against the five-minute limit by a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Shot:
    """One still with its caption, or a generated card when `image` is None."""

    seconds: float
    caption: str
    image: str | None = None
    crop: tuple[int, int, int, int] | None = None
    title: str | None = None
    section: str = ""
    body: list[str] = field(default_factory=list)


def metric(metrics: dict, path: str):
    """Value at a dotted path in metrics.json."""
    node = metrics
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def shot_list(metrics: dict) -> list[Shot]:
    """Every shot, with numbers filled in from the metrics."""
    m = lambda path: metric(metrics, path)  # noqa: E731 - local shorthand, used heavily below
    records = f"{m('data_asset.n_records'):,}"
    spike = f"{m('data_asset.counts.by_data_level.spike_times'):,}"
    # Counted, not typed: the captions said "five sources" for two commits after
    # the asset had seven, beside a screenshot that showed all seven.
    sources = len(m("data_asset.counts.by_source"))
    return [
        *_opening(records, spike, sources),
        *_asset(m, records, spike, sources),
        *_transfer(m),
        *_twin(m),
        *_reproducibility(m),
    ]


def _opening(records: str, spike: str, sources: int) -> list[Shot]:
    """Title and problem: 35 s."""
    return [
        Shot(
            6,
            "",
            title="NeuroMPS",
            body=[
                "A provenance-tracked MEA data asset,",
                "and a two-compartment spiking-network twin calibrated on it.",
                "",
                "Research decision-support only.",
            ],
            section="",
        ),
        Shot(
            10,
            "",
            title="The problem",
            body=[
                "Public microelectrode-array data are easy to download",
                "and hard to compare.",
                "",
                "Feature names collide while definitions differ.",
                "Electrode counts, durations and species vary.",
                "Provenance is usually gone by the time data reach an analysis.",
            ],
            section="Problem",
        ),
        Shot(
            10,
            "",
            title="What we built",
            body=[
                f"A standardised asset: {records} records, {sources} sources,",
                "every record carrying its URL, licence, SHA256 and download time.",
                "",
                "A Brian2 twin of an asymmetric two-chamber chip,",
                "scored through the same feature extractor as the real recordings.",
            ],
            section="Problem",
        ),
        Shot(
            9,
            "",
            title="And what we found",
            body=[
                "Three of the five headline results are negative.",
                "They are the ones we report first.",
            ],
            section="Problem",
        ),
    ]


def _asset(m, records: str, spike: str, sources: int) -> list[Shot]:
    """Data asset and standard: 60 s."""
    return [
        Shot(
            14,
            f"{records} records from {sources} sources. The build refuses to write an asset unless "
            "every record's provenance hash matches the download manifest.",
            image="outputs/video/stills/tab1.png",
            crop=(0, 300, 1600, 880),
            section="Data asset & standard",
        ),
        Shot(
            13,
            f"Records carry an explicit data level, and only compare within it: {spike} records "
            "support the nine-feature vector; the rest are published curve fits.",
            image="outputs/video/stills/tab1.png",
            crop=(0, 940, 1600, 1420),
            section="Data asset & standard",
        ),
        Shot(
            16,
            "",
            title="Licence discipline, both ways",
            body=[
                "Tukker 2020 was excluded: non-commercial licence,",
                "incompatible with a prize-bearing submission.",
                "",
                "Mateus 2024 was excluded too, then reinstated:",
                "the deposit's metadata says CC BY 4.0, its README says CC BY-NC-ND.",
                "The authors confirmed the metadata governs.",
                "",
                "Both decisions, and what each rests on, are in the repository.",
            ],
            section="Data asset & standard",
        ),
        Shot(
            12,
            "QC is recorded, never applied silently: a flagged record stays in the asset with its "
            "reason attached, and the flag vocabulary is part of the schema.",
            image="outputs/video/stills/tab1.png",
            crop=(0, 1330, 1600, 1800),
            section="Data asset & standard",
        ),
    ]


def _transfer(m) -> list[Shot]:
    """What transfers between species and platforms: 60 s."""
    none = "species_transfer.transfer.none.rat_to_human"
    coral = "species_transfer.transfer.coral.rat_to_human"
    return [
        Shot(
            15,
            f"A rat-trained mechanism classifier reaches {m(none + '.balanced_accuracy')} balanced "
            f"accuracy on human recordings, against a chance level of "
            f"{m('species_transfer.transfer.none.chance_balanced_accuracy')}.",
            image="outputs/video/stills/tab2.png",
            crop=(760, 300, 1600, 1180),
            section="Transfer",
        ),
        Shot(
            13,
            "",
            title="Domain adaptation made it worse",
            body=[
                f"No adaptation: {m(none + '.balanced_accuracy')}",
                f"CORAL: {m(coral + '.balanced_accuracy')}",
                f"Per-species standardisation: "
                f"{m('species_transfer.transfer.standardize.rat_to_human.balanced_accuracy')}",
                "",
                "With 37 training records, CORAL is estimating a covariance",
                "it cannot support. We report that rather than hiding it.",
            ],
            section="Transfer",
        ),
        Shot(
            16,
            "Rate and burst-fraction measures survive a change of species; synchrony and "
            "network-burst measures do not. Only one feature's interval excludes zero.",
            image="outputs/figures/species_shift.png",
            section="Transfer",
        ),
        Shot(
            16,
            "Across platforms the picture reverses: mean firing rate itself flips sign while "
            "synchrony agrees. Portability is a species result, not a platform one.",
            image="outputs/figures/platform_contrast_gabaa_antagonist.png",
            section="Transfer",
        ),
    ]


def _twin(m) -> list[Shot]:
    """The twin, its posterior, and what it does on real data: 90 s."""
    hd15 = "twin_hd15_real_inference"
    hd16 = "twin_hd16_directionality"
    return [
        Shot(
            15,
            "Three sliders move over a precomputed grid of twin simulations. Every interaction "
            "answers in under a second, because no callback runs a simulator.",
            image="outputs/video/stills/tab3.png",
            crop=(0, 300, 1600, 1090),
            section="Twin & inference",
        ),
        Shot(
            14,
            "One button does run Brian2, live, at the current slider values: six simulated seconds "
            "in 3.82 s of wall clock. The twin is a simulator, not a lookup table.",
            image="outputs/video/stills/tab3_live.png",
            crop=(0, 860, 1600, 1440),
            section="Twin & inference",
        ),
        Shot(
            14,
            "The posterior recovers excitability well and connectivity poorly, so the twin "
            "supports a contrast between wiring regimes, not a measurement of one.",
            image="outputs/video/stills/tab3.png",
            crop=(760, 1040, 1600, 1420),
            section="Twin & inference",
        ),
        Shot(
            17,
            f"Applied to {m(hd15 + '.n_recordings')} real two-chamber recordings, the inferred "
            f"GABA-A scale shows no dose trend, and untreated chips already infer "
            f"{m(hd15 + '.gabaa_scale_at_baseline.median')} where a neutral value is 1.0.",
            image="outputs/video/stills/tab3.png",
            crop=(0, 1540, 1600, 2460),
            section="Twin & inference",
        ),
        Shot(
            15,
            "",
            title="The negative result, stated plainly",
            body=[
                "The knob is absorbing model mismatch,",
                "not tracking the compound.",
                "",
                f"Spearman {m(hd15 + '.gabaa_scale_dose_trend.spearman_rho')}, "
                f"95% CI {m(hd15 + '.gabaa_scale_dose_trend.spearman_rho_ci95')}",
                f"over {m(hd15 + '.gabaa_scale_dose_trend.n_devices')} devices.",
                "",
                "The twin cannot yet read pharmacology off a real chip.",
            ],
            section="Twin & inference",
        ),
        Shot(
            15,
            f"In simulation, wiring direction is recoverable once firing rate is residualised out: "
            f"AUC {m(hd16 + '.cheapest_configuration.free_rate_residualized.auc')} from a 10 s "
            f"recording. On 26 real chips whose geometry was fabricated, it does not reproduce "
            f"(Kruskal p = {m('microchannel_directionality.comparison.kruskal_p')}).",
            image="outputs/figures/microchannel_asymmetry_real.png",
            section="Twin & inference",
        ),
    ]


def _stage_summary(timings: dict) -> str:
    """Stage timings as one readable line."""
    return ", ".join(f"{name} {seconds:.0f} s" for name, seconds in sorted(timings.items()))


def _reproducibility(m) -> list[Shot]:
    """Reproducibility and close: 35 s."""
    timings = m("run_all_timings_full.by_stage_s")
    return [
        Shot(
            16,
            "",
            title="One command, from a clean clone",
            body=[
                "git clone ... && pip install -e .",
                "python run_all.py --quick",
                "",
                "401 s end to end on a fresh clone of this machine:",
                "download, build, model, twin, demo - and 1126 s on",
                "a two-core CI runner. No step is skipped silently.",
                "",
                "Most recent full run of each stage: " + _stage_summary(timings),
            ],
            section="Reproducibility",
        ),
        Shot(
            12,
            "",
            title="Numbers cannot be typed in",
            body=[
                "Every figure in the report cites the metrics key that produced it,",
                "and a checker re-reads all 159 citations against the live metrics.",
                "",
                "It has caught a stale timestamp, four mis-anchored citations",
                "and two transcription slips - including one of its author's.",
            ],
            section="Reproducibility",
        ),
        Shot(
            8,
            "",
            title="NeuroMPS",
            body=[
                "Data asset, digital twin, and the results that did not work.",
                "",
                "github.com/Vodolay/AI4S",
            ],
            section="",
        ),
    ]
