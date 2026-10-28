"""Named definition sets (D37): the sidecar table, its checks, and where translation is possible.

Three things, one stage:

1. **The sidecar.** Every spike-time record of the asset, re-read with the ETL's own
   loaders (`etl.spike_trains`) and extracted under every set in
   `features/definitions.py`, written beside the asset as
   ``features_by_definition.parquet`` (`etl.definition_sidecar`).
2. **The checks.** The ``neuromps`` rows equal the asset's own features exactly; and on
   the Kapucu 2022 developmental plates, each set's median ratios against the
   depositors' meaRtools tables equal the numbers the settings ladder (S0, S3) and the
   registered logISI test (S4) already published.
3. **The translation table.** On the same plates - the only recordings where a
   depositor's own output exists to say what "their number" is - how a value under our
   definitions maps onto the same value under theirs: median ratio, Spearman rho, share
   within 10%, and a verdict per feature: "same number", "scale factor" or "no conversion".

This is an engineering stage, not a registered study: it computes no new comparison
with the depositors, only re-expresses the ladder's and the logISI test's evidence per
pair of definition sets. An ``experimental`` set is shown and used for no claim.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuromps.features.definitions import REFERENCE, REGISTRY, registry_summary
from neuromps.models.extractor_agreement import (
    DATA_ROOT,
    PLATES,
    RANK_AGREEMENT,
    SCALE_TOLERANCE,
    UNCHECKABLE,
    Definition,
    _ratio,
    paired_table,
)
from neuromps.models.extractor_ladder import ladder_stats

log = logging.getLogger("neuromps.models")

LABEL = (
    "REAL recordings: the asset's spike-time records under every named definition set "
    "(sidecar), and on the Kapucu 2022 developmental plates, where the depositors' meaRtools "
    "output exists, how our numbers translate into theirs; engineering (D37), not registered"
)
PAIRS = (("neuromps", "kapucu2022"), ("neuromps", "kapucu2022_logisi"))
VERDICTS = ("same number", "scale factor", "no conversion")
WITHIN = 0.10
# Where each set's agreement with the depositors was already published.
PUBLISHED = {
    "neuromps": ("settings_ladder", "S0_ours"),
    "kapucu2022": ("settings_ladder", "S3_rate_per_bursting_electrode"),
    "kapucu2022_logisi": ("logisi", "S4_logisi"),
}
TABLE_NAME = "feature_definitions.md"


def agreement_definitions() -> tuple[Definition, ...]:
    """The registry as the agreement study's definitions: noisy lists read per DIV."""
    return tuple(
        Definition(d.name, d.params, noisy="dropped_per_div" if d.drop_noisy else "kept")
        for d in REGISTRY.values()
    )


def verdict(median_ratio: float | None, rho: float | None) -> str:
    """Same number, scale factor, or no conversion, by the agreement study's thresholds."""
    if rho is None or rho < RANK_AGREEMENT:
        return "no conversion"
    if median_ratio is not None and abs(median_ratio - 1.0) <= SCALE_TOLERANCE:
        return "same number"
    return "scale factor"


def _rho(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return round(float(spearmanr(a, b).statistic), 4)


def pair_stats(reference: np.ndarray, target: np.ndarray) -> dict:
    """Target over reference on the same recordings: ratio, rank agreement, share within 10%."""
    ratios = _ratio(target, reference)
    median = round(float(np.median(ratios)), 4) if ratios.size else None
    rho = _rho(reference, target)
    within = round(float(np.mean(np.abs(ratios - 1.0) <= WITHIN)), 4) if ratios.size else None
    return {
        "n": int(len(reference)),
        "median_ratio": median,
        "spearman_rho": rho,
        "fraction_within_10pct": within,
        "verdict": verdict(median, rho),
    }


def _values(table: pd.DataFrame, name: str) -> pd.Series:
    keys = ["plate", "well", "div", "feature"]
    return table[table["variant"] == name].set_index(keys)["ours"]


def translation(table: pd.DataFrame) -> dict:
    """Pair -> feature -> statistics, on recordings both sets and the depositors define."""
    pooled = ladder_stats(table)
    out: dict[str, dict] = {}
    for source, target in PAIRS:
        both = pd.concat(
            {"a": _values(table, source), "b": _values(table, target)}, axis=1, join="inner"
        )
        entry = {}
        for feature, rows in both.groupby(level="feature"):
            stats = pair_stats(rows["a"].to_numpy(dtype=float), rows["b"].to_numpy(dtype=float))
            stats["median_ratio_by_plate"] = by_plate(rows)
            theirs = pooled.get(target, {}).get("pooled", {}).get(str(feature), {})
            stats["target_vs_depositors"] = {
                k: theirs.get(k) for k in ("median_ratio", "spearman_rho")
            }
            entry[str(feature)] = stats
        out[f"{source}_to_{target}"] = entry
    return out


def by_plate(rows: pd.DataFrame) -> dict[str, float | None]:
    """Median target/reference ratio per plate: the check that a pooled factor is one factor.

    A pooled ratio is a property of these recordings, not of the two definitions: where
    the plates disagree (hPSC against rat), a single factor would mislead on either.
    """
    out = {}
    for plate, subset in rows.groupby(level="plate"):
        ratios = _ratio(subset["b"].to_numpy(dtype=float), subset["a"].to_numpy(dtype=float))
        out[str(plate)] = round(float(np.median(ratios)), 4) if ratios.size else None
    return out


def verdicts(pairs: dict) -> dict:
    """Pair -> verdict -> features, for the summary line."""
    return {
        pair: {v: sorted(f for f, s in by_feature.items() if s["verdict"] == v) for v in VERDICTS}
        for pair, by_feature in pairs.items()
    }


def ladder_check(table: pd.DataFrame, published: dict) -> dict:
    """Whether each set's median ratios against the depositors equal the published ones.

    ``published`` is ``extractor_agreement`` as metrics.json holds it. Every scope and
    feature the published entry has is compared; a set with nothing published is skipped.
    """
    mine = ladder_stats(table)
    out = {}
    for name, (section, variant) in PUBLISHED.items():
        recorded = published.get(section, {}).get("by_variant", {}).get(variant)
        if not recorded:
            out[name] = {"compared_against": f"extractor_agreement.{section}", "skipped": True}
            continue
        pairs = [
            (scope, feature, entry.get("median_ratio"))
            for scope, by_feature in recorded.items()
            for feature, entry in by_feature.items()
        ]
        equal = [
            mine.get(name, {}).get(scope, {}).get(feature, {}).get("median_ratio") == value
            for scope, feature, value in pairs
        ]
        out[name] = {
            "compared_against": f"extractor_agreement.{section}.by_variant.{variant}",
            "n_compared": len(pairs),
            "n_equal": int(sum(equal)),
            "pass": bool(pairs) and all(equal),
        }
    return out


def _cell(stats: dict) -> str:
    plates = [v for v in stats.get("median_ratio_by_plate", {}).values() if v is not None]
    spread = f"{min(plates)}-{max(plates)}" if plates else "-"
    return (
        f"{stats['median_ratio']} | {spread} | {stats['spearman_rho']} "
        f"| {stats['fraction_within_10pct']} | {stats['verdict']}"
    )


def markdown_table(pairs: dict, n_recordings: int) -> str:
    """The translation table as markdown, generated from the metrics payload."""
    lines = [
        "# Translating our features into the Kapucu 2022 depositors' definitions",
        "",
        "REAL recordings: the Kapucu 2022 developmental plates, "
        f"{n_recordings} recordings where the depositors' meaRtools output exists. Generated by "
        "`neuromps.models.feature_definitions` from `outputs/metrics.json` key "
        "`feature_definitions.translation`. Ratio is the target set's value over ours (neuromps) "
        "on the same recording; rho is Spearman; within 10% is the share of recordings whose "
        f"ratio is within 10% of 1. Verdict: same number (ratio within {SCALE_TOLERANCE} of 1, "
        f"rho >= {RANK_AGREEMENT}), scale factor (rho >= {RANK_AGREEMENT}, ratio outside), "
        f"no conversion (rho < {RANK_AGREEMENT}). The last two columns say how close the target "
        "set itself is to the depositors' published values: a same-number verdict between two "
        "of our sets does not mean either matches the depositors. The per-plate column is the "
        "range of the median ratio over the three plates; a factor that differs by plate is a "
        "property of the cultures, not of the definitions. kapucu2022_logisi is experimental "
        "and supports no claim. Checked only on these plates; not a conversion for other labs.",
        "",
    ]
    for pair, by_feature in pairs.items():
        lines += [
            f"## {pair.replace('_to_', ' -> ')}",
            "",
            "| feature | n | median ratio | per plate | rho | within 10% | verdict "
            "| target/depositors ratio | target/depositors rho |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for feature, stats in by_feature.items():
            theirs = stats["target_vs_depositors"]
            lines.append(
                f"| `{feature}` | {stats['n']} | {_cell(stats)} | {theirs['median_ratio']} "
                f"| {theirs['spearman_rho']} |"
            )
        lines.append("")
    return "\n".join(lines)


NOT_IN_TIER = "kapucu2022 developmental plate spike files are not in this tier"


def translation_record(table: pd.DataFrame, published: dict) -> dict:
    """The translation section and the ladder check, from the developmental plates' table."""
    if table.empty:
        return {"status": NOT_IN_TIER}
    pairs = translation(table)
    return {
        "scope": (
            "Kapucu 2022 developmental plates, recordings with the depositors' meaRtools output"
        ),
        "n_plates": int(table["plate"].nunique()),
        "n_recordings": int(table.groupby(["plate", "well", "div"]).ngroups),
        "thresholds": {
            "same_number_ratio_within": SCALE_TOLERANCE,
            "rank_agreement": RANK_AGREEMENT,
            "within_share": WITHIN,
        },
        "not_covered": list(UNCHECKABLE),
        "pairs": pairs,
        "verdicts": verdicts(pairs),
        "ladder_check": ladder_check(table, published),
    }


def plates_in_tier(hashes: dict[str, str], root: Path = DATA_ROOT) -> tuple[str, ...]:
    """Developmental plates whose spike files this tier's manifest lists.

    The build's tier rule: a file on disk that this tier did not fetch is not part of it.
    The quick tier carries the plates' meaRtools tables but not their spike files.
    """
    out = []
    for plate in PLATES:
        spikes = sorted((root / plate / f"{plate}_spikes_noise_explogs").glob("*_DIV*_spikes.csv"))
        if spikes and all(path.name in hashes for path in spikes):
            out.append(plate)
    return tuple(out)


def sidecar_record(raw_root: Path, hashes: dict[str, str], quick: bool) -> dict:
    """Build, check and write the sidecar; returns its metrics section."""
    from neuromps.etl import definition_sidecar as sidecar
    from neuromps.paths import REPO_ROOT, asset_path, processed_dir

    asset = pd.read_parquet(asset_path())
    table = sidecar.build_table(raw_root, hashes, full=not quick, quick=quick)
    in_asset = table["record_id"].isin(set(asset["record_id"]))
    dropped = int(table.loc[~in_asset, "record_id"].nunique())
    table = table[in_asset].reset_index(drop=True)
    path = sidecar.write_table(table, processed_dir())
    spike_ids = set(asset.loc[asset["data_level"] == "spike_times", "record_id"])
    return {
        "file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "quick_share": quick,
        "n_rows": int(len(table)),
        "n_records": int(table["record_id"].nunique()),
        "records_read_not_in_asset": dropped,
        # The quick share reads a few records per source by design; a full run reads them all.
        "asset_spike_time_records_without_rows": len(spike_ids - set(table["record_id"])),
        "by_source": sidecar.counts(table, asset),
        "check_reference_equals_asset": sidecar.check_reference(table, asset),
    }


def checks_pass(sidecar: dict, translated: dict, quick: bool) -> bool:
    """The sidecar matches the asset (and covers it, on a full run); the ladder reproduces."""
    ladder = translated.get("ladder_check")
    reproduced = ladder is None or all(
        ladder.get(name, {}).get("pass") is True for name in ("neuromps", "kapucu2022")
    )
    covered = quick or sidecar["asset_spike_time_records_without_rows"] == 0
    return bool(sidecar["check_reference_equals_asset"]["pass"] and covered and reproduced)


def run(quick: bool) -> dict:
    """The whole metrics section."""
    from neuromps.etl.build import RAW_ROOT, manifest_hashes
    from neuromps.paths import metrics_path

    hashes = manifest_hashes(RAW_ROOT)
    sidecar = sidecar_record(RAW_ROOT, hashes, quick)
    published = {}
    if metrics_path().exists():
        published = json.loads(metrics_path().read_text()).get("extractor_agreement", {})
    plates = plates_in_tier(hashes)
    dev = paired_table(plates, definitions=agreement_definitions()) if plates else pd.DataFrame()
    translated = translation_record(dev, published)
    return {
        "label": LABEL,
        "status": "engineering decision D37; not a registered study",
        "reference": REFERENCE,
        "registry": registry_summary(),
        "claims_use": [d.name for d in REGISTRY.values() if d.status != "experimental"],
        "sidecar": sidecar,
        "translation": translated,
        "checks_pass": checks_pass(sidecar, translated, quick),
        "table": f"outputs/tables/{TABLE_NAME}",
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: build the sidecar, check it, and write the translation table."""
    import argparse

    from neuromps.metrics import update_metrics
    from neuromps.paths import outputs_dir

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="a few records per source")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload = run(args.quick)
    translated = payload["translation"]
    if "pairs" in translated:
        tables = outputs_dir() / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        text = markdown_table(translated["pairs"], translated["n_recordings"])
        (tables / TABLE_NAME).write_text(text + "\n")
    if not payload["checks_pass"]:
        log.error(
            "feature definitions: a built-in check failed; see feature_definitions."
            "sidecar.check_reference_equals_asset and translation.ladder_check"
        )
    update_metrics("feature_definitions", payload)
    summary = {k: v for k, v in payload.items() if k not in ("registry", "translation")}
    print(json.dumps({**summary, "verdicts": translated.get("verdicts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
