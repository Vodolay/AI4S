"""Sources evaluated and deliberately excluded from the data asset.

Each entry is written to MANIFEST.json as ``status="skipped"`` with no files on
every fetch run, so the exclusion and its reason stay part of the provenance
record. Full rationale: data/SOURCES.md.
"""

from neuromps.fetch.manifest import SourceRecord

EXCLUDED: dict[str, dict[str, str]] = {
    "iras_mea": {
        "landing_url": "https://doi.org/10.1093/toxsci/kfaa136",
        "license": "CC-BY-NC-4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "citation": (
            "Tukker AM, Wijnolts FMJ, de Groot A, Westerink RHS. Toxicol Sci 178(1):71-87 "
            "(2020). doi:10.1093/toxsci/kfaa136"
        ),
        "note": (
            "EXCLUDED 2026-09-16 (PI decision): non-commercial license is incompatible with a "
            "prize-bearing submission released under an open license. Revisit only if the "
            "competition rules clearly keep submissions under the entrant's own open license."
        ),
    },
}


def excluded_records() -> list[SourceRecord]:
    """Manifest records for excluded sources (no files)."""
    return [SourceRecord(name=k, status="skipped", **v) for k, v in EXCLUDED.items()]
