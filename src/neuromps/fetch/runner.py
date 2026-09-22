"""Run all registered fetchers and record results in MANIFEST.json."""

import argparse
import logging
import traceback
from collections.abc import Callable
from pathlib import Path

from neuromps.fetch.excluded import excluded_records
from neuromps.fetch.manifest import Manifest, SourceRecord
from neuromps.paths import manifest_path

log = logging.getLogger("neuromps.fetch")

Fetcher = Callable[[Path, Manifest, bool], SourceRecord]
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "data" / "raw"


def registry() -> dict[str, tuple[Fetcher, dict[str, str]]]:
    """Map source name -> (fetch function, static metadata used if it is blocked)."""
    from neuromps.fetch import (
        axon_diode,
        brewer2021,
        fofie2025,
        kapucu2022,
        mateus2024,
        screenneuropharm_bic,
        sharf2022,
        toxcast_mea,
        trujillo2019,
        vassallo2017,
    )

    mods = [
        kapucu2022,
        toxcast_mea,
        sharf2022,
        fofie2025,
        screenneuropharm_bic,
        axon_diode,
        mateus2024,
        trujillo2019,
        brewer2021,
        vassallo2017,
    ]
    return {m.SOURCE: (m.fetch, m.META) for m in mods}


def run_fetchers(
    raw_root: Path = RAW_ROOT, names: list[str] | None = None, full: bool = False
) -> Manifest:
    """Run fetchers; a failing source is recorded as ``blocked`` and never aborts the rest."""
    manifest = Manifest.load(manifest_path(raw_root))
    for record in excluded_records():
        if record.name in registry():
            raise RuntimeError(f"{record.name!r} is both registered and excluded")
        manifest.upsert(record)
        log.info("[%s] skipped (excluded, see data/SOURCES.md)", record.name)
    manifest.save()
    for name, (fetch, meta) in registry().items():
        if names and name not in names:
            continue
        log.info("[%s] fetching (full=%s)", name, full)
        try:
            record = fetch(raw_root, manifest, full)
        except Exception as exc:  # any failure (incl. SourceBlocked) is recorded, not fatal
            log.warning("[%s] BLOCKED: %s", name, exc)
            log.debug("".join(traceback.format_exception(exc)))
            previous = manifest.sources.get(name)
            record = SourceRecord(
                name=name,
                status="blocked",
                note=f"{type(exc).__name__}: {exc}",
                files=previous.files if previous else [],
                **meta,
            )
            if record.files:
                record.status = "partial"
        manifest.upsert(record)
        manifest.save()
        total = sum(f.bytes for f in record.files)
        log.info("[%s] status=%s files=%d bytes=%d", name, record.status, len(record.files), total)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m neuromps.fetch.runner [--full] [--only NAME ...]``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true", help="also fetch large raw recordings")
    p.add_argument("--only", nargs="*", help="restrict to these source names")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = run_fetchers(names=args.only, full=args.full)
    print(f"\n{'source':<22}{'status':<10}{'files':>6}{'MB':>10}  license")
    for s in manifest.sources.values():
        mb = sum(f.bytes for f in s.files) / 1e6
        print(f"{s.name:<22}{s.status:<10}{len(s.files):>6}{mb:>10.1f}  {s.license}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
