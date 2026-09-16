"""Run all registered fetchers and record results in MANIFEST.json."""

import argparse
import logging
import traceback
from collections.abc import Callable
from pathlib import Path

from neuromps.fetch.manifest import Manifest, SourceRecord

log = logging.getLogger("neuromps.fetch")

Fetcher = Callable[[Path, Manifest, bool], SourceRecord]
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "data" / "raw"


def registry() -> dict[str, tuple[Fetcher, dict[str, str]]]:
    """Map source name -> (fetch function, static metadata used if it is blocked)."""
    from neuromps.fetch import axon_diode, iras_mea, kapucu2022, toxcast_mea

    mods = [kapucu2022, toxcast_mea, iras_mea, axon_diode]
    return {m.SOURCE: (m.fetch, m.META) for m in mods}


def run_fetchers(
    raw_root: Path = RAW_ROOT, names: list[str] | None = None, full: bool = False
) -> Manifest:
    """Run fetchers; a failing source is recorded as ``blocked`` and never aborts the rest."""
    manifest = Manifest.load(raw_root / "MANIFEST.json")
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
    print(f"\n{'source':<16}{'status':<10}{'files':>6}{'MB':>10}  license")
    for s in manifest.sources.values():
        mb = sum(f.bytes for f in s.files) / 1e6
        print(f"{s.name:<16}{s.status:<10}{len(s.files):>6}{mb:>10.1f}  {s.license}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
