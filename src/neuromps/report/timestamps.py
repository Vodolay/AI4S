"""Internal timestamps for every registration: hashes, commits, a hash chain, an order check.

``python -m neuromps.report.timestamps`` writes `preregistration/TIMESTAMPS.json`:
- per registration file, its SHA256 and the commit (hash, date) that first added it;
- a hash chain over the files in commit order, whose root changes if any registration's
  content, commit or order changes;
- per study output that names a registration, whether the registration's commit is an
  ancestor of the first commit that added the output (registered before its results).
The counts (registrations, outputs checked, violations, registrations no output names) also go
to `outputs/metrics.json` under ``registration_timestamps``, so a report can cite them.
This is internal evidence (it rests on the repository's history), not a third-party timestamp.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT
from neuromps.qc.provenance import sha256_file
from neuromps.report.audit import manuscript_only

log = logging.getLogger("neuromps.report")
PREREG = REPO_ROOT / "preregistration"
OUT = PREREG / "TIMESTAMPS.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def first_commit(path: Path) -> tuple[str, str] | None:
    """(hash, ISO committer date) of the commit that first added `path`."""
    got = _git("log", "--diff-filter=A", "--format=%H %cI", "--", str(path)).splitlines()
    return tuple(got[-1].split()) if got else None


def chain(entries: list[dict]) -> list[str]:
    """Running SHA256 over (previous link, file hash, commit), in the given order."""
    links, prev = [], ""
    for e in entries:
        prev = hashlib.sha256(f"{prev}|{e['sha256']}|{e['commit']}".encode()).hexdigest()
        links.append(prev)
    return links


def registrations() -> list[dict]:
    """Every registration with its hash and first commit, in commit order."""
    out = []
    for path in sorted(PREREG.glob("*.json")):
        if path.name == OUT.name:
            continue
        found = first_commit(path)
        if found is None:
            continue
        out.append(
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "commit": found[0],
                "committed_at": found[1],
            }
        )
    out.sort(key=lambda e: (e["committed_at"], e["file"]))
    for e, link in zip(out, chain(out), strict=True):
        e["chain"] = link
    return out


def is_ancestor(a: str, b: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", a, b], cwd=REPO_ROOT, check=False
        ).returncode
        == 0
    )


def order_checks(regs: list[dict]) -> list[dict]:
    """Each output JSON naming a registration: was the registration committed first?"""
    commit_of = {e["file"]: e["commit"] for e in regs}
    out = []
    for path in sorted((REPO_ROOT / "outputs").rglob("*.json")):
        try:
            reg = json.loads(path.read_text()).get("registration")
        except (json.JSONDecodeError, AttributeError):
            continue
        if not isinstance(reg, dict) or "file" not in reg:
            continue
        name = Path(reg["file"]).name
        added = first_commit(path)
        if name not in commit_of or added is None:
            continue
        out.append(
            {
                "output": str(path.relative_to(REPO_ROOT)),
                "registration": name,
                "same_commit": commit_of[name] == added[0],
                "registered_first": is_ancestor(commit_of[name], added[0])
                and commit_of[name] != added[0],
            }
        )
    return out


def _counts(regs: list[dict], checks: list[dict]) -> dict:
    named = {c["registration"] for c in checks}
    return {
        "n_registrations": len(regs),
        "n_outputs_checked": len(checks),
        "n_violations": sum(not c["registered_first"] for c in checks),
        "not_covered": [r["file"] for r in regs if r["file"] not in named],
    }


def summary(regs: list[dict], checks: list[dict], manuscript_only: set[str] = frozenset()) -> dict:
    """The manifest in numbers, including the registrations the order check cannot see.

    ``submission`` repeats them without the registrations listed as manuscript-only.
    """
    shown = [r for r in regs if r["file"] not in manuscript_only]
    shown_checks = [c for c in checks if c["registration"] not in manuscript_only]
    return {**_counts(regs, checks), "submission": _counts(shown, shown_checks)}


def main(argv: list[str] | None = None) -> int:
    """CLI: write the timestamp manifest and print its root and any order violations."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    regs = registrations()
    checks = order_checks(regs)
    manifest = {
        "note": "Internal timestamps from this repository's history; not third-party verified.",
        "root": regs[-1]["chain"] if regs else None,
        "registrations": regs,
        "registered_before_results": checks,
        "violations": [c for c in checks if not c["registered_first"]],
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    update_metrics("registration_timestamps", summary(regs, checks, manuscript_only(PREREG)))
    print(
        f"{len(regs)} registrations; root {manifest['root']}; "
        f"{len(checks)} outputs checked; {len(manifest['violations'])} violations"
    )
    for v in manifest["violations"]:
        print("  VIOLATION", v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
