"""The study design, read from the committed pre-registration.

The analysis takes its conditions, doses and thresholds from
`preregistration/fly_chip_to_animal.json` rather than restating them, so what
was registered and what was run cannot drift apart. The file's SHA-256 and the
commit that added it are reported next to every result.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from neuromps.paths import REPO_ROOT
from neuromps.qc.provenance import sha256_file

REGISTRATION = REPO_ROOT / "preregistration" / "fly_chip_to_animal.json"


@dataclass(frozen=True)
class Condition:
    """One drug condition: a family (mechanism), a knob level and the overrides it sets."""

    family: str
    level: float
    overrides: dict = field(default_factory=dict)

    @property
    def dose(self) -> float:
        """How far the knob moved from untreated (0 for vehicle)."""
        return float(sum(abs(v - 1.0) for v in self.overrides.values()))

    @property
    def name(self) -> str:
        """Stable label, used in file names and tables."""
        return "vehicle" if self.family == "vehicle" else f"{self.family}_{self.level:g}"


def registration(path: Path = REGISTRATION) -> dict:
    """The registered design and predictions."""
    return json.loads(Path(path).read_text())


def conditions(spec: dict) -> list[Condition]:
    """Vehicle first, then every registered level of every family."""
    out = [Condition("vehicle", 1.0, {})]
    for family, entry in spec["design"]["conditions"].items():
        out += [Condition(family, lv, {entry["knob"]: lv}) for lv in entry["levels"]]
    return out


def provenance(path: Path = REGISTRATION) -> dict:
    """Hash of the registration and the first commit that contains it."""
    try:
        commit = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H %cI", "--", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        commit = []
    return {
        "file": str(Path(path).relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "registered_in_commit": commit[0] if commit else None,
        "committed_at": commit[1] if len(commit) > 1 else None,
    }
