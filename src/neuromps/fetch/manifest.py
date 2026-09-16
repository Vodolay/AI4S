"""`data/raw/MANIFEST.json`: per-source license and per-file provenance."""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Status = Literal["ok", "partial", "blocked", "skipped"]
SCHEMA_VERSION = 1
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass
class FileRecord:
    """One downloaded file. ``path`` is relative to ``data/raw``."""

    path: str
    url: str
    sha256: str
    bytes: int
    downloaded_at: str

    def validate(self) -> None:
        """Raise ValueError if the record is malformed."""
        if not _HEX64.match(self.sha256):
            raise ValueError(f"invalid sha256 for {self.path!r}: {self.sha256!r}")
        if self.bytes < 0:
            raise ValueError(f"negative size for {self.path!r}")


@dataclass
class SourceRecord:
    """Status, license and files for one data source."""

    name: str
    status: Status
    landing_url: str
    license: str
    license_url: str
    citation: str
    note: str = ""
    checked_at: str = field(default_factory=utc_now)
    files: list[FileRecord] = field(default_factory=list)

    def validate(self) -> None:
        """Raise ValueError if an ``ok``/``partial`` source lacks license or files."""
        for f in self.files:
            f.validate()
        if self.status in ("ok", "partial"):
            if not self.license.strip():
                raise ValueError(f"source {self.name!r} has files but no license")
            if not self.files:
                raise ValueError(f"source {self.name!r} is {self.status} but has no files")

    @classmethod
    def from_dict(cls, d: dict) -> "SourceRecord":
        """Build from the JSON representation."""
        files = [FileRecord(**f) for f in d.get("files", [])]
        return cls(**{**d, "files": files})


@dataclass
class Manifest:
    """In-memory view of MANIFEST.json."""

    path: Path
    sources: dict[str, SourceRecord] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load from disk, or return an empty manifest if the file is absent."""
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        raw = json.loads(path.read_text())
        srcs = {k: SourceRecord.from_dict(v) for k, v in raw.get("sources", {}).items()}
        return cls(path=path, sources=srcs)

    def upsert(self, record: SourceRecord) -> None:
        """Validate and insert/replace the record for ``record.name``."""
        record.validate()
        self.sources[record.name] = record

    def file_index(self) -> dict[str, FileRecord]:
        """Map relative path -> FileRecord across all sources."""
        return {f.path: f for s in self.sources.values() for f in s.files}

    def save(self) -> None:
        """Write atomically, sorted for stable diffs."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "sources": {k: asdict(self.sources[k]) for k in sorted(self.sources)},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.path)
