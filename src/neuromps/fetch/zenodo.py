"""Zenodo record metadata and file listings."""

from dataclasses import dataclass

from neuromps.fetch.download import get_json

API = "https://zenodo.org/api/records"


@dataclass
class ZenodoFile:
    """One file in a Zenodo record."""

    key: str
    url: str
    size: int
    md5: str | None


def record(record_id: str | int) -> dict:
    """Fetch the record metadata document."""
    return get_json(f"{API}/{record_id}")


def license_id(rec: dict) -> str:
    """Return the record's license id (e.g. ``cc-zero``), or '' if absent."""
    lic = rec.get("metadata", {}).get("license") or {}
    return str(lic.get("id", ""))


def files(rec: dict) -> dict[str, ZenodoFile]:
    """Map filename -> ZenodoFile for a record."""
    out = {}
    for f in rec.get("files", []):
        checksum = str(f.get("checksum", ""))
        md5 = checksum.removeprefix("md5:") if checksum.startswith("md5:") else None
        out[f["key"]] = ZenodoFile(f["key"], f["links"]["self"], int(f["size"]), md5)
    return out
