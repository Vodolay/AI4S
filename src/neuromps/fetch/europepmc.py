"""Open-access article full text and supplementary files via Europe PMC."""

import re
from pathlib import Path

import requests

from neuromps.fetch.common import fetch_many
from neuromps.fetch.download import USER_AGENT, SourceBlocked
from neuromps.fetch.manifest import FileRecord

API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_LICENSE = re.compile(r"<license\b[^>]*?xlink:href=\"([^\"]+)\"", re.S)
_LICENSE_REF = re.compile(r"<ali:license_ref[^>]*>([^<]+)</ali:license_ref>")


def require_open_license(xml: str) -> str:
    """Return the Creative Commons license URL declared in JATS XML, else raise SourceBlocked."""
    m = _LICENSE.search(xml) or _LICENSE_REF.search(xml)
    url = m.group(1).strip() if m else ""
    if "creativecommons.org" not in url:
        raise SourceBlocked(f"no Creative Commons license found in article XML ({url or 'none'})")
    return url


def fetch_article(
    pmcid: str,
    raw_root: Path,
    rel_dir: str,
    cached: dict[str, FileRecord],
    supplementary: bool = False,
) -> tuple[list[FileRecord], str]:
    """Download full-text XML (and optionally the supplementary zip) for an OA article.

    The license is verified from the XML before anything else is kept.
    Note: Europe PMC builds the supplementary zip on request, so its SHA256 can
    differ between downloads even when the contents are unchanged.
    """
    xml_url = f"{API}/{pmcid}/fullTextXML"
    r = requests.get(xml_url, headers={"User-Agent": USER_AGENT}, timeout=120)
    if r.status_code != 200:
        raise SourceBlocked(f"{pmcid}: full text not available (HTTP {r.status_code})")
    license_url = require_open_license(r.text)
    items = [(xml_url, f"{rel_dir}/{pmcid}.xml")]
    if supplementary:
        items.append((f"{API}/{pmcid}/supplementaryFiles", f"{rel_dir}/{pmcid}_supplementary.zip"))
    return fetch_many(items, raw_root, cached), license_url
