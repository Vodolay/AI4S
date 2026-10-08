"""Capture the demo app's tabs to PNG, so the video's screenshots regenerate.

The first set of stills was taken by hand. They then stayed on disk through a
density correction, a cross-laboratory result and an inference re-run, so the
video showed numbers the report had stopped making -- the exact failure the
citation checker exists to prevent in the prose.

This drives the app with headless Chrome instead. One command, no clicking, and
a still that disagrees with the metrics is a still that was not regenerated.

Usage:
    python -m neuromps.video.stills            # start the app, capture, stop
    python -m neuromps.video.stills --port 8501 --keep-running
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.video")

WIDTH, HEIGHT = 1600, 2800
# Streamlit renders client-side, so the page needs time after the document loads.
# This is Chrome's virtual clock, not wall time: it fast-forwards the page's own
# timers, which is why a shot of the live Brian2 run cannot be taken this way --
# that round trip costs real seconds the virtual clock does not wait for, and the
# attempt produced a file byte-identical to the un-run page.
SETTLE_MS = 25_000
ATTEMPTS = 6

# Each still, by the query string that produces it and the row its content must
# reach. The slugs are the app's own, so a tab that is renamed or reordered
# cannot silently capture the wrong page; the depth is what catches a page that
# painted its first column and was photographed before the second arrived. Both
# of those have happened, and a generic "is there ink on this page" check passed
# for both.
# name -> (query, virtual-clock budget, row the content must reach, seconds to
# wait after the warm-up load). The pause is for work the page starts on the
# server: the live Brian2 run takes about four real seconds, and Chrome hangs up
# long before that. It is cached on its parameter tuple, so the warm-up pays for
# the simulation and the shot that counts renders from the cache.
STILLS = {
    "models.png": ("?tab=models", SETTLE_MS, 1700, 0),
    "twin.png": ("?tab=twin", SETTLE_MS, 1600, 0),
    "twin_live.png": ("?tab=twin&live=1", SETTLE_MS, 2200, 20),
    "data.png": ("?tab=data", SETTLE_MS, 1500, 0),
}

CHROME_NAMES = ("google-chrome", "chromium", "chromium-browser", "google-chrome-stable")


def chrome() -> str:
    """The headless browser to drive, or a clear error naming what to install."""
    for name in CHROME_NAMES:
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(f"no headless browser found; looked for {', '.join(CHROME_NAMES)}")


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


def wait_for_server(port: int, timeout_s: float = 90.0) -> None:
    """Block until the app answers, or say how long it was given."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _port_open(port):
            return
        time.sleep(0.5)
    raise TimeoutError(f"the app did not open port {port} within {timeout_s:.0f} s")


def start_app(port: int) -> subprocess.Popen | None:
    """Start the demo app, or return None when something already serves the port.

    A server this script did not start will not have the live-on-load opt-in, so
    `twin_live.png` cannot be captured against it. That is said out loud rather
    than discovered as a blank frame.
    """
    if _port_open(port):
        log.warning(
            "port %d is already serving; capturing against it. twin_live.png needs a server "
            "started by this script - stop that one first if you want that still.",
            port,
        )
        return None
    executable = REPO_ROOT / ".venv" / "bin" / "streamlit"
    command = [
        str(executable) if executable.exists() else "streamlit",
        "run",
        str(REPO_ROOT / "app" / "app.py"),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    log.info("starting the app on port %d", port)
    # Only a server this script starts may run the simulation on page load. A
    # deployed app leaves this unset, so `?live=1` there does nothing.
    environment = {**os.environ, "NEUROMPS_ALLOW_LIVE_ON_LOAD": "1"}
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=environment
    )
    wait_for_server(port)
    return process


def capture(url: str, destination: Path, settle_ms: int, browser: str) -> Path:
    """Screenshot one page. Fails loudly rather than leaving a stale file in place."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = destination.stat().st_mtime if destination.exists() else 0.0
    subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--window-size={WIDTH},{HEIGHT}",
            f"--virtual-time-budget={settle_ms}",
            f"--screenshot={destination}",
            url,
        ],
        check=True,
        capture_output=True,
        timeout=settle_ms / 1000 + 90,
    )
    if not destination.exists() or destination.stat().st_mtime == before:
        raise RuntimeError(f"no screenshot was written for {url}")
    log.info("wrote %s (%d KB)", destination, destination.stat().st_size // 1024)
    return destination


def content_depth(path: Path) -> int:
    """The lowest row of the image that carries real ink.

    A page caught mid-render is almost the right size and almost the right
    colours - Streamlit's skeleton is three greys on white - so counting colours
    does not separate it from a finished one. How far down the page the content
    actually reaches does.
    """
    import numpy as np
    from PIL import Image

    pixels = np.asarray(Image.open(path).convert("L"))
    rows = np.where((pixels < 160).sum(axis=1) > 3)[0]
    return int(rows.max()) if len(rows) else 0


def capture_until_rendered(
    url: str, destination: Path, settle_ms: int, browser: str, want_depth: int
) -> Path:
    """Capture, check, and try again with more time if the page had not painted.

    Chrome's virtual clock fast-forwards the page's own timers but does not wait
    for Streamlit's websocket round trip, so a cold tab can be screenshotted
    mid-skeleton. The first version of this had no retry and wrote a blank frame
    for the slowest tab.
    """
    budget = settle_ms
    for attempt in range(1, ATTEMPTS + 1):
        capture(url, destination, budget, browser)
        depth = content_depth(destination)
        if depth >= want_depth:
            return destination
        log.warning(
            "%s reached row %d of a wanted %d on attempt %d; retrying",
            destination.name,
            depth,
            want_depth,
            attempt,
        )
        budget *= 2
        time.sleep(4.0)
    raise RuntimeError(
        f"{destination.name} never rendered past row {content_depth(destination)}, "
        f"wanted {want_depth}, after {ATTEMPTS} attempts"
    )


def main(argv: list[str] | None = None) -> int:
    """Capture every still the shot list uses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--keep-running", action="store_true", help="leave the app up afterwards")
    parser.add_argument("--only", help="capture one still by filename")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    browser = chrome()
    stills_dir = outputs_dir() / "video" / "stills"
    process = start_app(arguments.port)
    try:
        wanted = {arguments.only: STILLS[arguments.only]} if arguments.only else STILLS
        for name, (query, settle, depth, pause) in wanted.items():
            url = f"http://localhost:{arguments.port}/{query}"
            # One warm-up load so the app's caches are populated before the shot
            # that counts: a cold tab is the one that paints too late.
            capture(url, stills_dir / ".warmup.png", settle, browser)
            if pause:
                log.info("waiting %d s for %s to finish its server-side work", pause, name)
                time.sleep(pause)
            capture_until_rendered(url, stills_dir / name, settle, browser, depth)
        (stills_dir / ".warmup.png").unlink(missing_ok=True)
    finally:
        if process is not None and not arguments.keep_running:
            process.terminate()
            process.wait(timeout=20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
