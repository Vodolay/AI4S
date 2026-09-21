"""Run one twin simulation and write it to a file, for the app's live button.

This exists as its own entry point because Brian2 cannot be imported inside
Streamlit's script-runner thread: it registers a SIGINT handler, and
`signal.signal` only works in a main thread. A separate interpreter gets a main
thread of its own, and keeps the simulator out of the web server, so a slow or
failed run cannot take the app down.

    python -m neuromps.demo.simulate_one --ratio 4 --gabaa 1.0 --ampa 1.0 --out run.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    """Simulate at the given parameters and save the raster, features and summary."""
    from neuromps.demo.precompute import simulate_point

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--gabaa", type=float, required=True)
    p.add_argument("--ampa", type=float, required=True)
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--neurons", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    point = {
        "forward_backward_ratio": args.ratio,
        "gabaa_scale": args.gabaa,
        "ampa_scale": args.ampa,
    }
    out = simulate_point(point, args.duration, args.seed, args.neurons)
    np.savez_compressed(
        args.out,
        summary=out["summary"],
        features=np.array(json.dumps(out["features"])),
        coupling=np.array(json.dumps(out["coupling"])),
        **out["raster"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
