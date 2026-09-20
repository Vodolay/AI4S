"""One command: data -> features -> models -> twin -> figures.

Usage: python run_all.py [--quick | --full] [--stage fetch|data|models|twin|all]
Stages beyond ``fetch`` are implemented in later phases (see PLAN.md).
"""

import argparse
import logging
import sys
import time

STAGES = ["fetch", "data", "models", "twin"]


def main(argv: list[str] | None = None) -> int:
    """Run the requested pipeline stages in order."""
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="CI path, <10 min on CPU (default)")
    mode.add_argument("--full", action="store_true", help="complete run incl. large raw data")
    p.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stages = STAGES if args.stage == "all" else [args.stage]
    t0 = time.time()
    for stage in stages:
        if stage == "fetch":
            from neuromps.fetch.runner import run_fetchers

            run_fetchers(full=args.full)
        elif stage == "data":
            from neuromps.etl.build import main as build_data

            build_data(["--full"] if args.full else [])
        elif stage == "models":
            from neuromps.models.asymmetry import main as run_asymmetry
            from neuromps.models.conformal import main as run_uncertainty
            from neuromps.models.mechanism import main as run_models
            from neuromps.models.transfer import main as run_transfer

            run_models([])
            run_transfer([])
            run_uncertainty([])
            run_asymmetry([])
        else:
            logging.info("stage %r not implemented yet (later phase) - skipping", stage)
    logging.info("run_all finished in %.1f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
