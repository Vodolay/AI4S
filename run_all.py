"""One command: data -> features -> models -> twin -> figures.

Usage: python run_all.py [--quick | --full] [--stage fetch|data|models|twin|all]
Stages beyond ``fetch`` are implemented in later phases (see PLAN.md).
"""

import argparse
import logging
import sys
import time

STAGES = ["fetch", "data", "models", "twin", "demo"]


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
    timings: dict[str, float] = {}
    for stage in stages:
        stage_started = time.time()
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

            quick = ["--quick"] if args.quick else []
            run_models(quick)
            run_transfer(quick)
            run_uncertainty(quick)
            run_asymmetry([])
        elif stage == "twin":
            from neuromps.twin.calibrate import main as calibrate_twin
            from neuromps.twin.directionality import main as run_directionality
            from neuromps.twin.inference import main as run_inference
            from neuromps.twin.perturbation import main as run_knobs
            from neuromps.twin.real_inference import main as run_real_inference
            from neuromps.twin.validate import main as validate_twin

            quick = ["--quick"] if args.quick else []
            validate_twin(quick)
            calibrate_twin(quick)
            run_knobs(quick)
            run_inference(quick)
            run_real_inference(quick)
            run_directionality(quick)
        elif stage == "demo":
            from neuromps.demo.precompute import main as precompute_demo

            precompute_demo(["--quick"] if args.quick else [])
        else:
            logging.info("stage %r not implemented yet (later phase) - skipping", stage)
        timings[stage] = round(time.time() - stage_started, 1)
    elapsed = round(time.time() - t0, 1)
    logging.info("run_all finished in %.1f s", elapsed)
    _record_timings(timings, elapsed, quick=args.quick, stages=stages)
    return 0


def _record_timings(timings: dict, elapsed: float, quick: bool, stages: list[str]) -> None:
    """Write stage timings to metrics.json, so the report can cite them.

    The reproducibility section quoted a stage table that existed only in a log
    file, in a report whose opening paragraph promises every number has a key.
    """
    from neuromps.metrics import update_metrics

    update_metrics(
        "run_all_timings_quick" if quick else "run_all_timings_full",
        {
            "label": f"wall-clock seconds per stage, {'--quick' if quick else 'full'} run",
            "total_s": elapsed,
            "by_stage_s": timings,
            "stages_run": list(stages),
        },
    )


if __name__ == "__main__":
    sys.exit(main())
