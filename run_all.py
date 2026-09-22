"""One command: data -> features -> models -> twin -> figures.

Usage: python run_all.py [--quick | --full] [--stage fetch|data|models|twin|all]

``--quick`` is the CI path: the small-tier sources, fewer draws, under ten
minutes on CPU. It writes to its own run root (``outputs/quick/`` and
``data/processed/quick/``) so that running it cannot overwrite the full run
whose numbers the report cites. See :mod:`neuromps.paths`.
"""

import argparse
import logging
import sys
import time

from neuromps.paths import FULL, QUICK, outputs_dir, processed_dir, set_run_mode

STAGES = ["fetch", "data", "models", "twin", "demo"]


def main(argv: list[str] | None = None) -> int:
    """Run the requested pipeline stages in order."""
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="CI path, <10 min on CPU; writes to outputs/quick/ and data/processed/quick/",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="complete run incl. large raw data; writes to outputs/ and data/processed/",
    )
    p.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Before any stage module is imported, so every writer resolves the same root.
    set_run_mode(QUICK if args.quick else FULL)
    logging.info("writing this run to %s and %s", outputs_dir(), processed_dir())
    stages = STAGES if args.stage == "all" else [args.stage]
    t0 = time.time()
    timings: dict[str, float] = {}
    for stage in stages:
        stage_started = time.time()
        logging.info("stage %r starting", stage)
        if stage == "fetch":
            from neuromps.fetch.runner import run_fetchers

            run_fetchers(full=args.full)
        elif stage == "data":
            from neuromps.etl.build import main as build_data

            build_data(["--full"] if args.full else [])
        elif stage == "models":
            from neuromps.models.asymmetry import main as run_asymmetry
            from neuromps.models.baselines import main as run_baselines
            from neuromps.models.conformal import main as run_uncertainty
            from neuromps.models.mechanism import main as run_models
            from neuromps.models.microchannels import main as run_microchannels
            from neuromps.models.transfer import main as run_transfer

            quick = ["--quick"] if args.quick else []
            run_models(quick)
            run_transfer(quick)
            run_uncertainty(quick)
            run_baselines(quick)
            run_asymmetry([])
            # Skips itself when the 109 MB archive has not been fetched, which is
            # the case on the quick tier.
            run_microchannels(quick)
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
        # Printed as each stage ends, not only in the summary: a run killed by a
        # timeout writes no summary, and then the log is all there is to explain
        # where the time went.
        logging.info("stage %r finished in %.1f s", stage, timings[stage])
    elapsed = round(time.time() - t0, 1)
    logging.info("run_all finished in %.1f s; results in %s", elapsed, outputs_dir())
    _record_timings(timings, elapsed, quick=args.quick, stages=stages)
    return 0


def _record_timings(timings: dict, elapsed: float, quick: bool, stages: list[str]) -> None:
    """Write stage timings to metrics.json, so the report can cite them.

    The reproducibility section quoted a stage table that existed only in a log
    file, in a report whose opening paragraph promises every number has a key.
    """
    import json

    from neuromps.metrics import update_metrics
    from neuromps.paths import metrics_path

    # Merge rather than replace: `--stage models` must not erase the timing the
    # demo stage recorded on its own run, which the report cites.
    key = "run_all_timings_quick" if quick else "run_all_timings_full"
    previous = (
        json.loads(metrics_path().read_text()).get(key, {}) if metrics_path().exists() else {}
    )
    merged = {**previous.get("by_stage_s", {}), **timings}
    update_metrics(
        key,
        {
            "label": (
                f"wall-clock seconds per stage, most recent "
                f"{'--quick' if quick else 'full'} run of each"
            ),
            "total_s": elapsed,
            "by_stage_s": merged,
            "stages_run": list(stages),
        },
    )


if __name__ == "__main__":
    sys.exit(main())
