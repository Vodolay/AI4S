"""One command: data -> features -> models -> twin -> figures.

Usage: python run_all.py [--quick | --full] [--with-fly]
                         [--stage fetch|data|models|twin|demo|fly|embodied|all]

``--quick`` is the CI path: the small-tier sources, fewer draws, under ten
minutes on CPU. It writes to its own run root (``outputs/quick/`` and
``data/processed/quick/``) so that running it cannot overwrite the full run
whose numbers the report cites. See :mod:`neuromps.paths`.

The fly stage (the chip-to-animal study on the whole fly brain) is part of a
full run. A quick run leaves it out unless ``--with-fly`` is given: it needs a
440 MB download and about 7 GB of memory per worker, which would take the CI
path past its ten minutes on its own.

The embodied stage (every later fly study: the male-CNS replication and motor
readouts, the output-tract electrode, the embodied stages 1-5 and the rhythm-measure
audit; see :mod:`neuromps.fly.pipeline`) is part of a full run only. It needs the
optional sources flywire783, malecns, malecns_sizes, flymimic and flybody and the ``embodied``
extra (``pip install -e .[embodied]``), and takes a few hours on a many-core machine.
"""

import argparse
import logging
import sys
import time

from neuromps import harness
from neuromps.paths import FULL, QUICK, outputs_dir, processed_dir, set_run_mode

STAGES = ["fetch", "data", "models", "twin", "demo"]
OPTIONAL_STAGES = ["fly", "embodied"]
EMBODIED_SOURCES = ["flywire783", "malecns", "malecns_sizes", "flymimic", "flybody"]


def select_stages(stage: str, quick: bool, with_fly: bool) -> list[str]:
    """Stages to run: one by name, or all - the fly stage on a quick run only if asked,
    the embodied stage on full runs only."""
    if stage != "all":
        return [stage]
    if not quick:
        return [*STAGES, *OPTIONAL_STAGES]
    return [*STAGES, "fly"] if with_fly else list(STAGES)


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
    p.add_argument("--stage", choices=[*STAGES, *OPTIONAL_STAGES, "all"], default="all")
    p.add_argument(
        "--with-fly", action="store_true", help="also run the fly stage on a --quick run"
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Before any stage module is imported, so every writer resolves the same root.
    set_run_mode(QUICK if args.quick else FULL)
    logging.info("writing this run to %s and %s", outputs_dir(), processed_dir())
    stages = select_stages(args.stage, quick=args.quick, with_fly=args.with_fly)
    t0 = time.time()
    timings: dict[str, float] = {}
    for stage in stages:
        stage_started = time.time()
        logging.info("stage %r starting", stage)
        if stage == "fetch":
            from neuromps.fetch.runner import (
                incomplete_message,
                incomplete_sources,
                run_fetchers,
            )

            # A bare run is the full run, so it fetches the full tier: passing
            # `args.full` here left a bare run fetching the small tier into the
            # full run's manifest, replacing each source's entry with fewer files.
            manifest = run_fetchers(full=not args.quick)
            # A source the network cut short would otherwise surface an hour
            # later as a crash in the models stage, with nothing naming the cause.
            incomplete = incomplete_sources(manifest)
            if incomplete:
                raise SystemExit(incomplete_message(incomplete))
        elif stage == "data":
            from neuromps.etl.build import main as build_data

            build_data(["--full"] if args.full else [])
        elif stage == "models":
            from neuromps.models.asymmetry import main as run_asymmetry
            from neuromps.models.axonal import main as run_axonal
            from neuromps.models.baselines import main as run_baselines
            from neuromps.models.conformal import main as run_uncertainty
            from neuromps.models.cross_lab import main as run_cross_lab
            from neuromps.models.dose_response import main as run_dose_response
            from neuromps.models.extractor_agreement import main as run_extractor_agreement
            from neuromps.models.feature_definitions import main as run_feature_definitions
            from neuromps.models.hpsc_settling import main as run_settling
            from neuromps.models.hpsc_split_half import main as run_split_half
            from neuromps.models.mechanism import main as run_models
            from neuromps.models.microchannels import main as run_microchannels
            from neuromps.models.rate_only import main as run_rate_only
            from neuromps.models.toxcast_noise_floors import main as run_noise_floors
            from neuromps.models.transfer import main as run_transfer
            from neuromps.models.tunnels import main as run_tunnels
            from neuromps.report.architecture import main as draw_architecture

            quick = ["--quick"] if args.quick else []
            # Does the shared extractor reproduce the depositors' own numbers?
            draw_architecture(quick)
            run_extractor_agreement(quick)
            # D37: the asset's spike-time records under every named definition set (a
            # sidecar beside the asset), checked against it and against the ladder above.
            run_feature_definitions(quick)
            # Is the dose signal in the features the twin is given at all?
            run_dose_response(quick)
            # How far one rate change carries, and whether it carries across.
            run_rate_only(quick)
            run_models(quick)
            run_transfer(quick)
            run_uncertainty(quick)
            run_cross_lab(quick)
            run_baselines(quick)
            run_tunnels(quick)
            # Ground truth for the tunnel result: the axons themselves.
            run_axonal(quick)
            run_asymmetry([])
            # Skips itself when the 109 MB archive has not been fetched, which is
            # the case on the quick tier.
            run_microchannels(quick)
            # Real vehicle wells: what change a chip can see. Skips itself when the
            # ToxCast well table (built outside the fetch tiers) is absent.
            run_noise_floors(quick)
            # The same question for human iPSC wells: split-half noise against rat, one lab.
            # A quick run takes the two 48-well baselines only (the quick data tier).
            run_split_half(quick)
            # D35: is that human spread settling or steady noise? Same two plates, D32's
            # usable wells (read from its results file just written); a few seconds.
            run_settling(quick)
        elif stage == "twin":
            from neuromps.models.twin_toxcast_check import main as run_toxcast_check
            from neuromps.models.twin_toxcast_dose_response import main as run_toxcast_doses
            from neuromps.report.diagrams import main as draw_model_diagrams
            from neuromps.twin.calibrate import main as calibrate_twin
            from neuromps.twin.delta_inference import main as run_delta_inference
            from neuromps.twin.depression import main as run_depression
            from neuromps.twin.directionality import main as run_directionality
            from neuromps.twin.inference import main as run_inference
            from neuromps.twin.perturbation import main as run_knobs
            from neuromps.twin.real_inference import main as run_real_inference
            from neuromps.twin.recalibration import main as run_recalibration
            from neuromps.twin.validate import main as validate_twin

            quick = ["--quick"] if args.quick else []
            validate_twin(quick)
            calibrate_twin(quick)
            # Whether the calibration's residual is a missing mechanism.
            run_depression(quick)
            run_knobs(quick)
            # The same knobs against a laboratory the twin has never seen; reuses the
            # committed predictions, and skips itself without the ToxCast well table.
            run_toxcast_check(quick)
            # The same knobs over a dose grid, against the screen's concentration-response
            # wells; also reuses committed predictions, and skips itself without the table.
            run_toxcast_doses(quick)
            run_inference(quick)
            # D31: the registered recalibration of that posterior. Quick is a token
            # run on the quick cache; full reuses the recorded simulation chunks and
            # simulates only those a clean clone lacks.
            run_recalibration(quick)
            run_real_inference(quick)
            # The same question put to a change against each chip's own
            # baseline, rather than to one recording's absolute features.
            run_delta_inference(quick)
            run_directionality(quick)
            # Last, because the three model diagrams quote the calibration and the
            # inference this stage has just written, as well as the classifier the
            # models stage wrote before it.
            draw_model_diagrams(quick)
        elif stage == "demo":
            from neuromps.demo.precompute import main as precompute_demo

            precompute_demo(["--quick"] if args.quick else [])
        elif stage == "fly":
            from neuromps.fetch.runner import incomplete_message, run_fetchers
            from neuromps.fly.study import main as run_fly_study

            manifest = run_fetchers(names=["flywire783"], full=not args.quick)
            if manifest.sources["flywire783"].status != "ok":
                raise SystemExit(incomplete_message([manifest.sources["flywire783"]]))
            run_fly_study(["--quick"] if args.quick else [])
        elif stage == "embodied":
            _run_embodied(quick=args.quick)
        else:
            logging.info("stage %r not implemented yet (later phase) - skipping", stage)
        timings[stage] = round(time.time() - stage_started, 1)
        # Printed as each stage ends, not only in the summary: a run killed by a
        # timeout writes no summary, and then the log is all there is to explain
        # where the time went.
        logging.info("stage %r finished in %.1f s", stage, timings[stage])
    # Count every registered prediction's outcome, now that the stages have written them.
    from neuromps.report.audit import main as run_audit

    run_audit([])
    elapsed = round(time.time() - t0, 1)
    logging.info("run_all finished in %.1f s; results in %s", elapsed, outputs_dir())
    _record_timings(timings, elapsed, quick=args.quick, stages=stages)
    harness.record(timings, elapsed, quick=args.quick, stages=stages, all_stages=STAGES)
    return 0


def _run_embodied(quick: bool) -> None:
    """Every fly study after the chip-to-animal study (full runs only)."""
    import importlib.util

    from neuromps.fetch.runner import incomplete_message, run_fetchers
    from neuromps.fly import pipeline

    if quick:
        raise SystemExit("the embodied stage has no --quick variant; run it on a full run")
    if importlib.util.find_spec("mujoco") is None:
        raise SystemExit("the embodied stage needs the embodied extra: pip install -e .[embodied]")
    manifest = run_fetchers(names=EMBODIED_SOURCES, full=True)
    missing = [manifest.sources[n] for n in EMBODIED_SOURCES if manifest.sources[n].status != "ok"]
    if missing:
        raise SystemExit(incomplete_message(missing))
    pipeline.run(pipeline.select(pipeline.STEPS))


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
