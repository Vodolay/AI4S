"""Run the registered chip-to-animal study and score every prediction.

``python -m neuromps.fly.study`` runs the design in
`preregistration/fly_chip_to_animal.json` on the FlyWire 783 brain:

1. spontaneous recordings of each drug condition, read by the virtual array;
2. the published behaviour protocols under each condition;
3. pulsed stimulation of every region, for directionality;
4. the sugar -> MN9 replication on 630 and 783.

``--quick`` runs a reduced design (three conditions, one seed, shorter runs)
for the CI path; it scores nothing, because the registered tests are defined on
the full design. Everything written is **simulated**.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from neuromps.fly import connectome, design, direction, figures, jobs, scoring
from neuromps.metrics import update_metrics
from neuromps.parallel import worker_processes
from neuromps.paths import figures_dir, outputs_dir

log = logging.getLogger("neuromps.fly")
PROTOCOLS = ("sugar", "sugar_bitter", "jon_ce")
MEMORY_PER_WORKER_GB = 7.0


def memory_bounded_workers(cap: int = 16, per_worker_gb: float = MEMORY_PER_WORKER_GB) -> int:
    """Workers that fit in available memory: each holds the whole brain (about 7 GB)."""
    try:
        meminfo = Path("/proc/meminfo").read_text()
        kb = int(
            next(
                line.split()[1] for line in meminfo.splitlines() if line.startswith("MemAvailable")
            )
        )
        fits = max(1, int(kb / 1e6 / per_worker_gb))
    except (OSError, StopIteration, ValueError):
        fits = 2
    return min(worker_processes(cap=cap), fits)


def quick_spec(spec: dict) -> dict:
    """The registered design cut down to a CI-sized run."""
    d = json.loads(json.dumps(spec["design"]))
    d["spontaneous"].update(duration_s=2.0, seeds=1)
    d["behaviour"]["trials"] = 2
    d["evoked"].update(seeds=1, duration_s=1.0)
    d["conditions"] = {
        "gaba_block": {**d["conditions"]["gaba_block"], "levels": [0.2]},
        "ach_block": {**d["conditions"]["ach_block"], "levels": [0.4]},
    }
    return {**spec, "design": d}


def _context(spec: dict, con: connectome.Connectome) -> tuple[jobs.Context, list[str]]:
    a = spec["design"]["array"]
    from neuromps.fly import electrodes

    array = electrodes.build_array(
        con.neuropil, a["electrodes_per_region"], a["units_per_electrode"], a["array_seed"]
    )
    regions = sorted(set(array.region.values()))
    p = spec["design"]["pairs"]
    pairs = direction.pair_truth(connectome.neuropil_flow(con), regions, p["min_synapses"])
    ann = connectome.read_annotations()
    dn_ids = ann.loc[ann["super_class"] == "descending", "root_id"]
    dn = con.index_of([r for r in dn_ids if r in set(con.root_ids.tolist())])
    return jobs.Context(spec=spec, pairs=pairs, descending=dn), regions


def _pool(context: jobs.Context, workers: int, release: str = "783") -> ProcessPoolExecutor:
    return ProcessPoolExecutor(workers, initializer=jobs.init, initargs=(context, release))


def run_sweep(context: jobs.Context, regions: list[str], workers: int, quick: bool) -> dict:
    """All 783 simulations; returns raw per-job summaries."""
    d = context.spec["design"]
    conds = design.conditions(context.spec)
    seeds = {k: int(v.rstrip("+")) for k, v in context.spec["fresh_seeds"].items()}
    spont = [(c, seeds["spontaneous"] + k) for c in conds for k in range(d["spontaneous"]["seeds"])]
    trials = [
        (c, p, seeds["behaviour"] + k)
        for c in conds
        for p in PROTOCOLS
        for k in range(d["behaviour"]["trials"])
    ]
    evoked = [
        (r, seeds["evoked"] + k)
        for r in regions[: 8 if quick else None]
        for k in range(d["evoked"]["seeds"])
    ]
    started = time.time()
    with _pool(context, workers) as pool:
        f_spont = [
            pool.submit(jobs.spontaneous, c.overrides, s, c.family == "vehicle") for c, s in spont
        ]
        f_trial = [pool.submit(jobs.behaviour_trial, c.overrides, p, s) for c, p, s in trials]
        f_evoked = [pool.submit(jobs.evoked, r, s) for r, s in evoked]
        f_rep = [
            pool.submit(jobs.replication_trial, seeds["replication"] + k)
            for k in range(d["behaviour"]["trials"])
        ]
        out = {
            "spontaneous": [(c, s, f.result()) for (c, s), f in zip(spont, f_spont, strict=True)],
            "behaviour": [
                (c, p, s, f.result()) for (c, p, s), f in zip(trials, f_trial, strict=True)
            ],
            "evoked": [(r, s, f.result()) for (r, s), f in zip(evoked, f_evoked, strict=True)],
            "replication_783": [f.result() for f in f_rep],
        }
    log.info(
        "783 sweep: %d jobs in %.0f s",
        len(spont) + len(trials) + len(evoked),
        time.time() - started,
    )
    with _pool(context, min(workers, d["behaviour"]["trials"]), release="630") as pool:
        out["replication_630"] = list(
            pool.map(
                jobs.replication_trial,
                [seeds["replication"] + k for k in range(d["behaviour"]["trials"])],
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: run, score and write the study."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="CI-sized run; scores nothing")
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = design.registration()
    spec = quick_spec(spec) if args.quick else spec
    con = connectome.cached()
    context, regions = _context(spec, con)
    workers = args.workers or memory_bounded_workers()
    raw = run_sweep(context, regions, workers, args.quick)
    tables = scoring.tables(raw, context)
    result = scoring.score(tables, context, spec, scored=not args.quick)
    result["registration"] = design.provenance()
    result["deviations"] = {
        "regions_driven_whole_because_smaller_than_registered_drive": jobs.small_regions(
            con, regions, spec["design"]["evoked"]["neurons_driven"]
        )
    }
    out_dir = outputs_dir() / "fly"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables["conditions"].to_csv(out_dir / "conditions_SIMULATED.csv", index=False)
    (out_dir / "chip_to_animal_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    figures.all_figures(tables, result, figures_dir())
    update_metrics("fly_chip_to_animal", scoring.metrics_payload(result))
    for name, entry in result.get("predictions", {}).items():
        print(f"{name:55s} {entry['outcome']:>6}  {entry['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
