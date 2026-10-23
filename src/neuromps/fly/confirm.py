"""Score addendum 2: is network-burst rate the early chip sentinel, out of sample?

``python -m neuromps.fly.confirm`` reruns the parent design's spontaneous
recordings and behaviour trials on fresh seeds (FlyWire), then scores N1-N4,
reading the MaleCNS table the replication wrote for N3 and N4. Everything
written is **simulated**.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from neuromps.fly import analysis, connectome, design, jobs, scoring
from neuromps.fly.replication import table
from neuromps.fly.study import PROTOCOLS, _context, memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
ADDENDUM = REPO_ROOT / "preregistration" / "fly_confirm_network_bursts.json"
SEEDS = {"spontaneous": 700, "behaviour": 800}


def run(context: jobs.Context, workers: int) -> dict:
    """Fresh-seed spontaneous recordings and behaviour trials of every condition."""
    d = context.spec["design"]
    conds = design.conditions(context.spec)
    spont = [(c, SEEDS["spontaneous"] + k) for c in conds for k in range(d["spontaneous"]["seeds"])]
    trials = [
        (c, p, SEEDS["behaviour"] + k)
        for c in conds
        for p in PROTOCOLS
        for k in range(d["behaviour"]["trials"])
    ]
    started = time.time()
    with ProcessPoolExecutor(workers, initializer=jobs.init, initargs=(context, "783")) as pool:
        f_s = [pool.submit(jobs.spontaneous, c.overrides, s, False) for c, s in spont]
        f_b = [pool.submit(jobs.behaviour_trial, c.overrides, p, s) for c, p, s in trials]
        raw = {
            "spontaneous": [(c, s, f.result()) for (c, s), f in zip(spont, f_s, strict=True)],
            "behaviour": [(c, p, s, f.result()) for (c, p, s), f in zip(trials, f_b, strict=True)],
        }
    log.info("confirmation: %d jobs in %.0f s", len(spont) + len(trials), time.time() - started)
    return raw


def verdicts(frame: pd.DataFrame, spec: dict) -> dict:
    """Per family: first detection by network bursts and by firing rate, against behaviour."""
    d = spec["design"]
    drugs = frame[frame["family"] != "vehicle"].copy()
    drugs["abs_delta_network"] = drugs["delta.network_burst_rate_per_min"].abs()
    threshold, impair = (
        d["chip_detection"]["abs_log2_threshold"],
        d["behaviour_impairment_threshold"],
    )
    return {
        "network_bursts": analysis.sentinel(drugs, threshold, impair, "abs_delta_network"),
        "firing_rate": analysis.sentinel(drugs, threshold, impair, "abs_delta_mfr"),
    }


def _counts(v: dict) -> tuple[int, int, int]:
    """(families scored, network first-or-tie, firing rate strictly later)."""
    scored = [f for f, e in v["network_bursts"].items() if e["chip_first"] is not None]
    net = sum(v["network_bursts"][f]["chip_first"] for f in scored)
    late = sum(v["firing_rate"][f]["chip_first"] is False for f in scored)
    return len(scored), int(net), int(late)


def score(flywire: dict, malecns: dict | None, m0_passed: bool | None, addendum: dict) -> dict:
    """N1-N4 with values and outcomes."""
    n, net, late = _counts(flywire)
    found = {
        "N1_network_bursts_are_an_early_sentinel_flywire": ((net, n), n > 0 and net == n),
        "N2_firing_rate_lags_flywire": ((late, n), late >= 4),
    }
    if malecns is not None:
        mn, mnet, mlate = _counts(malecns)
        found["N3_network_bursts_are_an_early_sentinel_malecns"] = ((mnet, mn), mnet >= 4)
        found["N4_firing_rate_lags_malecns"] = ((mlate, mn), mlate >= 4)
    out = {}
    for name, ((k, total), ok) in found.items():
        outcome = "pass" if ok else "fail"
        if "malecns" in name and not m0_passed:
            outcome = "uninterpretable"
        out[name] = {
            "test": addendum["predictions"][name]["test"],
            "value": f"{k} of {total} families",
            "outcome": outcome,
        }
    return out


def _malecns() -> tuple[dict | None, bool | None]:
    out_dir = outputs_dir() / "fly"
    path = out_dir / "malecns_conditions_SIMULATED.csv"
    if not path.exists():
        return None, None
    result = json.loads((out_dir / "malecns_replication_SIMULATED.json").read_text())
    m0 = result["predictions"]["M0_feeding_circuit_exists"]["outcome"] == "pass"
    return verdicts(pd.read_csv(path), design.registration()), m0


def main(argv: list[str] | None = None) -> int:
    """CLI: run the fresh-seed FlyWire runs (unless --score-only) and score N1-N4."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--score-only", action="store_true", help="reuse the saved confirmation table")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec, addendum = design.registration(), json.loads(ADDENDUM.read_text())
    out_dir = outputs_dir() / "fly"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "confirm_conditions_SIMULATED.csv"
    if not args.score_only:
        context, _ = _context(spec, connectome.cached())
        frame = table(run(context, args.workers or memory_bounded_workers()), spec)
        frame.to_csv(path, index=False)
    frame = pd.read_csv(path)
    flywire = verdicts(frame, spec)
    malecns, m0 = _malecns()
    result = {
        "label": "SIMULATED confirmation of the network-burst sentinel (FlyWire, MaleCNS)",
        "predictions": score(flywire, malecns, m0, addendum),
        "flywire": flywire,
        "malecns": malecns,
        "registration": design.provenance(ADDENDUM),
    }
    (out_dir / "confirm_network_bursts_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("fly_confirm_network_bursts", scoring.clean(result))
    for name, entry in result["predictions"].items():
        print(f"{name:52s} {entry['outcome']:>15}  {entry['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
