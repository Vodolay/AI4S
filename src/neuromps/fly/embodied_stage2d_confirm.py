"""Embodied fly, stage 2d follow-up: confirming the cell the corrected measure selects.

``python -m neuromps.fly.embodied_stage2d_confirm`` runs the registered design
(`preregistration/embodied_stage2d_confirm_corrected.json`): stage 2d's confirmation at
glutamate excitatory everywhere, adaptation 16 mV and 100 ms, on fresh seeds, with the
audit's corrected rhythm measure in every test. Everything here is **simulated**.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

from neuromps.fly import design, fictive, scoring
from neuromps.fly import embodied_stage2c as s2c
from neuromps.fly import embodied_stage2d as s2d
from neuromps.fly.study import memory_bounded_workers
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

log = logging.getLogger("neuromps.fly")
REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage2d_confirm_corrected.json"
CELL = {"rule": "glut_all", "b_mv": 16.0, "tau_ms": 100.0}
SEEDS = range(1200, 1208)
METHOD = "audited"


def _trial(job) -> dict:
    spikes = s2c.simulate(job)
    return s2c.measure(job, spikes, keep_signals_seed=SEEDS[0], method=METHOD)


def main(argv: list[str] | None = None) -> int:
    """CLI: the registered confirmation."""
    from neuromps.paths import figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = json.loads(REGISTRATION.read_text())
    spec2a = json.loads(fictive.REGISTRATION.read_text())
    rule, b, t = CELL["rule"], CELL["b_mv"], CELL["tau_ms"]
    jobs = [
        (n, d, s, b, t)
        for n in (rule, f"{rule}_rewired")
        for d in s2c.CONFIRM_DRIVES
        for s in SEEDS
    ]
    with ProcessPoolExecutor(
        memory_bounded_workers(per_worker_gb=8.0), initializer=s2d._init
    ) as pool:
        rows = list(pool.map(_trial, jobs))
    table = s2c._frame(rows)
    main_rows = table[table["drive"] != "backward_walking"]
    predictions = s2c.confirm(
        main_rows[main_rows["network"] == rule],
        main_rows[main_rows["network"] == f"{rule}_rewired"],
        spec,
        spec2a,
    )
    result = {
        "label": "SIMULATED confirmation of the corrected-measure cell (stage 2d follow-up)",
        "cell": CELL,
        "predictions": predictions,
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly"
    table.to_csv(out / "embodied_stage2d_confirm_SIMULATED.csv", index=False)
    (out / "embodied_stage2d_confirm_SIMULATED.json").write_text(
        json.dumps(scoring.clean(result), indent=2) + "\n"
    )
    update_metrics("embodied_stage2d_confirm", scoring.clean(result))
    signals = next(r["signals"] for r in rows if r["network"] == rule and "signals" in r)
    s2c.plot_confirmation(
        signals,
        CELL,
        figures_dir() / "embodied_stage2d_confirm_SIMULATED.png",
        seed=SEEDS[0],
    )
    for name, entry in predictions.items():
        print(f"{name:46s} {entry['outcome']:>6}  {entry['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
