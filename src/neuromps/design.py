"""What can a two-chamber recording of this length, on this many electrodes, detect?

§17 sweeps a grid - electrode count against recording length, seven feature
sets, two arms - and reports the AUC with which symmetric and asymmetric wiring
separate. That grid answers a question a chip user asks *before* running an
experiment, and this exposes it as a lookup rather than leaving it in a figure.

**Three caveats travel with every answer, because without them the numbers
mislead.** The grid is simulated, so it measures identifiability under a model
we wrote rather than performance on tissue. The classifier was trained on the
same generative model that produced its test set, which is the friendliest
possible case. And §15 put the same question to real devices whose geometry was
fabricated rather than simulated and found no separation at all. Treat a row
here as an upper bound on what the measurement could do, not as a promise.

    python -m neuromps.design --electrodes 12 --duration 30
    python -m neuromps.design --target-auc 0.85
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("neuromps.design")
REPO_ROOT = Path(__file__).resolve().parents[2]

# The arm and feature set §17 reports as its result: rate residualised out, and
# drive left alone. The rate-matched arm fails its own validity check.
HEADLINE_ARM = "free_rate"
HEADLINE_FEATURES = "all_rate_residualized"
CAVEAT = (
    "SIMULATED upper bound: identifiability under a correct model, classifier trained on the "
    "same generative model, and the equivalent real-device test in report section 15 was null."
)


def sweep_rows(metrics: dict) -> list[dict]:
    """The §17 grid, or an empty list when the twin stage has not run."""
    return list(metrics.get("twin_hd16_directionality", {}).get("sweep") or [])


def _lower_bound(row: dict) -> float:
    """The interval's lower bound, which is what a configuration is chosen on."""
    interval = row.get("auc_ci95") or [None, None]
    return float(interval[0]) if interval and interval[0] is not None else float("nan")


def candidates(
    rows: list[dict], arm: str = HEADLINE_ARM, feature_set: str = HEADLINE_FEATURES
) -> list[dict]:
    """The grid restricted to the arm and feature set §17 reports."""
    return [r for r in rows if r.get("arm") == arm and r.get("feature_set") == feature_set]


def at_configuration(rows: list[dict], electrodes: int, duration_s: float, **kwargs) -> dict | None:
    """What the grid says about one electrode count and recording length."""
    for row in candidates(rows, **kwargs):
        if row["electrodes_per_compartment"] == electrodes and row["duration_s"] == duration_s:
            return {**row, "auc_ci95_lower": _lower_bound(row), "caveat": CAVEAT}
    return None


def cheapest_for(rows: list[dict], target_auc: float, **kwargs) -> dict | None:
    """The smallest recording whose interval *lower bound* clears the target.

    Selected on the lower bound rather than the point estimate, for the reason
    section 17 gives: choosing on the estimate picks whichever cell was lucky.
    Cost is electrodes times seconds, since both are what an experiment pays.
    """
    qualifying = [r for r in candidates(rows, **kwargs) if _lower_bound(r) >= target_auc]
    if not qualifying:
        return None
    best = min(qualifying, key=lambda r: r["electrodes_per_compartment"] * r["duration_s"])
    return {
        **best,
        "auc_ci95_lower": _lower_bound(best),
        "target_auc": target_auc,
        "cost_electrode_seconds": best["electrodes_per_compartment"] * best["duration_s"],
        "caveat": CAVEAT,
    }


def table(rows: list[dict], **kwargs) -> list[dict]:
    """The grid as a sorted table, for printing."""
    return sorted(
        (
            {
                "electrodes_per_compartment": r["electrodes_per_compartment"],
                "duration_s": r["duration_s"],
                "auc": r["auc"],
                "auc_ci95_lower": _lower_bound(r),
            }
            for r in candidates(rows, **kwargs)
        ),
        key=lambda r: (r["electrodes_per_compartment"], r["duration_s"]),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: read the §17 grid and answer a design question against it."""
    import argparse

    from neuromps.paths import for_reading, outputs_dir

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--electrodes", type=int, help="electrodes per compartment")
    p.add_argument("--duration", type=float, help="recording length in seconds")
    p.add_argument("--target-auc", type=float, help="AUC lower bound the design must clear")
    p.add_argument("--arm", default=HEADLINE_ARM)
    p.add_argument("--feature-set", default=HEADLINE_FEATURES)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    metrics_path = for_reading(outputs_dir() / "metrics.json")
    if not metrics_path.exists():
        print(f"no metrics at {metrics_path}; run `python run_all.py` first")
        return 1
    rows = sweep_rows(json.loads(metrics_path.read_text()))
    if not rows:
        print("no directionality sweep in metrics.json; the twin stage has not run")
        return 1
    options = {"arm": args.arm, "feature_set": args.feature_set}

    if args.target_auc is not None:
        answer = cheapest_for(rows, args.target_auc, **options)
        if answer is None:
            print(f"nothing in the grid reaches an AUC lower bound of {args.target_auc}")
            return 1
    elif args.electrodes is not None and args.duration is not None:
        answer = at_configuration(rows, args.electrodes, args.duration, **options)
        if answer is None:
            print("that configuration is not in the grid; available combinations:")
            for row in table(rows, **options):
                print(
                    f"  {row['electrodes_per_compartment']:3d} electrodes, "
                    f"{row['duration_s']:5.0f} s"
                )
            return 1
    else:
        for row in table(rows, **options):
            print(
                f"  {row['electrodes_per_compartment']:3d} electrodes  "
                f"{row['duration_s']:5.0f} s   AUC {row['auc']:.4f}  "
                f"lower bound {row['auc_ci95_lower']:.4f}"
            )
        print(f"\n{CAVEAT}")
        return 0

    if args.json:
        print(json.dumps(answer, indent=2))
        return 0
    print(
        f"{answer['electrodes_per_compartment']} electrodes per compartment, "
        f"{answer['duration_s']:.0f} s\n"
        f"  AUC {answer['auc']:.4f}, 95% CI lower bound {answer['auc_ci95_lower']:.4f}\n"
        f"  arm {answer['arm']}, features {answer['feature_set']}\n\n{CAVEAT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
