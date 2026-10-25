"""Embodied fly, stage 5: simulated movement directions against published fly findings.

``python -m neuromps.fly.embodied_stage5`` applies the registered comparison
(`preregistration/embodied_stage5_real_fly_directions.json`) to the curated findings
of the registered literature search (`real_fly_literature.json`, every candidate with
its classification) and the stage-4a medians: for each comparison, the simulated
direction of the mapped movement under the mapped family, and whether it agrees.
The simulated side is **simulated**; the published side is summarized, not reproduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neuromps.fly import design, scoring
from neuromps.metrics import update_metrics
from neuromps.paths import REPO_ROOT, outputs_dir

REGISTRATION = REPO_ROOT / "preregistration" / "embodied_stage5_real_fly_directions.json"
LITERATURE = Path(__file__).with_name("real_fly_literature.json")
EPS_MM, LINE = 0.01, 0.5
PRIMARY = ("pharmacological", "receptor-genetic")


def simulated_direction(family: str, movement: str, medians: dict, doses: dict) -> str:
    """Sign of the change at the family's lowest dose crossing the stage-4a line; or 'none'."""
    base = medians["vehicle"][movement]
    for condition, _dose in sorted(doses[family].items(), key=lambda kv: kv[1]):
        x = medians[condition][movement]
        if abs(np.log2((x + EPS_MM) / (base + EPS_MM))) >= LINE:
            return "increase" if x > base else "decrease"
    return "none"


def compare(literature: dict, medians: dict, doses: dict) -> list[dict]:
    """Every curated comparison with its simulated direction and agreement."""
    out = []
    for c in literature["comparisons"]:
        sim = simulated_direction(c["family"], c["movement"], medians, doses)
        out.append(c | {"simulated_direction": sim, "agrees": sim == c["published_direction"]})
    return out


def score(rows: list[dict], spec: dict) -> dict:
    """V1 and V2 over pharmacological and receptor-genetic comparisons."""
    primary = [r for r in rows if r["category"] in PRIMARY]
    agree = sum(r["agrees"] for r in primary)
    feeding = [
        r for r in primary if r["family"] == "gaba_block" and r["movement"].startswith("proboscis")
    ]
    found = {
        "V1_the_pharmacology_agrees": (
            {"comparisons": len(primary), "agree": agree},
            len(primary) >= 3,
            len(primary) > 0 and agree > len(primary) / 2,
        ),
        "V2_disinhibition_and_feeding_agree": (
            {"comparisons": len(feeding), "agree": sum(r["agrees"] for r in feeding)},
            len(feeding) >= 1,
            all(r["agrees"] for r in feeding),
        ),
    }
    return {
        n: {
            "test": spec["predictions"][n]["test"],
            "value": v,
            "outcome": ("pass" if ok else "fail") if enough else "uninterpretable",
        }
        for n, (v, enough, ok) in found.items()
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: score the registered comparison."""
    spec = json.loads(REGISTRATION.read_text())
    literature = json.loads(LITERATURE.read_text())
    stage4a = json.loads((outputs_dir() / "fly" / "embodied_stage4a_SIMULATED.json").read_text())
    from neuromps.fly import embodied_stage4a as s4a

    conds = s4a.conditions()
    doses = {
        f: dict(zip(g["condition"], g["dose"], strict=True))
        for f, g in conds[conds["family"] != "vehicle"].groupby("family")
    }
    rows = compare(literature, stage4a["medians"], doses)
    neuron = [r for r in rows if r["category"] == "transmitter-neuron"]
    result = {
        "label": "Simulated movement directions (stage 4a) against published fly findings",
        "predictions": score(rows, spec),
        "comparisons": [
            {
                k: r[k]
                for k in (
                    "id",
                    "category",
                    "family",
                    "movement",
                    "published_direction",
                    "simulated_direction",
                    "agrees",
                )
            }
            for r in rows
        ],
        "transmitter_neuron": {
            "comparisons": len(neuron),
            "agree": sum(r["agrees"] for r in neuron),
        },
        "candidates": {k: len(literature[k]) for k in ("impossible", "excluded", "inaccessible")},
        "registration": design.provenance(REGISTRATION),
    }
    out = outputs_dir() / "fly" / "embodied_stage5_SIMULATED.json"
    out.write_text(json.dumps(scoring.clean(result), indent=2) + "\n")
    update_metrics("embodied_stage5", scoring.clean(result))
    for name, entry in result["predictions"].items():
        print(f"{name:40s} {entry['outcome']:>15}  {entry['value']}")
    for r in result["comparisons"]:
        print(
            f"   {r['id']:26s} {r['category']:18s} published {r['published_direction']:8s} "
            f"simulated {r['simulated_direction']:8s} agrees {r['agrees']}"
        )
    print("transmitter-neuron", result["transmitter_neuron"], "candidates", result["candidates"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
