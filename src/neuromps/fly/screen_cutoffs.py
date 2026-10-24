"""How large a firing-rate change does a regulatory MEA screen need before it calls a hit?

The EPA's MEA screens (ToxCast, CCTE_Shafer endpoints, 20,375 records in the
asset) call a chemical active on an endpoint only when its fitted maximum
response passes the endpoint's cutoff. In the acute assay, the response is normalised to a
positive control at -100 (activity abolished), so a firing-rate cutoff of c
means suppression to (100 - c)% of control; in the developmental assay it is a
percent change of control. Converted to log2, these sit beside the fly study's
precision table: the firing-rate change at which behaviour had already failed.

Descriptive, not a prediction. The cutoff applies to a fitted curve's maximum
response over the tested range, not to one dose, so the comparison is of magnitudes only.
Everything from the fly is **simulated**; the cutoffs are **real**.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from neuromps.metrics import update_metrics
from neuromps.paths import metrics_path, processed_dir

log = logging.getLogger("neuromps.fly")
ENDPOINTS = {
    "acute_firing_rate": "CCTE_Shafer_MEA_acute_firing_rate_mean",
    "dev_firing_rate": "CCTE_Shafer_MEA_dev_firing_rate_mean",
}


def cutoff_as_log2(cutoff_percent: float) -> dict:
    """A percent-of-control cutoff as the log2 change it allows, down and up."""
    down = 1.0 - cutoff_percent / 100.0
    return {
        "down": None if down <= 0 else round(float(np.log2(down)), 3),
        "up": round(float(np.log2(1.0 + cutoff_percent / 100.0)), 3),
    }


def cutoffs(asset: pd.DataFrame) -> dict:
    """Per firing-rate endpoint: the cutoff, its log2 equivalents, chemicals and hits."""
    toxcast = asset[asset["source"] == "toxcast_mea"]
    out = {}
    for name, endpoint in ENDPOINTS.items():
        rows = toxcast[toxcast["curve.endpoint"] == endpoint]
        values = rows["curve.cutoff"].dropna().unique()
        if values.size != 1:
            raise ValueError(f"{endpoint}: expected one cutoff, found {values}")
        cutoff = float(values[0])
        out[name] = {
            "endpoint": endpoint,
            "cutoff_percent": round(cutoff, 2),
            "cutoff_log2": cutoff_as_log2(cutoff),
            "chemicals": int(rows["compound.name"].nunique()),
            "hits": int((rows["curve.hitcall"] >= 0.9).sum()),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: compare the screen cutoffs with the fly's precision table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asset = pd.read_parquet(processed_dir() / "neuromps.parquet")
    found = cutoffs(asset)
    metrics = json.loads(metrics_path().read_text())
    needed = metrics["fly_confirm_network_bursts"]["precision"][
        "abs_log2_mfr_change_at_behavioural_onset"
    ]["flywire_sweep"]
    largest = max(v for v in needed.values() if v is not None)
    acute_down = abs(found["acute_firing_rate"]["cutoff_log2"]["down"])
    found["largest_fly_change_at_behavioural_onset_log2"] = round(largest, 3)
    found["acute_cutoff_over_largest_fly_change"] = round(acute_down / largest, 2)
    update_metrics("fly_screen_cutoffs", found)
    print(json.dumps(found, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
