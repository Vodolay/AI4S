# Where each result lives

Every number in the technical report comes from a key in `outputs/metrics.json`, written by a
script under `src/`. This is the full index; the report carries a pointer to it.

| result | metrics key | script |
|------------------------------|------------------------------|----------------------------|
| asset counts and provenance | `data_asset`, `data_card` | `neuromps.etl.build` |
| mechanism classification, calibration | `mechanism_baselines` | `neuromps.models.mechanism` |
| conformal sets, ablations | `uncertainty_and_ablations` | `neuromps.models.conformal`, `.ablation` |
| species transfer and shift | `species_transfer` | `neuromps.models.transfer` |
| compartment asymmetry | `compartment_asymmetry` | `neuromps.models.asymmetry` |
| untreated baseline comparison | `baseline_activity` | `neuromps.models.baselines` |
| twin baseline and drive sweep | `twin_hd11_validation` | `neuromps.twin.validate` |
| twin calibration | `twin_hd12_calibration` | `neuromps.twin.calibrate` |
| drug knob directions | `twin_hd13_perturbation` | `neuromps.twin.perturbation` |
| posterior training and coverage | `twin_hd14_inference` | `neuromps.twin.inference` |
| posterior applied to real recordings | `twin_hd15_real_inference` | `neuromps.twin.real_inference` |
| directionality experiment | `twin_hd16_directionality` | `neuromps.twin.directionality` |
