# Example records

One record per data level and platform, used by `tests/test_schema.py` to keep the schema, the
standard and the code in sync.

**These are illustrative records, not measurements.** Identity, rights, biology, platform, compound
and provenance fields are real: they are taken from the actual sources, and every `sha256` matches an
entry in `data/raw/MANIFEST.json`. **Feature values are `null`** because the shared feature extractor
(`features/mea.py`) is written in HD4; they are not placeholders to be filled with plausible numbers.

The one exception is `toxcast_fitted_curve.json`, whose `curve` block holds real fitted values for
fipronil on `CCTE_Shafer_MEA_acute_firing_rate_mean` (aeid 2456), read from the ToxCast mc5-6 table.

Validate them with:

```bash
python -m neuromps.qc.schema schema/examples/*.json
```
