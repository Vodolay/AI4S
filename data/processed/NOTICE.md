# The committed data asset

`neuromps.parquet` (21,887 harmonized records) and `provenance.json` are the output of the build
recorded in `outputs/metrics.json` (generated 2026-09-22T11:20:51+00:00). They are committed so
that the hosted demo can show the record table; `python run_all.py` rebuilds them from the raw
sources in `data/raw/MANIFEST.json`, which records each source's URL, licence and SHA-256.

The records are derived from eight sources, each under a licence that permits redistribution.
Full citations are in `CITATIONS.md`; cite the original sources when reusing their records.

| Source | Licence |
|---|---|
| `brewer2021` | CC-BY-4.0 |
| `kapucu2022` | CC-BY-4.0 |
| `mateus2024` | CC-BY-4.0 (the deposit's metadata licence; see CITATIONS.md for the README's conflicting line) |
| `screenneuropharm_bic` | MIT (notice below) |
| `sharf2022` | CC0-1.0 |
| `toxcast_mea` | CC0-1.0 |
| `trujillo2019` | CC-BY-4.0 |
| `vassallo2017` | US public domain |

Research decision-support only: nothing in these records is an efficacy, safety, diagnostic or
clinical claim.

## `screenneuropharm_bic` licence notice

Records from `screenneuropharm_bic` are derived from github.com/ScreenNeuroPharm/Bicuculline
(doi:10.5281/zenodo.7665283), distributed under the following licence:

```
MIT License

Copyright (c) 2023 Marti1513

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
