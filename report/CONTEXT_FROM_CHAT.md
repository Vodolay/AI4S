# Context from planning conversations (Sep 14–20, 2026)

Decisions and framing agreed with Claude (chat) before and during the build. Written for whoever drafts the report.

## Competition
- AI4S Open Innovation: AI for Life Science — "AI + Organ-on-a-Chip" open innovation track, 5th Pazhou Algorithm Competition, hosted on Kaggle.
- Submission: complete runnable work, public code, demo video, technical report via Kaggle Writeup. Deadline Oct 10, 2026; internal target Oct 9.
- Supporting organization: CellShells Bioscience (neural organ-on-a-chip; founding team from a Sorbonne brain-on-chip lab; interests in neural asymmetric connectivity, axon guidance, data standardization/assetization, data IP frameworks, AI digital twins for predictive simulation in neural drug development).

## Why this project
- The general OoC field's bottleneck is standardization and reproducibility (no shared definitions, cell sources, QC). China published its first group standard (intestine-on-a-chip) in April 2024. Our draft data standard is modeled on that structure.
- No tidy public organ-on-chip datasets exist; public neural MEA data is comparatively rich. Hence the pivot from liver to neural.
- Existing public entries to this track used purely synthetic data; we deliberately anchor on real, openly licensed recordings.

## Narrative for judges
raw recordings → standardized, provenance-tracked data asset → calibrated digital twin → predictive inference.
Two layers: (A′) NeuroMPS-Data: schema + standard + real MEA data + rat→hPSC transfer study; (B′) NeuroMPS-Twin: Brian2 two-compartment asymmetric network + simulated MEA readout + sbi inference of connectivity/drug effects, calibrated on A′.

## Scope guardrails
- Research decision-support only. No efficacy, safety, diagnostic, or clinical claims anywhere.
- Every reported number traces to a script writing to outputs/metrics.json.
- Real vs simulated data always labeled in filenames and figure titles.
- One shared MEA feature extractor (features/mea.py) used by both layers.

## Data and licensing decisions
- License policy: CC0, CC BY, and permissive software licenses that allow commercial reuse (MIT etc.). Rationale: competition rules may grant organizers rights to submissions, and the supporting company is commercial; NC-licensed inputs could taint the derived data asset.
- Dropped: Tukker 2020 (CC BY-NC 4.0), optional source with marginal value.
- Dropped: a 3D hiPSC HD-MEA set with 10 chemicals ("available on request").
- Pending: Zenodo 14525182 (Mateus & Aguiar, i3S Porto; asymmetric microchannels MEA + imaging). Zenodo metadata says CC BY 4.0, README says CC BY-NC-ND. Emailed authors on Sep 18 for clarification. Fetcher exists but source is quarantined (status "pending_license"); excluded from run_all and all figures until confirmed. If ND is confirmed: cite paper only, use its published geometry/statistics as twin design parameters and qualitative validation targets.
- Core sources: Hyvärinen et al. Sci Data 2022 (hPSC vs rat cortical MEA, pharmacology); EPA ToxCast MEA acute (17 endpoints, 508 chemicals) and developmental (19 endpoints, 463 chemicals); Sharf et al. hiPSC organoid HD-MEA diazepam dose-response (CC0); Fofie et al. 2025 hiPSC nociceptor Axion dataset (CC0); ScreenNeuroPharm rat bicuculline dose-response (MIT).
- ToxCast cannot join the per-well table (different platform/feature definitions); kept as a separate platform in the asset. Effect-direction sign convention is per assay, verified against cytotoxicity anchors (acute 92.4%, dev 99.1% agreement) and re-checked by a test on every run.

## Regional context (for a possible "future work / collaborators" line, optional)
Quebec groups adjacent to this work: The Neuro's EDDU (T. Durcan; iPSC neurons, organoids, open science), C. Moraes (McGill, engineered tissues), D. Juncker + T. Kennedy (McGill, microfluidic axon-guidance gradients), IRCM axon-guidance labs, Mila/McGill neuro-AI groups for modeling.

## Report requirements agreed
- Technical report sections: problem, data standard, data asset, models, twin, results, limitations, reproducibility, AI-assistance disclosure.
- Kaggle Writeup summarizes the report. Video ≤5 min: 30 s problem, 60 s data asset + standard, 60 s transfer result, 90 s twin + inference, 30 s reproducibility.
- Honest reporting of negative results (e.g., if rat→hPSC transfer is weak) is explicitly acceptable and preferred.
