# Phase C checklist

- [ ] README: plain-English summary for a repair-company reader; dataset and licence; how to run; results tables with test set sizes; "How this project was built" (supported by DECISIONS, CORRECTIONS, PREDICTIONS).
- [ ] docs/RESULTS.md: every metric with how it was measured, read from committed outputs.
- [ ] Limitations: public web images differ from repair photos; 98.0% partial views; small test set (see the CIs); duplicates removed from val; review threshold tuned on validation; no undamaged class; confident errors (≥ 0.9997) bypass any review threshold; A13 finding worded exactly as in DATA_REPORT (visual inspection of a small sample).
- [ ] "How this would extend at a repair business" (future work): estimator corrections as new labels; the review queue; drift monitoring between sites; calibration (for example temperature scaling).
- [ ] PREDICTIONS.md: my reflection; Phase B predictions and actuals.
- [ ] Final check: pytest passes; evaluation re-runs from saved predictions; every number in README and RESULTS matches the outputs.
- [ ] Licence (D-027, option b): keep the full repo private; build a clean public copy without per-annotation data. Audit every committed file, including `splits/crops_manifest.csv`, `outputs/metrics/runs/*/val_probs.csv` and `test_probs.csv`, `outputs/metrics/a13_examples.json`, `splits/dup_review.csv`, `outputs/metrics/dup_candidates.json`, and the A13 table in `docs/DATA_REPORT.md`. Note in the public copy that these regenerate from the scripts once CarDD access is granted; link the public copy in my statement and offer the private repo on request.
