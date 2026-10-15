# Step 7B1 — Diagnose the utility predictor (no retraining)

## Purpose
The 200-frame development experiment showed higher small-object recall with predictor selection than random selection, but more false positives and slower processing. Diagnose which **video sequences** account for that gap before modifying the model. This is a *development analysis*, not a final scientific result.

## Files
- `scripts/diagnose_utility_predictor.py`: audits existing Step 7A experiment files and produces comparison summaries.
- `tests/test_utility_diagnosis.py`: unit/integrity tests; no GPU needed.
- This documentation is stored in `docs/`.

## Run inside Docker
```powershell
docker compose run --rm research python -m pytest -q tests/test_utility_diagnosis.py
```
Then:
```powershell
docker compose run --rm research python scripts/diagnose_utility_predictor.py --report reports/development/eval_200_v1 --manifest reports/development/split_200_v2/manifest.json --roi-csv reports/utility/probe_train_100_v1/roi_utility.csv --out reports/utility/step7b1_diagnosis_v1
```

## Review outputs
- `diagnosis.json`: complete audited summary and exploratory sequence-bootstrap intervals.
- `pairwise_comparisons.csv`: paired small-object gain and false-positive differences.
- `sequence_differences.csv`: comparison by filename-derived video sequence.
- `fp_penalty_sensitivity.csv`: ranking sensitivity if false-alarm penalty varies (analysis only).
- `worst_images_vs_random.csv`: images for later visual failure examination.

## Scientific safeguards
The program checks that all image-method records exist exactly once, original small-object GT is identical across methods, aggregate totals match Step 7A summary, and every evaluated image belongs to the frozen manifest. It refuses to overwrite an earlier report.

Utility uses `ΔsmallTP - 0.25×ΔFP`. A larger utility is better; adding true positives is positive, introducing false positives is negative. Bootstrap confidence intervals use **equal-weight video groups** (not trials as independent data). CIs including zero do **not** establish superiority. The CSV only contains TP/FP per image; it does **not** contain which ROI was selected or prediction scores, so it cannot prove *why* a certain image failed. That requires additional future instrumentation.

**Do not tune the predictor using the 200 FLIR frames and later call those frames a pristine test.** Additional original training-domain sequences and out-of-fold utility labels should be generated before new predictor training. Independent final testing requires a separate untouched set.
