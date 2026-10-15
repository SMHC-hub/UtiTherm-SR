# Step 11F diagnostic hotfix — preserve results on reference mismatch

## Problem

On the full 200-frame evaluation, an exact aggregate-control comparison against the frozen Step 10D report raised:

```
ValueError: Frozen full_large_fsrcnn_640 differs from Step10D reference on small_tp; investigate
```

The old script checked that assertion **before** writing any CSV/JSON, so the 200-frame evaluation was lost despite successful inference. The 8-frame smoke cannot verify 200-frame reference agreement.

## Scope of this hotfix

- Replaces `scripts/evaluate_bicubic_first_combined_step11f.py` with a version that saves its raw per-image detections, timing and summary *before* surfacing a reference mismatch.
- Performs the existing full-metrics strict reference check **unchanged**.
- Adds exact per-image cross-checking against `reports/development/step10d_full_200_v1/detection_per_image.csv` for both full Bicubic and full Large FSRCNN.
- Validates the Step10D reference CSV aggregates against its `summary.json`, checks image identity and prevents silent cancellation (e.g., +1 and -1 TP on separate images).
- If any control changes, records differences in `reference_control_differences.csv`, marks the report as `reference_full_controls_verified: false`, prints concise totals, then **raises an error**. In such a case the new metrics are **not verified** and must not be used as a reliable result.
- Does **not** change inference models, renderer, detector, split, selection, latency measurement, or experiment configuration.

## Installation

Extract the ZIP into `F:\UtiTherm-SR`, preserving relative paths. It includes the original frozen Step11F configuration for clarity. Do not change any checkpoint or freeze manifest.

## Check unit tests

```powershell
docker compose run --rm research python -m pytest -q tests/test_bicubic_first_combined_step11f.py tests/test_step11f_control_diagnostic_hotfix.py
```

Expected locally: 55 passed.

## Retry full run into a **new** report folder

```powershell
docker compose run --rm research python scripts/evaluate_bicubic_first_combined_step11f.py --limit 0 --repeats 3 --out reports/development/step11f_full_200_diagnostic_v2
```

If reference agreement fails again, this command will exit with an error **but still save**:

```
reports/development/step11f_full_200_diagnostic_v2/
    detection_per_image.csv
    timing_per_trial.csv
    summary.json
    reference_control_differences.csv      # only if discrepancies exist
```

Review summaries from PowerShell:

```powershell
$r = Get-Content .\reports\development\step11f_full_200_diagnostic_v2\summary.json -Raw | ConvertFrom-Json
$r.reference_full_controls_verified
$r.reference_integrity_failure
$r.reference_control_diagnostic | Format-List
Import-Csv .\reports\development\step11f_full_200_diagnostic_v2\reference_control_differences.csv | Select-Object -First 20 | Format-Table -AutoSize
```

The last `Import-Csv` is only needed when the reference check fails. If the run agrees with Step10D, `reference_full_controls_verified` will be true and the differences CSV will not be created.

## Interpreting the discrepancy

The full-frame Large FSRCNN branches in Step10D and Step11F both call `render_fsrcnn(...,'full')`. The frozen checkpoint hashes agree. A difference may involve CUDA numerical variation, an inference-order-dependent effect, a threshold-adjacent detection, or another implementation/runtime discrepancy; it is **not yet diagnosed**. Count how many image-level records differ and by how much before deciding how to fix reproducibility. Do **not** disable/relax the reference check and do **not** call the experiment successful while controls disagree.

The FLIR development set remains reused/exploratory. Even a verified Step11F run will not be an independent final test.
