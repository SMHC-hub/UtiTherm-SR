# Step 11I — deterministic post-NMS confidence and utility audit

## Objective

Determine whether many weak false positives observed in **Step 11H** disappear when applying fixed confidence cuts, and whether the **same thresholds applied to Full Bicubic and Contrast-4** preserve or sacrifice the small-object detections gained by selective neural SR.

The `fp_candidates.csv` file from Step 11H lists **only unpaired added FP candidates**. It does **not** store all detections or recovered small-object TPs. Applying confidence thresholds to that CSV would therefore be misleading. Step 11I instead regenerates **all** predictions under the already-verified deterministic protocol and matches *both* methods to the original ground truth.

## What this experiment actually measures

- The unchanged YOLO checkpoint is executed **once per full-resolution image at conf=0.25**, after the frozen Step 11G rendering procedure. Each method reuses its **exact same NMS-filtered prediction list** to re-evaluate confidence cutoffs **0.25, 0.30, 0.35, 0.40, 0.50**. The larger cutoffs are **offline post-NMS filtering**, *not* new calls to `YOLO.predict(conf=...)`.
- Ground-truth matching exactly duplicates the original Step 4 diagnostic protocol: sort detections by confidence descending, match 1:1 to same-class GT at **IoU>=0.50**, and count any unmatched post-threshold detection as FP. This is *not* mAP, ECE, or probability calibration.
- GT identity sets reveal how many small-object GT detections Contrast-4 **gained versus** and **lost relative to** Bicubic, even where their net TP counts are unchanged.
- On the first threshold (0.25), it requires **exact** agreement with the verified Step 11G detection counts per image, and requires identical rendered-image hashes, prediction fingerprints and selections for **all four original methods**, reproducing Step 11G warmup and first-repeat order.
- The full 200-image run must exactly recover the previously verified **ΔsmallTP=+15, ΔFP=+88, ΔU=-7.00 at conf .25**. If anything differs, fail closed, do not interpret.
- Exploratory **sequence-cluster bootstrap** intervals (17 sequences; 4,000 resamples); timing is *not* re-benchmarked because this is post-NMS bookkeeping, not a faster detector.
- Class-specific output includes the user-identified problematic classes **0=person, 2=car, 8=sign** and other classes. Note the class names come from the loaded detector, not handwritten mappings.

## Requirements

Extract ZIP into `F:\UtiTherm-SR` preserving paths. **All Step 11I files are new.** Preserve prior `configs`, `models`, `data`, `reports`, and previous `scripts`; no replacements are needed.

The original Step11F, Step11G, Step11H scripts must be present, along with:

```
reports/development/step11g_full_200_a/summary.json
reports/development/step11g_full_200_a/detection_per_image.csv
reports/development/step11g_full_200_a/fingerprints_per_image.csv
reports/development/step11g_full_200_b/summary.json
reports/development/step11g_full_200_b/detection_per_image.csv
reports/development/step11g_full_200_b/fingerprints_per_image.csv
```

Both Step11G runs were reported as exactly cross-container verified. Keep `CUBLAS_WORKSPACE_CONFIG=:4096:8` and RTX 5070 runtime the same.

## Usage — PowerShell

Tests (CPU):

```powershell
docker compose run --rm research python -m pytest -q tests/test_confidence_utility_step11i.py
```

Smoke, first 8 frozen frames (smoke metrics cannot be interpreted as population estimates):

```powershell
docker compose run --rm -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" research `
  python scripts/audit_confidence_utility_step11i.py `
  --limit 8 --out reports/development/step11i_smoke_8
```

Complete retrospective audit (only after smoke):

```powershell
docker compose run --rm -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" research `
  python scripts/audit_confidence_utility_step11i.py `
  --limit 0 --out reports/development/step11i_confidence_200_v1
```

Display summary:

```powershell
$r = Get-Content .\reports\development\step11i_confidence_200_v1\summary.json -Raw | ConvertFrom-Json
$r.results_by_confidence | Format-List
```

## Output files

- `summary.json`: version/sha provenance, fixed cutoffs, Bicubic/Contrast-4 small TP, FP, recall, *relative utility*, losses/gains by GT identity, exploratory CIs, warning labels.
- `per_image_confidence.csv`: per-image, per-threshold, per-method detection counts and small-object GT index sets.
- `per_class_confidence.csv`: class-level small GT, TP and FP at each threshold (person/car/sign and six other classes).
- `matched_small_tp_confidence_at_025.csv`: base-threshold matched small-TP confidence distributions for both methods (the higher-threshold matcher is rerun; this table itself must not be used to infer identity changes).
- `false_positive_confidence_at_025.csv`: base-threshold FPs for each method, not merely the unpaired added candidates from Step 11H.
- `reference_mismatches.csv`: **only on mismatch**; do not use results if found.

## Limitations / decisions

- **The FLIR development data have been inspected repeatedly.** The cutoffs here are predeclared for *diagnostic comparison*, **not** legitimate hyperparameter tuning or an independent benchmark. Even if utility becomes positive at a given threshold, selecting it post hoc on this set would be optimistic; validate on separately reserved video sequences.
- The original model was run with `conf=0.25`; evaluating 0.30–0.50 as offline post-NMS filters is not equivalent to retraining, calibration, or changing the detector's intrinsic confidence quality. YOLO confidence is not a calibrated probability by default.
- As a threshold rises, both matched TPs and FPs may be removed. The class-aware **greedy matching** can alter which GTs count as matched, so use the actual re-matcher rather than inferring retention by raw counts.
- Annotation omissions, correlated video frames, confidence selection, FP pairing, and multiple comparisons limit conclusions.
- **No new neural model, new region selector, new thresholds chosen for deployment, or verified improvement over Full Bicubic** is claimed. No new latency improvement is possible solely from this offline analysis.
