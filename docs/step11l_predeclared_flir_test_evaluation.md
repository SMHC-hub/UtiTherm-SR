# Step 11L — First predeclared, sequence-disjoint FLIR test evaluation

## Purpose and why it matters

Step11K successfully prepared **3,749 320×256, 8-bit thermal LR PNGs** with **3,749 YOLO labels** representing **62,015 supported-class bounding boxes**, across **8 video sequences** with no video IDs overlapping detector training or model-development validation. Step11J established 1,144/1,144 validation annotation conversions and 51/51 exact PIL Bicubic preprocessing matches. These are the first *sequence-disjoint* test frames for this pipeline, but **are part of the same FLIR ADAS v2 dataset family**, not an external cross-dataset validation.

This package makes **no model changes**. The test protocol is declared in `configs/flir_disjoint_test_evaluation_step11l.yaml` and should be left untouched after the first test run. The script reads but never changes the Step11K prepared dataset, pretrained weights or historical reports.

## Locked test plan

- Main comparison (primary): **Contrast top-four neural SR versus Full Bicubic at confidence 0.25**. This is the originally planned evaluation operating point and is the main hypothesis test.
- Secondary (development-identified threshold) comparison: **Contrast top-four versus Full Bicubic at confidence 0.40**. Threshold 0.40 came from previously inspected **development** results. It was not set from these test images.
- Secondary alternative-operating-point comparison: **Contrast at 0.40 versus Bicubic at 0.25** to avoid claiming superiority simply by comparing at the same, potentially suboptimal, threshold.
- Additional controls: Full Large FSRCNN and a fixed-seed random-four Bicubic-first selection at 0.25 and 0.40. The random control is a *single seed*, not multi-seed robustness.
- YOLO inference (pre-NMS and NMS settings) remains unchanged with input confidence `0.25`, NMS IoU `0.70`, imgsz `640`. Confidence `0.40` is **post-NMS filtering of the predictions obtained at 0.25**, not a YOLO rerun at 0.40. Evaluation matching uses **class-aware IoU ≥ 0.50**.
- Utility = small-object true positives − 0.25 × false positives. This utility is a bespoke detection diagnostic, not mAP. Compute total absolute utility, paired change, small-object recall, and deterministic GPU latency. Uncertainty uses a video-cluster bootstrap over only **8 independent sequences**; report exploratory 95% intervals and per-video counts. Do not treat adjacent video frames as independent evidence.
- Exactly the same renderer, contrast ranking, seed, detector checkpoint and Large FSRCNN checkpoint as the **cross-container verified** Step11G development experiment. The deterministic PyTorch protocol is unchanged.

## Required files from earlier verified steps

- `scripts/evaluate_bicubic_first_combined_step11f.py`
- `scripts/evaluate_bicubic_first_deterministic_step11g.py`
- `scripts/prepare_frozen_flir_test_step11k.py`
- The upstream image processing/SR/detector files used by Steps 11F/11G
- `configs/development_eval.yaml`
- `configs/bicubic_first_deterministic_step11g.yaml`
- `configs/flir_frozen_test_preparation_step11k.yaml`
- `reports/development/step11g_full_200_a/summary.json` and complete associated CSVs
- `reports/development/step11g_full_200_b/summary.json` and complete associated CSVs
- `reports/test/step11k_test_preparation_v1/summary.json` and `manifest.csv`
- `data/processed/yolo_flir_lr_x2_test_step11k_v1/images/test/*.png` and `labels/test/*.txt`
- Frozen checkpoints in `models/detection` and `models/sr`

## Execution: PowerShell from `F:\UtiTherm-SR`

**1. CPU unit tests**

```powershell
docker compose run --rm research python -m pytest -q tests/test_frozen_flir_test_step11l.py
```

**2. GPU smoke: one frame from each of the eight test videos.** Smoke is an engineering sanity check *within the sealed test protocol*. **Do not change model/threshold/selection rules based on smoke results**.

```powershell
docker compose run --rm -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" research `
  python scripts/evaluate_frozen_flir_test_step11l.py `
  --limit 8 --repeats 3 --out reports/test/step11l_smoke_a
```

**3. Cross-container smoke confirmation**, new output folder:

```powershell
docker compose run --rm -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" research `
  python scripts/evaluate_frozen_flir_test_step11l.py `
  --limit 8 --repeats 3 --out reports/test/step11l_smoke_b `
  --compare reports/test/step11l_smoke_a/summary.json
```

**4. Full 3,749 images, first deterministic reference:**

```powershell
docker compose run --rm -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" research `
  python scripts/evaluate_frozen_flir_test_step11l.py `
  --limit 0 --repeats 3 --out reports/test/step11l_full_a
```

**5. Full 3,749 images, cross-container verification:**

```powershell
docker compose run --rm -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" research `
  python scripts/evaluate_frozen_flir_test_step11l.py `
  --limit 0 --repeats 3 --out reports/test/step11l_full_b `
  --compare reports/test/step11l_full_a/summary.json
```

**Do not delete or overwrite any prior reports**. If an output directory already exists, the evaluator refuses to run. The full test is substantial because each of four methods is run three times per frame for reproducibility. Use an ordinary terminal with adequate GPU availability; tests use frame-at-a-time processing instead of loading all test images into RAM.

## Outputs

Each run saves these in its newly named report folder:

- `summary.json` with locked provenance and the three predeclared comparisons and two control comparisons; bootstrap intervals; absolute utility for all methods at both thresholds; reproducibility status.
- `detection_per_image_thresholds.csv` with counts for every (image, method, threshold).
- `fingerprints_per_image.csv` with SHA256 of exactly rendered uint8 images, exact YOLO prediction floats, plus chosen region IDs.
- `timing_per_trial.csv` with three GPU measurements per method per frame; each per-frame method median is used in the mean latency.
- Optional `repeat_mismatches.csv` / `cross_run_mismatches.csv` on any failure. A mismatched run is saved but marked unverified and exits with an error.

Note that test counts are not meant to equal Step11G's **development** counts; checks instead enforce **same models, same methods, same environment**, sealed test inputs, within-run repeat fingerprints, and optionally cross-container fingerprint agreement. An unexplained mismatch is a **failure**, never a tolerated error.

## Strict interpretation

1. First assess the **primary** contrast@0.25-vs-Bicubic@0.25 result. Only afterward discuss the two secondary comparisons as predeclared exploratory observations. Do not choose a new threshold using this test split.
2. A positive result at conf 0.40 alone does not imply superiority over Bicubic at a more appropriate operating point or a speed advantage.
3. The eight video groups limit statistical certainty even though thousands of frames are available. Bootstrap intervals are exploratory, not multiple-comparison-corrected hypothesis tests.
4. The same FLIR camera family and annotation limitations remain, including omitted COCO categories. A true cross-dataset test would require a different camera/dataset and ontology controls.
5. Do not use the results to retrain or tune and then reuse these test sequences as an untouched final test. Future adaptations should be assessed on a **new** independent test dataset.
