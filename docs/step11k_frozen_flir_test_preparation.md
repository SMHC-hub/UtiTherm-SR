# Step 11K — Frozen FLIR video-test input preparation

**No inference, no training, no threshold selection.** This package creates a separate, new LR test split; never modifies `data/raw`, the original processed datasets, models, or earlier reports.

## Why this is permitted now

The user ran Step11J successfully:

- Train 10,742 images / 133 video IDs; VAL 1,144 / 17; TEST 3,749 / 8. Video identifiers do not overlap.
- Test COCO annotation records: 62,317, of which **62,015** are retained in the 9-class mapping. The excluded 302 are COCO category 11 (hydrant, 277) and category 17 (dog, 25). This is not evidence that those objects do not exist; they are outside the trained detector's class set.
- The Step11J full validation-label parity is 1,144/1,144, and **PIL BICUBIC on COCO-referenced 8-bit thermal JPG** produced pixel-identical 320x256 LR frames for all 51 sampled validation images. Other candidate resampling methods failed parity.
- Step11J report has `ready_for_test_preparation=true`. This package verifies the report, original COCO hashes, and the frozen mapping again before any file output.

This remains FLIR ADAS v2 sequence-disjoint testing, **not an external cross-dataset test**. Test metadata/labels have been inspected, but no test prediction performance or threshold tuning should occur at this stage.

## Installed files (4, all new)

- `configs/flir_frozen_test_preparation_step11k.yaml`
- `scripts/prepare_frozen_flir_test_step11k.py`
- `tests/test_flir_frozen_test_preparation_step11k.py`
- `docs/step11k_frozen_flir_test_preparation.md`

Requires the Step11J configuration, audit script and successful saved report from prior steps.

## PowerShell execution (`F:\UtiTherm-SR`)

### Step A — CPU tests

```powershell
 docker compose run --rm research python -m pytest -q tests/test_flir_frozen_test_preparation_step11k.py
```

### Step B — Prepare all 3,749 frames (CPU only)

```powershell
 docker compose run --rm research python scripts/prepare_frozen_flir_test_step11k.py
```

Expected: 3,749 grayscale 320x256 `.png` frames and 3,749 matching `.txt` label files, 62,015 retained class-mapped objects across 8 video IDs. The generated label count will include empty-text annotations for images with no *supported* class objects. The script does not assume that COCO's 3,493 annotated images all have supported-class labels.

Output paths are new and **cannot overwrite** existing content:

- `data/processed/yolo_flir_lr_x2_test_step11k_v1/images/test/*.png`
- `data/processed/yolo_flir_lr_x2_test_step11k_v1/labels/test/*.txt`
- `reports/test/step11k_test_preparation_v1/summary.json`
- `reports/test/step11k_test_preparation_v1/manifest.csv`

Each manifest row records the original JPG's checksum, prepared PNG checksum, label checksum, video ID and count. The summary contains dataset totals and SHA256 for the ordered manifest. Written images and labels are round-trip-verified. Paths are sealed after preparation; a rerun with the same output destinations refuses to overwrite them.

### Step C — Read saved summary

```powershell
 Get-Content .\reports\test\step11k_test_preparation_v1\summary.json -Raw
```

### Step D — Optional all-files integrity check

```powershell
 docker compose run --rm research python scripts/prepare_frozen_flir_test_step11k.py --verify-only
```

`--verify-only` hashes and validates every prepared LR PNG and YOLO label file against the saved manifest; it doesn't write outputs.

## Next research gate (NOT part of this package)

**Before looking at test detections**, record a pre-registered evaluation manifest: fixed model checkpoint SHA256, 0.25 and 0.40 thresholds (chosen from earlier development results), original class-aware IoU 0.50 matching, FP penalty 0.25, fixed 4x4 contrast-top4 selector, halo10 feather8, deterministic cuDNN/CUBLAS settings, shared preprocessing, and sequence-cluster confidence intervals for 8 videos. Include Full Bicubic at both thresholds; for a fair systems claim, compare against the strongest validation-selected Bicubic operating point, not only the same threshold. Include Large FSRCNN and random-4 as separate controls if resources allow. **Do not select a winning method or threshold based on test results**.

### Cautions

- FLIR test COCO JPG is an 8-bit image, not its corresponding thermal TIFF; do not silently substitute TIFF.
- PIL BICUBIC here generates **low-resolution input**; full Bicubic at inference remains separate HR rendering, using the same code as before.
- Test annotation categories outside the detector's nine class set are excluded by the already-validated conversion; predictions on unlabeled categories may be counted as FPs.
- Test has only 8 video groups; sequence-cluster intervals will be uncertain.
- The script requires exact Step11J COCO/config hashes. If those differ, STOP and investigate instead of relaxing the checks.
