# Step 11J — Frozen FLIR Video Test Preparation: Gate 1

**Status:** Source validation **only**. This package deliberately does **not** train, infer, preprocess the test split, choose confidence, or inspect test performance. It establishes whether doing so can be justified without changing the pre-registered protocol.

## Verified information provided by user

- FLIR ADAS v2 train: **10,742 images, 133 videos**; validation: **1,144 images, 17 videos**; video thermal test: **3,749 images, 8 videos**.
- COCO test: **62,317 annotations**, **3,493 images with annotations**; 256 images have no COCO annotations.
- The 3 splits have no shared `extra_info.video_id` identifiers according to previous inventory. This script independently checks them.
- YOLO11n fixed checkpoint has 9 names: `person, bike, car, motor, bus, truck, other_vehicle, light, sign`.
- **These are test COCO category IDs, not trained YOLO class IDs.** The frozen mapping is:

| COCO category | COCO name | YOLO ID | YOLO name |
|---:|---|---:|---|
| 1 | person | 0 | person |
| 2 | bike | 1 | bike |
| 3 | car | 2 | car |
| 4 | motor | 3 | motor |
| 6 | bus | 4 | bus |
| 8 | truck | 5 | truck |
| 79 | other vehicle | 6 | other_vehicle |
| 10 | light | 7 | light |
| 12 | sign | 8 | sign |

`hydrant` (COCO 11), `dog` (COCO 17), and other excluded categories are NOT automatically mapped to known classes. The audit reports excluded annotations. One consequence: detections of real but excluded objects may be counted as FPs by the nine-class evaluation. Preserve this limitation.

## Usage from PowerShell in `F:\UtiTherm-SR`

Extract the ZIP at the project root. It installs **four new files**: YAML, Python, tests, and this document. It does not overwrite models or earlier work.

```powershell
# Step A — CPU unit tests
 docker compose run --rm research python -m pytest -q tests/test_flir_test_integrity_step11j.py

# Step B — metadata & training-conversion parity audit; CPU only
 docker compose run --rm research python scripts/audit_flir_test_integrity_step11j.py --out reports/test/step11j_integrity_audit_v1

# Step C — read audit (created even when parity fails)
 Get-Content .\reports\test\step11j_integrity_audit_v1\summary.json -Raw
```

**Expected local unit test result:** 50 passed. This result does NOT mean that the user's test dataset passes the audit; it must be run on the user's dataset.

The script will report:

- Confirmed 10,742/1,144/3,749 COCO images and 133/17/8 video IDs, zero shared video IDs, every referenced JPG exists and COCO dimensions are 640×512.
- Test COCO annotation coverage and nine-class mapping, including unmapped categories and crowd annotations.
- Image **pixel-exact** matching for six candidate LR degradation procedures against a deterministic sample of **51 existing processed validation PNGs**. These are hypotheses; none is assumed correct.
- Full **1,144-image validation** COCO-to-YOLO bounding-box parity, using clipped normalized coordinates, duplicate-safe per-image matching, and existing train-time val `.txt` labels.

`ready_for_test_preparation` is **true ONLY IF** at least one candidate downsampler matches every sampled validation image, and all validation label files agree with the proposed annotation rules. An integrity failure emits the report and exits with an error. Do not bypass it.

**Do not accidentally call this an external dataset**: it is a sequence-disjoint test split of the same FLIR ADAS v2 data source.

### Why the calibration matters

Source is the COCO-referenced **8-bit JPG**, not the paired radiometric TIFF. The detector/FSRCNN were trained on synthetic 8-bit LR images. We cannot infer from folder names that the original LR generator used PIL bicubic, OpenCV cubic, or other downsampling; mismatches change the experiment's input domain. Similarly, normalizing boxes sounds simple but clipping, class filtering, and duplicate handling may differ from the original converter. Calibration must precede test preparation.

The audit is intentionally read-only with respect to all raw and processed data and never inspects model performance on test videos. After it passes, the separate preparation phase will freeze a test manifest and produce a 320×256 LR test data directory with all 3,749 frames, including empty-label images, without touching the raw test source. Then a **single pre-registered** deterministic test evaluation should compare Full Bicubic and Contrast-4 at 0.25 and 0.40 and report their *absolute* utility, detection recall, false positives, and latency; no test-informed tuning or threshold selection.

**Limitations:** Video ID separation does not prove absence of near-duplicate physical scenes between video IDs. Detector confidence selection was influenced by earlier development experiments; test is for *confirmation*. The test has only 8 sequence groups for uncertainty estimation.
