# UtiTherm-SR — Step 6B: True ROI-Only Bicubic and Latency Pilot

## Objective

Step 6A tested detection benefits by generating a **full-frame Bicubic image** and
blending its pixels into 4 predictor-selected tiles. It processed the entire
640×512 image with Bicubic. This benchmark changes **only the preparation
method**, not the predictor, chosen tile indices, YOLO model, or detector
thresholds.

We now compute Bicubic only on four selected regions, with a **2-pixel LR halo**
to preserve OpenCV's 4-tap cubic interpolation. The code asserts the resulting
640×512 image is **pixel-for-pixel identical** to the matched full-frame
Bicubic selective method. This is important: a faster algorithm must not
change the tested detection intervention.

**Full-frame INTER_LINEAR is still calculated for the baseline image.**
Cubic output pixels include each ROI's halo, so the amount of cubic work can be
**more than 25%**, even though just 25% of the frame is enhanced. Avoid assuming
that 25% area produces 75% speed or GPU savings.

## Install

Extract the ZIP into `F:\UtiTherm-SR`, preserving folder structure:

- `src/utitherm/sr/roi_bicubic.py`
- `scripts/benchmark_roi_bicubic.py`
- `tests/test_roi_bicubic.py`
- `docs/step6b_roi_only_bicubic.md`

**Keep all existing files unchanged.** The benchmark reuses your existing
`configs/selective_sr_eval.yaml`, YOLO detector checkpoint, Step 5B predictor
checkpoint, and Step 6A helper functions.

## Antigravity PowerShell commands

Unit tests (CPU; no FLIR dataset or CUDA needed):

```powershell
docker compose run --rm research python -m pytest -q tests/test_roi_bicubic.py
```

Smoke test (4 predictor-validation TRAIN-domain frames):

```powershell
docker compose run --rm research python scripts/benchmark_roi_bicubic.py --config configs/selective_sr_eval.yaml --limit 4 --out reports/utility/roi_compute_smoke_v1
```

After reviewing this output, the complete 20-image internal pilot is:

```powershell
docker compose run --rm research python scripts/benchmark_roi_bicubic.py --config configs/selective_sr_eval.yaml --limit 20 --out reports/utility/roi_compute_val20_v1
```

Existing outputs cannot be overwritten; use a new `--out` name for retries.

## Four controlled conditions

1. `linear_640`: full-frame bilinear, without selection.
2. `full_bicubic_640`: full-frame Bicubic baseline.
3. `fullframe_selective4`: match Step 6A with full-frame cubic then feathered four ROIs.
4. `roi_only_selective4`: same four ROIs, but cubic is applied only to cropped LR
   neighborhoods. Pixel-perfect equality is mandatory.

The predictor runs **once for each image**, chooses identical four tiles for the
two selective methods, and is **not supplied with bounding boxes or GT labels**.
YOLO runs on all four policies using the same checkpoint and `imgsz=640`,
`conf=0.25`, NMS IoU 0.7. GT is only loaded **after** ROI selection.

## Metrics, outputs, limitations

- `summary.json` and `per_image.csv` contain pixel-equivalence confirmation,
  per-method small-object true positives, false positives, selector time,
  CPU preparation time, cubic-output pixel fraction, YOLO detector time, and
  exploratory pipeline time.
- The CPU preparation timing uses warmups, randomized method order and repeated
  in-memory calls. The detector timing uses a single warmed in-memory inference
  pass per condition. File I/O, image decoding and label processing are excluded.
- These timing measurements on 4 or 20 images are **exploratory**. The detector
  has already trained on these training-domain frames, and the predictor's
  checkpoint was selected with this internal validation split.
- Bicubic is a relatively cheap CPU operation. Multiple small crops and neural
  selector overhead may make ROI-only Bicubic **slower** than full-frame Bicubic.
  That is a valid result, not a failure; efficient ROI-only *learned* SR is a
  separate future experiment.
- Cross-run latency comparisons from Step 6A/4A are not standardized. Compare
  the four methods measured **within this benchmark**, and do not claim
  independent generalization or novel efficiency from the internal pilot.
