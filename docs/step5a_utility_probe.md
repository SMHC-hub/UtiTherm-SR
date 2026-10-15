# UtiTherm-SR — Phase 1, Step 5A: ROI utility-label probe

## Research purpose

Before training a utility predictor, experimentally measure whether **enhancing an individual region** changes small-object detection when a **fixed YOLO11n** is used. This is an *offline exploratory intervention study*, not yet an end-to-end selective-SR system.

## What the script does

1. Reads only `data/processed/yolo_flir_lr_x2/images/train` and its matching `labels/train` (never official detection `val`/`test`).
2. Converts each 320×256 grayscale LR frame into 640×512 via OpenCV `INTER_LINEAR` (common reference).
3. Computes a full 640×512 `INTER_CUBIC` interpolation reference once (for controlled probes only).
4. Creates a 4×4 grid over the HR frame. For each of 16 tiles, *only that tile* is changed from the linear reference to a feathered cubic version. No annotated bounding box decides where to enhance.
5. Applies **the same frozen YOLO11n checkpoint** to the reference frame and each altered frame, always at input 640.
6. Counts class-aware, one-to-one matches to ground-truth at IoU≥0.50, confidence≥0.25, using original LR pixel units to identify small objects.
7. Writes CSV rows with detection changes from each intervention and a defined exploratory score:

   `utility_score = Δ(number of matched small ground-truth boxes) − 0.25 × Δ(false positives)`

   Positive labels indicate an improvement *under this specific utility definition*. It is not an mAP difference or a guarantee of improvement.

## Install

Merge these **new** files into project root, without overwriting existing code:

- `configs/utility_probe.yaml`
- `scripts/probe_roi_utility.py`
- `tests/test_roi_utility_probe.py`
- `docs/step5a_utility_probe.md`

## Smoke test

```powershell
docker compose run --rm research python -m pytest -q tests/test_roi_utility_probe.py

docker compose run --rm research python scripts/probe_roi_utility.py --config configs/utility_probe.yaml --limit 6 --out reports/utility/probe_smoke_v1
```

Expected: 6 training frames, 16 interventions each, 96 rows in `roi_utility.csv`.

Outputs: `roi_utility.csv`, `baseline_per_image.csv`, `manifest.json`.

## Interpretation and limitations

- Results are **not** the official detection validation/test scores; this probe uses detector-training images and likely has in-sample bias.
- Training a trustworthy predictor later requires **sequence-grouped separation** and preferably out-of-fold/scout-model predictions so the utility labels are not scored solely with an in-sample detector.
- Many ROI interventions may produce zero improvement. Assess label balance and signal before building a predictor; do **not** assume improvement.
- A full-image cubic reference is calculated for controlled intervention generation; **no computation saving has yet been measured or claimed**.
- The exact interpolation / padding / rectangular-validation convention differs from the earlier `YOLO.val` baseline. Future comparisons must rerun all candidates under identical inference shapes and profiling conditions.
- Enhancement borders are feathered to minimize seams, though they may still affect predictions.
- If the smoke output shows positive labels, it does not prove a novel method; it only suggests a candidate learning target.
- Larger runs should sample hundreds of **sequence-diverse training frames**, not blindly process all 10k images (17 inferences per image) before examining signal and label quality.
