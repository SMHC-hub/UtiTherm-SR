# Step 6C2 — Selective FSRCNN x2 (research pilot)

**Objective:** Test whether cropped neural SR helps end-to-end performance when the Step 5B predictor chooses four of 16 regions. Use the same YOLO detector, 640x512 HR input, and selected regions for the two selective methods.

## Files

- `src/utitherm/sr/roi_fsrcnn.py`: crop-aware FSRCNN with LR context halo and Step 6B seam feathering.
- `scripts/benchmark_selective_fsrcnn.py`: matched detection and warm-started randomized repeated latency measurements.
- `tests/test_roi_fsrcnn.py`: CPU verification of neural patches and benchmark accounting.
- This document: `docs/step6c2_selective_fsrcnn.md`.

These are **new** files. Preserve all existing code and trained checkpoints.

## Use (Antigravity PowerShell / Docker)

```powershell
docker compose run --rm research python -m pytest -q tests/test_roi_fsrcnn.py

docker compose run --rm research python scripts/benchmark_selective_fsrcnn.py --config configs/selective_sr_eval.yaml --limit 4 --repeats 3 --out reports/utility/fsrcnn_selective_smoke_v1
```

**Only if the smoke test passes**, run the 20-image internal pilot:

```powershell
docker compose run --rm research python scripts/benchmark_selective_fsrcnn.py --config configs/selective_sr_eval.yaml --limit 20 --repeats 5 --out reports/utility/fsrcnn_selective_val20_v1
```

Uses `models/sr/fsrcnn_x2_v1.pt`, `models/utility/roi_utility_v1.pt`, and the existing YOLO checkpoint. No Docker rebuild or retraining is needed.

## Methods

1. `linear_640`: no learned SR; CPU linear resize.
2. `full_bicubic_640`: full CPU cubic resize.
3. `full_fsrcnn_640`: learned full-image SR.
4. `selective_fullframe_fsrcnn4`: run full FSRCNN, blend only four selected tiles.
5. `selective_roi_fsrcnn4`: run FSRCNN **only on four LR crops + a 16-pixel LR context halo**, blend the same tiles.

The two selective methods must be numerically equivalent within the logged 8-bit tolerance before benchmark measurements are accepted. If mismatch is greater than 1, DO NOT blindly raise the tolerance: inspect halo/border behavior first. The script reports cropping *input pixels*, not claimed FLOPs or power savings.

Ground-truth boxes are used only for offline diagnostic matching; selection sees only LR thermal pixels and tile geometry.

## Outputs

- `summary.json`: actual small-object TPs/false positives, stage latency, CPU/GPU benchmark metadata, ROI numeric equivalence, and limitations.
- `timing_per_trial.csv`: per-image repeated component-level synchronized timing and PyTorch allocated memory.
- `detection_per_image.csv`: detector TP/FP at confidence 0.25, class-aware IoU 0.5.
- `roi_equivalence.csv`: pixel-match audit and model LR input fractions.

## Scientific limitations

- Internal predictor-held-out frames are from **YOLO detector's training split**; never present as final independent evaluation.
- Predictor labels were generated from Bicubic interventions, so the model may select suboptimal FSRCNN regions.
- CUDA timing includes CPU↔GPU movement and SR model costs but excludes file I/O; repeated measurements are paired by image.
- Four individual tile calls incur launch/transfer overhead; cropping may be slower despite lower convolutional pixel count.
- Unlike earlier Step 4A PIL Bicubic, this benchmark uses OpenCV INTER_CUBIC. Compare within this run, not historical AP numbers.
- No standard mAP in this pilot, only class-aware TP/FP/recall. Final held-out FLIR evaluation and stricter controls remain pending.
