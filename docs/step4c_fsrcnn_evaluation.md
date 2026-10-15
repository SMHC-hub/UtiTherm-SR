# UtiTherm-SR Phase 1 — Step 4C: FSRCNN evaluation

## Purpose
Compare full-frame learned FSRCNN x2 against the **saved** bicubic x2 baseline and a detector-input 640 resize control, on identical FLIR validation files. **No training** is performed.

## Files
- `scripts/evaluate_fsrcnn_baseline.py`
- `configs/fsrcnn_eval.yaml`
- `tests/test_fsrcnn_evaluation.py`
- `docs/step4c_fsrcnn_evaluation.md`

Keep `scripts/evaluate_bicubic_baseline.py`, `src/utitherm/sr/fsrcnn.py`, and both existing model checkpoints in place. Dependencies (PyTorch, cv2, PyYAML, Pillow, Ultralytics) are already in Docker.

## 1. Run unit tests

```powershell
docker compose run --rm research python -m pytest -q tests/test_fsrcnn_evaluation.py
```

## 2. Smoke evaluation (first 30, same selection as Bicubic smoke)

```powershell
docker compose run --rm research python scripts/evaluate_fsrcnn_baseline.py --limit 30 --baseline-report reports/sr_baseline/bicubic_smoke_30/summary.json --out reports/sr_baseline/fsrcnn_eval_smoke_30
```

Expected output `FSRCNN enhanced 30/30 images`, `DONE ...`. Examine image reconstruction and summary before running 1,144 images.

## 3. Full validation (after smoke succeeds)

```powershell
docker compose run --rm research python scripts/evaluate_fsrcnn_baseline.py --limit 0 --baseline-report reports/sr_baseline/bicubic_full_val_v2/summary.json --out reports/sr_baseline/fsrcnn_eval_full_v1
```

Expected `FSRCNN enhanced 1144/1144 images`. Never reuse `--out`: runs intentionally refuse overwrites.

## Outputs
- `summary.json`: detection metrics copied from exact Step 4A reference + new FSRCNN detection; model SHA256; average PSNR/SSIM and FSRCNN forward timing
- `comparison.csv`: mAP50, mAP50–95, small-object recall, false positives, YOLO inference time
- `per_image_quality.csv`: per-image PSNR/SSIM for FSRCNN vs Bicubic against original **HR**; degradation mismatch audit
- `data/interim/fsrcnn_eval/<out-folder>/images/val/`: saved enhanced grayscale validation images for human visual inspection

## Research caveats
- Reconstruction quality is measured on the processed 640×512 HR images under synthetic x2 degradation; this is **not** real paired-sensor thermal fidelity.
- The LR validation images might have been produced with an interpolation method different from PIL bicubic used for FSRCNN training. Review `quality.lr_vs_pil_train_degradation_mae_u8` before concluding the SR model is weak or strong.
- SSIM is 11×11 Gaussian-window grayscale SSIM with standard constants; PSNR/SSIM are averaged per image.
- Official mAP follows Ultralytics 8.4.x; small recall is separately calculated using IoU≥0.50, confidence 0.25, original LR size bins.
- Step 4A's LR/bicubic results are **reused**, not rerun. The script checks image counts, detector SHA256 and LR source path; it does not cryptographically verify original image bytes across experiment dates.
- Model.val 'inference' times do **not** include SR. The script reports the FSRCNN forward stage separately; for publication, measure end-to-end latency with image decode, SR, detector, NMS, and preprocessing using repeated warm measurements and consistent device batches.
- The FLIR validation split is a model-selection resource, not a final independent test. Keep the test set held out.

## Filename-pairing fix (2026-10-14)
LR PNGs may correspond to HR JPEGs. The evaluator now matches by **exact filename stem** rather than matching the PNG extension. It checks every selected pair before writing enhanced images and refuses ambiguous names.

If an earlier run failed and created the folder `fsrcnn_eval_smoke_30`, use a new report name to preserve the failed run:

```powershell
docker compose run --rm research python -m pytest -q tests/test_fsrcnn_evaluation.py
docker compose run --rm research python scripts/evaluate_fsrcnn_baseline.py --limit 30 --baseline-report reports/sr_baseline/bicubic_smoke_30/summary.json --out reports/sr_baseline/fsrcnn_eval_smoke_30_v2
```
