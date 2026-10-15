# UtiTherm-SR / Phase 1 / Step 4B: FSRCNN ×2 Training

This is the **learned thermal super-resolution baseline**, not the proposed
utility predictor. FSRCNN learns to reconstruct full thermal frames from
synthetically downsampled thermal frames. It is trained on FLIR **training**
images only. No object detector is retrained in this step.

## Copy these files into the existing project

- `src/utitherm/sr/fsrcnn.py` — model
- `scripts/train_fsrcnn.py` — YAML-driven training script
- `configs/fsrcnn_x2.yaml` — configuration
- `tests/test_fsrcnn.py` — tests
- `docs/step4b_fsrcnn.md` — this note

Do not delete or replace earlier scripts/checkpoints. No Docker rebuild
required because the project directory is bind-mounted and dependencies
(torch, numpy, Pillow, PyYAML, pytest) are already in the environment.

## First: tests (PowerShell, project root)

```powershell
docker compose run --rm research python -m pytest -q tests/test_fsrcnn.py
```

Expected: `5 passed`.

## Second: training smoke test

```powershell
docker compose run --rm research python scripts/train_fsrcnn.py --config configs/fsrcnn_x2.yaml --smoke
```

Expected: one epoch, 64 training images maximum, up to 16 internal validation
images, outputs in `reports/sr_baseline/fsrcnn_train_v1_smoke/`.
It **does not** publish a model to `models/sr/`.

If output folder already exists, use a different `output` value in the YAML;
we never silently overwrite experiment checkpoints.

## Third: full FSRCNN training (after inspecting smoke output)

```powershell
docker compose run --rm research python scripts/train_fsrcnn.py --config configs/fsrcnn_x2.yaml
```

Produces:

- `reports/sr_baseline/fsrcnn_train_v1/best.pt`, `last.pt`
- `reports/sr_baseline/fsrcnn_train_v1/metrics.jsonl`
- `reports/sr_baseline/fsrcnn_train_v1/run_manifest.json`
- `models/sr/fsrcnn_x2_v1.pt` (copied from best checkpoint)

## Important research controls

- HR thermal **training** images (640×512) are degraded by PIL bicubic to
  synthetic LR (320×256). This should match the LR construction used for
  YOLO training and Bicubic evaluation. Verify before publishing.
- Internal patch-level validation holds out **video-sequence groups** from
  `images/train`; never uses official FLIR `images/val` or video test.
- PSNR is on fixed internal validation *patches*, not a replacement for
  a whole-image evaluation. SSIM is not measured yet.
- FSRCNN model output is unclipped when computing MSE loss; clamp to [0,1]
  when converting its output to 8-bit images for detection evaluation.
- Next deliverable: compare FSRCNN ×2 vs bicubic ×2 at the SAME YOLO `imgsz=640`,
  same detector checkpoint, same val images/labels, with image quality,
  false positives, small-object recall and end-to-end latency.
- Dataset and detector test set are not touched by the training script.
- Benchmark results from this step cannot establish *physical* thermal
  fidelity or the novelty of our selective SR method.
