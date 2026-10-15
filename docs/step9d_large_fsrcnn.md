# Step 9D1 — Train Large FSRCNN ×2

## Purpose and scientific limits

Step 9C found an **estimated** 0.927 ms SR+selector advantage for a **randomly initialized** Large FSRCNN model on the RTX 5070. That is *not* a proven detection, visual-quality, or complete-pipeline speed improvement. Step 9D1 trains that exact 148,009-parameter candidate and preserves the original trained model.

## Why no pre-filled YAML is in this ZIP

The existing `configs/fsrcnn_x2.yaml` contains the user's actual FLIR dataset paths, learning rate, split, and training settings. We have not seen those values, so inventing a new hard-coded YAML could silently change the experiment. `prepare_fsrcnn_large_step9d_config.py` creates **`configs/fsrcnn_large_x2_step9d.yaml`** from the current baseline on the user's machine, preserving all original values except architecture (`d=168,s=36,m=10`), a conservative batch cap of 8, and *new* output/checkpoint paths. The resulting YAML is a normal, editable file and is hashed by the established trainer.

## Install

Extract the ZIP directly into `F:\UtiTherm-SR` to add `scripts/`, `tests/`, and `docs/` files. The original `scripts/train_fsrcnn.py`, `configs/fsrcnn_x2.yaml`, `models/sr/fsrcnn_x2_v1.pt`, and prior results are unchanged.

## Run in PowerShell from `F:\UtiTherm-SR`

1. `docker compose run --rm research python -m pytest -q tests/test_fsrcnn_large_step9d.py`
2. `docker compose run --rm research python scripts/prepare_fsrcnn_large_step9d_config.py`
3. `Get-Content .\configs\fsrcnn_large_x2_step9d.yaml`
4. `docker compose run --rm research python scripts/train_fsrcnn_large_step9d.py --smoke`
5. Check smoke output, then run `docker compose run --rm research python scripts/train_fsrcnn_large_step9d.py`

The smoke run writes to `reports/sr/fsrcnn_large_x2_step9d_v1_smoke` and **does not** write the canonical checkpoint. The full run writes `reports/sr/fsrcnn_large_x2_step9d_v1/best.pt`, `metrics.jsonl`, `run_manifest.json`, and the separate canonical `models/sr/fsrcnn_large_x2_step9d_v1.pt`.

All output paths are exclusive: existing outputs are never silently overwritten. On reruns, select a new run name and update the generated configuration deliberately.

## Protocol

- Training domain: official FLIR HR `images/train`, same as original trainer.
- LR creation: PIL bicubic x2, exactly as original trainer; **synthetic LR**, not a measurement of real sensor degradation.
- Split: original `group_split`, deterministic, video-sequence-exclusive within the official TRAIN data.
- Training: same MSE, Adam, learning rate, patience, epochs, patch size and seed as the original run. Initial batch <=8 is a VRAM precaution and is recorded in YAML.
- Model: FSRCNN ×2 with `d=168`, `s=36`, `m=10`, **148,009 parameters**. No alterations to loss, detector or other checkpoints.
- Validation: internal held-out *patch* PSNR determines best epoch; this is not a final image-quality/detection score.

## Follow-up evaluation (not performed by this package)

1. Confirm internal validation and held-out full-image PSNR/SSIM on the same image set and preprocessing as the existing FSRCNN.
2. Evaluate full-frame Large-FSRCNN detection relative to linear, Bicubic and original FSRCNN on an audit-logged development split.
3. Validate a sufficient **halo** for the deeper SR model; halo 10 vs 16 tests for the original model do **not** transfer automatically.
4. Measure batch-ROI versus full-frame runtime using the **trained** Large weights, with the optimized selector included and interleaved control runs.
5. The utility predictor was trained on **Bicubic** interventions, so retraining/adapting utility labels for Large-FSRCNN is required before scientific claims about its selective detection quality.
6. Use a genuinely untouched external dataset for final generalization evidence. The repeatedly examined FLIR development set is **not** a pristine final holdout.

## Limitations / safety

- Larger FSRCNN may train more slowly and may produce worse PSNR/detection; Step 9C's speed result was from untrained models.
- The Docker memory cap should remain set near 12 GiB as in this project. PyTorch CUDA VRAM usage differs from host RAM; start with the smoke test.
- Avoid training on FLIR `images/val`; do not mix video groups between internal SR train/val.
- If the smoke run reports CUDA OOM, choose a smaller `batch` (e.g., 4), record the change, and use a fresh smoke-output directory; do not disable integrity/overwrite checks.
