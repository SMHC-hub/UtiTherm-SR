# Step 9D3: Stable Large-FSRCNN Training (UtiTherm-SR)

## Why
The Step 9D1 Large FSRCNN (d=168, s=36, m=10; 148,009 parameters) diverged at original baseline LR 0.001: training MSE reached 1,090,282 on epoch 8 and best internal validation PSNR was 5.37 dB at epoch 7. That checkpoint is **not** ready for detection evaluation.

The learning rate and optimization instability are hypotheses, not proven root causes. This is an explicit controlled retraining with the **same network architecture and sequence-aware dataset split**.

## Files
- `configs/fsrcnn_large_x2_stable_step9d3.yaml` — complete, real configuration; no generator required
- `scripts/train_fsrcnn_large_stable_step9d3.py` — independent new training entrypoint, imports the original `train_fsrcnn.py` dataset/split/evaluator
- `tests/test_fsrcnn_large_stable_step9d3.py` — CPU tests
- This document

## Stability changes
- LR reduced 0.001 → 0.0001; no model change.
- Clip gradients globally to L2 norm 1.0 before optimizer updates.
- Fail fast for non-finite gradients and extreme normalized-pixel batch MSE >10.0 (not silent data skipping).
- Reduce learning rate when validation MSE plateaus (factor .5, patience 4, floor 1e-6).
- Maximum 40 epochs, early stopping patience 10.
- Batch size 8 and 48x48 LR patches; read individual images, not entire FLIR dataset into memory.

**No guarantee** these adjustments will converge; do not promote the checkpoint on training success alone. The script deliberately aborts if loss exceeds the safety threshold and will not copy a canonical checkpoint in that case. Preserve failed output for diagnosis and use a new experiment name for the next attempt.

## Commands (Windows PowerShell from F:\\UtiTherm-SR)

```powershell
# Verify the new training entrypoint and safeguards
 docker compose run --rm research python -m pytest -q tests/test_fsrcnn_large_stable_step9d3.py

# One epoch, four train batches, two internal validation batches
 docker compose run --rm research python scripts/train_fsrcnn_large_stable_step9d3.py --smoke

# Full train only AFTER smoke is reviewed
 docker compose run --rm research python scripts/train_fsrcnn_large_stable_step9d3.py
```

If the `_smoke` report directory already exists, rerunning smoke is intentionally rejected. Rename/archive the old smoke report or change the **run-specific** `output` path in the new YAML **and** adapt the guarded `validate_config` naming rule in the script; do not delete research records without a backup.

## Output and reproducibility
- Smoke: `reports/sr/fsrcnn_large_x2_stable_step9d3_v1_smoke/`
- Full: `reports/sr/fsrcnn_large_x2_stable_step9d3_v1/`
- New checkpoint: `models/sr/fsrcnn_large_x2_stable_step9d3_v1.pt`
- `run_manifest.json`: source, split hashes, config hash, architecture, GPU
- `metrics.jsonl`: epoch MSE, validation PSNR, LR and gradient norm
- Checkpoint retains `state_dict` and `model_kwargs` for existing `load_sr_checkpoint()` compatibility.

## Interpretation and next steps
Compare internal held-out patch PSNR to the original baseline **only with identical validation pairing and degradation**. The dataset path and random seed are matched, but any baseline training protocol differences should be documented. After successful convergence, test full-image quality and end-to-end detection on a designated development set, with the same frozen detector and a V2 predictor that was trained for **Bicubic** utility (so transfer to FSRCNN is unproven). Reassess context halo for the new deeper model—halo 10 results obtained on the original light model do not automatically transfer.

The 200-frame FLIR validation split is development data already studied, not a pristine test set.
