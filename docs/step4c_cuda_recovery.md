# UtiTherm-SR — Step 4C CUDA OOM Recovery

## Why the previous full evaluation stopped

FSRCNN successfully enhanced the 1,144 validation images and YOLO's standard mAP validation completed. CUDA ran out of memory when the same Python process started the separate small-object recall / false-positive `size_recall` loop at batch=8. The previous run did **not** write the final `summary.json` or `comparison.csv`.

This patch improves memory handling by reducing validation batch to 2 and diagnostic prediction batch to 1 by default, releasing GPU allocations between evaluation phases, and offering safe reuse of the previously generated full-image FSRCNN outputs. No retraining or dataset changes are needed.

## Updated files

- Replace `scripts/evaluate_fsrcnn_baseline.py`.
- Replace `tests/test_fsrcnn_evaluation.py` (three additional recovery tests, 10 total).
- Add this document under `docs/`.

## 1. Verify the fix

```powershell
docker compose run --rm research python -m pytest -q tests/test_fsrcnn_evaluation.py
```

Expected: `10 passed`.

## 2. Complete the interrupted run without regenerating images

```powershell
docker compose run --rm research python scripts/evaluate_fsrcnn_baseline.py --limit 0 --resume-prepared --val-batch 2 --diag-batch 1 --baseline-report reports/sr_baseline/bicubic_full_val_v2/summary.json --out reports/sr_baseline/fsrcnn_eval_full_val_v1
```

The `--out` intentionally points to the **existing interrupted** results directory. `--resume-prepared` validates the image/label filenames and saved PSNR/SSIM CSV, skips SR generation, and reruns the mAP and the previously failed diagnostic evaluation. A new `val_fsrcnn_640_recovery` validation subfolder is created without overwriting the original incomplete report.

Expected start: `REUSING 1144 previously enhanced images and saved quality measurements`. Expected end: `DONE ... summary.json ... comparison.csv`.

### Memory fallback

If CUDA still reports out of memory, close other GPU programs (including GPU-heavy browsers, inference/training scripts), run `nvidia-smi`, and retry only after inspecting resource use. The default diagnostic batch is already 1, so a new OOM at batch=1 needs memory/process investigation rather than simply reducing the batch again.

## Scientific caveats

mAP and size-recall are different metrics; do not compare them as if they are interchangeable. The speed measurements from YOLO validation exclude file input and super-resolution. PSNR/SSIM against the HR reference assess synthetic-degradation reconstruction quality, not real camera fidelity. Dataset conversion duplicate labels and train/val leakage require separate checks before a publication claim.
