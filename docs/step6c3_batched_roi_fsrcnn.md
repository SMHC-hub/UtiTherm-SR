# Step 6C3 — Batched ROI FSRCNN (GPU efficiency experiment)

**Research stage:** exploratory engineering benchmark; **not a held-out final detector test**.

## Goal

Compare Step 6C2's four separate FSRCNN ROI forward passes with **one batch of four ROI crops**. Keep the trained detector, predictor and FSRCNN checkpoints unchanged.

The trained utility predictor chooses exactly 4 of the 16 LR 4×4 grid tiles. In Step 6C2, the ROI FSRCNN model received four variable-size LR crops, resulting in four separate GPU passes and significant overhead.

In Step 6C3, each chosen 80×64 LR tile is placed within a **112×96 LR window** (16-pixel context margin on each side when possible). At image borders, the window *shifts inside the image*; it does not add artificial pixels outside the physical frame. These fixed-size windows form a `[4,1,96,112]` tensor so a single FSRCNN forward pass can process all four. The same feather mask and linear baseline as Step 6C2 reconstruct the final 640×512 image.

**Trade-off:** Four windows process 43,008 LR input pixels (52.5% of an 81,920-pixel LR frame), versus about 46.1% using the smaller clipped crops of Step 6C2. One model call may still be faster despite more input pixels. We must **measure** it.

## New files

- `src/utitherm/sr/roi_fsrcnn_batched.py` — uniform-window crop logic, single-batch neural inference and feathered composition
- `scripts/benchmark_batched_fsrcnn.py` — randomized repeated comparison with Stage 6C2, pixel equivalence checks, detection TP/FP, GPU timing and memory
- `tests/test_batched_fsrcnn.py` — CPU unit tests including actual one-forward-pass assertion and comparison against serial/full-frame outputs
- `docs/step6c3_batched_roi_fsrcnn.md` — this document

Merge into your existing `F:\UtiTherm-SR` folder. Keep older scripts, config, all datasets, and the model checkpoints unchanged. No Docker rebuild should be necessary.

## Commands (Antigravity PowerShell)

```powershell
docker compose run --rm research python -m pytest -q tests/test_batched_fsrcnn.py
```

```powershell
docker compose run --rm research python scripts/benchmark_batched_fsrcnn.py --config configs/selective_sr_eval.yaml --limit 4 --repeats 3 --out reports/utility/fsrcnn_batched_smoke_v1
```

If the smoke experiment passes:

```powershell
docker compose run --rm research python scripts/benchmark_batched_fsrcnn.py --config configs/selective_sr_eval.yaml --limit 20 --repeats 5 --out reports/utility/fsrcnn_batched_val20_v1
```

Always choose a new `--out` path for reruns. `--limit 0` uses all 20 eligible validation sequences in the *current* 100-image ROI measurement data; it does **not** expand the evaluation to an independent 100–300-image set.

## What is compared?

1. Original LR to 640 (linear resize)
2. Full Bicubic
3. Full FSRCNN
4. Selective FSRCNN using full-frame FSRCNN computation
5. Selective FSRCNN with **four sequential** cropped forward passes (Step 6C2)
6. Selective FSRCNN with **one batched** cropped forward pass (Step 6C3)

All methods use the exact same YOLO detector and image order controls. The three selective methods use the same predictor-chosen tiles per image.

## Outputs

`summary.json`, `timing_per_trial.csv`, `roi_equivalence.csv`, `detection_per_image.csv` under the specified `--out` directory.

Summary includes mean of per-image median timings, paired bootstrap uncertainty based on unique image units, each method's TP/FP at confidence 0.25, maximum 8-bit image deviation from the full-frame reference, and PyTorch peak GPU allocation.

**Equivalence guard:** default `--max-diff 1` requires all output pixels to be within one grayscale level of reference. If this fails, inspect image mismatch and crop boundary handling; do **not** relax tolerance merely to force the benchmark to pass. Slight quantization differences can affect YOLO predictions and must be reported.

## Caveats for publication

- The selector was trained on **Bicubic ROI outcomes**, not FSRCNN ROI outcomes. Transfer of utility may fail.
- The current 20 sequence-held-out images are **inside the detector training domain**; these are not independent detector evaluation data. The predictor checkpoint was chosen using this internal validation set.
- The method reduces the number of separate SR forward passes from 4 to 1, but **may not** reduce wall-clock latency, GPU memory or actual operations.
- All processing runs per-image; no claims about stream throughput, device energy or sensor fidelity are supported.
- Once architecture is stable, collect additional independent 100–300 image sequences and reserve a truly untouched final test set.
