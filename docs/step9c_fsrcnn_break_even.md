# UtiTherm-SR — Step 9C: FSRCNN Architecture Break-Even Screen

## Purpose

Determine whether more expensive FSRCNN architectures could make batched, four-region processing faster than running super-resolution on the complete thermal frame. This is a **GPU computational-feasibility screen only**. Do not cite it as proof of detection-accuracy improvement, or as a trained model comparison.

## Package contents

- `scripts/benchmark_fsrcnn_break_even_step9c.py` — benchmark; original Step 9C command preserved.
- `configs/fsrcnn_break_even_step9c.yaml` — dedicated experiment parameters and untrained architecture candidates.
- `tests/test_fsrcnn_break_even_step9c.py` — CPU-only data/settings/model tests.
- `docs/step9c_fsrcnn_break_even.md` — protocol and limitations.

The package does not replace existing `configs/selective_sr_eval.yaml`, FSRCNN weights, or `src/utitherm/sr` modules. It reads the former to discover `train_lr`, and loads the existing trained checkpoint for the original-model baseline.

## Usage (PowerShell from `F:\UtiTherm-SR`)

Install/extract the above files into the matching project folders. Do not replace existing checkpoints.

**Validate first:**

```powershell
docker compose run --rm research python -m pytest -q tests/test_fsrcnn_break_even_step9c.py
```

**Full default screen:**

```powershell
docker compose run --rm research python scripts/benchmark_fsrcnn_break_even_step9c.py --out reports/performance/fsrcnn_break_even_step9c_v1.json
```

**Quick sanity run:**

```powershell
docker compose run --rm research python scripts/benchmark_fsrcnn_break_even_step9c.py --frames 2 --repeats 3 --warmup 2 --out reports/performance/fsrcnn_break_even_step9c_smoke_v1.json
```

Existing CLI override arguments still work. Edit YAML for durable changes; override with CLI only for short tests. You can use another YAML via `--settings path/to/settings.yaml`. The benchmark refuses to overwrite output files.

## Experiment controls

- Thermal LR inputs: 320×256 grayscale; sourced exclusively from the `train_lr` directory in the existing `configs/selective_sr_eval.yaml` by default.
- One trained FSRCNN ×2 checkpoint (original); other variants are randomly initialized and **never trained** in this step.
- Same four fixed grid tiles `[0, 5, 10, 15]` for every candidate. This intentionally removes selector-ranking variability.
- Context halo 16 pixels in the ROI path; rendered output size 640×512.
- Full SR vs batched four-ROI SR, single-frame sequential GPU timings using synchronization and warmup.
- Repeated measurements, randomized image order and alternated method order.
- The 1.107 ms selector cost is an **external estimate** from Step 8B, not measured simultaneously.
- Report records settings SHA256, original FSRCNN checkpoint SHA256, GPU, timing, parameter counts, and limitations.

## Interpretation

`median_sr_saving_ms` = measured full-frame SR median − measured ROI SR median.

`estimated_full_minus_roi_with_selector_ms` = that saving − external selector estimate.

A positive estimated margin suggests a candidate for follow-up, **not** a proven end-to-end speed improvement. Small differences may reflect GPU scheduling/measurement noise. The model is untrained, so visual similarity, PSNR, SSIM, or YOLO TP/FP cannot be inferred from this benchmark.

**Before promoting any candidate:** train on TRAIN only; retest receptive-field halo equivalence; preserve the detector and split; evaluate actual combined-region detection utility and interleaved, same-run full-pipeline latency on separated scenes. Keep the frozen 200-image FLIR validation set for development only, not a pristine final test.

## Important limitations

1. Larger FSRCNN networks here are randomly initialized; output images are meaningless for quality evaluation.
2. ROI input pixels (43,008) are not equivalent to measured FLOPs or latency.
3. Larger receptive fields may invalidate the 16-pixel halo; retest before quality claims.
4. RTX 5070 timing results will depend on CUDA/cuDNN, GPU clocks and batch shape.
5. The current selective-FSRCNN region selector was learned from Bicubic utility labels; those labels would not validate a new FSRCNN candidate.
