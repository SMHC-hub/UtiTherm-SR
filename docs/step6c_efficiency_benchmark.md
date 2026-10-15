# UtiTherm-SR — Step 6C1: Controlled Repeated Efficiency Benchmark

## Why

The Step 6B 20-image pilot reported ~17.24 ms for ROI-only selective Bicubic
versus ~12.35 ms for full Bicubic. Those timings came from a small number of
single detector passes. Before optimizing anything, we need repeatable,
synchronized end-to-end timings to determine where the time is spent.

## Install

Copy these **new files** into your existing `F:\UtiTherm-SR` project:

- `scripts/benchmark_step6c.py`
- `tests/test_step6c_benchmark.py`
- `docs/step6c_efficiency_benchmark.md`

All existing model checkpoints, scripts, and files stay unchanged. This script
uses the existing `configs/selective_sr_eval.yaml` and the Step 6B helpers.

## Antigravity PowerShell

Run tests first (no GPU required):

```powershell
docker compose run --rm research python -m pytest -q tests/test_step6c_benchmark.py
```

Then four-image smoke benchmark:

```powershell
docker compose run --rm research python scripts/benchmark_step6c.py --config configs/selective_sr_eval.yaml --limit 4 --repeats 3 --out reports/utility/efficiency_smoke_v1
```

After reviewing the smoke output, full 20-image predictor-validation benchmark:

```powershell
docker compose run --rm research python scripts/benchmark_step6c.py --config configs/selective_sr_eval.yaml --limit 20 --repeats 5 --out reports/utility/efficiency_val20_v1
```

New report directories are required for each run. You must have already
installed Step 6A and Step 6B. If a report path exists, choose a different
`--out` name; **do not delete old research outputs**.

## Conditions

- `linear_640`: LR image enlarged using linear interpolation.
- `full_bicubic_640`: whole-frame Bicubic interpolation.
- `fullframe_selective4`: predict four tiles; blend from full-frame Bicubic.
- `roi_only_selective4`: predict same four tiles; compute Bicubic only around
  selected tile neighborhoods.

The last two outputs **must match pixel for pixel** and have matching detector
TP/FP counts on the first pass. Detector checkpoint and input size are fixed.

## What is measured

Each condition includes warmup, randomized ordering, repeated predictions, and
CUDA synchronization around each stage. Timing tracks:

1. `selector_ms` — construct tile inputs, run neural predictor, choose four.
2. `prep_ms` — generate the HR image (including selected tile blending).
3. `detector_ms` — YOLO inference with its Python prediction and output transfer.
4. `total_ms` — sequential end-to-end stages excluding file reads and GT work.

Saved outputs: `timing_per_trial.csv`, `detection_first_pass.csv`, `summary.json`.
For each method, repeated timings are collapsed to a per-image median, then
summarized across images. Approximate paired bootstrap confidence intervals are
reported; with 20 images they are **exploratory, not definitive**. GPU memory
peak is recorded.

## Scientific limits

- These 20 predictor-held-out sequences are still inside YOLO's **TRAIN** domain.
  They are NOT an independent detector test, and predictor checkpoint selection
  used this validation split.
- Same-resolution full Bicubic is cheap; ROI selector overhead may dominate.
  Do not claim the method saves compute unless the measured **total** pipeline
  latency and uncertainty support it.
- Excludes disk I/O and dataset decoding; host scheduling and GPU clocks may
  still change timings; repeat runs before writing publication claims.
- This is CPU Bicubic ROI enhancement, **not ROI-only FSRCNN**. An efficiency
  advantage from selective learned SR remains a separate hypothesis.

**Next decision:** If selector cost dominates, work on optimizing its tile
encoding/dispatch. If full Bicubic still wins, evaluate ROI-only *learned* SR
under the same accuracy and latency protocol rather than forcing a false speed
claim for Bicubic.
