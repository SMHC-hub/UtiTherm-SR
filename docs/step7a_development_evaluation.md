# UtiTherm-SR — Step 7A: Larger, Sequence-Audited Development Evaluation

**Purpose:** Determine whether the frozen utility predictor generalizes from 100 TRAIN images to 200 FLIR validation images (different sequence IDs), and compare real detector outcomes and timing. This is a **development** evaluation because FLIR validation data was used in Step 4. It cannot serve as an untouched final test.

## Files

- `configs/development_eval.yaml` — experiment configuration
- `scripts/prepare_development_split.py` — deterministic, label-blind, sequence-aware subset and audit
- `scripts/evaluate_development_holdout.py` — 8-policy detector and synchronized timing benchmark
- `tests/test_development_holdout.py` — checks for separation, split immutability and metric integrity

## Run in Docker

```powershell
# Step 1: Test
 docker compose run --rm research python -m pytest -q tests/test_development_holdout.py

# Step 2: Freeze a candidate split (200 images, at most three per sequence)
 docker compose run --rm research python scripts/prepare_development_split.py --config configs/development_eval.yaml --target 200 --out reports/development/split_200_v1

# Step 3: Four-image smoke run on the FROZEN split
 docker compose run --rm research python scripts/evaluate_development_holdout.py --config configs/development_eval.yaml --manifest reports/development/split_200_v1/manifest.json --limit 4 --repeats 3 --out reports/development/eval_smoke_4_v1

# Step 4: All 200 image frames. Run only after smoke checks.
 docker compose run --rm research python scripts/evaluate_development_holdout.py --config configs/development_eval.yaml --manifest reports/development/split_200_v1/manifest.json --limit 0 --repeats 3 --out reports/development/eval_200_v1
```

Do not edit the frozen manifest or reuse output directories; new experiment = new unique output folder. The split script errors instead of silently selecting fewer images. If your validation set has fewer eligible sequences under the 3-frame cap, inspect the audit and deliberately adjust `max_per_sequence` or target; do not silently relax train/validation separation.

## What this measures

Eight pipelines all use the same YOLO11n checkpoint and 640-sized detector input: linear resize; full Bicubic; full FSRCNN; fixed/random/predictor selected Bicubic; random/predictor selected batched FSRCNN. The selected methods enhance 4/16 regions. Small-object diagnostic recall and false positives are at IoU 0.5 and confidence 0.25. Ground-truth labels are read **only after inference and selection**. Speed is synchronized and reported separately from disk I/O. Bootstrapping resamples **video sequences**, not repeated trials.

This script does **not** measure official AP or independent final-test generalization, and the existing model was selected using predictor-internal validation. Also, even if name-derived sequences do not overlap, scene duplication needs additional visual/manual checks. These limitations must be disclosed in a paper. For a final untouched test, reserve new held-out sequences or an additional annotated public dataset before future tuning.
