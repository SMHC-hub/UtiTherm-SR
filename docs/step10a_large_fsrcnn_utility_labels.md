# Step 10A — Large FSRCNN-specific single-ROI utility labels

## Purpose

The Step 9H sequence-cluster analysis did **not** establish a reliable advantage for the V2 selector over random selection. V2 was trained on Bicubic-enhancement labels. Step 10A tests a concrete alternative: collect **Large FSRCNN ×2, halo 10** region-specific detection-utility measurements using the *same ROI rendering method* as Step 9G.

This collects **labels**, not a new predictor or final results.

## Contents

- `configs/large_fsrcnn_utility_step10a.yaml` — frozen choices and existing project paths
- `scripts/collect_large_fsrcnn_utility_step10a.py` — resumable inference collector
- `tests/test_large_fsrcnn_utility_step10a.py` — CPU protocol/logic tests
- This document

## Invariants

1. Read the existing frozen 300-image Step 7B2A **TRAIN** manifest; audit its image/annotation SHA256 hashes, exclusion of V1 ROI sequences, and separation from FLIR VAL sequences via existing `verify_manifest`.
2. The detector and settings are unchanged: YOLO11n, 640×512, confidence 0.25, NMS 0.7, 4×4 ROI grid, feather 8, penalty 0.25.
3. The SR checkpoint is the trained Large FSRCNN from Step 9D3, with architecture `(scale=2, d=168, s=36, m=10)`.
4. Each intervention upscales the full 320×256 LR frame with `INTER_LINEAR`, then replaces **one of 16 ROI tiles** using `render_batched_fsrcnn(..., selected=[tile], feather=8, halo=10)`.
5. The detector predicts on the baseline and each of 16 interventions. Offline GT is used only to measure TP/FP labels; it never enters selector inference.
6. The saved CSV matches the column structure of Step 7B2A's `roi_utility.csv`. **Do not mix SR-specific targets with Bicubic targets as if they were the same intervention.**
7. Each image's results are saved as a JSON shard using an atomic rename; `--resume` requires an identical run-manifest including the SR checkpoint's SHA256.
8. Original datasets, V1/V2 labels, V1/V2 predictor checkpoints, detector checkpoint, and FLIR VAL development results are never modified.

## Run from PowerShell project root

Extract the archive so `configs/`, `scripts/`, `tests/`, and `docs/` merge with your existing folders.

### Unit tests

```powershell
docker compose run --rm research python -m pytest -q tests/test_large_fsrcnn_utility_step10a.py
```

### Four-image smoke test — 64 counterfactual interventions

```powershell
docker compose run --rm research python scripts/collect_large_fsrcnn_utility_step10a.py --limit 4 --out reports/utility/large_sr_labels_smoke_4_step10a_v1
```

### Full frozen sample — 300 images, 4,800 interventions

```powershell
docker compose run --rm research python scripts/collect_large_fsrcnn_utility_step10a.py --limit 0 --out reports/utility/large_sr_labels_300_step10a_v1
```

### Resume if interrupted

Use **exactly the same arguments** as the previous run, adding `--resume`:

```powershell
docker compose run --rm research python scripts/collect_large_fsrcnn_utility_step10a.py --limit 0 --resume --out reports/utility/large_sr_labels_300_step10a_v1
```

Do not change config, checkpoint, manifest, or image limit while resuming. Invalid or partially written shards cause an explicit failure rather than silent use.

## Outputs

- `run_manifest.json`: input configuration, frozen manifest, detector and SR SHA256 IDs
- `shards/<image>.json`: completed per-image metrics for resumability
- `roi_utility.csv`: 16 counterfactual utility records per image
- `baseline_per_image.csv`: baseline linear-upscaled detector counts
- `summary.json`: counts and scientific limitations

## Scientific limitations

- **Detector-in-sample:** the detector was trained on these FLIR TRAIN images. This may make the utility labels overoptimistic. For publication-grade claims, retrain **out-of-fold detectors** for utility-label generation and verify on a truly independent evaluation set.
- All 16 outcomes come from *single-region* interventions; summing their utility scores is not guaranteed to equal the detection utility of enhancing four regions jointly.
- Newly trained SR-specific predictor must have a fresh **sequence-grouped train/validation/test split**. Do not repeatedly tune on the previously analyzed FLIR 200-image development set.
- No accuracy or runtime improvement is implied by producing additional labels. Future comparisons must include full Bicubic, full trained Large FSRCNN, random ROI selection, and learned ROI selection on identical image sets.
