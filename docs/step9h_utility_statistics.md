# Step 9H — Exploratory paired detection-utility statistics

## Purpose

Analyze the seven methods in the frozen 200-image / 17-sequence FLIR *development* evaluation (Step 9G), without rerunning GPU inference. This is **not** independent final testing.

## Files

- `configs/large_fsrcnn_utility_step9h.yaml` — frozen source, comparisons, penalty, cluster bootstrap settings
- `scripts/compare_large_fsrcnn_step9h.py` — reproducible audit and paired analysis
- `tests/test_large_fsrcnn_utility_step9h.py` — synthetic unit tests

## Run in project Docker environment

```powershell
docker compose run --rm research python -m pytest -q tests/test_large_fsrcnn_utility_step9h.py
docker compose run --rm research python scripts/compare_large_fsrcnn_step9h.py --out reports/development/large_step9h_stats_v1
```

## Interpretation

For each paired image, define `delta_utility = delta_small_tp - 0.25 * delta_fp`. Sum across all frames for total utility and divide by 200 for mean per image. For exploratory confidence intervals, resample **17 complete video-sequence clusters** with replacement 10,000 times, and compute the image-weighted mean for each resample. Report percentile 2.5/97.5 boundaries and sequence-level wins/ties/losses.

If a CI includes zero, there is insufficient evidence for a dependable advantage. Even if it excludes zero, this repeatedly studied FLIR development set cannot establish final independent performance. The utility FP penalty is subjective; robustness at multiple penalties should be evaluated before a final claim. Detector input, checkpoint and threshold controls are those of Step 9G. This analysis does not compute official mAP, evaluate new model weights, or improve performance.

## Outputs

- `summary.json`: hashes, method pairs, exact deltas, 95% exploratory cluster-bootstrap intervals and limitations
- `sequence_differences.csv`: per-video utility differences to diagnose which sequences drive results

The script refuses to overwrite an existing output directory, duplicate method/image rows, inconsistent images/ground truth, or unexpected image/sequence counts. It doesn't modify model checkpoints or training results.
