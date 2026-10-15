# Step 7B2A — Fresh Utility Data for V2

We are not training V2 yet. First freeze new *detector-TRAIN-domain* examples whose
video sequence identifiers do not appear in the previous 100-image V1 utility
experiment. The prep script also excludes sequence collisions with FLIR VAL.

1. Test: `python -m pytest -q tests/test_utility_v2_data.py`
2. Freeze sample: `python scripts/prepare_utility_v2_manifest.py --config configs/utility_v2_data.yaml --target 300 --out reports/utility/v2_fresh_split_300_v1`
3. Smoke label collection (6 frames; different output):
   `python scripts/collect_utility_v2_labels.py --config configs/utility_v2_data.yaml --manifest reports/utility/v2_fresh_split_300_v1/manifest.json --limit 6 --out reports/utility/v2_labels_smoke_6_v1`
4. After verifying smoke output, full label collection:
   `python scripts/collect_utility_v2_labels.py --config configs/utility_v2_data.yaml --manifest reports/utility/v2_fresh_split_300_v1/manifest.json --limit 0 --out reports/utility/v2_labels_300_v1`

If the full labeling run gets interrupted, execute the same command with
`--resume` and keep all inputs/paths unchanged. Each complete image is saved as
an atomic JSON progress shard. Do not mix partial CSV files manually.

**Research cautions:** The detector has already seen these training-domain
frames. The labels may be optimistically biased. A publication-grade model
needs out-of-fold detector labels and independent final evaluation.
Sequence IDs from filenames do not guarantee independence of physical scenes.
This phase does not improve inference speed automatically; it collects data.

This package adds new files only; does not replace your V1 predictor/checkpoint.
The Step 5A scoring and ROI blending implementations are intentionally reused.
