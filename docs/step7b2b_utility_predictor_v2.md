# Step 7B2B — Utility Predictor V2

V2 is **backwards compatible** with the Step 5B / Step 6 utility selector: it uses the same four-head `UtilityPredictor` class and `encode_tile` preprocessing. It does **not** replace `models/utility/roi_utility_v1.pt`.

## Files (add only; do not replace old files)

- `configs/utility_predictor_v2.yaml`
- `scripts/train_utility_predictor_v2.py`
- `tests/test_utility_predictor_v2.py`
- `docs/step7b2b_utility_predictor_v2.md`

## Prerequisites

Existing project files required:

- `scripts/train_utility_predictor.py`, `src/utitherm/utility/predictor.py`
- `reports/utility/probe_train_100_v1/roi_utility.csv` (100 frames, 1,600 ROI records)
- `reports/utility/v2_labels_300_v1/roi_utility.csv` (300 frames, 4,800 ROI records)
- `models/utility/roi_utility_v1.pt`
- `data/processed/yolo_flir_lr_x2/images/train`

No additional packages are necessary beyond the existing `torch`, `opencv-python`, `numpy`, `scikit-learn`, `pyyaml`, `pytest`.

## Run in Antigravity PowerShell at `F:\UtiTherm-SR`

**1. Unit tests**

```powershell
docker compose run --rm research python -m pytest -q tests/test_utility_predictor_v2.py
```

**2. Small smoke training**

```powershell
docker compose run --rm research python scripts/train_utility_predictor_v2.py --config configs/utility_predictor_v2.yaml --smoke
```

**3. Full training after the smoke run passes**

```powershell
docker compose run --rm research python scripts/train_utility_predictor_v2.py --config configs/utility_predictor_v2.yaml
```

This writes new artifacts to `reports/utility/predictor_v2/` and, if not already present, `models/utility/roi_utility_v2.pt`. The script refuses to overwrite an existing experiment directory or canonical checkpoint. To rerun, pass a new `--out` and set a new canonical path in the config if needed.

## What changes vs V1?

1. More data: 100 original + 300 new frames (6,400 ROI measurements).
2. Three-way sequence split: original V1 sequences + some new groups for V2 training; different new groups for V2 early stopping; **separate new groups for the V1-versus-V2 comparison**.
3. BCE classification loss **plus pairwise region ranking loss** based on measured marginal utility, so harmful regions rank below beneficial ones.
4. Compare V1 and V2 on the **same frozen new test-domain sequences** using top-4 ranking proxy, all four head average-precision values, and sequence-bootstrap exploratory intervals.

## Important interpretation limits

- These are **detector-TRAIN-domain images**. Neither this test-domain split nor its comparison is a final independent YOLO evaluation.
- The predictor's target is single-ROI Bicubic utility; *summing four single-ROI values does not equal the actual combined-four-ROI detection result*.
- Improvements in top-4 ranking proxy do not demonstrate mAP improvement, false-positive reduction, computational savings, or research novelty.
- V2 retains the same inference architecture and expected inference cost. Faster inference needs separate engineering work.
- We will evaluate actual combined four-ROI detection on the FLIR development set only **after** reviewing the V1-V2 report, using the original frozen V1 as a direct baseline.
- Out-of-fold detector-generated training labels, additional random seeds, external data and an untouched final test remain needed before publication.
