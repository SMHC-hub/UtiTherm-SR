# UtiTherm-SR — Phase 1, Step 5B: First utility predictor

## What this is

Exploratory **four-head** small CNN using the **100 training images and 1,600 counterfactual ROI records** from Step 5A. **No new YOLO or FSRCNN training is needed.**

Inputs to the model (available at inference without annotations):
- One low-resolution (80×64) thermal tile
- A resized view of the whole low-resolution thermal image (context)
- Four normalized tile-position/size coordinates

Labels (used offline for training only):
1. `small_gain`: a new small true positive after local enhancement
2. `fp_reduce`: fewer false positives after enhancement
3. `small_loss`: a previously detected small object is lost
4. `fp_increase`: more false positives after enhancement

Combined ranking is **a heuristic**: `P(small_gain) − P(small_loss) + 0.25 × [P(fp_reduce) − P(fp_increase)]`. This predicts a *ranking score* for tiles, **not a calibrated numerical detection gain**. Fixed 4-of-16 budget is 25% of tiles (not proven 25% of SR compute).

## Files

```
configs/utility_predictor.yaml
src/utitherm/utility/predictor.py
scripts/train_utility_predictor.py
tests/test_utility_predictor.py
docs/step5b_utility_predictor.md
```

## Commands (project root, Antigravity PowerShell)

Unit tests:

```powershell
docker compose run --rm research python -m pytest -q tests/test_utility_predictor.py
```

One-epoch smoke training (up to 40 sequence IDs):

```powershell
docker compose run --rm research python scripts/train_utility_predictor.py --config configs/utility_predictor.yaml --smoke
```

Full prototype **only after smoke succeeds**:

```powershell
docker compose run --rm research python scripts/train_utility_predictor.py --config configs/utility_predictor.yaml
```

Output folder `reports/utility/predictor_v1` (smoke in `predictor_v1_smoke`). Checkpoint also copied to `models/utility/roi_utility_v1.pt` after full training only. **Never overwrite prior experiment folders**; specify a new `--out` for new experiments.

Output records: `best.pt`, `summary.json`, `split_manifest.json`, `history.csv`, `val_predictions.csv`. Check `val_head_metrics` average precision and `budget_ranking_proxy.mean_marginal_utility_per_image` (predicted vs random vs first four vs oracle).

## Scientific limits

- Train/validation are **different video-sequence IDs within the YOLO training data**, not the official FLIR detection validation set. The original YOLO checkpoint already trained on those frames, so ROI labels are **in-sample with respect to the detector**.
- Only 100 images (9.1% positive overall utility labels) makes generalization estimates high variance; four event heads can each have even fewer positives. A split by sequence identifier reduces exact sequence leakage but not repeated similar scenes.
- Ground-truth labels are used **only to compute Step 5A training targets**, never predictor inputs.
- Budget comparison sums **single-tile marginal effects** and is just a **ranking proxy**. It does not demonstrate multi-tile efficacy, full-frame detection AP gain or SR compute savings. The whole frame bicubic image is currently precomputed in Step 5A.
- Before novel method claims or publication: gather more sequence-diverse data; produce ROI targets from a detector not trained on those images (out-of-fold); evaluate on an independent held-out split; compare budgeted multi-ROI selection against random, uncertainty, confidence, full-frame SR, and no SR with end-to-end latency and VRAM.
