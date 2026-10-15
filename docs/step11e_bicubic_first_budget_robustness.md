# Step 11E — Bicubic-first region budget and abstention robustness

## Purpose and interpretation

The preceding Step 11D cheap-feature audit found poor out-of-sequence transfer for Extra Trees and logistic gated models. On the 46-image TEST split, `contrast_top4` had a modest positive retrospective utility proxy (+0.0924/image) but was negative on 39-image VAL (-0.0577/image). Selecting more than zero regions is **not** established to beat full Bicubic. We now investigate fixed **0, 1, 2, 4** budgets, with contrast (LR tile intensity standard deviation) and edge (LR mean Sobel gradient magnitude) ranking. No new model fitting, YOLO inference or neural SR inference.

**Critical limitation:** Utility per image is **SUM of previously measured SINGLE-TILE utility values**. This is a *marginal utility proxy*, **NOT** detection with 1, 2 or 4 simultaneous enhanced tiles. Global detection matching and NMS can make tile interventions nonadditive. Even a hindsight oracle positive proxy is not guaranteed as actual combined-ROI gain.

## Package

```
configs/bicubic_first_budget_step11e.yaml
scripts/audit_bicubic_first_budget_step11e.py
tests/test_bicubic_first_budget_step11e.py
docs/step11e_bicubic_first_budget_robustness.md
```

Extract into `F:\UtiTherm-SR` *while preserving folders*. **The existing Step 11D script is a runtime dependency:** `scripts/audit_bicubic_first_predictability_step11d.py` must remain intact. Step 11E directly imports its validated CSV loader and exact 20-feature extraction; Step 11D is neither edited nor rerun.

## Run

```powershell
docker compose run --rm research python -m pytest -q tests/test_bicubic_first_budget_step11e.py

docker compose run --rm research python scripts/audit_bicubic_first_budget_step11e.py

Get-Content .\reports\utility\step11e_budget_robustness_v1\summary.json -Raw
```

No GPU necessary. This reads original LR frames from `data/processed/yolo_flir_lr_x2/images/train` and labels from `reports/utility/step11a_full_300_v1/incremental_roi_utility.csv`. It uses exactly Step 11D's `sequence_split.json` (23 TRAIN / 5 VAL / 5 TEST video sequences). By design, it *does not* fit a new model. It recomputes intensity-std and Sobel-gradient features for VAL+TEST images.

## Source verification

- Requires Step 11A source summary's SHA256 to match ROI CSV bytes, and both Step 11D saved summary/split to agree with that checksum.
- Validates original 300 images, 33 sequences, all 4,800 source interventions, full Bicubic background, tile geometry, per-tile detection gain/FP formulas and the 0.25 FP penalty.
- Checks no image/sequence leakage across the *unchanged saved* train/val/test split.
- **Replays per-image** `contrast_top4`, `edge_top4`, `random4_expectation`, `first4`, and Bicubic-zero against Step 11D's existing test CSV (tolerance only for six-digit CSV rounding); fails closed on mismatch.
- Output default is new, separate folder and refuses overwrite without explicit `--overwrite`.
- No modification to datasets or original Step 11A/Step 11D outputs.

## Policies

- `bicubic_budget0`: zero extra neural region enhancement; zero *incremental* ROI utility by definition (full Bicubic processing cost excluded).
- `contrast_top{k}` for k=1,2,4, rank by **LR tile intensity std**; ties by lower tile index.
- `edge_top{k}` for k=1,2,4, rank by **LR tile mean Sobel magnitude**.
- `random{k}_expectation`: analytic mean of uniformly choosing k distinct tiles. Not an actual combined k-tile outcome or timing.
- `oracle_top{k}_hindsight`: retrospectively sorts measured single-ROI labels; can select negative utility tiles.
- `oracle_gated{k}_hindsight`: hindsight select up to k only if label >0. Not deployable. These oracles should **not** be called combined-region bounds.
- `contrast_val_selected_gated`: allow no neural enhancement. Select both k in {0,1,2,4} and a within-frame contrast z-score threshold in {0,0.5,1,1.5,2} using **only VAL** utility. Strict threshold condition. Highest mean val utility wins, then fewer selected regions; k=0 wins zero-utility ties. Evaluate that one prechosen rule on TEST. Caveat: 39 val frames/5 groups are too few to avoid rule-selection overfitting.

Printed results report utility per image. They are retrospective lookup estimates; model inputs for contrast/edge come only from LR pixels, with no labels at ranking time. The label file is used only to evaluate the selected regions.

## Outputs

```
reports/utility/step11e_budget_robustness_v1/
    summary.json                 # source locks, val-selected gate, per-method results + test CIs
    per_image_budget_proxy.csv   # retrospective utility on val/test, NOT combined detections
```

The summary includes image wins/ties/losses and **video-sequence-block bootstrap 95% intervals** for test utility proxy comparisons against Bicubic budget 0 and random expectation. Five TEST video groups give wide, exploratory intervals. The TEST groups and `contrast_top4` outcomes were previously examined in Step 11D; **do not treat Step 11E as an untouched test or choose a newly preferred k after inspecting TEST**.

`mean_nominal_processed_lr_pixels_per_image = 8400 * selected_regions / num_images`; this is a symbolic accounting of the per-tile crop pixels recorded in the original *single* ROI labeling experiment (8400 LR pixels), not an actual GPU benchmark. Actual batched ROI memory or wall-clock time **must** be measured separately; the feature extractor cost also needs timing. A 4×4 ROI budget on a 320×256 LR source is not automatically faster than Bicubic, which is already strong.

## Research decision

If validation-picked gated policy enhances zero regions, that is a legitimate negative result: our current cheap features cannot establish a safe advantage over Bicubic. Do not keep tuning TEST scores. If a fixed rule is promising, perform an actual combined-ROI YOLO evaluation on truly independent held-out sequences (ideally a detector not trained on those images), with real end-to-end latency and standard mAP. Do not call these retrospective proxies a novel object detection improvement.
