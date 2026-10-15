# Step 11D — Cheap, pre-SR region predictability over full Bicubic

## Objective

**Before training V4**, determine whether region utility can be predicted from cheap information observable in the **low-resolution thermal image**, with no YOLO or FSRCNN inference needed to produce the features. The fixed target is the Step 11A experiment: replacing **one tile** of a full-Bicubic image with Large-FSRCNN ×2 halo-10 / feather-8 and measuring `ΔTP_small − 0.25 × ΔFP`.

This is **only a screening experiment**, not a learned deployment policy or a test of four simultaneous enhanced regions. The source is 300 detector-training-domain frames / 33 video groups, and the detector and SR model were trained on FLIR train images. These data are not independent generalization evidence.

## Contents

```
configs/bicubic_first_predictability_step11d.yaml
scripts/audit_bicubic_first_predictability_step11d.py
tests/test_bicubic_first_predictability_step11d.py
docs/step11d_bicubic_first_predictability.md
```

Extract while preserving relative paths into `F:\UtiTherm-SR`.

### Tests

```powershell
docker compose run --rm research python -m pytest -q tests/test_bicubic_first_predictability_step11d.py
```

### Full lightweight feature audit (no GPU required)

```powershell
docker compose run --rm research python scripts/audit_bicubic_first_predictability_step11d.py
```

The script expects these existing files:

```
reports/utility/step11a_full_300_v1/incremental_roi_utility.csv
reports/utility/step11a_full_300_v1/summary.json
data/processed/yolo_flir_lr_x2/images/train/*.png
```

It verifies the Step 11A ROI CSV's SHA256 against the Step 11A summary, validates 300 images, 33 groups, 16 ROIs/image, each utility formula, expected baseline, and exact row-major tile geometry. It refuses missing or changed inputs. Full-frame LR images are read only for extracting features.

### Features and models

20 pre-SR LR image/ROI features: intensity mean, std, percentiles, bright/dark fraction, entropy, Sobel gradient statistics, Laplacian, frame context, and tile position. No YOLO bounding boxes/predictions, no neural SR outputs, no labels as inputs.

Sequence-disjoint split (fixed seed 143): **23 train, 5 validation, 5 test video sequences**. Counts of frames per subset may differ.

Rank-only policies: `edge_top4`, `contrast_top4`, `extra_trees_top4`, `logistic_top4`, `first4`, exact expected random4. Baseline: `bicubic_no_neural_sr` = utility 0, neural-enhancement compute 0. Two **optional-abstention** policies (`extra_trees_gated`, `logistic_gated`) choose **up to four tiles**, and can choose none. Their score thresholds are selected **only** using 5 validation sequences from a prespecified candidate grid. The 5 test sequences are evaluated once; do **not** adjust thresholds/models based on the test report. Outcome counts under these rankings are direct lookup of previously measured **single-region** outcomes, summed as a diagnostic **marginal proxy**. Exact random4 is an expectation, *not* a four-region image.

`oracle_top4_hindsight` and `oracle_gated_hindsight` use utility labels (ground truth after measuring interventions), and are **not deployable**. Neither is a guarantee of attainable multi-ROI utility.

### Outputs

```
reports/utility/step11d_predictability_v1/
  summary.json           # aggregate train/val/test proxy and test sequence bootstrap CIs
  sequence_split.json    # exact sequence/image split and checksum
  test_per_image.csv     # per-image proxy outcomes for paired audit
```

The default output folder must not already exist. Do not overwrite results casually. Explicit `--overwrite` is needed to replace an existing report and must not be used for retuning test outcomes.

### Decision rule

Do **not** claim success because a classifier has good positive-label AP or because the hindsight oracle is positive. Examine **held-out test marginal utility** vs zero/Bicubic and exact random expectation, especially the gated policy. If utility advantage is small, nonpositive, or has a sequence-level CI covering 0, **avoid V4**. Even a positive proxy only authorizes a later *actual* combined-ROI detector evaluation + target-GPU latency measurement against full Bicubic, full Large FSRCNN and a cheap heuristic. Selection overhead must be included. An independent detector-domain test is required for scientific claims.

### Caveats

- Source observations are detector-TRAIN domain; neither frame selection nor detector is out-of-fold.
- This is **NOT actual combined four-ROI performance**. Adding marginal gains can be invalid when regions interact through NMS and image context.
- Random method is an analytic expected *sum of isolated effects*, not empirical randomized deployment.
- Sequence bootstrap with only five held-out sequences is exploratory.
- Feature extraction and selector runtime is not yet benchmarked on target hardware; “cheap” refers to input/architecture, **not measured speed**.
- Do not tune on the repeatedly inspected FLIR 200-frame development evaluation split.
