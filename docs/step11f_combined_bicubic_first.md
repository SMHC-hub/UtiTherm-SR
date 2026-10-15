# Step 11F — Actual combined four-tile Bicubic-first Large FSRCNN

## Motivation and scope
Step 11A's 4,800 training-domain measurements and Steps 11D/11E's retrospective per-tile score sums **do not show what happens when four neural patches are applied together**. Step 11F runs the real combined render and fixed detector on every frame.

The chosen contrast heuristic is **not learned from labels**. It ranks the 16 cells of a 4×4 grid using the LR tile's pixel standard deviation, exactly the Step 11D `tile_std` feature. It uses the top 4 cells, with lower tile index as the deterministic tie-break. The random control uses the same deterministic image-specific `choose_tiles('random4',...)` policy as Step 10D.

## Methods

1. `full_bicubic_640`: full-frame OpenCV Bicubic interpolation; the strong control.
2. `full_large_fsrcnn_640`: trained full-frame Large FSRCNN.
3. `random4_bicubic_first_h10`: Large FSRCNN neural patches blended **over full-frame Bicubic** in four random cells, using halo=10 and feather=8.
4. `contrast4_bicubic_first_h10`: same renderer, highest-contrast four LR cells.

The Step 11A `render_bicubic_first_fsrcnn` is used, **not** the earlier linear-background renderer. Every method runs the same detector `yolo11n_lr_baseline_v1.pt`. Full-frame Bicubic/FSRCNN control totals on all 200 images must match Step 10D (`summary.json`) before results are saved.

## Installation

Extract the ZIP at `F:\UtiTherm-SR` while preserving the `configs`, `scripts`, `tests`, `docs` folders. You must already have the complete Step 11A and Step 10D dependencies installed.

## Test and run

```powershell
docker compose run --rm research python -m pytest -q tests/test_bicubic_first_combined_step11f.py
docker compose run --rm research python scripts/evaluate_bicubic_first_combined_step11f.py --limit 8 --repeats 3 --out reports/development/step11f_smoke_8_v1
docker compose run --rm research python scripts/evaluate_bicubic_first_combined_step11f.py --limit 0 --repeats 3 --out reports/development/step11f_full_200_v1
```

Use a **new --out** path for every attempt. The script will not overwrite existing results.

## Measurement protocol

- Frozen Step 7A validation-development manifest: 200 frames, 17 video sequences; validates images, labels, detector and original model hashes, original train/val sequence exclusions.
- Large FSRCNN checkpoint SHA256 included in output. Training data and frozen manifests are read-only.
- GPU synchronized around stages; 2 warmups, 3 repeats by default, randomized method order per repeat, means of per-image medians.
- `contrast_scores` is computed **inside selection timing**. It uses just the `tile_std` feature, not all 20 Step 11D features. For smoke runs (`--limit<=8`) every contrast score/ranking is checked against Step 11D's original `extract_frame_features` and `choose_indices`.
- Detection reports class-aware matched small-object TP, all TP, FP, small GT, and small-object recall with confidence threshold 0.25 and matching IoU 0.50. These are *diagnostics*, not COCO mAP.
- Utility `ΔsmallTP - 0.25 × ΔFP`. Confidence intervals resample *video sequences*, rather than treating correlated frames as independent. Only 17 video groups; results remain exploratory.
- Outputs `detection_per_image.csv`, `timing_per_trial.csv`, `summary.json`, separate from previous experiments.

## Research limitations

The validation development set has repeatedly influenced past modeling decisions. Do not treat results as a pristine or independently confirmatory test. Contrast top-4 was selected following the training-domain Step 11E analysis; there is model-selection risk. A single deterministic random-four assignment is not a full random-seed study. The method processes 4 ROI patches with overhead and **cannot be faster than plain Bicubic by assumption**. Honest success requires evidence of a meaningful detector gain with justified latency costs and later truly independent data. No novelty or publication claim should be made from this run alone.
