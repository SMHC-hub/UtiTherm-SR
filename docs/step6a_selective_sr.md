# UtiTherm-SR — Step 6A: End-to-End Selective Enhancement Pilot

## Scientific objective

Does selecting four *jointly enhanced* thermal regions with our trained Step 5B
predictor lead to better **observed** YOLO11n small-object detection than
random selection or choosing the top-left four tiles?

All candidates enter the **same** fixed detector at 640 input size. The test
uses images from the 20 Step 5B predictor-validation sequences, *within the
YOLO training domain*. Ground truth is not a selector input.

## Install files

Copy `configs/selective_sr_eval.yaml`, `scripts/evaluate_selective_sr.py`,
`tests/test_selective_sr.py`, and `docs/step6a_selective_sr.md` into their
matching existing project folders. Do not replace Step 5A/5B files.

## First commands (Antigravity PowerShell)

```powershell
docker compose run --rm research python -m pytest -q tests/test_selective_sr.py

docker compose run --rm research python scripts/evaluate_selective_sr.py --config configs/selective_sr_eval.yaml --limit 4 --out reports/utility/selective_smoke_v1
```

For the complete predictor-held-out pilot, once the smoke test is reviewed:

```powershell
docker compose run --rm research python scripts/evaluate_selective_sr.py --config configs/selective_sr_eval.yaml --limit 0 --with-map --out reports/utility/selective_holdout_v1
```

## Five conditions

- `linear_640`: full-frame LR->640 linear interpolation; no selective change.
- `fixed_first4`: bicubic blended into first four grid tiles.
- `random4`: deterministic random four grid tiles; one seed is only exploratory.
- `predictor4`: top four predicted tile utility scores.
- `full_bicubic_640`: full-image bicubic interpolation.

ROI tiles are 160 x 128 in a 4 x 4 grid. Selective policies change 25% of
HR *pixel area*, with an 8-pixel feathered boundary. Whole-frame bicubic is
computed before composing the four ROI areas to match Step 5A interventions.
**Therefore this script does not establish any SR computation savings.**
A true ROI-only interpolator and rigorous hardware benchmarking come next.

## Outputs

- `reports/utility/selective_smoke_v1/summary.json` — observed multi-region
  small-object TP, recall, false positives, combined utility and rough timing.
- `reports/utility/selective_smoke_v1/per_image.csv` — selections and per-image
  detector outcomes; quality checks and audit trail.
- `data/interim/selective_eval/selective_smoke_v1/<policy>/images/val` — rendered
  policy images with corresponding unchanged YOLO labels.
- `--with-map` computes Ultralytics mAP50/mAP50-95 for each policy, separately
  from the custom IoU=0.50/conf=0.25 diagnostic.

## Key limitations

1. Predictor was trained on 80 different sequence IDs but the model checkpoint
   was selected using these 20 internal validation sequences. The detector had
   already been trained on these training-domain frames. Not an unbiased test.
2. Small differences on 4 or 20 frames can be sampling noise. Report both
   utility deltas and negative/harmful cases; never claim proven superiority.
3. The Step 4A historical Bicubic report uses PIL interpolation; Step 5A/6A
   uses OpenCV cubic interpolation. Compare only policies generated in this run.
4. Timings exclude file I/O, rendering to disk, and expensive dataframe/label
   logic. CUDA is synchronized, but timings are not yet a fair latency study.
5. Random selection should later be repeated over many seeds, and a
   confidence-based ROI selection control must be added.
6. To support publication: out-of-fold detector labeling, independent scene
   groups, held-out FLIR test, rigorous mAP/FP tradeoff, compute and budgets.
