# Step 10D — Paired V2 vs V3 end-to-end detection evaluation

## Purpose
Compare learned utility predictors V2 (trained using Bicubic interventions) and V3 (trained using **Large FSRCNN halo=10** interventions) using exactly the same trained Large FSRCNN and frozen FLIR development set. Random four-region selection, full-frame Large FSRCNN, full Bicubic and linear are controls.

## Files
- `configs/large_v2_v3_step10d.yaml`: artifact paths and frozen protocol.
- `scripts/evaluate_v2_v3_large_step10d.py`: evaluation and paired statistics.
- `tests/test_large_v2_v3_step10d.py`: invariant and analysis tests.

The original `scripts/evaluate_large_fsrcnn_step9g.py` and all original checkpoints/configs are left untouched.

## Commands

```powershell
docker compose run --rm research python -m pytest -q tests/test_large_v2_v3_step10d.py
docker compose run --rm research python scripts/evaluate_v2_v3_large_step10d.py --limit 8 --repeats 3 --out reports/development/step10d_smoke_8_v1
docker compose run --rm research python scripts/evaluate_v2_v3_large_step10d.py --limit 0 --repeats 3 --out reports/development/step10d_v2_v3_200_v1
```

`--out` **must** name a new directory. The script refuses to overwrite any output. CUDA GPU 0 is required.

## Comparisons
The legacy Step 9G method labels `predictor_large_roi4_h10/h16` are retained here to identify **V2**, making its results directly comparable with Step 9G. The new labels `predictor_v3_large_roi4_h10/h16` identify **V3**.

Comparison metrics: class-aware small-object TP at IoU 0.5 and conf 0.25, false positives, utility = ΔTP_small − 0.25×ΔFP, measured end-to-end GPU latency (excluding disk IO and loading), sequence-cluster exploratory bootstrap confidence intervals, sequence wins/ties/losses.

All seven Step 9G modes plus two V3 modes are measured in one evaluation process. The method order is randomized by image and repeat using the frozen seed. V2 and V3 each have their own trained-source CSV SHA256 verification via the existing predictor loader. The original frozen Step 7A checkpoint SHA256 checks are preserved, along with frozen image/sequence checks. Both fast scorers are audited against the original scorer in the eight-image smoke run.

## Limitations

- FLIR validation was previously used for development. This is **not independent final testing**; multiple prior looks at the 200 frames can bias subsequent decisions.
- The two utility predictors use different training targets. The matched V2/V3 predictor-level test from Step 10C remains separate from this combined-region end-to-end comparison.
- These predictors' utility labels use detector-TRAIN images. They are not detector out-of-fold predictions; training-domain detection leakage may affect conclusions.
- Utility weights (FP penalty 0.25), tile budget 4, halo 10/16 and detector conf=0.25 are preselected development settings.
- Repeats reduce timing noise but are **not** independent test samples. Sequence-level bootstrap uses only 17 IDs, which may not represent distinct scenes.
- Full Bicubic is still the principal affordable efficiency baseline. Higher Large-SR image quality alone is not evidence of better detection.
- Do not modify or replace model checkpoints, reinterpret negative utility proxy values as combined-region results, or claim significance if CI includes zero.
