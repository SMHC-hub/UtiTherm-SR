# Step 9F3 — Full-frame Large FSRCNN pixel equivalence audit

## Purpose

Step 9F2 compared four-tile batched SR **halo 10/12** against batched **halo 16**, yielding a max one-level difference; that reference is not true full-frame SR. This audit compares each selected tile directly with the corresponding area of **full-frame Large FSRCNN x2 output** using identical inputs and quantization.

## Data and provenance

- FLIR thermal **training-domain**, V1 predictor held-out sequences (up to 20 images), not independent final evaluation.
- V2 predictor checkpoint; hash-verified by the existing predictor loader against the fresh V2 CSV.
- Trained stable Large FSRCNN checkpoint; model weights never modified.
- Four V2-selected tiles; no ground-truth boxes enter inference.
- Model outputs converted to uint8 by `clamp(0,1)*255` and `round` before comparison.
- Compare only 4 ROI tile interiors; no background or blend pixels dilute the error statistics.

## Commands

From `F:\UtiTherm-SR`:

```powershell
docker compose run --rm research python -m pytest -q tests/test_large_fsrcnn_fullframe_equiv_step9f.py

docker compose run --rm research python scripts/check_large_fsrcnn_fullframe_equiv_step9f.py --out reports/performance/large_fsrcnn_equiv_smoke_step9f_v1.json

docker compose run --rm research python scripts/check_large_fsrcnn_fullframe_equiv_step9f.py --limit 0 --out reports/performance/large_fsrcnn_equiv_20_step9f_v1.json
```

The default YAML limit is 4; `--limit 0` evaluates all existing held-out training-domain images (usually 20). Reports refuse overwrite.

## Interpretation

- `max_abs_pixel_difference_vs_full` of 0 means exact uint8 patch match for all selected tiles sampled.
- If max difference 1, output is *near*-equivalent within one 8-bit intensity; **not** exact equality and **not** evidence of identical detections.
- If halo 16 also differs, do not assume halo 16 is an exact full-frame reference.
- If any halo exceeds threshold, do not silently accept it. Review error magnitude/distribution and run a full detection comparison.

This audit does **not** measure speed or detection quality; Step 9F4 must compare **full Large FSRCNN**, **selective Large FSRCNN** with chosen halo, and **full Bicubic** end-to-end, at fixed detector settings with paired per-image sequence analysis. Development evaluation is not a pristine test. As a reminder, utility V2 was trained on Bicubic interventions; not optimized specifically for Large FSRCNN.
