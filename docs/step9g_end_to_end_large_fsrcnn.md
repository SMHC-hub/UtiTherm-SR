# Step 9G — Large FSRCNN: detection × efficiency

## Research question

Under a fixed four-of-sixteen tile budget, does V2-guided batched Large FSRCNN
improve diagnostic small-object detection utility and reduce true end-to-end latency
relative to full-frame Large FSRCNN? Does it compete with **full Bicubic**?

## Design

Seven policies use exactly the same FLIR 200-image *development* set across
17 video-sequence IDs, the same YOLO11n detector at 640×512, confidence 0.25,
NMS IoU 0.7, diagnostic matching IoU 0.5, seed 42 and utility
`ΔsmallTP − 0.25 × ΔFP`.

- `linear_640` — control; no SR
- `full_bicubic_640` — cheap and strong baseline
- `full_large_fsrcnn_640` — **trained** 148,009-param Large FSRCNN
- `random_large_roi4_h16` and `predictor_large_roi4_h16`
- `random_large_roi4_h10` and `predictor_large_roi4_h10`

The predictor is **V2**, using the optimized FastScorer from Step 8B. The ROI
enhancement is **batched** with identical feather blending, and exactly four
tiles. Random selections are deterministic by filename/seed and the same
between halo choices. SR weights and detector weights do not change across
methods. Ground truth is read only after all predictions are produced.

## Install

Extract the package at `F:\UtiTherm-SR`, merging `configs/`, `scripts/`,
`tests/`, and `docs/`. Do **not** replace original Step 7A or Step 8C files.

### Unit tests

```powershell
docker compose run --rm research python -m pytest -q tests/test_large_fsrcnn_step9g.py
```

### Smoke test (8 frames; never use for scientific conclusions)

```powershell
docker compose run --rm research python scripts/evaluate_large_fsrcnn_step9g.py --limit 8 --repeats 3 --out reports/development/large_step9g_smoke_8_v1
```

### Full frozen development test (200 frames)

```powershell
docker compose run --rm research python scripts/evaluate_large_fsrcnn_step9g.py --limit 0 --repeats 3 --out reports/development/large_step9g_200_v1
```

The output directory must not already exist. If a run fails after partial
results are written, choose a new output directory; never overwrite reports.

## Files and verification

- `configs/large_fsrcnn_step9g.yaml` — explicitly fixes checkpoints and protocol.
- `scripts/evaluate_large_fsrcnn_step9g.py` — includes timing and summary statistics.
- `tests/test_large_fsrcnn_step9g.py` — CPU tests for mode selection, halos,
  protocol checks, paired utility and cluster bootstrap calculations.
- `reports/development/.../summary.json` — main result and SHA256 provenance.
- `reports/development/.../detection_per_image.csv` — per-image TP / FP counts.
- `reports/development/.../timing_per_trial.csv` — synchronized timing repeats.

The script validates Step 7A's **original** frozen detector, predictor V1,
FSRCNN V1, utility CSV, manifest and image/annotation hashes. It then loads
Large FSRCNN and Predictor V2 **as explicitly new evaluation artifacts**,
checking the V2 predictor's training CSV provenance, and reports hashes for
both without altering the frozen manifest.

## Metrics and limitations

- Class-aware small-object TP / recall at diagnostic confidence 0.25,
  matching IoU 0.50, and false positive count. **Not mAP.**
- Utility `(TPa−TPb) − 0.25×(FPa−FPb)`. Favor more small TPs and fewer FPs.
- End-to-end latency includes selector + SR/upscaling + detector, with GPU
  synchronization, **excluding disk image loading and model load**.
- Repeated trials are reduced to a median per image; video sequences are the
  bootstrap clusters (not independent adjacent frames). CIs are exploratory.
- **No claim of final held-out generalization**: the FLIR validation domain
  was already studied in Step 4 and during model/architecture selection.
- Predictor V2 was trained on **Bicubic ROI** intervention utility, not the
  large-model FSRCNN counterfactual. Transfer may be poor and should not be
  assumed to work.
- A model may be faster than full Large FSRCNN but still slower or less useful
  than full Bicubic. Compare **both** baselines.
- Mask composition can affect detection even when raw ROI pixels differ by
  at most one 8-bit value from the full-frame model. Actual detector metrics
  are authoritative for the tested policy.
- GPU differences between runs can affect timings; in-run paired timings are
  preferable to comparing means from different Docker runs.
- The 4×4 fixed geometry may not suit all small objects; large improvements
  require stronger labels and independent external data.

## Acceptance criteria

1. Unit tests and smoke test pass.
2. Verified SHA256 checks and no source mutation.
3. FP, TP, and latency compared against full Large and full Bicubic.
4. Paired improvement should be positive with a robust sequence-level CI
   excluding zero before claiming reliable superiority. A 17-sequence CI
   is still exploratory.
5. Freeze research decisions before collecting a **genuinely independent**
   final test set, ideally with different camera scenes and capture conditions.
