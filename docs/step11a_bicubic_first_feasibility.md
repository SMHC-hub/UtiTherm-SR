# Step 11A — Bicubic-first Large FSRCNN incremental utility feasibility

**Purpose:** Test whether a *single tile* of trained Large FSRCNN can improve detector utility **above full-frame Bicubic**. This is a controlled **feasibility probe**, not a training phase or a demonstrated production selector.

## Files
- `configs/bicubic_first_feasibility_step11a.yaml` — frozen settings and references
- `src/utitherm/sr/bicubic_first_fsrcnn_step11a.py` — true Bicubic-background compositor (new)
- `scripts/collect_bicubic_first_feasibility_step11a.py` — resuming, SHA256-locked collector
- `tests/test_bicubic_first_feasibility_step11a.py` — CPU numerical and protocol tests
- `docs/step11a_bicubic_first_feasibility.md` — this protocol

## Fundamental correction versus Step 10A

Step 10A evaluated neural ROIs on a **linear-upscaled** canvas. Its renderer `render_batched_fsrcnn()` initializes `INTER_LINEAR`; it cannot be directly reused here. Instead we obtain *raw neural tile patches* from `batched_fsrcnn_tiles(...)` and blend with the original `_blend()` routine on top of `cv2.INTER_CUBIC` across the entire image. Neural inference/cropping/halo matches Step 9G; only the background changes. All pixels outside selected tiles are exactly Bicubic.

## Exact frozen protocol

- Original FLIR ADAS v2 320×256 LR thermal input, 640×512 output.
- Same SHA256-locked **300 training images / 33 sequences** from `v2_fresh_split_300_v2`, with manifest image, annotation and detector hash verification.
- Model: trained Large FSRCNN `fsrcnn_large_x2_stable_step9d3_v1.pt` (architecture `scale=2,d=168,s=36,m=10`), no parameter changes.
- Detection model: frozen YOLO11n checkpoint from Step 7B2A manifest, confidence 0.25, NMS IoU 0.7; matched class-aware small-TP/FP detection scoring at IoU 0.50.
- Full-frame Bicubic baseline is predicted **once per frame**, then 16 independent interventions, **one neural tile at a time**, using 4×4 tiled 160×128 HR grid, halo10 LR, feather8 HR.
- Utility: `ΔsmallTP − 0.25 × ΔFP` **relative to full Bicubic**; not relative to linear. No GT-guided inference-time tile selection.
- Single ROI processed LR pixels = `(80 + 2×10) × (64 + 2×10) = 8,400`; computed pixel counts **do not equal FLOPs or measured runtime**.

## Commands (from `F:\UtiTherm-SR`)

```powershell
# Tests (run before any GPU work)
docker compose run --rm research python -m pytest -q tests/test_bicubic_first_feasibility_step11a.py

# 4-image smoke / 64 single-ROI interventions
docker compose run --rm research python scripts/collect_bicubic_first_feasibility_step11a.py --limit 4 --out reports/utility/step11a_smoke_4_v1

# Optional larger pilot / 640 interventions
docker compose run --rm research python scripts/collect_bicubic_first_feasibility_step11a.py --limit 40 --out reports/utility/step11a_pilot_40_v1

# Only after smoke passes: entire frozen training sample / 4,800 interventions
docker compose run --rm research python scripts/collect_bicubic_first_feasibility_step11a.py --limit 0 --out reports/utility/step11a_full_300_v1

# Interrupted? Resume only the exact same run; source and checkpoint hashes must match.
docker compose run --rm research python scripts/collect_bicubic_first_feasibility_step11a.py --limit 0 --resume --out reports/utility/step11a_full_300_v1
```

Output files:

`run_manifest.json`, `shards/*.json`, `bicubic_baseline_per_image.csv`, `incremental_roi_utility.csv`, `summary.json`.

The CSV explicitly labels `baseline_kind=full_bicubic_640`. It is **not interchangeable** with Step 10A Large-SR-vs-linear labels or Step 7B2A Bicubic-vs-linear labels. Keep these source datasets separate.

## Interpret results conservatively

Key feasibility signals: number/fraction of positive isolated neural interventions, count of images with *at least one* positive tile, net small TP/FP changes, and marginal oracle-top4 score on already-measured interventions. A substantial positive fraction would justify studying **learnability of incremental benefit** in a new training/OOF setup. A low fraction or strongly negative effects would argue against training a new selector.

**Critically:** Summing top-four isolated utilities is a *ranking diagnostic* using ground-truth intervention results, not achievable prediction, not measured multi-tile performance, and not necessarily an upper bound on four-ROI detection utility due to ROI interactions. It is not a deployment pipeline latency measurement; 16 independent GPU patch calls and 17 YOLO predictions are made per training frame for counterfactual collection.

These 300 images were used before, YOLO and Large FSRCNN trained on the same larger FLIR train domain; this study is **exploratory, not independent or out-of-fold**. Do not tune repeatedly on repeatedly inspected FLIR validation. Before publication claims: acquire independent scene-disjoint test frames, make detector out-of-fold for utility labels, evaluate top-4 combined outcomes and end-to-end runtime with an inference-time selector, full Bicubic, and random controls, and report standard mAP plus utility/uncertainty.
