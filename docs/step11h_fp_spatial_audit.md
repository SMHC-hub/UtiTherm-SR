# Step 11H — deterministic spatial FP candidate audit

## Why

Full Bicubic vs Contrast-4 on 200 reused FLIR development frames gave **ΔsmallTP +15, ΔFP +88, ΔU -7.0** in Step11G, reproducible across two Docker executions. Top 20 frames show ΔFP positive 74 in aggregate; this is **gross concentration among those selected frames**, not 74/88 independent FPs. We need box-level spatial diagnostics before changing models or ranking rules.

## Scientific scope

Descriptive retrospective audit **only**, not causal attribution, independent evidence, or standard mAP. Object matching follows the exact existing detection protocol: class-aware confidence-sorted 1-to-1 greedy GT IoU>=0.5, confidence>=.25. Extra contrast FP candidates are those **unpaired with a baseline FP** under *separate* class-aware FP-vs-FP greedy IoU>=0.50 pairing. Unpaired original Bicubic FP candidates are counted as removed, and added minus removed must equal measured net ΔFP. Threshold sensitivity and 1:1 pairing ambiguities matter.

For each unpaired contrast FP, compute HR box centre and overlap with selected HR tiles. Classify centroid as `selected_interior`, `selected_boundary_band` (within 8 HR px of an actual edited tile boundary), or `outside_selected_tiles`. Note the **halo is CNN input context; the edited output pixels are confined to tile rectangles**. Bounding-box overlap proportion is separately recorded and is not part of the mutually exclusive centre zone. Association with selected ROI/boundary **does NOT establish physical causation** because detector receptive field and NMS are global. Some annotations may omit real objects.

## Prerequisites

- Previous Step 11G full A and B runs exist and `step11g_full_200_b/summary.json` has `cross_container_verified: true` and no mismatches.
- Original frozen models, dataset, and Step11F/G scripts/configs untouched.
- Extract this ZIP at `F:\UtiTherm-SR`; add *only* new Step11H files.
- Set `CUBLAS_WORKSPACE_CONFIG` for deterministic GPU operations.

## Commands (Windows PowerShell)

Run tests:

```powershell
docker compose run --rm research python -m pytest -q tests/test_spatial_false_positives_step11h.py
```

Run the complete **200-image descriptive spatial audit** (one first-repeat equivalent per frame, same Step11G first-repeat inference order). A 200-image run is preferable to selecting only the top 20 after seeing their outcomes.

```powershell
docker compose run --rm -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" research `
  python scripts/audit_spatial_false_positives_step11h.py `
  --out reports/development/step11h_spatial_fp_v1
```

Inspect:

```powershell
Get-Content .\reports\development\step11h_spatial_fp_v1\summary.json -Raw
Import-Csv .\reports\development\step11h_spatial_fp_v1\per_sequence_spatial_fp.csv |
  Sort-Object {[int]$_.net_delta_fp} -Descending | Select-Object -First 8 | Format-Table -AutoSize
```

### Outputs

- `summary.json`: fingerprints match state, net counts, inside/boundary/outside FP candidate counts, caveats
- `fp_candidates.csv`: every new-unpaired FP candidate, class, confidence, box, selected ROI, zone, bbox overlap
- `per_image_spatial_fp.csv`: original/difference counts, paired/new/removed, zone counts, utility
- `per_sequence_spatial_fp.csv`: concentration across video sequences
- `reference_mismatches.csv` (only if found): any checksum, prediction, tile, or detection-count mismatch. **Any mismatch triggers error; do not interpret spatial report.**

The script requires a new output directory, never overwrites prior research. For debugging only `--limit N` can be used, but it does not establish full results and requires new out folder. The report is verified only if `reference_fingerprint_match=true` and `mismatch_count=0` on `images=200`.

### Limitations / next decision

We may find FPs overlap enhanced tiles more often, but that alone is not a causal test. A follow-up *controlled intervention* would be needed to test whether reducing feather, local differences, or confidence effects truly resolves them. Do not tune on reused development data and call the result final. Preserving a fresh sequence-held-out evaluation and detection utility remains critical.
