# UtiTherm-SR — Step 11G: Deterministic, Combined-ROI End-to-End Evaluation

## Evidence prompting this phase

The same Full Large FSRCNN thermal image produced a few 1-gray-level changes between GPU renders with nondeterministic cuDNN settings. The resulting YOLO false-positive count alternated between 9 and 10 for one frame while small TP stayed 11. The Step 10D→Step 11F reference discrepancy was one FP on this frame. A controlled single-frame experiment with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and deterministic PyTorch/cuDNN settings gave **10/10 identical rendered image SHA256s and TP=11, FP=8**; another container reproduced the same. This verifies reproducibility for **that one frame**, not all 200 or all four policies.

## Scope and integrity

- Evaluates **real combined four-ROI** enhanced images, not the Step11E marginal single-ROI proxy.
- Uses the same four arms and source code renderer as Step11F: full Bicubic, full Large FSRCNN, random-four Bicubic-first Large FSRCNN, and contrast-four Bicubic-first Large FSRCNN (halo 10, feather 8).
- Locks the prior frozen **200 images / 17 sequences from FLIR VAL**, training provenance, detector and Large FSRCNN checkpoint hashes against the original Step10D metadata.
- Does **not** remove or modify any Step10D or Step11F integrity test.
- New protocol **must not** demand identical counts to Step10D: deterministic GPU kernels can change output counts even for frozen checkpoints. Instead, it establishes a new **same-protocol A/B repeat and cross-container comparison**.
- During each run, with 3 repetitions of all methods, requires identical 512×640 rendered image byte hashes, identical exact serialized YOLO prediction outputs (boxes, class IDs, confidences) and selected tile indices for each image/method. Any difference saves a `repeat_mismatches.csv` and exits with an error after the report is written.
- Second run compares exact **per-image detection counts, pixel hashes, prediction hashes, selections and numbers of predictions** with the first run, plus checkpoint SHA256, config SHA256, image manifest, library versions, deterministic flags. Differences saved to `cross_run_mismatches.csv`, failing closed.
- Timing measures synchronized selector, SR and detector execution. SHA256 computation occurs **after** the timed window. The deterministic algorithms may change absolute runtime; do not compare the old Step11F GPU timings as if hardware/software protocols were identical. Disk I/O and model loading excluded.
- This is **development / retrospective**, **NOT final untouched test**. Confidence at least .25, NMS IoU .7, class-aware matching IoU .5, utility = Δsmall TP − .25×ΔFP. Not standard mAP. With 17 video groups, bootstrapped CIs remain exploratory. Random four uses one fixed seed.

## Package placement

Extract ZIP into **`F:\UtiTherm-SR`**, preserving paths, and add:

```
configs/bicubic_first_deterministic_step11g.yaml
scripts/evaluate_bicubic_first_deterministic_step11g.py
tests/test_bicubic_first_deterministic_step11g.py
docs/step11g_deterministic_bicubic_first_evaluation.md
```

**Prerequisites:** Keep Step11F original plus the diagnostic hotfix in `scripts/evaluate_bicubic_first_combined_step11f.py`, all existing trained models, project source, manifests and reports. Do not delete older files. The ZIP does not overwrite any Step11F files.

## Command A — Unit tests (CPU, no detector inference)

```powershell
docker compose run --rm research python -m pytest -q tests/test_bicubic_first_deterministic_step11g.py
```

Expected **52 passed** (verified locally).

## Command B — Smoke reference, fresh Docker container

PowerShell; `-e` is passed to **docker compose run**, not Python:

```powershell
docker compose run --rm -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" research `
  python scripts/evaluate_bicubic_first_deterministic_step11g.py `
  --limit 8 --repeats 3 `
  --out reports/development/step11g_smoke_a
```

The first run is **internally repeat-verified only**. It does not claim an A/B comparison (`cross_container_verified: false`). Run into a **new** output directory.

## Command C — Cross-container smoke reference verification

```powershell
docker compose run --rm -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" research `
  python scripts/evaluate_bicubic_first_deterministic_step11g.py `
  --limit 8 --repeats 3 `
  --out reports/development/step11g_smoke_b `
  --compare reports/development/step11g_smoke_a/summary.json
```

Expected `VERIFIED: repeat fingerprints and cross-run counts/render/prediction hashes agree.` If not, inspect `summary.json` plus any mismatch CSV; do not claim reproducibility or loosen checks.

## Command D — Full reference, all 200 frames (only after smoke A/B pass)

```powershell
docker compose run --rm -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" research `
  python scripts/evaluate_bicubic_first_deterministic_step11g.py `
  --limit 0 --repeats 3 `
  --out reports/development/step11g_full_200_a
```

## Command E — Full cross-container verification

```powershell
docker compose run --rm -e "CUBLAS_WORKSPACE_CONFIG=:4096:8" research `
  python scripts/evaluate_bicubic_first_deterministic_step11g.py `
  --limit 0 --repeats 3 `
  --out reports/development/step11g_full_200_b `
  --compare reports/development/step11g_full_200_a/summary.json
```

## Reading outputs

```powershell
$r = Get-Content .\reports\development\step11g_smoke_b\summary.json -Raw | ConvertFrom-Json
$r.repeats_verified
$r.cross_container_verified
$r.results.methods | Format-List
$r.results.comparisons | Format-List
```

Report files:

```
summary.json                    # provenance, deterministic flags, per-method counts, utility CIs
fingerprints_per_image.csv      # pixel SHA, prediction SHA, selected tiles for each image/method
detection_per_image.csv         # matched TP/FP per image/method
timing_per_trial.csv            # 3 repeats per image/method, runtime and memory
repeat_mismatches.csv           # only if within-run violation
cross_run_mismatches.csv        # only if A/B disagreement
```

If a deterministic CUDA operation is unsupported, the test should error clearly and **must not** silently fall back to nondeterministic execution. Do not change FP penalty, checkpoints, confidence or NMS thresholds to make the results look better. If contrast-four does not beat Bicubic on utility and latency, report this limitation rather than overclaiming novelty.
