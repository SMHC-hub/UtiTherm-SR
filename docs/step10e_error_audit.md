# Step 10E — Paired error, sequence and latency audit

**Purpose:** Analyze existing Step 10D results without retraining a model, touching source files, loading CUDA, or rerunning detection.

## Files in this add-on

- `configs/large_sr_error_audit_step10e.yaml`: locked input paths, comparisons and bootstrap settings.
- `scripts/audit_large_sr_step10e.py`: reads the two Step 10D CSVs and original summary; validates counts, repeats, method pairing, provenance and timing aggregation.
- `tests/test_large_sr_error_audit_step10e.py`: offline CPU tests including a synthetic full audit and integrity checks.
- `docs/step10e_error_audit.md`: these instructions.

## Instructions: Windows PowerShell, from `F:\UtiTherm-SR`

1. Extract the ZIP **into the project root**, preserving directories. The Step 10D inputs must already exist at `reports/development/step10d_full_200_v1/`.

2. Run the unit tests:

   ```powershell
   docker compose run --rm research python -m pytest -q tests/test_large_sr_error_audit_step10e.py
   ```

3. Run the read-only audit:

   ```powershell
   docker compose run --rm research python scripts/audit_large_sr_step10e.py
   ```

4. Inspect the output:

   ```powershell
   Get-Content .\reports\development\step10e_error_audit_v1\summary.json -Raw
   ```

   For worst-case images:

   ```powershell
   Import-Csv .\reports\development\step10e_error_audit_v1\worst_images.csv | Select-Object -First 20 | Format-Table -AutoSize
   ```

## Output files

| Output | Contents |
|---|---|
| `summary.json` | Per-method exact totals, timing decomposition, all pairwise comparisons, sequence-bootstrap CIs, FP penalty sensitivity, source SHA256 hashes and limitations. |
| `paired_per_image.csv` | Every image's `delta_small_tp`, `delta_fp`, `delta_utility`, latency difference and coarse TP–FP tradeoff for each comparison. |
| `paired_per_sequence.csv` | Same contrasts aggregated by each of the 17 sequences. |
| `worst_images.csv` | Most negative utility frames for each pair, for qualitative future inspection (not a pristine test). |
| `timing_by_method.csv` | Mean and P50/P95 of per-image median total time, CPU/GPU stages, processed LR pixels and PyTorch memory metric. |

## Statistical protocol

- Each comparison is **method A minus method B**, same frozen 200 frames, same 17 sequences.
- Single-image utility difference is `Δsmall_tp - 0.25 * Δfp`; the weighted group total equals the sum across images.
- Utility confidence intervals describe mean utility difference **per image**, via resampling whole video sequences (not correlated frames) with replacement. Bootstrap samples are weighted by number of frames sampled within each sequence.
- Latency difference uses differences in **per-image median** timing across three repeats, again bootstrapped by sequence.
- V3 gains vs V2 need to be weighed against increased FP. The default FP penalty 0.25 is **not a learned or validated optimum**: the sensitivity table checks weights 0, 0.10, 0.25, 0.50, 1.0.
- Median-timing components do **not** have to sum exactly to median total latency due to measurement overhead and independently medians; `mean_unattributed_overhead_ms` is descriptive.
- An interval crossing zero does **not** prove no effect. The same development images have already influenced research decisions, so no p-values or independent generalization claims should be made from this audit.

## What these CSVs can—and cannot—answer

They **can** show where V3's aggregate small-object TP count improves or declines, how FP changes, which sequences dominate pairwise differences, how outcomes vary with FP penalties, and the measured latency cost of selection and SR.

They **cannot** identify exactly which object was rescued or lost, which tile was selected, or whether interactions between tiles caused a detection gain. They store aggregate counts only. To test combined-region nonadditivity, a dedicated future experiment must record selected tile IDs and matched per-object detections and evaluate interventions on **training-domain or appropriate independent data**, without adapting decisions on this repeatedly studied FLIR development set.

## Research recommendation / decision rule

First examine the image/sequence distributions and utility-penalty sensitivity. If V3's gains are restricted to a few sequences or disappear at modest FP penalties, do **not** train another utility model on the same held-out data. Consider designing an intervention experiment that begins with **Full Bicubic** as baseline, and measures whether selective Large FSRCNN adds value over Bicubic, on appropriately separated data.
