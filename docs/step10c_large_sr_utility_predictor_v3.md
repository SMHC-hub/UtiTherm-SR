# UtiTherm-SR — Step 10C: Large-FSRCNN-specific ROI utility predictor V3

## Purpose

Train the **same four-head UtilityPredictor CNN** used for V1/V2, but learn exclusively from **trained Large FSRCNN ×2, halo=10** interventions collected in Step 10A. V2 was trained on Bicubic interventions, which have partly different helpful regions. The new source has 300 video frames, 33 video sequences, and 4,800 individually probed tiles.

**Do not combine** old 100-image V1 or 300-image Bicubic V2 ROI label CSVs with these Large-SR labels. V3 starts with **random weights**, not V2 pretrained weights. V2 used a sequence split on the same 300-frame sample. V3 preserves the same seed and fractions and verifies V2’s recorded TRAIN/VAL/TEST memberships so the **five V3 TEST sequences are also held out from V2 training**. V3 still starts from scratch to isolate SR-specific learning.

## Files to copy into project root

- `configs/large_sr_predictor_step10c.yaml`
- `scripts/train_large_sr_predictor_step10c.py`
- `tests/test_large_sr_predictor_step10c.py`
- `docs/step10c_large_sr_utility_predictor_v3.md`

The script imports existing project code `train_utility_predictor.py`, `train_utility_predictor_v2.py`, and `src/utitherm/utility/predictor.py` — no existing scripts need changing.

## Commands in PowerShell (project root F:\\UtiTherm-SR)

First run unit/integration tests:

```powershell
 docker compose run --rm research python -m pytest -q tests/test_large_sr_predictor_step10c.py
```

Then smoke-train on 12 complete video sequences:

```powershell
 docker compose run --rm research python scripts/train_large_sr_predictor_step10c.py --smoke
```

For all 300 images (after checking the smoke results):

```powershell
 docker compose run --rm research python scripts/train_large_sr_predictor_step10c.py
```

Outputs (smoke uses `_smoke` suffix):

- `reports/utility/large_sr_predictor_v3_step10c_v1/best.pt`
- `reports/utility/large_sr_predictor_v3_step10c_v1/split_manifest.json`
- `reports/utility/large_sr_predictor_v3_step10c_v1/history.csv`
- `reports/utility/large_sr_predictor_v3_step10c_v1/test_selections.csv`
- `reports/utility/large_sr_predictor_v3_step10c_v1/summary.json`
- Full run only: `models/utility/roi_utility_v3_large_h10.pt`

**Run safety:** the trainer refuses to overwrite existing output directories or a canonical checkpoint. Use `--out <new-directory>` for a new experiment. Never delete experiment folders to conceal failures.

## Experimental protocol

- Source: verified Step 10A 300-image frozen collection, whose `run_manifest.json` records the exact Large FSRCNN checkpoint SHA256 and halo-10 intervention protocol.
- Train: ~23 video sequences; validation: ~5 sequences; test: ~5 sequences. Counts of **images** can vary because videos contain different frame counts.
- Use `val` combined BCE + pairwise ranking loss for checkpoint selection and early stopping. **Evaluate `test` only after choosing `best.pt`.** No V2 checkpoint is used at train time.
- Evaluate heads `small_gain`, `fp_reduce`, `small_loss`, and `fp_increase` using average precision; selection goal is top four marginal utilities.
- Test report includes random-expected top-4 utility, oracle top-4 utility, and sequence-cluster bootstrap interval for selector vs the *expected* utility of four random tiles. It also compares V2 against V3 on the **identical held-out predictor TEST sequences** after verifying V2's split manifest and source CSV hash, but this is exploratory, not external or detector-independent testing.
- Output checkpoints use the existing production loader format: `model`, `heads`, `input_size`, `source_csv_sha256`; add SR provenance metadata.

## Critical limitations

1. These labels are from images the YOLO detector **already trained on**, so predictor-held-out video sequences are **not detector-independent**.
2. V2 was trained on **a subset** of the same 300-image sample (with separate VAL and TEST groups). We require identical V2/V3 TEST sequence membership; this supports a paired utility-predictor test but prior project decisions have already used V2 TEST results, so **it is not pristine independent confirmation**.
3. The top-4 offline scoring sums **single-ROI** measured changes; it is a **proxy**, not the effect of enhancing four regions at once.
4. Do not tune repeatedly on the 17-sequence FLIR development split; an untouched external or out-of-fold detector test is still needed for research publication.
5. SR-trained utility V3 may fail to improve detection utility over random or full Bicubic; report this explicitly if observed.

## Next milestone

After V3 training, compare V3, V2, random, full Large FSRCNN and full Bicubic **end-to-end** at the same region budget and detector settings. Start with independent sequence splits where possible. Report small-object TP, FP, `delta_small_tp - 0.25 * delta_fp`, latency, and sequence-level intervals. Do not change the existing frozen 200-image development set or declare statistically established improvement unless supported.
