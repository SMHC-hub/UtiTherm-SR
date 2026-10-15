UtiTherm-SR Step 4: Bicubic baseline (manual Antigravity / Docker workflow)

1. Copy scripts/evaluate_bicubic_baseline.py into F:\UtiTherm-SR\scripts\
2. Copy tests/test_bicubic_baseline.py into F:\UtiTherm-SR\tests\
3. DO NOT replace any existing file or run docker compose build.

Commands, from F:\UtiTherm-SR:
  docker compose run --rm research python -m pytest -q tests/test_bicubic_baseline.py
  docker compose run --rm research python scripts/evaluate_bicubic_baseline.py --limit 30 --out reports/sr_baseline/bicubic_smoke_30

When smoke passes, run the FULL 1144-image comparison using a NEW output directory:
  docker compose run --rm research python scripts/evaluate_bicubic_baseline.py --limit 0 --out reports/sr_baseline/bicubic_full_val

Outputs:
  reports/sr_baseline/<run>/comparison.csv
  reports/sr_baseline/<run>/summary.json
  data/interim/bicubic_eval/<run>/{lr,bicubic,nearest}/ (generated validation only)

Five diagnostic conditions on precisely the same images and checkpoint:
  lr_320                  LR @ YOLO input size 320 (reference)
  lr_640_resize_control   LR @ input size 640 (YOLO internal interpolation control)
  bicubic_320             Bicubic x2 @ YOLO input size 320 (network-size control)
  bicubic_640             Bicubic x2 @ YOLO input size 640 (SR baseline)
  nearest_640_control     Nearest x2 @ YOLO input size 640 (interpolation control)

Keep the same validation labels: YOLO normalized bounding boxes are scale invariant.
This is a validation-stage experiment only, not a held-out test. Results are NOT
proof of novelty. FPS/latency outputs from validator are NOT end-to-end timing.
