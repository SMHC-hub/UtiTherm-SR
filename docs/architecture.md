# UtiTherm-SR architecture and implementation gates

Input LR thermal frame -> scout detector -> high-recall candidates -> learned utility predictor -> compute-budget controller -> selective x2 SR -> refined detector -> merge + measure.

**Gate 0:** Windows driver / WSL2 / Docker / CUDA GPU verification.

**Gate 1:** FLIR ADAS download audit: images, modality, modes/bit-depth, COCO JSON, categories, true paths, train/val/test split integrity, leakage concerns. No automatic split or annotation assumptions.

**Gate 2:** Synthetic LR/HR ×2 preview with explicit degradation. Starting dataset is 8-bit thermal, not 14-bit TIFF. Validate original, LR and bicubic sizes.

**Gate 3:** Convert verified COCO boxes to training labels, freeze class-map and split lists, fine-tune YOLO11n; report image-level and small-object AP.

**Gate 4:** Bicubic and FSRCNN×2 full-frame SR versus same detector/input/evaluation, paired frames and labels. If detection does not improve, diagnose rather than claim benefit.

**Gate 5:** High-recall candidate ROI generation -> measured per-ROI detection benefit labels on TRAINING folds only; utility predictor.

**Gate 6:** Budget optimization and end-to-end latency including scout, SR, refined detector, crop/paste/merge. Compare random, confidence ROI, full frame and learned ROI baselines.

**Constraints:** No GT boxes for final ROI selection; preserve test holdout; don't train on validation/test benefit labels; report candidate-recall ceiling and small-object misses; only claim novelty supported by controlled experiments. Planned Phase 2 fidelity treatment remains modular and out of scope.
