# Examples and Quickstart

This directory provides working demonstration scripts illustrating the UtiTherm-SR pipeline.

## Demo: `demo_pipeline.py`

Demonstrates:
1. Creating or loading a 320×256 grayscale thermal low-resolution (LR) image.
2. Generating a 640×512 full-frame Bicubic baseline.
3. Partitioning the coordinate space into candidate region-of-interest (ROI) tiles.
4. Ranking candidate tiles by thermal contrast (standard deviation).
5. Running the trained Large FSRCNN model (`d=168, s=36, m=10`) on the top-4 selected tiles.
6. Compositing the neural super-resolved patches over the immutable Bicubic base with feathered boundary blending.
7. Running live inference with the fine-tuned YOLO11n detector on both full Bicubic and selective-SR outputs.
8. Comparing live detection counts (including small objects) and displaying the frozen Step 11L benchmark reference.

### Required Checkpoints

The demo requires the trained model checkpoints:
- `models/sr/fsrcnn_large_x2_stable_step9d3_v1.pt` (Large FSRCNN)
- `models/detection/yolo11n_lr_baseline_v1.pt` (Fine-tuned YOLO11n)

See [`docs/MODEL_CHECKPOINTS.md`](../docs/MODEL_CHECKPOINTS.md) for architecture, SHA-256 hashes, and training instructions.

### Running with Docker (Recommended)

```bash
docker compose run --rm research python examples/demo_pipeline.py
```

### Running Locally (requires ultralytics and trained checkpoints)

```bash
python examples/demo_pipeline.py
```

Output visual artifacts are written to `examples/output/`:
- `01_input_lr_320x256.png`: Input thermal image.
- `02_full_bicubic_640x512.png`: Full Bicubic baseline.
- `03_selective_sr_composite_640x512.png`: Composited output (Bicubic base with neural SR on top-4 contrast regions).
- `04_selective_vs_bicubic_diff.png`: False-color JET difference map highlighting enhanced regions.
- `05_bicubic_detections.png`: Annotated YOLO11n detections on the full Bicubic baseline.
- `06_selective_sr_detections.png`: Annotated YOLO11n detections on the selective-SR composite.
