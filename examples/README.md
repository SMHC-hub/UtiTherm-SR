# Examples and Quickstart

This directory provides working demonstration scripts illustrating the UtiTherm-SR pipeline.

## Demo: `demo_pipeline.py`

Demonstrates:
1. Creating or loading a 320×256 grayscale thermal low-resolution (LR) image.
2. Generating a 640×512 full-frame Bicubic baseline.
3. Partitioning the coordinate space into candidate region-of-interest (ROI) tiles.
4. Ranking candidate tiles by thermal contrast (standard deviation).
5. Running a Large FSRCNN model on the top-4 selected tiles.
6. Compositing the neural super-resolved patches over the immutable Bicubic base with feathered boundary blending.
7. Calculating the diagnostic detection utility score: `Utility = Small TP - 0.25 × FP`.

### Running with Docker (Recommended)

```bash
docker compose run --rm research python examples/demo_pipeline.py
```

### Running Locally

```bash
python examples/demo_pipeline.py
```

Output images are written to `examples/output/`:
- `01_input_lr_320x256.png`: Input thermal image.
- `02_full_bicubic_640x512.png`: Full Bicubic baseline.
- `03_selective_sr_composite_640x512.png`: Composited output (Bicubic base with neural SR on top-4 contrast regions).
- `04_selective_vs_bicubic_diff.png`: False-color difference map highlighting enhanced regions.
