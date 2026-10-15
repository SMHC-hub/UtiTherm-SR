# Model Checkpoints

This document describes all trained model checkpoints used in the UtiTherm-SR Phase 1 research pipeline, their acquisition, paths, architecture parameters, and redistribution restrictions.

> [!IMPORTANT]
> Model checkpoint files are **not redistributed** in this repository (they are excluded by `.gitignore`).
> The trained checkpoints are the result of original Phase 1 research training runs and are not publicly hosted.
> To reproduce results from scratch, follow the training instructions below.

---

## Checkpoint Index

| File | Role | Architecture | Size | SHA-256 |
|---|---|---|---|---|
| `models/sr/fsrcnn_large_x2_stable_step9d3_v1.pt` | **Primary SR model** (used in Step 11L frozen evaluation) | FSRCNN `d=168 s=36 m=10 scale=2` | ~592 KB | `445b4adc8ad53c18274cf7d101ca54ac1e5ab7bea35c85bd53b5688ff729a5ae` |
| `models/sr/fsrcnn_large_x2_step9d_v1.pt` | SR model checkpoint (Step 9D, earlier variant) | FSRCNN `d=168 s=36 m=10 scale=2` | ~592 KB | `2a6b761010eaead4242748d37cb5b867db015cf017592db352bdc481d64e8b27` |
| `models/sr/fsrcnn_x2_v1.pt` | Small FSRCNN baseline | FSRCNN `d=56 s=12 m=4 scale=2` | ~58 KB | `a456c4e4bc726a982f73821f1f54affc8a96936bae61e5e4a89bf72d2ef8b42b` |
| `models/detection/yolo11n_lr_baseline_v1.pt` | YOLO11n fine-tuned on FLIR ADAS thermal classes | YOLO11n (Ultralytics) | ~5.2 MB | `890ed93b8d6466244ebe703c0955414b535e37672db19dce08da6551b816677c` |
| `models/utility/roi_utility_v3_large_h10.pt` | ROI utility predictor (Step 10C/D) | MLP (see Step 10C docs) | ~130 KB | `01ef3998b8912bbd784789101502f460ecc81c9581c44b5e3e814756b1f9c886` |

---

## Architecture Details

### FSRCNN (Fast Super-Resolution CNN)

The FSRCNN architecture follows Dong et al. (2016) with custom hyperparameters tuned during Phase 1 research.

The **Large FSRCNN** used in the primary frozen evaluation (Step 11L) has the following architecture:

```
scale : 2          (×2 upsampling)
d     : 168        (feature extraction channels)
s     : 36         (shrinking / mid-layer channels)
m     : 10         (number of non-linear mapping layers)
```

Network structure:
- **Feature extraction:** `Conv2d(1→168, 5×5)` + PReLU
- **Shrinking:**          `Conv2d(168→36, 1×1)` + PReLU
- **Non-linear mapping:** 10 × (`Conv2d(36→36, 3×3)` + PReLU)
- **Expanding:**          `Conv2d(36→168, 1×1)` + PReLU
- **Upsampling:**         `ConvTranspose2d(168→1, 9×9, stride=2)`

Input: `[B, 1, H, W]` float32 in `[0, 1]`<br>
Output: `[B, 1, 2H, 2W]` float32 (unclipped, clip to `[0, 1]` before display)

The **Small FSRCNN** baseline (`fsrcnn_x2_v1.pt`) uses `d=56, s=12, m=4`.

### YOLO11n

The detection model is the `yolo11n` architecture (Ultralytics, nano variant) fine-tuned on the FLIR ADAS v2 dataset using 8,862 training images. Weights are stored in the Ultralytics `.pt` format. The model was frozen for all Step 11L evaluations.

---

## Checkpoint Metadata

Each FSRCNN checkpoint stores training metadata under the following keys:

| Key | Description |
|---|---|
| `model_kwargs` | `dict` — constructor arguments (`scale`, `d`, `s`, `m`) |
| `epoch` | Training epoch at which this checkpoint was saved |
| `val_mse` | Validation MSE on the FLIR training split holdout |
| `val_psnr_db` | Validation PSNR (dB) |
| `train_split_sha256` | SHA-256 of the training split file (reproducibility audit) |
| `config_sha256` | SHA-256 of the training config YAML |

To inspect metadata without loading model weights:

```python
import torch
ckpt = torch.load("models/sr/fsrcnn_large_x2_stable_step9d3_v1.pt",
                  map_location="cpu", weights_only=True)
print("Architecture:", ckpt["model_kwargs"])
print("Epoch:", ckpt["epoch"])
print("Val PSNR:", ckpt["val_psnr_db"], "dB")
```

**Primary checkpoint metadata** (`fsrcnn_large_x2_stable_step9d3_v1.pt`):

```
model_kwargs : {'scale': 2, 'd': 168, 's': 36, 'm': 10}
epoch        : 40
val_psnr_db  : 33.977 dB
val_mse      : 4.003e-04
```

---

## Acquiring Checkpoints

### Option A — Train from Scratch (Recommended for Reproducibility)

1. Download and register the FLIR ADAS v2 dataset from [Teledyne FLIR](https://www.flir.com/oem/adas/adas-dataset-form/).
2. Run the dataset audit and preprocessing pipeline:
   ```bash
   docker compose run --rm research python scripts/audit_flir_dataset.py
   ```
3. Train the Large FSRCNN (Step 9D3 variant):
   ```bash
   docker compose run --rm research python scripts/train_fsrcnn.py \
     --config configs/fsrcnn_large_step9d3.yaml
   ```
4. Fine-tune YOLO11n on the thermal training split:
   ```bash
   docker compose run --rm research python scripts/train_yolo_baseline.py \
     --config configs/yolo_training.yaml
   ```

Training reproduces the exact checkpoint SHA-256 values listed above when using the same data split (verified by `train_split_sha256` field).

### Option B — Contact the Authors

For academic collaboration or verification purposes, contact the project maintainer via the [GitHub repository](https://github.com/SMHC-hub/UtiTherm-SR/issues).

---

## Redistribution Restrictions

| Checkpoint | Redistribution Status |
|---|---|
| FSRCNN checkpoints | Original research weights — **not redistributed publicly**. Train from scratch using the provided scripts and configuration. |
| YOLO11n fine-tuned weights | Derived from Ultralytics YOLO11n (AGPL-3.0 base). Fine-tuned on FLIR ADAS v2, which requires a registration agreement with Teledyne FLIR. **Not redistributed.** |
| FLIR ADAS v2 dataset | Copyright Teledyne FLIR LLC. Obtain via [official registration](https://www.flir.com/oem/adas/adas-dataset-form/). Raw images and labels are not included in this repository. |

---

## Verifying Existing Checkpoints

To verify checkpoint integrity against the SHA-256 hashes listed above:

```bash
# PowerShell (Windows)
Get-FileHash models\sr\fsrcnn_large_x2_stable_step9d3_v1.pt -Algorithm SHA256

# bash (Linux / macOS / WSL)
sha256sum models/sr/fsrcnn_large_x2_stable_step9d3_v1.pt
```

Expected output for the primary SR checkpoint:
```
445b4adc8ad53c18274cf7d101ca54ac1e5ab7bea35c85bd53b5688ff729a5ae
```
