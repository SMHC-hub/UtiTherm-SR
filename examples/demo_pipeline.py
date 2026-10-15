"""UtiTherm-SR Quickstart Demo.

Demonstrates the core UtiTherm-SR pipeline end-to-end on a single image:
1. Synthetic or real LR thermal input (320×256)
2. Full-frame Bicubic upsampling base (640×512)
3. Grid tile extraction and contrast-based ROI ranking
4. Selective FSRCNN super-resolution on top-k tiles (trained checkpoint)
5. Feathered composition over the immutable Bicubic base
6. Real YOLO11n inference on both Bicubic and Selective-SR composites
7. Detection count comparison and output images

This script is a **single-image demonstration** of the pipeline mechanics.
It does NOT reproduce the Step 11L frozen test results; those require the full
FLIR ADAS v2 dataset and are locked in ``reports/published/``.

Usage (inside Docker — recommended):
    docker compose run --rm research python examples/demo_pipeline.py
    docker compose run --rm research python examples/demo_pipeline.py \\
        --image path/to/image.png --output-dir examples/output

Usage (host, with ultralytics installed):
    python examples/demo_pipeline.py [--image ...] [--output-dir ...]

Required checkpoints (trained, not redistributed):
    models/sr/fsrcnn_large_x2_stable_step9d3_v1.pt   -- Large FSRCNN
    models/detection/yolo11n_lr_baseline_v1.pt        -- Fine-tuned YOLO11n
See docs/MODEL_CHECKPOINTS.md for SHA-256 hashes and acquisition instructions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure src/ is on the path when running from project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from utitherm.sr.fsrcnn import FSRCNN
from utitherm.sr.bicubic_first_fsrcnn_step11a import render_bicubic_first_fsrcnn

# ---------------------------------------------------------------------------
# Default checkpoint paths
# ---------------------------------------------------------------------------
_DEFAULT_SR_CKPT = _ROOT / "models" / "sr" / "fsrcnn_large_x2_stable_step9d3_v1.pt"
_DEFAULT_DET_CKPT = _ROOT / "models" / "detection" / "yolo11n_lr_baseline_v1.pt"


# ---------------------------------------------------------------------------
# Checkpoint loaders
# ---------------------------------------------------------------------------

def load_fsrcnn(checkpoint: Path, device: torch.device) -> FSRCNN:
    """Load the trained Large FSRCNN from *checkpoint*.

    The checkpoint embeds ``model_kwargs`` so the correct architecture
    (``d=168, s=36, m=10``) is always instantiated from saved metadata.

    Exits with a non-zero status and an explicit error message if the
    checkpoint is absent — does NOT silently fall back to random weights,
    because random weights would produce visually misleading SR output.
    """
    if not checkpoint.is_file():
        print(
            f"\n[ERROR] FSRCNN checkpoint not found: {checkpoint}\n"
            "        This demo requires a trained checkpoint to run.\n"
            "        See docs/MODEL_CHECKPOINTS.md for acquisition instructions.\n"
            "        To train from scratch: docker compose run --rm research "
            "python scripts/train_fsrcnn.py --config configs/fsrcnn_large_step9d3.yaml"
        )
        sys.exit(1)

    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=True)
    kwargs = ckpt.get("model_kwargs", {"scale": 2, "d": 168, "s": 36, "m": 10})
    model = FSRCNN(**kwargs)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    meta = {k: ckpt[k] for k in ("epoch", "val_psnr_db", "val_mse") if k in ckpt}
    print(
        f"[+] FSRCNN loaded  "
        f"d={kwargs.get('d')} s={kwargs.get('s')} m={kwargs.get('m')}  "
        f"from: {checkpoint.name}"
    )
    if meta:
        psnr = meta.get("val_psnr_db", "?")
        print(f"    epoch={meta.get('epoch','?')}  val_psnr={psnr:.4f} dB")
    return model


def load_yolo(checkpoint: Path):
    """Load the fine-tuned YOLO11n detection model from *checkpoint*.

    Exits with an explicit error if the checkpoint is absent, rather than
    silently downloading a default (non-thermal, non-fine-tuned) model.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "\n[ERROR] ultralytics is not installed in this environment.\n"
            "        Run inside Docker: docker compose run --rm research python examples/demo_pipeline.py\n"
            "        Or install: pip install ultralytics>=8.3,<9"
        )
        sys.exit(1)

    if not checkpoint.is_file():
        print(
            f"\n[ERROR] YOLO checkpoint not found: {checkpoint}\n"
            "        This demo requires the fine-tuned thermal YOLO11n checkpoint.\n"
            "        See docs/MODEL_CHECKPOINTS.md for acquisition instructions."
        )
        sys.exit(1)

    model = YOLO(str(checkpoint))
    print(f"[+] YOLO11n loaded from: {checkpoint.name}")
    print(f"    Classes: {list(model.names.values())}")
    return model


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def create_synthetic_thermal_scene(height: int = 256, width: int = 320) -> np.ndarray:
    """Generate a synthetic thermal LR frame with warm targets and background gradients."""
    img = np.zeros((height, width), dtype=np.float32)
    y_coords = np.linspace(30, 90, height)[:, None]
    img += y_coords
    img[int(height * 0.55):, :] += 20.0
    # Warm targets
    img[120:145, 150:162] += 120.0   # central pedestrian-like blob
    img[100:112, 80:86] += 95.0      # small distant target
    img[130:165, 230:270] += 110.0   # vehicle on right
    np.random.seed(42)
    noise = np.random.normal(0, 3.0, (height, width))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def generate_grid_tiles(width: int = 640, height: int = 512,
                        grid_x: int = 4, grid_y: int = 4) -> list[tuple[int, int, int, int]]:
    """Partition a 640×512 HR frame into regular rectangular tiles."""
    w_tile = width // grid_x
    h_tile = height // grid_y
    return [
        (gx * w_tile, gy * h_tile, (gx + 1) * w_tile, (gy + 1) * h_tile)
        for gy in range(grid_y)
        for gx in range(grid_x)
    ]


def rank_tiles_by_contrast(lr_image: np.ndarray,
                           tiles: list[tuple[int, int, int, int]]) -> list[int]:
    """Rank tiles descending by LR-domain pixel std-deviation (thermal contrast)."""
    scores = [
        (float(np.std(lr_image[y0 // 2:y1 // 2, x0 // 2:x1 // 2])), idx)
        for idx, (x0, y0, x1, y1) in enumerate(tiles)
    ]
    scores.sort(key=lambda s: s[0], reverse=True)
    return [idx for _, idx in scores]


def run_yolo_inference(yolo, image_bgr: np.ndarray, conf: float = 0.25) -> dict:
    """Run YOLO inference on a BGR uint8 image.  Returns a summary dict."""
    results = yolo.predict(image_bgr, conf=conf, verbose=False, imgsz=640)
    r = results[0]
    boxes = r.boxes
    n_total = len(boxes)
    # Small object: bounding box area < 32^2 = 1024 px in 640×512 space
    areas = (boxes.xywh[:, 2] * boxes.xywh[:, 3]).cpu().numpy() if n_total > 0 else np.array([])
    n_small = int((areas < 1024).sum())
    det_summary = []
    if n_total > 0:
        for cls_id, conf_v, area in zip(
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
            areas,
        ):
            det_summary.append({
                "class": yolo.names[cls_id],
                "conf": float(conf_v),
                "area_px": float(area),
                "small": bool(area < 1024),
            })
    return {
        "total_detections": n_total,
        "small_detections": n_small,
        "conf_threshold": conf,
        "detections": det_summary,
    }


def annotate_image(image_bgr: np.ndarray, yolo_result) -> np.ndarray:
    """Draw YOLO bounding boxes on a BGR image (returns a new copy)."""
    return yolo_result[0].plot(img=image_bgr.copy())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UtiTherm-SR pipeline demonstration")
    parser.add_argument("--image", type=Path, default=None,
                        help="Optional path to 320×256 grayscale thermal image")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"),
                        help="Output directory for visual comparisons")
    parser.add_argument("--sr-checkpoint", type=Path, default=_DEFAULT_SR_CKPT,
                        help=f"Trained Large FSRCNN checkpoint. Default: {_DEFAULT_SR_CKPT}")
    parser.add_argument("--det-checkpoint", type=Path, default=_DEFAULT_DET_CKPT,
                        help=f"Fine-tuned YOLO11n checkpoint. Default: {_DEFAULT_DET_CKPT}")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("   UtiTherm-SR: Region-Selective Thermal Super-Resolution Demo   ")
    print("=================================================================")

    # 1 — Load or synthesize LR input
    if args.image and args.image.is_file():
        lr = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
        if lr.shape != (256, 320):
            print(f"[!] Resizing input image from {lr.shape} to (256, 320)")
            lr = cv2.resize(lr, (320, 256), interpolation=cv2.INTER_AREA)
        print(f"[+] Loaded input image: {args.image} (320×256 uint8)")
    else:
        lr = create_synthetic_thermal_scene(256, 320)
        print("[+] Generated synthetic 320×256 thermal LR scene")

    # 2 — Full-frame Bicubic baseline
    bicubic_full = cv2.resize(lr, (640, 512), interpolation=cv2.INTER_CUBIC)
    print("[+] Rendered Full Bicubic baseline (640×512)")

    # 3 — Partition and rank tiles
    tiles = generate_grid_tiles(640, 512, 4, 4)
    ranked = rank_tiles_by_contrast(lr, tiles)
    selected_k4 = ranked[:4]
    print(f"[+] Contrast-ranked top-4 tile indices: {selected_k4}")

    # 4 — Load trained FSRCNN (exits if checkpoint absent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")
    sr_model = load_fsrcnn(args.sr_checkpoint, device)

    # 5 — Selective neural SR composite
    with torch.no_grad():
        selective_composite, num_patches = render_bicubic_first_fsrcnn(
            lr=lr,
            tiles=tiles,
            selected=selected_k4,
            model=sr_model,
            device=device,
            feather=8,
            halo=10,
            bicubic=bicubic_full,
        )
    print(f"[+] Composited {num_patches} SR patches over Bicubic base (8px feather)")

    # Convert grayscale outputs to BGR for YOLO and saving
    bicubic_bgr = cv2.cvtColor(bicubic_full, cv2.COLOR_GRAY2BGR)
    selective_bgr = cv2.cvtColor(selective_composite, cv2.COLOR_GRAY2BGR)

    # 6 — Load YOLO and run real inference (exits if checkpoint absent)
    yolo = load_yolo(args.det_checkpoint)

    print(f"\n[+] Running YOLO11n inference (conf={args.conf}) ...")
    bicubic_yolo_result = yolo.predict(bicubic_bgr, conf=args.conf, verbose=False, imgsz=640)
    selective_yolo_result = yolo.predict(selective_bgr, conf=args.conf, verbose=False, imgsz=640)

    bic_info = run_yolo_inference(yolo, bicubic_bgr, args.conf)
    sel_info = run_yolo_inference(yolo, selective_bgr, args.conf)

    # 7 — Save output images
    cv2.imwrite(str(args.output_dir / "01_input_lr_320x256.png"), lr)
    cv2.imwrite(str(args.output_dir / "02_full_bicubic_640x512.png"), bicubic_full)
    cv2.imwrite(str(args.output_dir / "03_selective_sr_composite_640x512.png"), selective_composite)

    diff = cv2.absdiff(selective_composite, bicubic_full)
    diff_vis = cv2.applyColorMap(diff * 5, cv2.COLORMAP_JET)
    cv2.imwrite(str(args.output_dir / "04_selective_vs_bicubic_diff.png"), diff_vis)

    bicubic_annotated = annotate_image(bicubic_bgr, bicubic_yolo_result)
    selective_annotated = annotate_image(selective_bgr, selective_yolo_result)
    cv2.imwrite(str(args.output_dir / "05_bicubic_detections.png"), bicubic_annotated)
    cv2.imwrite(str(args.output_dir / "06_selective_sr_detections.png"), selective_annotated)
    print(f"[+] Output images saved to: {args.output_dir.resolve()}")

    # 8 — Print live single-image detection results
    print("\n========== Live Single-Image YOLO Detection Results ==========")
    print(f"  Input: {'synthetic scene' if not args.image else args.image.name}")
    print(f"  Confidence threshold: {args.conf}")
    print()
    print(f"  {'Method':<25} | {'Total Dets':>10} | {'Small Dets':>10}")
    print("  " + "-" * 52)
    print(f"  {'Full Bicubic Baseline':<25} | {bic_info['total_detections']:>10} | {bic_info['small_detections']:>10}")
    print(f"  {'Contrast 4-Region SR':<25} | {sel_info['total_detections']:>10} | {sel_info['small_detections']:>10}")
    print()
    print("NOTE: This is a SINGLE synthetic image — results are illustrative,")
    print("      not comparable to the 3,749-image frozen Step 11L evaluation.")

    print("\n========== Reference: Frozen Step 11L Results (3,749 images) ===========")
    print("Source: reports/published/step11l_full_3749_summary.json")
    print("Formula: Utility = Small_TP - 0.25 × FP  (conf=0.25, GPU-deterministic)")
    print()
    ref = [
        ("Full Bicubic Baseline",  28401, 16722, 24220.50, "10.31 ms"),
        ("Full Large FSRCNN",      27671, 16516, 23542.00, "32.74 ms"),
        ("Random 4 Regions",       28302, 16825, 24095.75, "18.94 ms"),
        ("Contrast 4 Regions",     28391, 17199, 24091.25, "19.31 ms"),
    ]
    print(f"  {'Method':<25} | {'Small TP':>9} | {'FP':>7} | {'Utility':>10} | {'Latency':>10}")
    print("  " + "-" * 73)
    for name, stp, fp, util, lat in ref:
        print(f"  {name:<25} | {stp:>9} | {fp:>7} | {util:>10.2f} | {lat:>10}")
    print("==========================================================================")
    print("Finding: Full Bicubic achieved highest utility and lowest latency.")
    print("         Neural SR increased false positives on this benchmark.")
    print("Done!")


if __name__ == "__main__":
    main()
