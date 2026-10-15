"""UtiTherm-SR Quickstart Demo.

Demonstrates the core UtiTherm-SR pipeline:
1. Synthetic / Real LR thermal input (320x256)
2. Full-frame Bicubic upsampling base (640x512)
3. Grid tile extraction and contrast-based ROI ranking
4. Selective FSRCNN super-resolution on top-k tiles
5. Feathered composition over the immutable Bicubic base
6. Downstream detection utility metric calculation

Usage:
    python examples/demo_pipeline.py [--image path/to/image.png] [--output-dir examples/output]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from utitherm.sr.fsrcnn import FSRCNN
from utitherm.sr.bicubic_first_fsrcnn_step11a import render_bicubic_first_fsrcnn


def create_synthetic_thermal_scene(height: int = 256, width: int = 320) -> np.ndarray:
    """Generate a realistic synthetic thermal LR frame with warm targets and background gradients."""
    img = np.zeros((height, width), dtype=np.float32)
    # Add ambient gradient (ground warmer than sky)
    y_coords = np.linspace(30, 90, height)[:, None]
    img += y_coords
    # Add road surface texture
    img[int(height * 0.55):, :] += 20.0
    # Add synthetic warm targets (pedestrians / vehicles)
    # Target 1 (center)
    img[120:145, 150:162] += 120.0
    # Target 2 (small distant target)
    img[100:112, 80:86] += 95.0
    # Target 3 (right side vehicle)
    img[130:165, 230:270] += 110.0
    # Add slight thermal sensor noise
    np.random.seed(42)
    noise = np.random.normal(0, 3.0, (height, width))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def generate_grid_tiles(width: int = 640, height: int = 512, grid_x: int = 4, grid_y: int = 4) -> list[tuple[int, int, int, int]]:
    """Partition 640x512 HR coordinate frame into regular rectangular tiles."""
    w_tile = width // grid_x
    h_tile = height // grid_y
    tiles = []
    for gy in range(grid_y):
        for gx in range(grid_x):
            x0 = gx * w_tile
            y0 = gy * h_tile
            x1 = x0 + w_tile
            y1 = y0 + h_tile
            tiles.append((x0, y0, x1, y1))
    return tiles


def rank_tiles_by_contrast(lr_image: np.ndarray, tiles: list[tuple[int, int, int, int]]) -> list[int]:
    """Rank tiles by standard deviation (contrast / thermal variation) in the LR frame."""
    scores = []
    for idx, (x0, y0, x1, y1) in enumerate(tiles):
        # Map HR coordinates back to LR coordinates (scale = 2)
        lx0, ly0, lx1, ly1 = x0 // 2, y0 // 2, x1 // 2, y1 // 2
        patch = lr_image[ly0:ly1, lx0:lx1]
        contrast = float(np.std(patch))
        scores.append((contrast, idx))
    # Sort descending by contrast
    scores.sort(key=lambda s: s[0], reverse=True)
    return [idx for _, idx in scores]


def compute_utility(small_tp: int, fp: int, penalty: float = 0.25) -> float:
    """Compute UtiTherm detection utility score: Small TP - 0.25 * FP."""
    return float(small_tp) - penalty * float(fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="UtiTherm-SR pipeline demonstration")
    parser.add_argument("--image", type=Path, default=None, help="Optional path to 320x256 grayscale thermal image")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"), help="Output directory for visual comparisons")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("   UtiTherm-SR: Region-Selective Thermal Super-Resolution Demo   ")
    print("=================================================================")

    # 1. Load or synthesize input
    if args.image and args.image.is_file():
        lr = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
        if lr.shape != (256, 320):
            print(f"[!] Resizing input image from {lr.shape} to (256, 320)")
            lr = cv2.resize(lr, (320, 256), interpolation=cv2.INTER_AREA)
        print(f"[+] Loaded input image: {args.image} (320x256 uint8)")
    else:
        lr = create_synthetic_thermal_scene(256, 320)
        print("[+] Generated synthetic 320x256 thermal LR scene (pedestrians, vehicle, road gradient)")

    # 2. Full-frame Bicubic baseline
    bicubic_full = cv2.resize(lr, (640, 512), interpolation=cv2.INTER_CUBIC)
    print("[+] Rendered Full Bicubic baseline (640x512)")

    # 3. Partition grid tiles & rank by contrast
    tiles = generate_grid_tiles(width=640, height=512, grid_x=4, grid_y=4)
    ranked_indices = rank_tiles_by_contrast(lr, tiles)
    selected_k4 = ranked_indices[:4]
    print(f"[+] Partitioned image into {len(tiles)} candidate tiles (4x4 grid)")
    print(f"[+] Contrast-ranked top-4 selected tile indices: {selected_k4}")

    # 4. Instantiate Large FSRCNN
    # Large FSRCNN architecture: d=64, s=16, m=6
    model = FSRCNN(scale=2, d=64, s=16, m=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[+] Initialized Large FSRCNN model on device: {device}")

    # 5. Composite selected neural patches over Bicubic base
    with torch.no_grad():
        selective_composite, num_patches = render_bicubic_first_fsrcnn(
            lr=lr,
            tiles=tiles,
            selected=selected_k4,
            model=model,
            device=device,
            feather=8,
            halo=10,
            bicubic=bicubic_full,
        )
    print(f"[+] Composited {num_patches} neural SR patches over immutable Bicubic base with 8px feathering")

    # 6. Save visualizations
    cv2.imwrite(str(args.output_dir / "01_input_lr_320x256.png"), lr)
    cv2.imwrite(str(args.output_dir / "02_full_bicubic_640x512.png"), bicubic_full)
    cv2.imwrite(str(args.output_dir / "03_selective_sr_composite_640x512.png"), selective_composite)

    # Difference map (amplified for visibility)
    diff = cv2.absdiff(selective_composite, bicubic_full)
    diff_vis = cv2.applyColorMap(diff * 5, cv2.COLORMAP_JET)
    cv2.imwrite(str(args.output_dir / "04_selective_vs_bicubic_diff.png"), diff_vis)

    print(f"[+] Visual artifacts saved to: {args.output_dir.resolve()}")

    # 7. Demonstrate Utility Metric Calculation
    print("\n----------------- Detection Utility Metric -----------------")
    print("Formula: Utility = Small_TP - 0.25 * FP")
    example_cases = [
        ("Full Bicubic Baseline", 28401, 16722),
        ("Full Large FSRCNN", 27671, 16516),
        ("Random 4 Regions", 28302, 16825),
        ("Contrast 4 Regions", 28391, 17199),
    ]
    print(f"{'Method':<25} | {'Small TP':>10} | {'FP':>10} | {'Utility Score':>15}")
    print("-" * 68)
    for name, stp, fp in example_cases:
        u = compute_utility(stp, fp)
        print(f"{name:<25} | {stp:>10} | {fp:>10} | {u:>15.2f}")
    print("------------------------------------------------------------")
    print("Scientific Observation: Full Bicubic achieved highest utility and lowest latency.")
    print("Neural super-resolution in ROI regions introduced false positives that penalized overall utility.")
    print("Done!")


if __name__ == "__main__":
    main()
