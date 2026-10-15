"""Step 9B: batched ROI FSRCNN halo sweep, on training-domain development images.

Compares halos 10, 12, and 16 while holding images, selected tiles, FSRCNN
checkpoint, device, and output blending fixed. This is an SR-only benchmark,
not a detection performance test or independent generalization claim.

Run from project root inside its existing research container:
    python scripts/benchmark_batched_fsrcnn_halo_step9b.py --limit 4 \\
       --out reports/performance/halo_step9b_smoke_v1.json
    python scripts/benchmark_batched_fsrcnn_halo_step9b.py --limit 0 --repeats 7 \\
       --out reports/performance/halo_step9b_20_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_selective_fsrcnn import load_sr_checkpoint
from evaluate_selective_sr import (
    choose_tiles,
    get_eval_image_names,
    predict_tile_scores,
    predictor_model,
)
from probe_roi_utility import roi_grid
from utitherm.sr.roi_fsrcnn_batched import render_batched_fsrcnn

HALOS = (10, 12, 16)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def synchronized_render(frame, tiles, chosen, sr, device, halo):
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    out, pixels = render_batched_fsrcnn(
        frame, tiles, chosen, sr, device, feather=8, halo=halo
    )
    torch.cuda.synchronize(device)
    return out, pixels, (time.perf_counter_ns() - start) / 1e6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/selective_sr_eval.yaml"))
    p.add_argument("--sr-checkpoint", type=Path, default=Path("models/sr/fsrcnn_x2_v1.pt"))
    p.add_argument("--predictor-checkpoint", type=Path, default=None)
    p.add_argument("--predictor-source-csv", type=Path, default=None)
    p.add_argument("--limit", type=int, default=4, help="4 for smoke, 0 for all available frames")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.limit < 0 or not 3 <= a.repeats <= 20 or not 1 <= a.warmup <= 10:
        p.error("Require --limit>=0, --repeats 3..20, --warmup 1..10")
    if a.out.exists():
        p.error(f"Refusing to overwrite existing report: {a.out}")
    if not torch.cuda.is_available():
        p.error("Requires CUDA to benchmark the existing pipeline")
    for path in (a.config, a.sr_checkpoint):
        if not path.is_file():
            p.error(f"Missing file: {path}")
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    lr_root = Path(cfg["train_lr"])
    csv_path = Path(cfg["utility_csv"])
    split_path = Path(cfg["split_manifest"])
    predictor_path = a.predictor_checkpoint or Path(cfg["predictor_checkpoint"])
    predictor_csv_path = a.predictor_source_csv or csv_path
    if a.predictor_checkpoint is not None and a.predictor_source_csv is None:
        p.error("When overriding predictor checkpoint, also provide --predictor-source-csv")
    names = get_eval_image_names(csv_path, split_path, a.limit)
    if not names:
        p.error("No eligible images")
    device = torch.device("cuda:0")
    cv2.setNumThreads(1)
    selector, input_size = predictor_model(predictor_path, predictor_csv_path, device)
    sr = load_sr_checkpoint(a.sr_checkpoint, device)
    tiles = roi_grid(640, 512, 4, 4)
    prepared = []
    for name in names:
        gray = cv2.imread(str(lr_root / name), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (256, 320) or gray.dtype != np.uint8:
            raise ValueError(f"Invalid thermal input: {name}")
        scores = predict_tile_scores(
            selector, gray, tiles, input_size, device, float(cfg["false_positive_penalty"])
        )
        chosen = choose_tiles("predictor4", name, 4, len(tiles), int(cfg["seed"]), scores)
        if len(chosen) != 4:
            raise ValueError(f"Expected four regions: {name}")
        prepared.append((name, gray, chosen))

    measurements = []
    per_image = []
    for idx, (name, gray, chosen) in enumerate(prepared, 1):
        for _ in range(a.warmup):
            for h in HALOS:
                synchronized_render(gray, tiles, chosen, sr, device, h)

        rendered = {}
        px_by_halo = {}
        timings = {h: [] for h in HALOS}
        for rep in range(a.repeats):
            order = list(HALOS)
            random.Random(2026 + rep * 997 + idx).shuffle(order)
            for h in order:
                out, px, ms = synchronized_render(gray, tiles, chosen, sr, device, h)
                timings[h].append(ms)
                px_by_halo[h] = px
                if h not in rendered:
                    rendered[h] = out
                measurements.append({
                    "image": name, "halo": h, "repeat": rep, "sr_ms": ms,
                    "lr_pixels": px,
                })

        ref = rendered[16]
        for h in HALOS:
            # The comparison includes blending boundaries as well as tile interiors.
            d = np.abs(rendered[h].astype(np.int16) - ref.astype(np.int16))
            per_image.append({
                "image": name,
                "halo": h,
                "selected": chosen,
                "lr_pixels": px_by_halo[h],
                "median_sr_ms": statistics.median(timings[h]),
                "mae_vs_halo16_8bit": float(d.mean()),
                "max_diff_vs_halo16_8bit": int(d.max()),
                "fraction_pixels_changed_vs_halo16": float(np.count_nonzero(d) / d.size),
            })
        if idx == 1 or idx % 5 == 0 or idx == len(prepared):
            print(f"Measured {idx}/{len(prepared)} images with batched ROI FSRCNN")

    results = {}
    for h in HALOS:
        rows = [r for r in per_image if r["halo"] == h]
        results[str(h)] = {
            "lr_pixels": sorted(set(r["lr_pixels"] for r in rows)),
            "median_sr_ms_across_image_medians": round(
                float(statistics.median(r["median_sr_ms"] for r in rows)), 5
            ),
            "mean_sr_ms_across_image_medians": round(
                float(statistics.mean(r["median_sr_ms"] for r in rows)), 5
            ),
            "maximum_pixel_difference_vs_halo16": max(r["max_diff_vs_halo16_8bit"] for r in rows),
            "mean_mae_vs_halo16_8bit": round(
                float(statistics.mean(r["mae_vs_halo16_8bit"] for r in rows)), 7
            ),
            "mean_fraction_pixels_changed_vs_halo16": round(
                float(statistics.mean(r["fraction_pixels_changed_vs_halo16"] for r in rows)), 7
            ),
        }

    report = {
        "task": "Step 9B batched FSRCNN ROI halo SR-only sweep",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "num_images": len(prepared),
        "repeats": a.repeats,
        "halo_values": list(HALOS),
        "images_source": "V1 predictor-held-out training-domain sequences (engineering development only)",
        "checkpoints": {
            "sr": {"path": str(a.sr_checkpoint), "sha256": sha256_file(a.sr_checkpoint)},
            "selector": {"path": str(predictor_path), "sha256": sha256_file(predictor_path)},
            "selector_training_csv": {"path": str(predictor_csv_path), "sha256": sha256_file(predictor_csv_path)},
        },
        "results": results,
        "per_image": per_image,
        "latency_trials": measurements,
        "limitations": [
            "SR-only latency: detector and predictor excluded, no end-to-end claim.",
            "Pixel agreement with halo=16 is not a detection-quality metric.",
            "Training-domain images and previously used sequences: not an untouched final test.",
            "GPU timing influenced by host load and kernel scheduling; results descriptive.",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nBATCHED ROI HALO SWEEP (SR stage only, not full detection):")
    for h in HALOS:
        r = results[str(h)]
        print(
            f"halo={h:2d} | LR pixels={r['lr_pixels']} | "
            f"median SR={r['median_sr_ms_across_image_medians']:.3f} ms | "
            f"max diff={r['maximum_pixel_difference_vs_halo16']} | "
            f"mean MAE={r['mean_mae_vs_halo16_8bit']:.5f} | "
            f"changed fraction={r['mean_fraction_pixels_changed_vs_halo16']:.5f}"
        )
    print(f"SAVED — {a.out}")


if __name__ == "__main__":
    main()
