"""Step 5A: exploratory ROI intervention labels for UtiTherm-SR (TRAIN ONLY).

This does NOT train the utility predictor and does NOT demonstrate compute savings.
For each TRAIN LR image, hold the detector fixed, compare a linear-upscaled
640x512 frame with frames containing ONE locally bicubic-enhanced tile.
The small-object detection change is the label, not an input feature.

Example:
 python scripts/probe_roi_utility.py --config configs/utility_probe.yaml --limit 6 --out reports/utility/probe_smoke_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from evaluate_bicubic_baseline import ground_truth, match_predictions

REQUIRED = {"train_lr", "checkpoint", "seed", "grid_rows", "grid_cols", "feather_hr_px",
            "imgsz", "confidence", "nms_iou", "false_positive_penalty", "device"}


def roi_grid(width: int, height: int, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    """Partition [0,W)x[0,H) with exact coverage and no overlaps."""
    if min(width, height, rows, cols) < 1 or rows > height or cols > width:
        raise ValueError("Invalid image size or grid")
    x = [j * width // cols for j in range(cols + 1)]
    y = [j * height // rows for j in range(rows + 1)]
    return [(x[c], y[r], x[c + 1], y[r + 1])
            for r in range(rows) for c in range(cols)]


def enhanced_tile(linear: np.ndarray, cubic: np.ndarray,
                  tile: tuple[int, int, int, int], feather: int = 8) -> np.ndarray:
    """Replace only tile interior with bicubic, blending its border for fewer seams."""
    if linear.shape != cubic.shape or linear.ndim != 2 or linear.dtype != np.uint8 or cubic.dtype != np.uint8:
        raise ValueError("Expected equal-shaped uint8 grayscale images")
    x0, y0, x1, y1 = tile
    h, w = linear.shape
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or feather < 0:
        raise ValueError("Invalid tile or feather")
    out = linear.copy()
    if feather == 0:
        out[y0:y1, x0:x1] = cubic[y0:y1, x0:x1]
        return out
    th, tw = y1 - y0, x1 - x0
    # Deterministic smooth ramp: mask=0 on boundary, mask=1 in tile interior.
    yy = np.minimum(np.arange(th), np.arange(th)[::-1]).astype(np.float32)
    xx = np.minimum(np.arange(tw), np.arange(tw)[::-1]).astype(np.float32)
    ay = np.clip(yy / feather, 0.0, 1.0)
    ax = np.clip(xx / feather, 0.0, 1.0)
    alpha = ay[:, None] * ax[None, :]
    base = linear[y0:y1, x0:x1].astype(np.float32)
    target = cubic[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.rint(base * (1.0 - alpha) + target * alpha).clip(0, 255).astype(np.uint8)
    return out


def sequence_id(path: Path) -> str:
    """Match the project's video-sequence naming convention."""
    return path.stem.split("-frame-", 1)[0] if "-frame-" in path.stem else path.stem


def diverse_subset(paths: list[Path], limit: int, seed: int) -> list[Path]:
    """Deterministically select across video sequences; 0 means all paths."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    paths = sorted(paths)
    if limit == 0 or len(paths) <= limit:
        return paths
    by_seq: dict[str, list[Path]] = {}
    for p in paths:
        by_seq.setdefault(sequence_id(p), []).append(p)
    rng = random.Random(seed)
    keys = sorted(by_seq)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(by_seq[k])
    selection = []
    while len(selection) < limit and keys:
        next_keys = []
        for k in keys:
            if by_seq[k]:
                selection.append(by_seq[k].pop())
                if len(selection) == limit:
                    break
            if by_seq[k]:
                next_keys.append(k)
        keys = next_keys
    return sorted(selection)


def prediction_records(model, img: np.ndarray, cfg: dict) -> list[dict]:
    """Single-image inference prevents the prior giant-list CUDA OOM bug."""
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError("Require uint8 grayscale image")
    frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    results = model.predict(
        source=frame, imgsz=int(cfg["imgsz"]), conf=float(cfg["confidence"]),
        iou=float(cfg["nms_iou"]), max_det=300, device=str(cfg["device"]),
        verbose=False, save=False, stream=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one prediction result, got {len(results)}")
    boxes = results[0].boxes
    if boxes is None:
        return []
    # Ground truth uses ORIGINAL LR 320x256 coordinates, not the upscaled HR pixels.
    sy, sx = 256.0 / img.shape[0], 320.0 / img.shape[1]
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()
    clss = boxes.cls.detach().cpu().tolist()
    return [{"box": (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy),
             "conf": float(q), "cls": int(k)} for b, q, k in zip(xyxy, confs, clss)]


def detection_counts(gts: list[dict], preds: list[dict], threshold: float) -> dict:
    matched, fps = match_predictions(gts, preds, threshold)
    return {"small_tp": sum(i in matched and gt["size"] == "small" for i, gt in enumerate(gts)),
            "all_tp": len(matched), "fp": int(fps),
            "small_gt": sum(gt["size"] == "small" for gt in gts)}


def utility(base: dict, perturbed: dict, fp_penalty: float) -> dict:
    """Global detection gain from ONE tile; no ROI chosen using ground truth."""
    if fp_penalty < 0 or not math.isfinite(fp_penalty):
        raise ValueError("fp_penalty must be finite and nonnegative")
    dtp = perturbed["small_tp"] - base["small_tp"]
    dall = perturbed["all_tp"] - base["all_tp"]
    dfp = perturbed["fp"] - base["fp"]
    # Score is a defined exploratory training target, not standard AP or utility.
    score = dtp - fp_penalty * dfp
    return {"delta_small_tp": int(dtp), "delta_all_tp": int(dall),
            "delta_fp": int(dfp), "utility_score": round(float(score), 6),
            "positive_label": int(score > 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/utility_probe.yaml"))
    ap.add_argument("--limit", type=int, default=6, help="Maximum training images; 0=all (expensive)")
    ap.add_argument("--out", type=Path, default=Path("reports/utility/probe_smoke_v1"))
    args = ap.parse_args()
    if args.limit < 0:
        ap.error("--limit cannot be negative")
    if not args.config.is_file():
        ap.error(f"Missing config: {args.config}")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or REQUIRED - cfg.keys():
        ap.error(f"Missing config keys: {sorted(REQUIRED - (cfg.keys() if isinstance(cfg, dict) else set()))}")
    root = Path(cfg["train_lr"])
    checkpoint = Path(cfg["checkpoint"])
    if root.name != "train" or root.parent.name != "images" or "yolo_flir_lr_x2" not in root.parts:
        ap.error("Safety check: train_lr must be data/processed/yolo_flir_lr_x2/images/train")
    if not root.is_dir() or not checkpoint.is_file():
        ap.error(f"Missing training images or checkpoint: {root}, {checkpoint}")
    if (args.out / "manifest.json").exists() or (args.out / "roi_utility.csv").exists():
        ap.error(f"Output already exists: {args.out}; select a new --out")
    files = sorted(p for p in root.glob("*.png") if p.is_file())
    if not files:
        ap.error(f"No PNG files in {root}")
    files = diverse_subset(files, args.limit, int(cfg["seed"]))
    rows, image_stats = [], []
    from ultralytics import YOLO
    import torch
    if not torch.cuda.is_available():
        ap.error("CUDA GPU not available")
    model = YOLO(str(checkpoint))
    tiles = roi_grid(640, 512, int(cfg["grid_rows"]), int(cfg["grid_cols"]))
    label_root = root.parent.parent / "labels" / "train"
    if not label_root.is_dir():
        ap.error(f"Missing training labels: {label_root}")
    threshold = float(cfg["confidence"])
    if not 0 < threshold < 1:
        ap.error("confidence must be in (0,1)")
    for i, path in enumerate(files, 1):
        label_path = label_root / f"{path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing ground truth for {path.name}")
        gts = ground_truth(label_path)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (256, 320):
            raise ValueError(f"Expected 320x256 grayscale training image: {path}")
        base = cv2.resize(gray, (640, 512), interpolation=cv2.INTER_LINEAR)
        cubic = cv2.resize(gray, (640, 512), interpolation=cv2.INTER_CUBIC)
        base_metrics = detection_counts(gts, prediction_records(model, base, cfg), threshold)
        image_stats.append({"image": path.name, "sequence": sequence_id(path), **base_metrics})
        for tile_index, tile in enumerate(tiles):
            altered = enhanced_tile(base, cubic, tile, int(cfg["feather_hr_px"]))
            new_metrics = detection_counts(gts, prediction_records(model, altered, cfg), threshold)
            target = utility(base_metrics, new_metrics, float(cfg["false_positive_penalty"]))
            rows.append({
                "image": path.name, "sequence": sequence_id(path), "tile_index": tile_index,
                "tile_x0_hr": tile[0], "tile_y0_hr": tile[1], "tile_x1_hr": tile[2], "tile_y1_hr": tile[3],
                "enhanced_fraction": round((tile[2]-tile[0])*(tile[3]-tile[1])/(640*512), 6),
                "baseline_small_tp": base_metrics["small_tp"], "baseline_fp": base_metrics["fp"],
                "candidate_small_tp": new_metrics["small_tp"], "candidate_fp": new_metrics["fp"],
                **target,
            })
        print(f"Measured {i}/{len(files)} TRAIN images; {len(rows)} ROI interventions", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "roi_utility.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "baseline_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_stats[0].keys()))
        writer.writeheader()
        writer.writerows(image_stats)
    manifest = {
        "task": "Step 5A exploratory intervention labels for tile utility",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(root), "split": "train ONLY", "num_images": len(files),
        "num_tiles_per_image": len(tiles), "num_interventions": len(rows),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config": cfg, "positive_labels": sum(r["positive_label"] for r in rows),
        "nonzero_delta_small_tp": sum(r["delta_small_tp"] != 0 for r in rows),
        "notes": [
            "No detection validation/test images are read.",
            "Ground-truth labels are used ONLY offline to score interventions.",
            "At runtime tile candidates are a fixed grid; no ground-truth ROI selection.",
            "The baseline is explicit OpenCV linear upscaling and is not identical to all Ultralytics resize pipelines.",
            "This stage computes full-frame bicubic once per image to enable interventions; it DOES NOT demonstrate compute savings.",
            "Targets are in-sample relative to the already TRAIN-trained YOLO checkpoint: use out-of-fold detectors or a true held-out training split for rigorous subsequent utility-model evaluation.",
            "Frame/sequence correlation must be accounted for when splitting future utility datasets.",
            "Utility score = delta_small_TP - false_positive_penalty * delta_FP at IoU=0.5, confidence as configured.",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"DONE — {args.out/'roi_utility.csv'} | {len(rows)} interventions | "
          f"positive: {manifest['positive_labels']} | nonzero small TP changes: {manifest['nonzero_delta_small_tp']}")


if __name__ == "__main__":
    main()
