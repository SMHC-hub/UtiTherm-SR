"""Evaluate size-stratified recall and confidence sensitivity for UtiTherm-SR.

Uses only the *validation* partition. Size categories are based on the
320x256 low-resolution image's pixel area, NOT the original HR image.
Matching: class-aware one-to-one greedy assignment, IoU >= 0.50.
These diagnostics are not official COCO AP or Ultralytics macro recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

CLASSES = [
    "person", "bike", "car", "motor", "bus", "truck",
    "other_vehicle", "light", "sign",
]
THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50)
IOU_THRESHOLD = 0.50


def category(box: tuple[float, float, float, float]) -> str:
    """COCO-style pixel-area bins measured on the actual LR image."""
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def read_labels(path: Path, width: int, height: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YOLO label: {path}")
    gts = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 columns at {path}:{line_no}")
        cls_float, xc, yc, bw, bh = map(float, parts)
        if not all(map(math.isfinite, (cls_float, xc, yc, bw, bh))):
            raise ValueError(f"Non-finite values at {path}:{line_no}")
        cls = int(cls_float)
        if cls != cls_float or cls < 0 or cls >= len(CLASSES):
            raise ValueError(f"Unexpected class ID at {path}:{line_no}")
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Invalid normalized box at {path}:{line_no}")
        box = (
            max(0.0, (xc - bw / 2) * width),
            max(0.0, (yc - bh / 2) * height),
            min(float(width), (xc + bw / 2) * width),
            min(float(height), (yc + bh / 2) * height),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Non-positive clipped box at {path}:{line_no}")
        gts.append({"cls": cls, "box": box, "size": category(box)})
    return gts


def match_gt(gts: list[dict], predictions: list[dict], threshold: float) -> set[int]:
    """Greedy confidence-ordered, class-aware, one-to-one IoU matching."""
    matched: set[int] = set()
    for prediction in sorted(predictions, key=lambda p: p["conf"], reverse=True):
        if prediction["conf"] < threshold:
            continue
        best_idx, best_iou = None, IOU_THRESHOLD
        for idx, gt in enumerate(gts):
            if idx in matched or gt["cls"] != prediction["cls"]:
                continue
            overlap = iou(gt["box"], prediction["box"])
            if overlap >= best_iou:
                best_iou, best_idx = overlap, idx
        if best_idx is not None:
            matched.add(best_idx)
    return matched


def diagnose(gt: dict, predictions: list[dict], threshold: float) -> str:
    preds = [p for p in predictions if p["conf"] >= threshold]
    same = [iou(gt["box"], p["box"]) for p in preds if p["cls"] == gt["cls"]]
    wrong = [iou(gt["box"], p["box"]) for p in preds if p["cls"] != gt["cls"]]
    if wrong and max(wrong) >= IOU_THRESHOLD:
        return "possible_wrong_class"
    if same and max(same) >= IOU_THRESHOLD:
        return "overlapping_or_competing_annotation"
    if same and max(same) >= 0.10:
        return "possible_localization_error"
    return "no_matching_candidate"


def preview(path: Path, gt: list[dict], matched: set[int], out: Path) -> None:
    with Image.open(path) as src:
        img = src.convert("RGB")
    draw = ImageDraw.Draw(img)
    for idx, item in enumerate(gt):
        color = "#33FF66" if idx in matched else "#FF3333"
        draw.rectangle(item["box"], outline=color, width=2)
    img = img.resize((img.width * 2, img.height * 2), Image.Resampling.NEAREST)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/detection/yolo11n_lr_baseline_v1.pt")
    parser.add_argument("--data", default="data/processed/yolo_flir_lr_x2")
    parser.add_argument("--out", default="reports/detection_evaluation/small_object_v1")
    parser.add_argument("--limit", type=int, default=0, help="0 = complete validation set")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--previews", type=int, default=6)
    args = parser.parse_args()

    images_dir = Path(args.data) / "images" / "val"
    labels_dir = Path(args.data) / "labels" / "val"
    model_path = Path(args.model)
    out_dir = Path(args.out)
    if not model_path.is_file():
        parser.error(f"Missing trained model: {model_path}")
    if not images_dir.is_dir() or not labels_dir.is_dir():
        parser.error("Validation image/label directories not found")
    paths = sorted(images_dir.glob("*.png"))
    if not paths:
        parser.error(f"No PNG files under {images_dir}")
    if args.limit < 0 or args.batch < 1:
        parser.error("--limit must be >= 0 and --batch must be >= 1")
    if args.limit:
        paths = paths[:args.limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    counts = {t: {s: [0, 0] for s in ("all", "small", "medium", "large")}
              for t in THRESHOLDS}
    per_class = defaultdict(lambda: [0, 0])
    failure_modes = Counter()
    missed_rows = []
    sample_candidates = []
    total_gt = 0

    # Run the GPU inference once at the lowest confidence, then sweep thresholds.
    results = model.predict(
        source=[str(p) for p in paths], imgsz=320, batch=args.batch,
        device=args.device, conf=min(THRESHOLDS), iou=0.7,
        max_det=300, stream=True, verbose=False, save=False,
    )
    for image_no, result in enumerate(results, 1):
        img_path = Path(result.path)
        w, h = result.orig_shape[1], result.orig_shape[0]
        gts = read_labels(labels_dir / (img_path.stem + ".txt"), w, h)
        total_gt += len(gts)
        boxes = result.boxes
        predictions = []
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            predictions = [
                {"box": tuple(box), "conf": float(conf), "cls": int(cls)}
                for box, conf, cls in zip(xyxy, confs, classes)
            ]

        default_matched = set()
        for threshold in THRESHOLDS:
            matched = match_gt(gts, predictions, threshold)
            for idx, gt in enumerate(gts):
                for key in ("all", gt["size"]):
                    counts[threshold][key][0] += 1  # GT objects
                    counts[threshold][key][1] += idx in matched  # matched objects
            if threshold == 0.25:
                default_matched = matched

        for idx, gt in enumerate(gts):
            per_class[gt["cls"]][0] += 1
            per_class[gt["cls"]][1] += idx in default_matched
            if idx not in default_matched:
                reason = diagnose(gt, predictions, 0.25)
                failure_modes[reason] += 1
                missed_rows.append({
                    "image": img_path.name, "class": CLASSES[gt["cls"]],
                    "size": gt["size"], "possible_failure": reason,
                    "x1": round(gt["box"][0], 2), "y1": round(gt["box"][1], 2),
                    "x2": round(gt["box"][2], 2), "y2": round(gt["box"][3], 2),
                })
        missed_count = len(gts) - len(default_matched)
        if gts and missed_count:
            sample_candidates.append((missed_count, str(img_path), gts, default_matched))
        if image_no % 100 == 0 or image_no == len(paths):
            print(f"Evaluated {image_no}/{len(paths)} validation images", flush=True)

    threshold_metrics = {}
    for threshold in THRESHOLDS:
        threshold_metrics[str(threshold)] = {
            size: {
                "ground_truth": gt_n,
                "matched": hit_n,
                "missed": gt_n - hit_n,
                "recall": round(hit_n / gt_n, 6) if gt_n else None,
            }
            for size, (gt_n, hit_n) in counts[threshold].items()
        }

    report = {
        "purpose": "Size-stratified validation recall and threshold diagnostics",
        "checkpoint": str(model_path), "data": str(args.data),
        "images_evaluated": len(paths), "ground_truth_objects": total_gt,
        "iou_threshold": IOU_THRESHOLD,
        "note": "Object-size bins use LOW-RESOLUTION image area. These are diagnostic micro recall values, not official COCO AP or Ultralytics macro recall. Thresholds explored on validation data; do not report optimized values as held-out test results.",
        "size_bins_lr_pixels": {"small": "area < 32^2", "medium": "32^2 <= area < 96^2", "large": "area >= 96^2"},
        "threshold_results": threshold_metrics,
        "class_recall_at_0_25": {
            CLASSES[k]: {"gt": v[0], "matched": v[1],
                         "recall": round(v[1] / v[0], 6) if v[0] else None}
            for k, v in sorted(per_class.items())
        },
        "possible_failure_counts_at_0_25": dict(failure_modes),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out_dir / "missed_objects.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "image", "class", "size", "possible_failure", "x1", "y1", "x2", "y2"
        ))
        writer.writeheader()
        writer.writerows(missed_rows)

    for rank, (_, name, gts, matched) in enumerate(
        sorted(sample_candidates, key=lambda row: row[0], reverse=True)[:args.previews], 1
    ):
        preview(Path(name), gts, matched, out_dir / "missed_previews" / f"{rank:02d}_{Path(name).name}")

    print("\nValidation recall @ IoU 0.50 (class-aware one-to-one matching)")
    for threshold, sizes in threshold_metrics.items():
        print(f"Threshold {threshold}: " + ", ".join(
            f"{size}={metrics['recall']:.1%}" if metrics["recall"] is not None else f"{size}=N/A"
            for size, metrics in sizes.items()
        ))
    print(f"Saved evaluation to: {out_dir}")
    print("Missed-object previews: RED = missed, GREEN = matched GT boxes")


if __name__ == "__main__":
    main()
