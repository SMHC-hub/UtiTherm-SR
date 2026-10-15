"""UtiTherm-SR Phase 1 / Step 4: controlled bicubic-vs-LR detection evaluation.

Run inside the EXISTING UtiTherm-SR Docker environment, at /workspace:
  python scripts/evaluate_bicubic_baseline.py --limit 30 --out reports/sr_baseline/bicubic_smoke_30
  python scripts/evaluate_bicubic_baseline.py --limit 0 --out reports/sr_baseline/bicubic_full_val

Do not train a detector. Uses saved YOLO11n. Always the same validation images,
normalized bounding boxes, 9 classes, checkpoint, confidence and NMS settings.
Object-size bins are computed in ORIGINAL 320x256 LR pixel units for ALL cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import time
from pathlib import Path

import yaml
from PIL import Image

CASES = (
    ("lr_320", "lr", 320),
    ("lr_640_resize_control", "lr", 640),
    ("bicubic_320", "bicubic", 320),
    ("bicubic_640", "bicubic", 640),
    ("nearest_640_control", "nearest", 640),
)
CONF_LEVELS = (0.01, 0.05, 0.25)
IOU_MATCH = 0.5
LR_SIZE = (320, 256)


def object_size(box: tuple[float, float, float, float]) -> str:
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def ground_truth(label_path: Path) -> list[dict]:
    """Parse normalized YOLO labels into the ORIGINAL LR 320x256 coordinate space."""
    records = []
    for n, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label at {label_path}:{n}")
        cls_id, xc, yc, bw, bh = map(float, parts)
        if not all(math.isfinite(x) for x in (cls_id, xc, yc, bw, bh)):
            raise ValueError(f"Non-finite label at {label_path}:{n}")
        if int(cls_id) != cls_id or not 0 <= cls_id <= 8:
            raise ValueError(f"Invalid class at {label_path}:{n}")
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Invalid normalized box at {label_path}:{n}")
        w, h = LR_SIZE
        rect = (max(0, (xc - bw / 2) * w), max(0, (yc - bh / 2) * h),
                min(w, (xc + bw / 2) * w), min(h, (yc + bh / 2) * h))
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            raise ValueError(f"Invalid clipped box at {label_path}:{n}")
        records.append({"cls": int(cls_id), "box": rect, "size": object_size(rect)})
    return records


def match_predictions(gts: list[dict], preds: list[dict], threshold: float) -> tuple[set[int], int]:
    """Class-aware, confidence-sorted one-to-one matching at IoU >= .50."""
    matched: set[int] = set()
    false_positives = 0
    for pred in sorted(preds, key=lambda d: d["conf"], reverse=True):
        if pred["conf"] < threshold:
            continue
        best_index, best_overlap = None, IOU_MATCH
        for index, gt in enumerate(gts):
            if index in matched or gt["cls"] != pred["cls"]:
                continue
            overlap = box_iou(gt["box"], pred["box"])
            if overlap >= best_overlap:
                best_index, best_overlap = index, overlap
        if best_index is None:
            false_positives += 1
        else:
            matched.add(best_index)
    return matched, false_positives


def write_data_yaml(folder: Path, training_data_dir: Path, names: dict) -> Path:
    conf = {
        "path": str(folder.resolve()),
        "train": str((training_data_dir / "images" / "train").resolve()),
        "val": "images/val",
        "names": names,
    }
    file = folder / "data.yaml"
    file.write_text(yaml.safe_dump(conf, sort_keys=False), encoding="utf-8")
    return file


def prepare_datasets(source: Path, dest: Path, limit: int) -> tuple[dict[str, Path], dict]:
    """Prepare identical labeled validation subsets with different interpolation."""
    originals = source / "images" / "val"
    source_labels = source / "labels" / "val"
    source_yaml = source / "data.yaml"
    if not originals.is_dir() or not source_labels.is_dir() or not source_yaml.is_file():
        raise FileNotFoundError(f"Missing source validation images, labels or YAML in {source}")
    files = sorted(originals.glob("*.png"))
    if not files:
        raise ValueError(f"No validation PNGs found in {originals}")
    if limit:
        files = files[:limit]
    names = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))["names"]
    paths = {}
    for style in ("lr", "bicubic", "nearest"):
        folder = dest / style
        (folder / "images" / "val").mkdir(parents=True, exist_ok=True)
        (folder / "labels" / "val").mkdir(parents=True, exist_ok=True)
        paths[style] = write_data_yaml(folder, source, names)
    durations = []
    for idx, original in enumerate(files, 1):
        label = source_labels / f"{original.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label: {label}")
        with Image.open(original) as img:
            gray = img.convert("L")
            if gray.size != LR_SIZE:
                raise ValueError(f"Expected LR {LR_SIZE}, got {gray.size}: {original}")
            tic = time.perf_counter()
            bicubic = gray.resize((640, 512), Image.Resampling.BICUBIC)
            durations.append((time.perf_counter() - tic) * 1000)
            nearest = gray.resize((640, 512), Image.Resampling.NEAREST)
            bicubic.save(dest / "bicubic" / "images" / "val" / original.name)
            nearest.save(dest / "nearest" / "images" / "val" / original.name)
        shutil.copy2(original, dest / "lr" / "images" / "val" / original.name)
        for style in ("lr", "bicubic", "nearest"):
            shutil.copy2(label, dest / style / "labels" / "val" / label.name)
        if idx % 100 == 0 or idx == len(files):
            print(f"Prepared {idx}/{len(files)} images", flush=True)
    return paths, {
        "image_count": len(files),
        "image_selection": "sorted filenames from the same FLIR LR validation set",
        "lr_size": list(LR_SIZE), "upscaled_size": [640, 512],
        "bicubic_resize_ms_mean_excluding_IO": round(statistics.mean(durations), 4),
        "bicubic_resize_ms_median_excluding_IO": round(statistics.median(durations), 4),
        "notes": "Synthetic resize only; saving images, file reads, loader wait and inference excluded from resize timing.",
    }


def size_recall(model, directory: Path, size: int, batch: int, device: str) -> dict:
    """Diagnostic class-aware micro recall and FP; NOT official COCO size-AP."""
    imgs = sorted((directory / "images" / "val").glob("*.png"))
    stats = {
        t: {s: {"gt": 0, "matched": 0} for s in ("all", "small", "medium", "large")}
        for t in CONF_LEVELS
    }
    total_fp = {t: 0 for t in CONF_LEVELS}
    outputs = model.predict(
        source=str((directory / "images" / "val").resolve()),
        imgsz=size, batch=batch, device=device,
        conf=min(CONF_LEVELS), iou=0.7, max_det=300, stream=True,
        verbose=False, save=False,
    )
    for result in outputs:
        image_name = Path(result.path).name
        gts = ground_truth(directory / "labels" / "val" / f"{Path(image_name).stem}.txt")
        orig_h, orig_w = result.orig_shape
        scale_x, scale_y = LR_SIZE[0] / orig_w, LR_SIZE[1] / orig_h
        boxes = result.boxes
        predictions = []
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            for rect, conf, cls in zip(xyxy, confidences, classes):
                predictions.append({
                    "box": (rect[0] * scale_x, rect[1] * scale_y,
                            rect[2] * scale_x, rect[3] * scale_y),
                    "conf": float(conf), "cls": int(cls),
                })
        for t in CONF_LEVELS:
            matched, fp = match_predictions(gts, predictions, t)
            total_fp[t] += fp
            for idx, gt in enumerate(gts):
                for category in ("all", gt["size"]):
                    stats[t][category]["gt"] += 1
                    stats[t][category]["matched"] += int(idx in matched)
    return {
        str(t): {
            "false_positives": total_fp[t],
            "groups": {
                s: {**v, "recall": round(v["matched"] / v["gt"], 6) if v["gt"] else None}
                for s, v in stats[t].items()
            },
        }
        for t in CONF_LEVELS
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=30, help="First N validation images (0 = all)")
    ap.add_argument("--out", type=Path, default=Path("reports/sr_baseline/bicubic_smoke_30"))
    ap.add_argument("--source", type=Path, default=Path("data/processed/yolo_flir_lr_x2"))
    ap.add_argument("--checkpoint", type=Path, default=Path("models/detection/yolo11n_lr_baseline_v1.pt"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    if args.limit < 0 or args.batch < 1:
        ap.error("--limit must be >= 0 and --batch >= 1")
    if not args.checkpoint.is_file():
        ap.error(f"Checkpoint missing: {args.checkpoint}")
    if (args.out / "summary.json").exists():
        ap.error(f"Experiment already exists at {args.out}; choose a new --out")

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Docker")
    dest = Path("data/interim/bicubic_eval") / args.out.name
    data_yamls, prep = prepare_datasets(args.source, dest, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.checkpoint))
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source": str(args.source),
        "gpu": torch.cuda.get_device_name(0),
        "images": prep["image_count"],
        "preparation": prep,
        "comparison_cases": {},
        "notes": [
            "All evaluations use same validation images, annotations and detector checkpoint.",
            "mAP is standard Ultralytics validation AP; size recall is separate IoU=0.50 class-aware diagnostic.",
            "Size definitions are always based on ORIGINAL 320x256 LR label area (<32^2 small, <96^2 medium).",
            "Confidence 0.001 for AP; diagnostic recall/FP use 0.01, 0.05 and 0.25.",
            "Baseline LR @640 includes detector-internal interpolation; nearest @640 controls input resolution.",
            "Speed timings from validator exclude file I/O, generated-image preparation and full pipeline overhead.",
            "Do not use validation-selected settings as an independent held-out test result.",
        ],
    }
    rows = []
    for label, style, imgsz in CASES:
        print(f"\nEVALUATING {label} (input {style}; model imgsz={imgsz})", flush=True)
        metrics = model.val(
            data=str(data_yamls[style].resolve()), imgsz=imgsz,
            batch=args.batch, device=args.device, workers=2,
            conf=0.001, iou=0.7, max_det=300,
            plots=False, save=False, verbose=False,
            project=str(args.out.resolve()), name=f"val_{label}", exist_ok=True,
        )
        diagnostic = size_recall(model, dest / style, imgsz, args.batch, args.device)
        speed = {k: round(float(v), 4) for k, v in metrics.speed.items()}
        entry = {
            "data_style": style,
            "imgsz": imgsz,
            "precision_macro": round(float(metrics.box.mp), 6),
            "recall_macro": round(float(metrics.box.mr), 6),
            "map50": round(float(metrics.box.map50), 6),
            "map50_95": round(float(metrics.box.map), 6),
            "validator_speed_ms_per_image": speed,
            "diagnostic_size_recall": diagnostic,
        }
        report["comparison_cases"][label] = entry
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        rows.append({
            "case": label, "imgsz": imgsz,
            "precision_macro": entry["precision_macro"],
            "recall_macro": entry["recall_macro"],
            "map50": entry["map50"], "map50_95": entry["map50_95"],
            "small_recall_conf_025": diagnostic["0.25"]["groups"]["small"]["recall"],
            "fp_conf_025": diagnostic["0.25"]["false_positives"],
            "inference_ms_per_image": speed.get("inference"),
        })
        print(f"{label}: mAP50={entry['map50']:.3f}, mAP50-95={entry['map50_95']:.3f}, "
              f"small recall@0.25={rows[-1]['small_recall_conf_025']}, "
              f"FP@0.25={rows[-1]['fp_conf_025']}", flush=True)

    with (args.out / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDONE — see {args.out / 'comparison.csv'} and {args.out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
