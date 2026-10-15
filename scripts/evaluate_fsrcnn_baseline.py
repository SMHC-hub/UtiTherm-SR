"""UtiTherm-SR Step 4C: full-image FSRCNN vs bicubic + fixed YOLO11n.

Run from repository root inside Docker. Does not train either model.
The original LR 320x256 validation images and 640x512 processed HR images
are required. HR images are used ONLY for SR quality metrics.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import shutil
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from utitherm.sr.fsrcnn import FSRCNN
try:
    from scripts.evaluate_bicubic_baseline import size_recall, write_data_yaml
except ModuleNotFoundError:  # python scripts/evaluate_fsrcnn_baseline.py
    from evaluate_bicubic_baseline import size_recall, write_data_yaml

LR_SIZE = (320, 256)
HR_SIZE = (640, 512)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def psnr_u8(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape or pred.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("PSNR inputs must be same-shaped uint8 arrays")
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return math.inf if mse == 0 else 10.0 * math.log10(255.0**2 / mse)


def ssim_u8(pred: np.ndarray, target: np.ndarray) -> float:
    """Grayscale 11x11 Gaussian-window SSIM (K1=.01 K2=.03), mean of valid pixels."""
    if pred.shape != target.shape or pred.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("SSIM inputs must be same-shaped uint8 arrays")
    if pred.ndim != 2 or min(pred.shape) < 11:
        raise ValueError("SSIM expects grayscale images at least 11x11")
    x, y = pred.astype(np.float64), target.astype(np.float64)
    blur = lambda a: cv2.GaussianBlur(a, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cov = blur(x * y) - mx * my
    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    score = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(np.mean(score[5:-5, 5:-5]))


def validate_reference(report: dict, count: int, checkpoint_hash: str, source: Path) -> None:
    if report.get("images") != count:
        raise ValueError(f"Bicubic report images={report.get('images')}, FSRCNN selection={count}; use matching --baseline-report")
    if report.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Detection checkpoint differs from Bicubic baseline!")
    if Path(report.get("source", "")) != source:
        raise ValueError("LR source differs from Bicubic report!")
    for name in ("lr_320", "lr_640_resize_control", "bicubic_640"):
        if name not in report.get("comparison_cases", {}):
            raise ValueError(f"Missing comparison case {name} in baseline report")


def load_fsrcnn(path: Path, device: torch.device) -> tuple[FSRCNN, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"state_dict", "model_kwargs"} <= payload.keys():
        raise ValueError("Expected FSRCNN Step 4B checkpoint with state_dict and model_kwargs")
    kwargs = payload["model_kwargs"]
    if not isinstance(kwargs, dict) or kwargs.get("scale") != 2:
        raise ValueError("Expected FSRCNN x2 checkpoint")
    model = FSRCNN(**kwargs)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def infer_u8(model: FSRCNN, lr: Image.Image, device: torch.device) -> tuple[np.ndarray, float]:
    lr_np = np.asarray(lr, dtype=np.float32)
    tensor = torch.from_numpy(lr_np.copy())[None, None].to(device) / 255.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        sr = model(tensor)
        if sr.shape != (1, 1, 512, 640):
            raise ValueError(f"Unexpected FSRCNN output shape: {tuple(sr.shape)}")
        sr = sr.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()[0, 0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return sr, (time.perf_counter() - t0) * 1000


def index_hr_images(hr_images: Path) -> dict[str, Path]:
    """Locate corresponding HR images by stem, independent of JPEG/PNG suffix.

    Reject ambiguous stems rather than silently pairing with the wrong target.
    Only support ordinary 8-bit visible/thermal image formats here; do not
    accidentally use 16-bit source TIFFs as the reconstruction reference.
    """
    allowed = {".png", ".jpg", ".jpeg"}
    index: dict[str, Path] = {}
    for candidate in sorted(hr_images.iterdir()):
        if not candidate.is_file() or candidate.suffix.lower() not in allowed:
            continue
        previous = index.get(candidate.stem)
        if previous is not None:
            raise ValueError(
                f"Ambiguous HR image stem '{candidate.stem}': {previous.name} and {candidate.name}"
            )
        index[candidate.stem] = candidate
    return index


def prepare(conf: dict, out: Path, limit: int, model: FSRCNN, device: torch.device) -> tuple[Path, dict]:
    source_lr = Path(conf["source_lr"])
    source_hr = Path(conf["source_hr"])
    lr_images = source_lr / "images" / "val"
    hr_images = source_hr / "images" / "val"
    lr_labels = source_lr / "labels" / "val"
    for path in (lr_images, hr_images, lr_labels, source_lr / "data.yaml"):
        if not path.exists():
            raise FileNotFoundError(f"Missing required dataset path: {path}")
    files = sorted(lr_images.glob("*.png"))
    if not files:
        raise ValueError("No LR validation PNG files")
    if limit:
        files = files[:limit]
    hr_index = index_hr_images(hr_images)
    # Preflight every selected pair before writing evaluation output. LR and HR
    # may have different image extensions after COCO-to-YOLO conversion.
    for f in files:
        label = lr_labels / (f.stem + ".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Missing LR annotation: {label}")
        if f.stem not in hr_index:
            raise FileNotFoundError(
                f"No HR image with stem {f.stem!r} in {hr_images} "
                f"(supported: .png/.jpg/.jpeg; found {len(hr_index)} HR images)"
            )
    dest = Path("data/interim/fsrcnn_eval") / out.name
    if dest.exists():
        raise FileExistsError(f"Intermediate dataset already exists: {dest}; choose a different --out")
    (dest / "images" / "val").mkdir(parents=True)
    (dest / "labels" / "val").mkdir(parents=True)
    names = yaml.safe_load((source_lr / "data.yaml").read_text(encoding="utf-8"))["names"]
    dataset_yaml = write_data_yaml(dest, source_lr, names)
    quality_rows = []
    timings = []
    mismatch = []
    for i, f in enumerate(files, 1):
        label = lr_labels / (f.stem + ".txt")
        target_file = hr_index[f.stem]
        with Image.open(f) as im_lr, Image.open(target_file) as im_hr:
            lr = im_lr.convert("L")
            hr = im_hr.convert("L")
            if lr.size != LR_SIZE or hr.size != HR_SIZE:
                raise ValueError(f"Invalid image dimensions: {f.name}: LR={lr.size} HR={hr.size}")
            sr, ms = infer_u8(model, lr, device)
            baseline = np.asarray(lr.resize(HR_SIZE, Image.Resampling.BICUBIC), dtype=np.uint8)
            hr_np = np.asarray(hr, dtype=np.uint8)
            # Audit whether detection LR matches the synthetic training degradation.
            training_lr = np.asarray(hr.resize(LR_SIZE, Image.Resampling.BICUBIC), dtype=np.uint8)
            mismatch.append(float(np.mean(np.abs(training_lr.astype(np.int16) - np.asarray(lr, dtype=np.int16)))))
        Image.fromarray(sr).save(dest / "images" / "val" / f.name)
        shutil.copy2(label, dest / "labels" / "val" / label.name)
        timings.append(ms)
        quality_rows.append({"image": f.name, "fsrcnn_psnr_db": psnr_u8(sr, hr_np),
                             "bicubic_psnr_db": psnr_u8(baseline, hr_np),
                             "fsrcnn_ssim": ssim_u8(sr, hr_np),
                             "bicubic_ssim": ssim_u8(baseline, hr_np),
                             "fsrcnn_sr_ms": round(ms, 5), "lr_synthetic_mismatch_mae": round(mismatch[-1], 6)})
        if i % 100 == 0 or i == len(files):
            print(f"FSRCNN enhanced {i}/{len(files)} images", flush=True)
    with (out / "per_image_quality.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(quality_rows[0]))
        writer.writeheader()
        writer.writerows(quality_rows)
    def avg(field: str) -> float:
        vals = [v[field] for v in quality_rows]
        # inf PSNR in a perfect image is not meaningful for dataset mean
        return round(statistics.mean(vals), 5) if all(math.isfinite(x) for x in vals) else math.inf
    quality = {"images": len(files), "files_sha256": hashlib.sha256("\n".join(f.name for f in files).encode()).hexdigest(),
               "fsrcnn_mean_psnr_db": avg("fsrcnn_psnr_db"), "bicubic_mean_psnr_db": avg("bicubic_psnr_db"),
               "fsrcnn_mean_ssim": avg("fsrcnn_ssim"), "bicubic_mean_ssim": avg("bicubic_ssim"),
               "lr_vs_pil_train_degradation_mae_u8": round(statistics.mean(mismatch), 6),
               "fsrcnn_inference_ms_mean": round(statistics.mean(timings), 5),
               "fsrcnn_inference_ms_median": round(statistics.median(timings), 5),
               "timing_notes": "FSRCNN forward + GPU-to-CPU output and sync, excludes host-to-GPU input transfer, file I/O, bicubic generation and YOLO detector", 
               "quality_notes": "Full-image grayscale PSNR/SSIM vs HR official validation; HR never used as inference input."}
    return dataset_yaml, quality



def reuse_prepared(conf: dict, out: Path, limit: int) -> tuple[Path, dict]:
    """Resume a failed run without repeating 1,144 expensive SR reconstructions.

    Require the expected LR validation names, enhanced files, labels and saved
    per-image quality measurements to agree before reusing any existing data.
    """
    source_lr = Path(conf["source_lr"])
    expected = sorted(p.name for p in (source_lr / "images" / "val").glob("*.png"))
    if not expected:
        raise ValueError("No LR validation PNG files")
    if limit:
        expected = expected[:limit]

    dest = Path("data/interim/fsrcnn_eval") / out.name
    yaml_path = dest / "data.yaml"
    image_dir = dest / "images" / "val"
    labels_dir = dest / "labels" / "val"
    quality_path = out / "per_image_quality.csv"
    for required in (yaml_path, image_dir, labels_dir, quality_path):
        if not required.exists():
            raise FileNotFoundError(f"Cannot resume; missing {required}")
    actual_images = sorted(p.name for p in image_dir.glob("*.png"))
    if actual_images != expected:
        raise ValueError(f"Cannot resume: expected {len(expected)} enhanced images with matching names, "
                         f"found {len(actual_images)} or differing filenames")
    expected_labels = {Path(name).stem + ".txt" for name in expected}
    actual_labels = {p.name for p in labels_dir.glob("*.txt")}
    if actual_labels != expected_labels:
        raise ValueError("Cannot resume: enhanced annotation filenames do not match source images")
    if any(not (source_lr / "labels" / "val" / label).is_file() for label in expected_labels):
        raise FileNotFoundError("Cannot resume: source annotation missing")

    data_conf = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if Path(data_conf.get("path", "")).resolve() != dest.resolve():
        raise ValueError("Cannot resume: prepared dataset YAML points to another dataset")
    with quality_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(expected) or sorted(row["image"] for row in rows) != expected:
        raise ValueError("Cannot resume: quality CSV does not match prepared validation images")

    def average(key: str, digits: int = 5) -> float:
        vals = [float(row[key]) for row in rows]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"Cannot resume: invalid {key} in quality CSV")
        return round(statistics.mean(vals), digits)

    quality = {
        "images": len(expected),
        "files_sha256": hashlib.sha256("\n".join(expected).encode()).hexdigest(),
        "fsrcnn_mean_psnr_db": average("fsrcnn_psnr_db"),
        "bicubic_mean_psnr_db": average("bicubic_psnr_db"),
        "fsrcnn_mean_ssim": average("fsrcnn_ssim"),
        "bicubic_mean_ssim": average("bicubic_ssim"),
        "lr_vs_pil_train_degradation_mae_u8": average("lr_synthetic_mismatch_mae", 6),
        "fsrcnn_inference_ms_mean": average("fsrcnn_sr_ms"),
        "fsrcnn_inference_ms_median": round(statistics.median(float(row["fsrcnn_sr_ms"]) for row in rows), 5),
        "timing_notes": "FSRCNN forward + GPU-to-CPU output and sync; excludes input transfer, I/O and detector",
        "quality_notes": "Full-image PSNR/SSIM vs HR validation; recovered from existing per_image_quality.csv",
    }
    print(f"REUSING {len(expected)} previously enhanced images and saved quality measurements", flush=True)
    return yaml_path, quality


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/fsrcnn_eval.yaml"))
    ap.add_argument("--limit", type=int, default=30, help="0 = all 1144 validation images")
    ap.add_argument("--out", type=Path, default=Path("reports/sr_baseline/fsrcnn_smoke_30"))
    ap.add_argument("--baseline-report", type=Path, default=Path("reports/sr_baseline/bicubic_smoke_30/summary.json"))
    ap.add_argument("--resume-prepared", action="store_true",
                    help="Reuse enhanced images and per_image_quality.csv from a previously interrupted run")
    ap.add_argument("--val-batch", type=int, default=2,
                    help="YOLO validation batch size (default 2 for safer GPU use)")
    ap.add_argument("--diag-batch", type=int, default=1,
                    help="Diagnostic prediction batch size (default 1 to prevent CUDA OOM)")
    args = ap.parse_args()
    if args.limit < 0 or args.val_batch < 1 or args.diag_batch < 1:
        ap.error("--limit must be >= 0, and both batch sizes must be >= 1")
    if args.resume_prepared:
        if not args.out.is_dir():
            ap.error("--resume-prepared requires the existing interrupted --out directory")
        if (args.out / "summary.json").exists():
            ap.error("Evaluation is already complete; refusing to overwrite summary.json")
    elif args.out.exists():
        ap.error(f"Output already exists: {args.out}; choose new --out or use --resume-prepared")
    conf = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint_sr = Path(conf["sr_checkpoint"])
    checkpoint_det = Path(conf["detection_checkpoint"])
    for path in (checkpoint_sr, checkpoint_det, args.baseline_report):
        if not path.is_file():
            ap.error(f"Missing checkpoint/report: {path}")
    source = Path(conf["source_lr"])
    images = sorted((source / "images" / "val").glob("*.png"))
    if args.limit:
        images = images[:args.limit]
    original = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    detector_hash = sha256_file(checkpoint_det)
    validate_reference(original, len(images), detector_hash, source)
    if not torch.cuda.is_available():
        ap.error("CUDA unavailable for FSRCNN experiment")
    if args.resume_prepared:
        dataset_yaml, quality = reuse_prepared(conf, args.out, args.limit)
        ckpt = torch.load(checkpoint_sr, map_location="cpu", weights_only=True)
    else:
        device = torch.device("cuda:0")
        net, ckpt = load_fsrcnn(checkpoint_sr, device)
        args.out.mkdir(parents=True, exist_ok=False)
        dataset_yaml, quality = prepare(conf, args.out, args.limit, net, device)
        del net
        gc.collect()
        torch.cuda.empty_cache()
    # Reuse Step 4A Ultralytics validation settings and EXACT diagnostic matching code.
    from ultralytics import YOLO
    detector = YOLO(str(checkpoint_det))
    print("EVALUATING fsrcnn_640 (fixed YOLO11n, imgsz=640)", flush=True)
    metrics = detector.val(data=str(dataset_yaml.resolve()), imgsz=640, batch=args.val_batch,
                           device="0", workers=2, conf=0.001, iou=0.7, max_det=300,
                           plots=False, save=False, verbose=False,
                           project=str(args.out.resolve()),
                           name="val_fsrcnn_640_recovery" if args.resume_prepared else "val_fsrcnn_640",
                           exist_ok=False)
    # Extract plain Python scalars, then destroy the validator, detector and
    # its GPU allocations before the second prediction-intensive stage.
    fsrnn_entry = {"data_style": "fsrcnn", "imgsz": 640,
                   "precision_macro": round(float(metrics.box.mp), 6),
                   "recall_macro": round(float(metrics.box.mr), 6),
                   "map50": round(float(metrics.box.map50), 6),
                   "map50_95": round(float(metrics.box.map), 6),
                   "validator_speed_ms_per_image": {k: round(float(v), 4) for k, v in metrics.speed.items()}}
    del metrics, detector
    gc.collect()
    torch.cuda.empty_cache()
    print(f"DIAGNOSTIC: running class-aware small-object recall at batch={args.diag_batch}", flush=True)
    diagnostic_detector = YOLO(str(checkpoint_det))
    diagnostic = size_recall(diagnostic_detector, dataset_yaml.parent, 640, args.diag_batch, "0")
    fsrnn_entry["diagnostic_size_recall"] = diagnostic
    del diagnostic_detector
    gc.collect()
    torch.cuda.empty_cache()
    cases = {k: original["comparison_cases"][k] for k in ("lr_320", "lr_640_resize_control", "bicubic_640")}
    cases["fsrcnn_640"] = fsrnn_entry
    summary = {"images": len(images), "source": str(source), "detector_checkpoint": str(checkpoint_det),
               "detector_checkpoint_sha256": detector_hash, "fsrcnn_checkpoint": str(checkpoint_sr),
               "fsrcnn_checkpoint_sha256": sha256_file(checkpoint_sr),
               "fsrcnn_checkpoint_epoch": ckpt.get("epoch"), "bicubic_reference": str(args.baseline_report),
               "quality": quality, "comparison_cases": cases,
               "notes": ["Same validation images/annotations and detector as Step 4A", "All size recalls use original 320x256 LR GT sizes", 
                         "Validator timings exclude SR cost and image I/O; SR cost separately reported", 
                         "Quality metrics use full official HR validation images; not internal training patch PSNR", 
                         "Caution: selection of settings on validation cannot substitute held-out independent test"]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rows = []
    for name, c in cases.items():
        rows.append({"case": name, "map50": c["map50"], "map50_95": c["map50_95"],
                     "small_recall_conf_025": c["diagnostic_size_recall"]["0.25"]["groups"]["small"]["recall"],
                     "false_positives_conf_025": c["diagnostic_size_recall"]["0.25"]["false_positives"],
                     "detector_inference_ms": c["validator_speed_ms_per_image"].get("inference")})
    with (args.out / "comparison.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"DONE — {args.out / 'summary.json'}; {args.out / 'comparison.csv'}", flush=True)
    print(f"FSRCNN: PSNR {quality['fsrcnn_mean_psnr_db']:.2f}dB, SSIM {quality['fsrcnn_mean_ssim']:.4f}, "
          f"mAP50 {fsrnn_entry['map50']:.3f}, small recall {rows[-1]['small_recall_conf_025']:.3f}", flush=True)


if __name__ == "__main__":
    main()
