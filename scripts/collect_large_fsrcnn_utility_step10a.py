"""Step 10A: Large FSRCNN x2 single-ROI counterfactual utility labels.

Reuse Step 7B2A frozen detector TRAIN-domain sequence manifest and original
scoring code. Interventions reproduce Step 9G Large FSRCNN ROI processing,
but enhance exactly ONE tile at a time, to create training targets.

Commands (from the UtiTherm-SR project root inside research Docker):
  python scripts/collect_large_fsrcnn_utility_step10a.py --limit 4 --out reports/utility/large_sr_labels_smoke_4_step10a_v1
  python scripts/collect_large_fsrcnn_utility_step10a.py --limit 0 --out reports/utility/large_sr_labels_300_step10a_v1
  python scripts/collect_large_fsrcnn_utility_step10a.py --limit 0 --resume --out reports/utility/large_sr_labels_300_step10a_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_utility_v2_manifest import verify_manifest, sha256, sequence_id  # noqa: E402
from probe_roi_utility import detection_counts, prediction_records, roi_grid, utility  # noqa: E402
from evaluate_bicubic_baseline import ground_truth  # noqa: E402
from benchmark_selective_fsrcnn import load_sr_checkpoint  # noqa: E402
from utitherm.sr.roi_fsrcnn_batched import render_batched_fsrcnn  # noqa: E402

SCHEMA = "STEP_10A_LARGE_FSRCNN_H10_SINGLE_ROI_LABELS_V1"
COLUMNS = (
    "image", "sequence", "tile_index", "tile_x0_hr", "tile_y0_hr",
    "tile_x1_hr", "tile_y1_hr", "enhanced_fraction", "baseline_small_tp",
    "baseline_fp", "candidate_small_tp", "candidate_fp", "delta_small_tp",
    "delta_all_tp", "delta_fp", "utility_score", "positive_label",
)


def validate_config(exp: dict, base: dict, frozen: dict) -> None:
    """Pin training/evaluation choices before spending GPU time."""
    if not isinstance(exp, dict) or not isinstance(base, dict):
        raise ValueError("YAML must contain mapping objects")
    if (int(exp.get("halo", -1)) != 10 or int(exp.get("budget_tiles", -1)) != 1
            or exp.get("intervention") != "large_fsrcnn_x2_roi_h10_linear_background_feather8"):
        raise ValueError("Unsupported Step 10A intervention protocol")
    if (int(base.get("grid_rows", -1)), int(base.get("grid_cols", -1))) != (4, 4):
        raise ValueError("Step 10A requires a frozen 4x4 tile grid")
    settings = {"imgsz": 640, "confidence": 0.25, "nms_iou": 0.7,
                "false_positive_penalty": 0.25, "feather_hr_px": 8}
    for k, value in settings.items():
        if abs(float(base.get(k, -999)) - value) > 1e-9:
            raise ValueError(f"Frozen detector/intervention setting changed: {k}")
    if str(base.get("device")) != "0":
        raise ValueError("Expected CUDA device 0")
    if (int(frozen.get("num_images", -1)) != int(exp.get("expected_images", -2))
            or int(frozen.get("unique_sequences", -1)) != int(exp.get("expected_sequences", -2))):
        raise ValueError("Frozen training sample differs from Step 10A config")


def validate_sr_checkpoint(checkpoint: Path) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "state_dict" not in state:
        raise ValueError("SR checkpoint has no state_dict")
    kw = state.get("model_kwargs")
    if not isinstance(kw, dict) or any(int(kw.get(k, -1)) != v
                                           for k, v in {"scale": 2, "d": 168, "s": 36, "m": 10}.items()):
        raise ValueError("Not the trained Step 9D3 Large FSRCNN architecture")
    return {"model_kwargs": kw, "epoch": state.get("epoch")}


def provenance(config_file: Path, base_file: Path, frozen_file: Path,
               detector_file: Path, sr_file: Path, image_names: list[str],
               exp: dict) -> dict:
    return {
        "protocol": SCHEMA,
        "selected_images": image_names,
        "configuration_sha256": sha256(config_file),
        "base_configuration_sha256": sha256(base_file),
        "frozen_manifest_sha256": sha256(frozen_file),
        "detector_checkpoint_sha256": sha256(detector_file),
        "sr_checkpoint_sha256": sha256(sr_file),
        "halo": int(exp["halo"]),
        "intervention": exp["intervention"],
        "num_tiles": 16,
    }


def shard_is_valid(item: dict, expected_name: str) -> bool:
    """Refuse partial/corrupted or cross-image shards during resume."""
    try:
        if (item.get("image") != expected_name or
                item.get("sequence") != sequence_id(expected_name)):
            return False
        base = item["baseline"]
        if not all(k in base for k in ("small_tp", "fp", "all_tp", "small_gt")):
            return False
        rows = item["rows"]
        if len(rows) != 16 or {int(r["tile_index"]) for r in rows} != set(range(16)):
            return False
        for r in rows:
            if not set(COLUMNS).issubset(r):
                return False
            if r["image"] != expected_name or r["sequence"] != sequence_id(expected_name):
                return False
            if int(r["baseline_small_tp"]) != int(base["small_tp"]) or int(r["baseline_fp"]) != int(base["fp"]):
                return False
            dtp = int(r["candidate_small_tp"]) - int(r["baseline_small_tp"])
            dfp = int(r["candidate_fp"]) - int(r["baseline_fp"])
            score = dtp - .25 * dfp
            if (int(r["delta_small_tp"]) != dtp or int(r["delta_fp"]) != dfp or
                    abs(float(r["utility_score"]) - score) > 1e-6 or
                    int(r["positive_label"]) != int(score > 0)):
                return False
        return True
    except (TypeError, KeyError, ValueError, OverflowError):
        return False


def make_rows(gray: np.ndarray, image_name: str, gts: list[dict],
              detector, sr, device, cfg: dict, tiles: list[tuple[int,int,int,int]],
              halo: int) -> tuple[dict, list[dict]]:
    """One single-tile Large-FSRCNN intervention per ROI; same Step 9G renderer."""
    if gray.shape != (256, 320) or gray.dtype != np.uint8 or len(tiles) != 16:
        raise ValueError("Require 320x256 uint8 LR image and 16 intervention tiles")
    linear = cv2.resize(gray, (640, 512), interpolation=cv2.INTER_LINEAR)
    threshold = float(cfg["confidence"])
    baseline = detection_counts(gts, prediction_records(detector, linear, cfg), threshold)
    rows = []
    for index, rect in enumerate(tiles):
        frame, _ = render_batched_fsrcnn(gray, tiles, [index], sr, device,
                                        feather=int(cfg["feather_hr_px"]), halo=halo)
        found = detection_counts(gts, prediction_records(detector, frame, cfg), threshold)
        change = utility(baseline, found, float(cfg["false_positive_penalty"]))
        rows.append({
            "image": image_name, "sequence": sequence_id(image_name),
            "tile_index": index, "tile_x0_hr": rect[0], "tile_y0_hr": rect[1],
            "tile_x1_hr": rect[2], "tile_y1_hr": rect[3], "enhanced_fraction": 0.0625,
            "baseline_small_tp": baseline["small_tp"], "baseline_fp": baseline["fp"],
            "candidate_small_tp": found["small_tp"], "candidate_fp": found["fp"],
            **change,
        })
    return baseline, rows


def save_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    """Atomically save completed CSV (safe for interrupted collection)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> None:
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    exp = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_path = Path(exp["base_collection_config"])
    frozen_path = Path(exp["frozen_manifest"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    validate_config(exp, base, frozen)
    all_paths = verify_manifest(frozen, base)  # verifies SHA256 image + label + V1 exclusions
    images = all_paths[:args.limit] if args.limit else all_paths
    if not images:
        raise ValueError("Manifest selection returned no images")
    detector_file = Path(base["detector_checkpoint"])
    sr_file = Path(exp["large_sr_checkpoint"])
    sr_info = validate_sr_checkpoint(sr_file)
    record = provenance(args.config, base_path, frozen_path, detector_file,
                        sr_file, [p.name for p in images], exp)
    manifest_out = args.out / "run_manifest.json"
    if args.resume:
        if (not manifest_out.is_file() or
                json.loads(manifest_out.read_text(encoding="utf-8")) != record):
            raise ValueError("Unsafe resume: run manifest/provenance changed")
    else:
        if args.out.exists():
            raise FileExistsError(f"Output exists: {args.out}; use --resume with matching run")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for official Step 10A collection")
    if not args.resume:
        (args.out / "shards").mkdir(parents=True)
        manifest_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Step 10A: {len(images)} detector-TRAIN images; "
          f"{len({sequence_id(p.name) for p in images})} sequences; "
          f"SR checkpoint epoch={sr_info['epoch']}; halo=10", flush=True)
    cv2.setNumThreads(1)
    device = torch.device("cuda:0")
    sr = load_sr_checkpoint(sr_file, device)
    from ultralytics import YOLO
    detector = YOLO(str(detector_file))
    tiles = roi_grid(640, 512, 4, 4)
    lab_dir = Path(base["train_lr"]).parent.parent / "labels" / "train"
    saved = []
    for pos, img in enumerate(images, 1):
        shard_file = args.out / "shards" / f"{img.stem}.json"
        if shard_file.exists():
            shard = json.loads(shard_file.read_text(encoding="utf-8"))
            if not shard_is_valid(shard, img.name):
                raise ValueError(f"Invalid shard; refusing to skip: {shard_file}")
        else:
            gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (256, 320) or gray.dtype != np.uint8:
                raise ValueError(f"Bad LR thermal image: {img}")
            gts = ground_truth(lab_dir / f"{img.stem}.txt")
            baseline, rows = make_rows(gray, img.name, gts, detector, sr, device,
                                       base, tiles, int(exp["halo"]))
            shard = {"image": img.name, "sequence": sequence_id(img.name),
                     "baseline": baseline, "rows": rows}
            if not shard_is_valid(shard, img.name):
                raise RuntimeError(f"New shard validation failed: {img.name}")
            tmp = shard_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(shard, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, shard_file)
        saved.append(shard)
        if pos == 1 or pos % 10 == 0 or pos == len(images):
            print(f"Collected {pos}/{len(images)} images, {pos*16} interventions", flush=True)
    rows = [r for shard in saved for r in shard["rows"]]
    baselines = [dict(image=shard["image"], sequence=shard["sequence"],
                      **shard["baseline"]) for shard in saved]
    save_csv(args.out / "roi_utility.csv", rows, COLUMNS)
    save_csv(args.out / "baseline_per_image.csv", baselines,
             ("image", "sequence", "small_tp", "all_tp", "fp", "small_gt"))
    summary = {
        "task": "Step 10A Large FSRCNN single-region utility measurements",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": SCHEMA, "images": len(images),
        "sequences": len({sequence_id(p.name) for p in images}),
        "roi_interventions": len(rows),
        "positive_utility": sum(float(r["utility_score"]) > 0 for r in rows),
        "small_gain": sum(int(r["delta_small_tp"]) > 0 for r in rows),
        "small_loss": sum(int(r["delta_small_tp"]) < 0 for r in rows),
        "sr_model_sha256": record["sr_checkpoint_sha256"],
        "detector_sha256": record["detector_checkpoint_sha256"],
        "single_roi_not_combined": True,
        "limitations": [
            "Targets collected from detector TRAIN images; detector is not out-of-fold.",
            "No FLIR validation images used for label collection.",
            "16 independent single-tile interventions, not observed four-region utility.",
            "Halo=10 trained Large FSRCNN, feather=8, linear background.",
            "No independent final test or demonstrated inference-speed benefit.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"DONE — {args.out/'roi_utility.csv'} | {len(rows)} interventions | "
          f"positive={summary['positive_utility']}; gains={summary['small_gain']}; "
          f"losses={summary['small_loss']}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   default=Path("configs/large_fsrcnn_utility_step10a.yaml"))
    p.add_argument("--limit", type=int, default=4, help="0=all frozen 300 images")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Resume only a matching existing run")
    run(p.parse_args())


if __name__ == "__main__":
    main()
