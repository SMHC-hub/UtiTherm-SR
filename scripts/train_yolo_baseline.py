"""Reproducible low-resolution FLIR YOLO11n detector baseline.

Run from the project root inside the GPU Docker container:
    python scripts/train_yolo_baseline.py --config configs/yolo_lr_baseline.yaml
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the UtiTherm-SR detector baseline")
    parser.add_argument(
        "--config", default="configs/yolo_lr_baseline.yaml", help="Experiment YAML path"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "model" not in config or "train" not in config:
        raise ValueError("Config requires 'model' and 'train' sections")

    train_args = dict(config["train"])
    data_file = Path(train_args["data"])
    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_file}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside Docker; do not start CPU training")

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Dataset: {data_file}", flush=True)
    print(f"Experiment: {train_args['name']}", flush=True)

    model = YOLO(config["model"])
    model.train(**train_args)

    # Ultralytics may increment the experiment directory if the name already exists.
    run_dir = Path(model.trainer.save_dir)
    metadata = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "experiment": config,
        "python_pytorch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved training outputs: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
