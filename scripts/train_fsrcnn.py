"""Train FSRCNN x2 on FLIR thermal TRAIN images only.

From project root in Docker:
    python scripts/train_fsrcnn.py --config configs/fsrcnn_x2.yaml --smoke
    python scripts/train_fsrcnn.py --config configs/fsrcnn_x2.yaml

Internal validation is split by video sequence from official TRAIN, never
by loading any file from official FLIR validation or detection evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utitherm.sr.fsrcnn import FSRCNN

SUFFIXES = {".jpg", ".jpeg", ".png"}


def sequence_id(path: Path) -> str:
    """Group video-derived frames, preventing near-duplicate train/val leakage."""
    stem = path.stem
    if "-frame-" in stem:
        return stem.split("-frame-", 1)[0]
    # Non-video sources are grouped individually; explicitly log this caveat.
    return stem


def group_split(paths: list[Path], seed: int, val_fraction: float) -> tuple[list[Path], list[Path]]:
    """Deterministic, mutually exclusive, group-aware split."""
    if not (0 < val_fraction < 0.5):
        raise ValueError("val_fraction must be > 0 and < 0.5")
    groups: dict[str, list[Path]] = {}
    for path in sorted(paths):
        groups.setdefault(sequence_id(path), []).append(path)
    if len(groups) < 2:
        raise ValueError("Need at least two independent sequences for train/validation")
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    val_keys = set()
    n_val = 0
    target = max(1, round(len(paths) * val_fraction))
    for key in ordered[:-1]:
        if n_val >= target:
            break
        val_keys.add(key)
        n_val += len(groups[key])
    train = sorted(p for k in ordered if k not in val_keys for p in groups[k])
    valid = sorted(p for k in ordered if k in val_keys for p in groups[k])
    assert train and valid
    assert not ({sequence_id(p) for p in train} & {sequence_id(p) for p in valid})
    return train, valid


class ThermalPairs(Dataset):
    """Read one HR thermal frame at a time, generate PIL bicubic x2 LR, crop patches.

    No cache of 10k full-resolution images; respects a 12-GiB Docker RAM limit.
    Val patch location is deterministic, with no augmentations.
    """
    def __init__(self, files: list[Path], lr_patch: int, training: bool, seed: int):
        if not files or lr_patch < 8:
            raise ValueError("Need images and lr_patch>=8")
        self.files = files
        self.lr_patch = lr_patch
        self.training = training
        self.seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.files[index]
        with Image.open(path) as raw:
            if raw.mode not in ("L", "RGB"):
                raise ValueError(f"Expected 8-bit image, got {raw.mode} at {path}")
            hr = raw.convert("L")
        if hr.width % 2 or hr.height % 2:
            raise ValueError(f"Image dimensions must be divisible by 2: {path}")
        lr = hr.resize((hr.width // 2, hr.height // 2), Image.Resampling.BICUBIC)
        if lr.width < self.lr_patch or lr.height < self.lr_patch:
            raise ValueError(f"LR image too small for patch: {path}")
        if self.training:
            x = random.randint(0, lr.width - self.lr_patch)
            y = random.randint(0, lr.height - self.lr_patch)
            flip = random.random() < 0.5
        else:
            # Stable across worker count, Python processes and training epochs.
            digest = hashlib.sha256(f"{path.name}|{self.seed}".encode()).digest()
            x = int.from_bytes(digest[:8], "big") % (lr.width - self.lr_patch + 1)
            y = int.from_bytes(digest[8:16], "big") % (lr.height - self.lr_patch + 1)
            flip = False
        low = lr.crop((x, y, x + self.lr_patch, y + self.lr_patch))
        high = hr.crop((2*x, 2*y, 2*(x+self.lr_patch), 2*(y+self.lr_patch)))
        if flip:
            low = low.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        a = torch.from_numpy(np.array(low, dtype=np.float32, copy=True)).unsqueeze(0) / 255.0
        b = torch.from_numpy(np.array(high, dtype=np.float32, copy=True)).unsqueeze(0) / 255.0
        return a, b


def mse_psnr(mse: float) -> float:
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def sha256_names(paths: list[Path]) -> str:
    return hashlib.sha256("\n".join(p.name for p in sorted(paths)).encode()).hexdigest()


def atomic_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             max_batches: int = 0) -> float:
    model.eval()
    sqerr, pixels = 0.0, 0
    with torch.inference_mode():
        for j, (lr, hr) in enumerate(loader):
            if max_batches and j >= max_batches:
                break
            pred = model(lr.to(device, non_blocking=True))
            target = hr.to(device, non_blocking=True)
            sqerr += torch.sum((pred - target).square()).item()
            pixels += target.numel()
    if pixels == 0:
        raise ValueError("Validation produced no image patches")
    return sqerr / pixels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fsrcnn_x2.yaml"))
    parser.add_argument("--smoke", action="store_true", help="1 epoch, max 4 train/2 val batches, separate output")
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"Missing config: {args.config}")
    conf = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    required = {"train_hr", "output", "canonical_checkpoint", "epochs", "batch", "lr_patch",
                "learning_rate", "seed", "val_fraction", "workers", "patience", "d", "s", "m"}
    if not isinstance(conf, dict) or required - conf.keys():
        parser.error(f"Config keys missing: {sorted(required - (conf.keys() if isinstance(conf,dict) else set()))}")
    if (conf["epochs"] < 1 or conf["batch"] < 1 or conf["workers"] < 0
            or conf["patience"] < 1 or conf["learning_rate"] <= 0):
        parser.error("Invalid training configuration values")
    source = Path(conf["train_hr"])
    if not source.is_dir():
        parser.error(f"Missing HR training folder: {source}")
    # This path assertion avoids accidentally training on official detection validation.
    if source.name != "train" or source.parent.name != "images":
        parser.error("train_hr must refer to the images/train directory")
    paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    if len(paths) < 2:
        parser.error("Not enough training images")

    seed = int(conf["seed"])
    train_paths, val_paths = group_split(paths, seed=seed, val_fraction=float(conf["val_fraction"]))
    actual_train = train_paths[:64] if args.smoke else train_paths
    actual_val = val_paths[:16] if args.smoke else val_paths
    root_out = Path(conf["output"])
    out = root_out.parent / f"{root_out.name}_smoke" if args.smoke else root_out
    if out.exists():
        parser.error(f"Experiment output already exists: {out}; change 'output' in YAML")
    canonical = Path(conf["canonical_checkpoint"])
    if not args.smoke and canonical.exists():
        parser.error(f"Canonical checkpoint already exists: {canonical}; choose a new run name")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU unavailable. FSRCNN training requires GPU in this workflow")
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0")
    train_ds = ThermalPairs(actual_train, int(conf["lr_patch"]), True, seed)
    val_ds = ThermalPairs(actual_val, int(conf["lr_patch"]), False, seed)
    train_loader = DataLoader(train_ds, batch_size=int(conf["batch"]), shuffle=True,
                              num_workers=int(conf["workers"]), pin_memory=True,
                              persistent_workers=bool(conf["workers"]), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(conf["batch"]), shuffle=False,
                            num_workers=int(conf["workers"]), pin_memory=True,
                            persistent_workers=bool(conf["workers"]), drop_last=False)
    net_args = {k: int(conf[k]) for k in ("d", "s", "m")}
    model = FSRCNN(scale=2, **net_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(conf["learning_rate"]))
    epochs = 1 if args.smoke else int(conf["epochs"])
    # No image folders or original FLIR data are changed.
    out.mkdir(parents=True)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "FSRCNN x2 synthetic FLIR thermal super-resolution training",
        "config": conf,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "training_images": len(actual_train), "internal_validation_images": len(actual_val),
        "full_training_split_images": len(train_paths), "full_internal_val_split_images": len(val_paths),
        "train_filename_sha256": sha256_names(actual_train),
        "internal_val_filename_sha256": sha256_names(actual_val),
        "train_groups": len({sequence_id(p) for p in actual_train}),
        "val_groups": len({sequence_id(p) for p in actual_val}),
        "data_source": str(source), "split_policy": "sequence-group-exclusive within official FLIR training",
        "degradation": "PIL BICUBIC 640x512 -> 320x256; synthetic x2",
        "evaluation": "Internal training-split held-out patches only; NOT FLIR detection validation",
        "validation_selection_caveat": "Validation patches fixed, train patches stochastic; results are patch-level, not full-image PSNR",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "parameters": sum(p.numel() for p in model.parameters()),
        "smoke": args.smoke,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"GPU: {manifest['gpu']} | parameters: {manifest['parameters']:,}", flush=True)
    print(f"Train/held-out images: {len(actual_train)}/{len(actual_val)}", flush=True)
    print(f"Report directory: {out}", flush=True)
    best_mse, stale = float("inf"), 0
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        total_loss, train_pixels = 0.0, 0
        for j, (lr, hr) in enumerate(train_loader):
            if args.smoke and j >= 4:
                break
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            pred = model(lr)
            loss = torch.mean((pred - hr).square())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * hr.numel()
            train_pixels += hr.numel()
        train_mse = total_loss / max(1, train_pixels)
        val_mse = evaluate(model, val_loader, device, max_batches=2 if args.smoke else 0)
        improved = val_mse < best_mse - 1e-9
        if improved:
            best_mse, stale = val_mse, 0
        else:
            stale += 1
        payload = {"state_dict": model.state_dict(), "model_kwargs": {"scale": 2, **net_args},
                   "epoch": epoch, "val_mse": val_mse, "val_psnr_db": mse_psnr(val_mse),
                   "train_split_sha256": manifest["train_filename_sha256"],
                   "config_sha256": manifest["config_sha256"]}
        atomic_checkpoint(out / "last.pt", payload)
        if improved:
            atomic_checkpoint(out / "best.pt", payload)
        row = {"epoch": epoch, "train_mse": round(train_mse, 8), "val_mse": round(val_mse, 8),
               "val_psnr_db": round(mse_psnr(val_mse), 4),
               "epoch_seconds": round(time.perf_counter() - start, 2)}
        logs.append(row)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch:02}/{epochs} train MSE={train_mse:.6f} "
              f"held-out PSNR={row['val_psnr_db']:.2f}dB "
              f"best={mse_psnr(best_mse):.2f}dB", flush=True)
        if stale >= int(conf["patience"]):
            print(f"Early stopping: no improvement for {stale} epochs", flush=True)
            break
    manifest["best_internal_val_psnr_db"] = round(mse_psnr(best_mse), 4)
    manifest["epochs_completed"] = len(logs)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.smoke:
        import shutil
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "best.pt", canonical)
        print(f"Canonical model checkpoint: {canonical}", flush=True)
    print(f"DONE — {out / 'best.pt'} | internal-val best PSNR {mse_psnr(best_mse):.2f} dB", flush=True)


if __name__ == "__main__":
    main()
