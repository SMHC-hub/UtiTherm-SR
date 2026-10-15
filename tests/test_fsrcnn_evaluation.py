"""Step 4C unit tests (CPU-only)."""
import numpy as np
import pytest
import torch
from PIL import Image

from scripts.evaluate_fsrcnn_baseline import (
    infer_u8, index_hr_images, load_fsrcnn, psnr_u8, ssim_u8, validate_reference, reuse_prepared,
)
from utitherm.sr.fsrcnn import FSRCNN


def test_perfect_quality():
    img = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32).astype(np.uint8)
    assert psnr_u8(img, img) == float("inf")
    assert ssim_u8(img, img) == pytest.approx(1.0, abs=1e-10)


def test_noise_reduces_quality():
    img = np.full((32, 32), 100, dtype=np.uint8)
    noisy = img.copy()
    noisy[8:24, 8:24] = 130
    assert psnr_u8(img, noisy) < 30
    assert ssim_u8(img, noisy) < 1.0


def test_shape_dtype_guard():
    with pytest.raises(ValueError):
        psnr_u8(np.zeros((12,12), dtype=np.uint8), np.zeros((13,13), dtype=np.uint8))
    with pytest.raises(ValueError):
        ssim_u8(np.zeros((8,8), dtype=np.uint8), np.zeros((8,8), dtype=np.uint8))


def test_load_and_infer_checkpoint(tmp_path):
    net = FSRCNN(scale=2, d=8, s=4, m=1)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": net.state_dict(), "model_kwargs": {"scale":2, "d":8,"s":4,"m":1}, "epoch":1}, path)
    loaded, payload = load_fsrcnn(path, torch.device("cpu"))
    assert payload["epoch"] == 1
    arr, elapsed = infer_u8(loaded, Image.new("L", (320,256), 42), torch.device("cpu"))
    assert arr.shape == (512, 640)
    assert arr.dtype == np.uint8 and elapsed >= 0


def test_reference_guard(tmp_path):
    ref = {"images":30, "checkpoint_sha256":"a", "source":"data/processed/yolo_flir_lr_x2",
           "comparison_cases":{"lr_320":{},"lr_640_resize_control":{},"bicubic_640":{}}}
    from pathlib import Path
    validate_reference(ref, 30, "a", Path("data/processed/yolo_flir_lr_x2"))
    with pytest.raises(ValueError):
        validate_reference(ref, 1144, "a", Path("data/processed/yolo_flir_lr_x2"))


def test_hr_pairing_matches_stems_across_extensions(tmp_path):
    (tmp_path / "example.png").write_bytes(b"fake PNG")
    (tmp_path / "video-frame-000816.jpg").write_bytes(b"fake JPG")
    (tmp_path / "upper-case.JPEG").write_bytes(b"fake JPEG")
    index = index_hr_images(tmp_path)
    assert index["video-frame-000816"].name == "video-frame-000816.jpg"
    assert index["example"].suffix == ".png"
    assert index["upper-case"].suffix == ".JPEG"


def test_hr_pairing_rejects_ambiguous_stems(tmp_path):
    (tmp_path / "image.png").write_bytes(b"fake PNG")
    (tmp_path / "image.jpg").write_bytes(b"fake JPG")
    with pytest.raises(ValueError, match="Ambiguous HR image stem"):
        index_hr_images(tmp_path)


def _make_interrupted_run(tmp_path, monkeypatch):
    """A small fake interrupted run, with known filenames and cached quality."""
    from pathlib import Path
    import csv
    import yaml
    monkeypatch.chdir(tmp_path)
    source = Path("data/processed/yolo_flir_lr_x2")
    src_images = source / "images/val"
    src_labels = source / "labels/val"
    img_dir = Path("data/interim/fsrcnn_eval/failed_run/images/val")
    label_dir = img_dir.parent.parent / "labels/val"
    for p in (src_images, src_labels, img_dir, label_dir):
        p.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        (src_images / f"{stem}.png").write_bytes(b"some source pixels")
        (src_labels / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1")
        (img_dir / f"{stem}.png").write_bytes(b"enhanced pixels")
        (label_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1")
    dest = img_dir.parent.parent
    (dest / "data.yaml").write_text(yaml.safe_dump({"path": str(dest.resolve()), "val": "images/val"}))
    out = Path("reports/sr_baseline/failed_run")
    out.mkdir(parents=True)
    with (out / "per_image_quality.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["image", "fsrcnn_psnr_db", "bicubic_psnr_db",
                                         "fsrcnn_ssim", "bicubic_ssim", "fsrcnn_sr_ms",
                                         "lr_synthetic_mismatch_mae"])
        w.writeheader()
        w.writerows([
            {"image": "a.png", "fsrcnn_psnr_db": "30", "bicubic_psnr_db": "31",
             "fsrcnn_ssim": "0.8", "bicubic_ssim": "0.85", "fsrcnn_sr_ms": "4",
             "lr_synthetic_mismatch_mae": "0.123456"},
            {"image": "b.png", "fsrcnn_psnr_db": "32", "bicubic_psnr_db": "33",
             "fsrcnn_ssim": "0.9", "bicubic_ssim": "0.95", "fsrcnn_sr_ms": "5",
             "lr_synthetic_mismatch_mae": "0.654321"},
        ])
    return {"source_lr": str(source)}, out, dest


def test_resume_existing_prepared_data(tmp_path, monkeypatch):
    conf, out, dest = _make_interrupted_run(tmp_path, monkeypatch)
    yaml_path, quality = reuse_prepared(conf, out, 0)
    assert yaml_path == dest / "data.yaml"
    assert quality["images"] == 2
    assert quality["fsrcnn_mean_psnr_db"] == 31.0
    assert quality["bicubic_mean_ssim"] == pytest.approx(0.9)
    assert quality["fsrcnn_inference_ms_median"] == 4.5
    assert quality["lr_vs_pil_train_degradation_mae_u8"] == 0.388889


def test_resume_rejects_missing_enhanced_image(tmp_path, monkeypatch):
    conf, out, dest = _make_interrupted_run(tmp_path, monkeypatch)
    (dest / "images/val/a.png").unlink()
    with pytest.raises(ValueError, match="expected 2 enhanced images"):
        reuse_prepared(conf, out, 0)


def test_resume_rejects_incomplete_quality_csv(tmp_path, monkeypatch):
    conf, out, _ = _make_interrupted_run(tmp_path, monkeypatch)
    quality_path = out / "per_image_quality.csv"
    rows = quality_path.read_text().splitlines()
    quality_path.write_text("\n".join(rows[:2]) + "\n")
    with pytest.raises(ValueError, match="quality CSV does not match"):
        reuse_prepared(conf, out, 0)
