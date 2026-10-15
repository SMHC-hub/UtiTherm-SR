"""Small, traceable synthetic LR/HR thermal image pair generator.

This implements a *synthetic* degradation benchmark only. It does not claim to
simulate real thermal sensor noise or reconstruct genuine sensor resolution.
"""
from __future__ import annotations
from pathlib import Path
import json
from PIL import Image
from utitherm.sr.bicubic import downsample, bicubic_upscale

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def prepare_pairs(src: Path, out: Path, *, scale: int = 2, max_images: int = 20) -> dict:
    src, out = Path(src).resolve(), Path(out).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Input image folder does not exist: {src}")
    if out == src or src in out.parents:
        raise ValueError("Choose a different output folder from the input folder")
    if scale < 2 or max_images < 1:
        raise ValueError("scale must be >=2 and max_images >=1")
    paths = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
    if not paths:
        raise ValueError("No 8-bit JPEG or PNG images found. Check --src; TIFFs intentionally require a separate pipeline.")
    entries = []
    for p in paths[:max_images]:
        with Image.open(p) as im:
            if im.mode not in {"L", "RGB"}:
                raise ValueError(f"Unsupported image mode {im.mode} in {p}; do not silently convert high-bit-depth thermal data")
            hr = im.convert("L")
            lr = downsample(hr, scale=scale)
            up = bicubic_upscale(lr, scale=scale)
            if up.size != hr.size:
                raise AssertionError("Bicubic baseline must match HR dimensions")
            relative_name = p.relative_to(src).with_suffix(".png")
            for label, result in (("hr", hr), ("lr", lr), ("bicubic", up)):
                dest = out / label / relative_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                result.save(dest)
            entries.append({
                "source": p.relative_to(src).as_posix(),
                "hr": (Path("hr") / relative_name).as_posix(),
                "lr": (Path("lr") / relative_name).as_posix(),
                "bicubic": (Path("bicubic") / relative_name).as_posix(),
                "hr_size": list(hr.size),
                "lr_size": list(lr.size),
                "scale": scale,
            })
    manifest = {"input_folder": str(src), "output_folder": str(out),
                "scale": scale, "count": len(entries),
                "method": "PIL bicubic downsample then PIL bicubic upscale; synthetic benchmark",
                "entries": entries}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
