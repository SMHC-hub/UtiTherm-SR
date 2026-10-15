"""Read-only audit of *actual* FLIR downloads; never assume a specific layout."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _modality(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if any("thermal" in p or p in {"lwir", "infrared", "ir"} for p in parts):
        return "thermal_path"
    if any(p in {"rgb", "visible", "color"} for p in parts):
        return "visible_path"
    return "unclassified_path"


def inspect_flir(root: Path, sample_size: int = 20) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"FLIR dataset directory not found: {root}")
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    images = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
    json_files = [p for p in paths if p.suffix.lower() == ".json"]
    groups = Counter(_modality(p.relative_to(root)) for p in images)
    suffixes = Counter(p.suffix.lower() for p in images)
    top_level = Counter(p.relative_to(root).parts[0] for p in images)
    examples = []
    invalid_samples = []
    if images:
        stride = max(1, len(images) // sample_size)
        sampled = images[::stride][:sample_size]
        for p in sampled:
            try:
                with Image.open(p) as im:
                    examples.append({
                        "path": p.relative_to(root).as_posix(),
                        "format": im.format,
                        "mode": im.mode,
                        "width": im.width,
                        "height": im.height,
                        "modality_by_path": _modality(p.relative_to(root)),
                    })
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                invalid_samples.append({"path": p.relative_to(root).as_posix(), "error": str(exc)})

    coco_annotations = []
    other_json = []
    json_errors = []
    for p in json_files:
        relative = p.relative_to(root).as_posix()
        if p.stat().st_size > 250_000_000:
            json_errors.append({"path": relative, "reason": "Too large for safe metadata-only audit (>250MB)"})
            continue
        try:
            with p.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and all(isinstance(data.get(k), list) for k in ("images", "annotations", "categories")):
                category_names = [{"id": c.get("id"), "name": c.get("name")} for c in data["categories"] if isinstance(c, dict)]
                example_filenames = [item.get("file_name") for item in data["images"][:5] if isinstance(item, dict)]
                invalid_bboxes = sum(
                    1 for item in data["annotations"]
                    if not isinstance(item, dict) or not isinstance(item.get("bbox"), list)
                    or len(item["bbox"]) != 4
                    or any(not isinstance(v, (float, int)) for v in item["bbox"])
                    or item["bbox"][2] <= 0 or item["bbox"][3] <= 0
                )
                coco_annotations.append({
                    "path": relative,
                    "images": len(data["images"]),
                    "annotations": len(data["annotations"]),
                    "invalid_bounding_boxes": invalid_bboxes,
                    "categories": category_names,
                    "sample_image_file_names": example_filenames,
                })
            else:
                other_json.append(relative)
        except (ValueError, UnicodeError, OSError, TypeError) as exc:
            json_errors.append({"path": relative, "error": str(exc)})

    warnings = []
    if not images:
        warnings.append("No supported images found. Download/extract FLIR ADAS manually first.")
    if not coco_annotations:
        warnings.append("No MS COCO style annotation JSON found; locate the correct annotation release.")
    if groups["unclassified_path"]:
        warnings.append("Some image paths are not explicitly marked thermal or RGB; confirm their modality manually.")
    return {
        "root": str(root),
        "image_count": len(images),
        "image_extensions": dict(suffixes),
        "path_modality_counts": dict(groups),
        "top_level_image_counts": dict(top_level),
        "image_samples": examples,
        "unreadable_samples": invalid_samples,
        "coco_annotation_files": coco_annotations,
        "other_json_files": other_json[:30],
        "json_errors": json_errors,
        "warnings": warnings,
        "audit_scope": "Read-only filesystem and metadata audit; image/annotation mapping and split leakage need separate validation.",
    }
