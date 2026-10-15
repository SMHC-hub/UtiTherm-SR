
import json
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path("data/raw/FLIR_ADAS_v2")
OUT = Path("data/processed/yolo_flir")

# COCO category ID -> YOLO class ID
CLASSES = {
    1: 0,   # person
    2: 1,   # bike
    3: 2,   # car
    4: 3,   # motor
    6: 4,   # bus
    8: 5,   # truck
    79: 6,  # other vehicle
    10: 7,  # light
    12: 8,  # sign
}

NAMES = [
    "person", "bike", "car", "motor", "bus",
    "truck", "other_vehicle", "light", "sign"
]

for split, folder in [
    ("train", "images_thermal_train"),
    ("val", "images_thermal_val"),
]:
    source = ROOT / folder
    coco = json.loads((source / "coco.json").read_text())

    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in CLASSES:
            grouped[ann["image_id"]].append(ann)

    used_names = set()
    label_count = 0

    for img in coco["images"]:
        filename = Path(img["file_name"]).name
        if filename in used_names:
            raise ValueError(f"Duplicate filename: {filename}")
        used_names.add(filename)

        src = source / img["file_name"]
        if not src.is_file():
            raise FileNotFoundError(src)

        shutil.copy2(src, img_dir / filename)

        W, H = img["width"], img["height"]
        lines = []

        for ann in grouped[img["id"]]:
            x, y, w, h = ann["bbox"]
            x1 = max(0, min(W, x))
            y1 = max(0, min(H, y))
            x2 = max(0, min(W, x + w))
            y2 = max(0, min(H, y + h))

            if x2 <= x1 or y2 <= y1:
                continue

            cls = CLASSES[ann["category_id"]]
            cx = (x1 + x2) / (2 * W)
            cy = (y1 + y2) / (2 * H)
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H

            lines.append(
                f"{cls} {cx:.6f} {cy:.6f} "
                f"{bw:.6f} {bh:.6f}"
            )
            label_count += 1

        (lbl_dir / f"{Path(filename).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    print(f"{split}: {len(coco['images'])} images, "
          f"{label_count} labels")

yaml = [
    "path: /workspace/data/processed/yolo_flir",
    "train: images/train",
    "val: images/val",
    "names:",
]
yaml += [f"  {i}: {name}" for i, name in enumerate(NAMES)]

(OUT / "data.yaml").write_text("\n".join(yaml) + "\n")

print("COCO to YOLO conversion completed.")
