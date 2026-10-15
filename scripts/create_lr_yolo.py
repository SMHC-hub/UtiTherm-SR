
from pathlib import Path
from PIL import Image
import shutil

SRC = Path("data/processed/yolo_flir")
DST = Path("data/processed/yolo_flir_lr_x2")

for split in ("train", "val"):
    src_images = SRC / "images" / split
    src_labels = SRC / "labels" / split
    dst_images = DST / "images" / split
    dst_labels = DST / "labels" / split

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    files = sorted(src_images.glob("*.jpg"))

    for i, file in enumerate(files, 1):
        label = src_labels / f"{file.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(label)

        output = dst_images / f"{file.stem}.png"

        if not output.exists():
            with Image.open(file) as img:
                img = img.convert("L")
                lr = img.resize(
                    (img.width // 2, img.height // 2),
                    Image.Resampling.BICUBIC
                )
                lr.save(output)

        shutil.copy2(label, dst_labels / label.name)

        if i % 1000 == 0:
            print(f"{split}: {i}/{len(files)}", flush=True)

    print(f"{split}: {len(files)} completed", flush=True)

# Keep the same classes and use the new dataset path
yaml = (SRC / "data.yaml").read_text()
yaml = yaml.replace(
    "/workspace/data/processed/yolo_flir",
    "/workspace/data/processed/yolo_flir_lr_x2"
)
(DST / "data.yaml").write_text(yaml)

print("Low-resolution YOLO dataset completed!")

