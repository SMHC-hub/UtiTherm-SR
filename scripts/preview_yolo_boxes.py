
from pathlib import Path
from PIL import Image, ImageDraw

root = Path("data/processed/yolo_flir")
labels = root / "labels/train"

label = next(
    p for p in sorted(labels.glob("*.txt"))
    if p.stat().st_size > 0
)

image_path = root / "images/train" / (label.stem + ".jpg")
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)
W, H = image.size

for line in label.read_text().splitlines():
    cls, xc, yc, w, h = map(float, line.split())
    x1, y1 = (xc-w/2)*W, (yc-h/2)*H
    x2, y2 = (xc+w/2)*W, (yc+h/2)*H

    draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
    draw.text((x1, y1), str(int(cls)), fill="yellow")

out = Path("reports/yolo_bbox_preview.jpg")
out.parent.mkdir(parents=True, exist_ok=True)
image.save(out)
print(f"Saved: {out}")
