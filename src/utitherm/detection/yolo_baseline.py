"""Pretrained YOLO11n baseline wrapper. Fine-tuning awaits validated class mapping."""


def load_yolo11n(weights: str = "yolo11n.pt"):
    from ultralytics import YOLO
    return YOLO(weights)


def train_yolo11n(data_yaml: str, epochs: int = 30, image_size: int = 640,
                  batch: int = 8, device: int = 0):
    """Train only after verifying a YOLO dataset's split and category mapping."""
    from pathlib import Path
    if not Path(data_yaml).is_file():
        raise FileNotFoundError("Validated YOLO dataset YAML missing; run FLIR audit and convert labels first")
    model = load_yolo11n()
    return model.train(data=data_yaml, epochs=epochs, imgsz=image_size, batch=batch,
                       device=device, amp=True, workers=2, seed=42)
