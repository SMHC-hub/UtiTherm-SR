"""Offline tests for controlled Step 4 evaluation. No GPU/dataset required."""
import importlib.util
from pathlib import Path

from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_bicubic_baseline.py"
spec = importlib.util.spec_from_file_location("evaluate_bicubic_baseline", MODULE)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_area_categories_are_fixed_in_lr_pixels():
    assert m.object_size((0, 0, 3, 7)) == "small"
    assert m.object_size((0, 0, 32, 32)) == "medium"
    assert m.object_size((0, 0, 96, 96)) == "large"


def test_iou_extremes():
    a = (2, 4, 20, 40)
    assert m.box_iou(a, a) == 1.0
    assert m.box_iou(a, (21, 0, 32, 20)) == 0.0


def test_matching_counts_false_positives():
    gts = [{"cls": 2, "box": (0, 0, 10, 10)}]
    preds = [
        {"cls": 2, "box": (0, 0, 10, 10), "conf": 0.9},
        {"cls": 2, "box": (0, 0, 10, 10), "conf": 0.7},
        {"cls": 1, "box": (0, 0, 10, 10), "conf": 0.8},
    ]
    matched, fp = m.match_predictions(gts, preds, 0.25)
    assert matched == {0}
    assert fp == 2


def test_label_normalization_and_small_area(tmp_path):
    p = tmp_path / "label.txt"
    p.write_text("1 0.500000 0.500000 0.031250 0.031250\n")
    gt = m.ground_truth(p)[0]
    assert gt["cls"] == 1
    assert gt["size"] == "small"
    assert gt["box"] == (155.0, 124.0, 165.0, 132.0)


def test_preparation_keeps_matching_labels_and_sizes(tmp_path):
    src = tmp_path / "source"
    imgd, lbld = src / "images" / "val", src / "labels" / "val"
    imgd.mkdir(parents=True)
    lbld.mkdir(parents=True)
    Image.new("L", (320, 256), 99).save(imgd / "first.png")
    (lbld / "first.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (src / "data.yaml").write_text("names: {0: person}\n")
    paths, manifest = m.prepare_datasets(src, tmp_path / "out", 1)
    assert len(paths) == 3
    assert manifest["image_count"] == 1
    for style, size in (("lr", (320, 256)), ("bicubic", (640, 512)), ("nearest", (640, 512))):
        with Image.open(tmp_path / "out" / style / "images" / "val" / "first.png") as img:
            assert img.size == size
        assert (tmp_path / "out" / style / "labels" / "val" / "first.txt").read_text() == (lbld / "first.txt").read_text()
