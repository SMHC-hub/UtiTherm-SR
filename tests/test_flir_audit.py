import json
from pathlib import Path
from PIL import Image
from utitherm.data.flir_audit import inspect_flir


def test_audit_detects_images_categories_and_bad_boxes(tmp_path: Path):
    thermal = tmp_path / "train" / "thermal_8_bit"
    thermal.mkdir(parents=True)
    Image.new("L", (640, 512)).save(thermal / "sample.jpg")
    annotation = {
        "images": [{"id": 2, "file_name": "thermal_8_bit/sample.jpg", "width": 640, "height": 512}],
        "categories": [{"id": 3, "name": "person"}],
        "annotations": [
            {"id": 7, "image_id": 2, "category_id": 3, "bbox": [4, 4, 40, 50]},
            {"id": 8, "image_id": 2, "category_id": 3, "bbox": [4, 4, 0, 50]},
        ],
    }
    (tmp_path / "train" / "labels.json").write_text(json.dumps(annotation), encoding="utf-8")
    report = inspect_flir(tmp_path)
    assert report["image_count"] == 1
    assert report["path_modality_counts"]["thermal_path"] == 1
    assert len(report["coco_annotation_files"]) == 1
    assert report["coco_annotation_files"][0]["invalid_bounding_boxes"] == 1
    assert report["coco_annotation_files"][0]["categories"][0]["name"] == "person"


def test_missing_dataset_is_an_error(tmp_path: Path):
    try:
        inspect_flir(tmp_path / "not_here")
        assert False, "Missing dataset should raise"
    except FileNotFoundError:
        pass
