from pathlib import Path
import json
import pytest
from PIL import Image
from utitherm.data.pairs import prepare_pairs
from utitherm.sr.bicubic import downsample, bicubic_upscale


def test_bicubic_has_matching_size():
    image = Image.new("L", (640, 512), color=125)
    lr = downsample(image, scale=2)
    assert lr.size == (320, 256)
    assert bicubic_upscale(lr, scale=2).size == image.size


def test_pair_manifest_and_files(tmp_path: Path):
    source = tmp_path / "raw" / "thermal_8_bit"
    source.mkdir(parents=True)
    Image.new("L", (80, 64), color=120).save(source / "demo.png")
    out = tmp_path / "out"
    report = prepare_pairs(source, out, max_images=1)
    assert report["count"] == 1
    assert (out / "manifest.json").exists()
    assert json.loads((out / "manifest.json").read_text())["entries"][0]["scale"] == 2
    with Image.open(out / "lr" / "demo.png") as lr:
        assert lr.size == (40, 32)
    with Image.open(out / "bicubic" / "demo.png") as bi:
        assert bi.size == (80, 64)


def test_rejects_16bit_source(tmp_path: Path):
    source = tmp_path / "thermal"
    source.mkdir()
    Image.new("I;16", (80, 64)).save(source / "raw.png")
    with pytest.raises(ValueError, match="Unsupported image mode"):
        prepare_pairs(source, tmp_path / "out")


def test_no_output_inside_source(tmp_path: Path):
    source = tmp_path / "thermal"
    source.mkdir()
    Image.new("L", (64, 64)).save(source / "sample.png")
    with pytest.raises(ValueError, match="different output folder"):
        prepare_pairs(source, source / "processed")
