"""Unit + synthetic end-to-end smoke tests for Step 10C."""
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
# Only in local package testing; in real project the V2/5B scripts are already present.
LEGACY = ROOT.parent / "source"
if (LEGACY / "scripts").exists():
    sys.path.insert(0, str(LEGACY / "scripts"))
    sys.path.insert(0, str(LEGACY / "src"))

from train_large_sr_predictor_step10c import (PROTOCOL, bootstrap_random_expectation,
                                               check_collection, split_large,
                                               train, sha256, verify_v2_paired_split)
from train_utility_predictor import load_measurements
from utitherm.utility.predictor import HEADS, UtilityPredictor


def fake_rows(n=12):
    rows = []
    for seq in range(n):
        name = f"video-seq{seq:02d}-frame-000001.png"
        for tile in range(16):
            rows.append({"image": name, "sequence": f"video-seq{seq:02d}",
                         "tile_index": tile, "score": 1.0 if tile == 0 else 0.0})
    return rows


def test_split_no_overlap():
    a, b, c, groups = split_large(fake_rows(33), 143, .15, .15)
    assert set(groups["train"]).isdisjoint(groups["val"])
    assert set(groups["train"]).isdisjoint(groups["test"])
    assert set(groups["val"]).isdisjoint(groups["test"])
    assert len(a) + len(b) + len(c) == 33 * 16
    assert len(groups["train"]) == 23
    assert len(groups["val"]) == len(groups["test"]) == 5


def test_split_deterministic():
    x = split_large(fake_rows(33), 143, .15, .15)[3]
    assert x == split_large(fake_rows(33), 143, .15, .15)[3]


def test_split_diff_seed():
    assert split_large(fake_rows(33), 143, .15, .15)[3] != split_large(fake_rows(33), 144, .15, .15)[3]


def test_split_minimum_sequences():
    with pytest.raises(ValueError, match="10 fresh sequences"):
        split_large(fake_rows(9), 143, .15, .15)


def test_bad_split_fraction():
    with pytest.raises(ValueError):
        split_large(fake_rows(33), 143, -.1, .15)


def synthetic_manifest(rows, ckpt, protocol=PROTOCOL, halo=10, intervention="large_fsrcnn_x2_roi_h10_linear_background_feather8"):
    return {"protocol": protocol, "intervention": intervention, "halo": halo,
            "num_tiles": 16, "selected_images": sorted({r["image"] for r in rows}),
            "sr_checkpoint_sha256": sha256(ckpt)}


def check_fixture(tmp_path):
    ckpt = tmp_path / "large.pt"
    ckpt.write_bytes(b"large fsrcnn checkpoint")
    csv_path = tmp_path / "roi_utility.csv"
    csv_path.write_text("fake", encoding="utf-8")
    rows = fake_rows(12)
    cfg = {"expected_images": 12, "expected_sequences": 12}
    return rows, ckpt, csv_path, cfg, synthetic_manifest(rows, ckpt)


def test_provenance_ok(tmp_path):
    r, ck, csvfile, cfg, manifest = check_fixture(tmp_path)
    assert check_collection(r, manifest, cfg, csvfile, ck)["regions"] == 192


def test_bad_label_protocol(tmp_path):
    r, ck, csvfile, cfg, m = check_fixture(tmp_path)
    m["protocol"] = "BICUBIC"
    with pytest.raises(ValueError, match="provenance"):
        check_collection(r, m, cfg, csvfile, ck)


def test_wrong_halo(tmp_path):
    r, ck, csvfile, cfg, m = check_fixture(tmp_path)
    m["halo"] = 16
    with pytest.raises(ValueError, match="halo"):
        check_collection(r, m, cfg, csvfile, ck)


def test_wrong_intervention(tmp_path):
    r, ck, csvfile, cfg, m = check_fixture(tmp_path)
    m["intervention"] = "bicubic"
    with pytest.raises(ValueError, match="interventions"):
        check_collection(r, m, cfg, csvfile, ck)


def test_changed_sr_weights(tmp_path):
    r, ck, csvfile, cfg, m = check_fixture(tmp_path)
    ck.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint hash"):
        check_collection(r, m, cfg, csvfile, ck)


def test_changed_images(tmp_path):
    r, ck, csvfile, cfg, m = check_fixture(tmp_path)
    m["selected_images"] = m["selected_images"][:-1]
    with pytest.raises(ValueError, match="do not match"):
        check_collection(r, m, cfg, csvfile, ck)


def test_reject_wrong_expected_image_count(tmp_path):
    r, ck, csvfile, cfg, m = check_fixture(tmp_path)
    cfg["expected_images"] = 11
    with pytest.raises(ValueError, match="complete"):
        check_collection(r, m, cfg, csvfile, ck)


def test_reject_renamed_csv(tmp_path):
    r, ck, csvfile, cfg, m = check_fixture(tmp_path)
    with pytest.raises(ValueError, match="roi_utility.csv"):
        check_collection(r, m, cfg, tmp_path / "mix.csv", ck)


def test_random_expectation_exact():
    rows = fake_rows(12)
    selections = {f"video-seq{i:02d}-frame-000001.png":
                  {"sequence": f"video-seq{i:02d}", "utility": 1.0}
                  for i in range(12)}
    report = bootstrap_random_expectation(selections, rows, 4, 143, draws=200)
    assert report["delta_predicted_minus_expected_random_per_image"] == .75
    assert report["num_sequences"] == 12
    assert report["sequence_bootstrap_exploratory_95ci"] == [.75, .75]


def test_bootstrap_determinism():
    rows = fake_rows(12)
    sel = {f"video-seq{i:02d}-frame-000001.png":
           {"sequence": f"video-seq{i:02d}", "utility": float(i % 4)}
           for i in range(12)}
    assert bootstrap_random_expectation(sel, rows, 4, 5, draws=200) == bootstrap_random_expectation(sel, rows, 4, 5, draws=200)


def test_bootstrap_bad_shapes():
    with pytest.raises(ValueError):
        bootstrap_random_expectation({}, fake_rows(), 4, 1)
    with pytest.raises(ValueError):
        bootstrap_random_expectation({"absent": {"sequence": "any", "utility": 1.}}, fake_rows(), 4, 1)


def test_torch_checkpoint_interface():
    model = UtilityPredictor()
    assert tuple(model(torch.zeros(2, 2, 64, 80), torch.zeros(2, 4)).shape) == (2, len(HEADS))


@pytest.fixture
def synthetic_project(tmp_path):
    # 12 sequence-separated frames, 16 correctly positioned ROI labels each.
    root = tmp_path / "data" / "processed" / "yolo_flir_lr_x2" / "images" / "train"
    root.mkdir(parents=True)
    label_dir = tmp_path / "utility"
    label_dir.mkdir()
    csvfile = label_dir / "roi_utility.csv"
    names = []
    fields = ["image", "sequence", "tile_index", "tile_x0_hr", "tile_y0_hr", "tile_x1_hr", "tile_y1_hr",
              "delta_small_tp", "delta_fp", "utility_score", "positive_label"]
    with csvfile.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for seq in range(12):
            name = f"video-seq{seq:02d}-frame-000001.png"
            names.append(name)
            img = np.full((256, 320), 64 + seq, dtype=np.uint8)
            cv2.imwrite(str(root / name), img)
            for tile in range(16):
                tp = int(tile == (seq % 16)) - int(tile == 15)
                fp = -int(tile == 4) + int(tile == 8)
                score = tp - .25 * fp
                writer.writerow(dict(image=name, sequence=f"video-seq{seq:02d}", tile_index=tile,
                                 tile_x0_hr=(tile % 4) * 160, tile_y0_hr=(tile // 4) * 128,
                                 tile_x1_hr=(tile % 4 + 1) * 160, tile_y1_hr=(tile // 4 + 1) * 128,
                                 delta_small_tp=tp, delta_fp=fp, utility_score=score,
                                 positive_label=int(score > 0)))
    sr = tmp_path / "large_sr.pt"
    sr.write_bytes(b"mock SR hash used only for provenance testing")
    manifest = label_dir / "run_manifest.json"
    manifest.write_text(json.dumps(dict(protocol=PROTOCOL,
        intervention="large_fsrcnn_x2_roi_h10_linear_background_feather8",
        halo=10, num_tiles=16, sr_checkpoint_sha256=sha256(sr), selected_images=names)),encoding="utf-8")
    from train_large_sr_predictor_step10c import split_large
    fresh_rows = load_measurements(csvfile)
    v2_groups = split_large(fresh_rows, 143, .15, .15)[3]
    v2_split_path = tmp_path / "v2_split.json"
    v2_split_path.write_text(json.dumps({"sequences": v2_groups}), encoding="utf-8")
    v2_checkpoint = tmp_path / "v2.pt"
    torch.save({"model": UtilityPredictor().state_dict(), "heads": list(HEADS),
                "input_size": [80, 64], "source_csv_sha256": sha256(csvfile)}, v2_checkpoint)
    cfg = yaml.safe_load((ROOT / "configs/large_sr_predictor_step10c.yaml").read_text())
    cfg.update(annotations_csv=str(csvfile), collection_manifest=str(manifest), sr_checkpoint=str(sr),
               train_lr=str(root), expected_images=12, expected_sequences=12,
               output=str(tmp_path / "v3"), canonical_checkpoint=str(tmp_path / "v3_model.pt"),
               v2_checkpoint=str(v2_checkpoint), v2_source_csv=str(csvfile),
               v2_split_manifest=str(v2_split_path),
               device="cpu", batch_images=2, workers=0)
    conf = tmp_path / "v3_config.yaml"
    conf.write_text(yaml.safe_dump(cfg))
    return conf


def test_synthetic_smoke_training_and_checkpoint_compatibility(synthetic_project):
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        args = Namespace(config=synthetic_project, smoke=True, out=None)
        out = train(args)
        result = json.loads((out / "summary.json").read_text())
        assert result["train_val_test_sequences"] == {"train": 8, "val": 2, "test": 2}
        assert not Path(yaml.safe_load(synthetic_project.read_text())["canonical_checkpoint"]).exists()
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        assert tuple(ck["heads"]) == tuple(HEADS)
        assert ck["input_size"] == [80, 64]
        assert ck["source_csv_sha256"] == sha256(Path(yaml.safe_load(synthetic_project.read_text())["annotations_csv"]))
        assert ck["halo"] == 10
        assert result["test"]["ranking_proxy"]["num_images"] >= 2
        assert (out / "test_selections.csv").is_file()
        with pytest.raises(FileExistsError):
            train(args)
    finally:
        torch.set_num_threads(prior_threads)


def test_v2_paired_split_audit(synthetic_project):
    cfg = yaml.safe_load(synthetic_project.read_text())
    rows = load_measurements(Path(cfg["annotations_csv"]))
    split = split_large(rows, 143, .15, .15)[3]
    meta = json.loads(Path(cfg["v2_split_manifest"]).read_text())
    result = verify_v2_paired_split(split, meta, Path(cfg["v2_checkpoint"]),
                                    Path(cfg["v2_source_csv"]), rows, rows)
    assert result["matched_test_sequences"] == 2


def test_v2_rejects_incompatible_split(synthetic_project):
    cfg = yaml.safe_load(synthetic_project.read_text())
    rows = load_measurements(Path(cfg["annotations_csv"]))
    split = split_large(rows, 143, .15, .15)[3]
    meta = json.loads(Path(cfg["v2_split_manifest"]).read_text())
    meta["sequences"]["test"] = ["other"]
    with pytest.raises(ValueError, match="comparison not paired"):
        verify_v2_paired_split(split, meta, Path(cfg["v2_checkpoint"]),
                               Path(cfg["v2_source_csv"]), rows, rows)


def test_v2_rejects_wrong_source_hash(synthetic_project):
    cfg = yaml.safe_load(synthetic_project.read_text())
    rows = load_measurements(Path(cfg["annotations_csv"]))
    split = split_large(rows, 143, .15, .15)[3]
    meta = json.loads(Path(cfg["v2_split_manifest"]).read_text())
    Path(cfg["v2_source_csv"]).write_text("changed", encoding="utf8")
    with pytest.raises(ValueError):
        verify_v2_paired_split(split, meta, Path(cfg["v2_checkpoint"]),
                               Path(cfg["v2_source_csv"]), rows, rows)


def test_full_synthetic_training_includes_v2_matched_comparison(synthetic_project):
    cfg = yaml.safe_load(synthetic_project.read_text())
    cfg["epochs"] = 1
    synthetic_project.write_text(yaml.safe_dump(cfg))
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        out = train(Namespace(config=synthetic_project, smoke=False, out=None))
        summary = json.loads((out / "summary.json").read_text())
        assert summary["matched_v2_comparison"]["v2_provenance"]["matched_test_sequences"] == 2
        assert summary["matched_v2_comparison"]["exploratory_paired_v3_minus_v2"]["num_sequences"] == 2
        ck = Path(cfg["canonical_checkpoint"])
        assert ck.exists()
        assert torch.load(ck, weights_only=True, map_location="cpu")["utility_protocol"] == PROTOCOL
    finally:
        torch.set_num_threads(prior)
