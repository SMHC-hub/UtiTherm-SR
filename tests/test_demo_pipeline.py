"""Regression tests for examples/demo_pipeline.py.

Tests verify:
- The demo script can be imported without errors.
- load_fsrcnn() raises SystemExit when checkpoint is absent.
- load_fsrcnn() successfully loads the trained checkpoint when present.
- load_yolo() raises SystemExit when checkpoint is absent (ultralytics available).
- load_yolo() raises ImportError-triggered SystemExit when ultralytics missing.
- The full synthetic pipeline (SR only, no YOLO) runs without errors.
- run_yolo_inference() correctly counts small detections (area < 32^2 pixels).
- Checkpoint architecture kwargs are read from file (not hard-coded).

These tests run entirely on CPU and do not require the FLIR dataset.
GPU tests and YOLO tests requiring ultralytics are skipped on hosts without it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure project root and src/ are importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

# Lazy import of the demo module (avoids importing ultralytics at collection time)
import importlib


def _import_demo():
    spec = importlib.util.spec_from_file_location(
        "demo_pipeline", _ROOT / "examples" / "demo_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# FSRCNN checkpoint loading
# ---------------------------------------------------------------------------

class TestLoadFSRCNN:
    """Unit tests for load_fsrcnn()."""

    def test_missing_checkpoint_exits_with_error(self, tmp_path):
        demo = _import_demo()
        device = torch.device("cpu")
        with pytest.raises(SystemExit) as exc_info:
            demo.load_fsrcnn(tmp_path / "nonexistent.pt", device)
        assert exc_info.value.code != 0, "Exit code should be non-zero on missing checkpoint"

    def test_loads_trained_checkpoint_when_present(self):
        """Verify the trained FSRCNN checkpoint loads with correct architecture."""
        ckpt_path = _ROOT / "models" / "sr" / "fsrcnn_large_x2_stable_step9d3_v1.pt"
        if not ckpt_path.is_file():
            pytest.skip("Trained FSRCNN checkpoint not present in this environment")

        demo = _import_demo()
        device = torch.device("cpu")
        model = demo.load_fsrcnn(ckpt_path, device)

        # Architecture must come from checkpoint metadata, not hard-coded values
        assert model is not None
        # Verify d=168 from first conv weight
        first_conv = model.features[0]
        assert first_conv.weight.shape[0] == 168, (
            f"Expected d=168 from checkpoint metadata, got {first_conv.weight.shape[0]}"
        )
        # Verify model runs forward pass
        x = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 64, 64)
        assert torch.isfinite(out).all()

    def test_architecture_from_checkpoint_metadata(self, tmp_path):
        """Architecture must be read from model_kwargs, not assumed as d=64,s=16,m=6."""
        from utitherm.sr.fsrcnn import FSRCNN
        # Create a dummy checkpoint with a small architecture
        model = FSRCNN(scale=2, d=32, s=8, m=2)
        ckpt_path = tmp_path / "dummy_fsrcnn.pt"
        torch.save({
            "model_kwargs": {"scale": 2, "d": 32, "s": 8, "m": 2},
            "state_dict": model.state_dict(),
            "epoch": 1,
            "val_psnr_db": 30.0,
            "val_mse": 0.001,
        }, ckpt_path)

        demo = _import_demo()
        device = torch.device("cpu")
        loaded = demo.load_fsrcnn(ckpt_path, device)

        first_conv = loaded.features[0]
        assert first_conv.weight.shape[0] == 32, (
            "load_fsrcnn must use model_kwargs from checkpoint, not hard-coded d=168"
        )


# ---------------------------------------------------------------------------
# YOLO checkpoint loading
# ---------------------------------------------------------------------------

class TestLoadYOLO:
    """Unit tests for load_yolo()."""

    def test_missing_yolo_checkpoint_exits(self, tmp_path):
        """SystemExit when YOLO checkpoint file is absent."""
        demo = _import_demo()
        with pytest.raises(SystemExit) as exc_info:
            demo.load_yolo(tmp_path / "no_yolo.pt")
        assert exc_info.value.code != 0

    def test_missing_ultralytics_exits_gracefully(self, tmp_path):
        """SystemExit with helpful message when ultralytics is not installed."""
        # Create a dummy .pt file so it passes the is_file() check
        dummy = tmp_path / "yolo.pt"
        dummy.write_bytes(b"dummy")

        demo = _import_demo()
        with patch.dict(sys.modules, {"ultralytics": None}):
            with pytest.raises(SystemExit) as exc_info:
                demo.load_yolo(dummy)
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# run_yolo_inference() helper
# ---------------------------------------------------------------------------

class TestRunYoloInference:
    """Unit tests for the run_yolo_inference() counting helper."""

    def _make_mock_yolo_result(self, boxes_xywh: list[tuple[float, float, float, float]],
                                classes: list[int]):
        """Build a mock ultralytics Results object."""
        mock_result = MagicMock()
        n = len(boxes_xywh)
        if n == 0:
            mock_result.boxes.xywh = torch.zeros((0, 4))
            mock_result.boxes.cls = torch.zeros((0,))
            mock_result.boxes.conf = torch.zeros((0,))
        else:
            mock_result.boxes.xywh = torch.tensor(boxes_xywh, dtype=torch.float32)
            mock_result.boxes.cls = torch.tensor(classes, dtype=torch.float32)
            mock_result.boxes.conf = torch.full((n,), 0.5)
        mock_result.boxes.__len__ = lambda self: n

        mock_yolo = MagicMock()
        mock_yolo.predict.return_value = [mock_result]
        mock_yolo.names = {0: "person", 1: "car"}
        return mock_yolo

    def test_empty_detections(self):
        demo = _import_demo()
        mock_yolo = self._make_mock_yolo_result([], [])
        image = np.zeros((512, 640, 3), dtype=np.uint8)
        result = demo.run_yolo_inference(mock_yolo, image, conf=0.25)
        assert result["total_detections"] == 0
        assert result["small_detections"] == 0

    def test_small_object_classification(self):
        """Objects with area < 32^2 = 1024 px must be counted as small."""
        demo = _import_demo()
        # area = w * h; small: 20*40=800 < 1024; large: 50*50=2500 > 1024
        boxes = [(100, 100, 20, 40), (300, 200, 50, 50)]
        mock_yolo = self._make_mock_yolo_result(boxes, [0, 1])
        image = np.zeros((512, 640, 3), dtype=np.uint8)
        result = demo.run_yolo_inference(mock_yolo, image, conf=0.25)
        assert result["total_detections"] == 2
        assert result["small_detections"] == 1

    def test_all_small_objects(self):
        demo = _import_demo()
        # 31*31 = 961 < 1024
        boxes = [(50, 50, 31, 31), (200, 200, 31, 31)]
        mock_yolo = self._make_mock_yolo_result(boxes, [0, 0])
        image = np.zeros((512, 640, 3), dtype=np.uint8)
        result = demo.run_yolo_inference(mock_yolo, image, conf=0.25)
        assert result["small_detections"] == 2


# ---------------------------------------------------------------------------
# Synthetic pipeline smoke test (CPU, no YOLO)
# ---------------------------------------------------------------------------

class TestSyntheticPipeline:
    """End-to-end SR pipeline on synthetic data, CPU only."""

    def test_create_synthetic_scene(self):
        demo = _import_demo()
        scene = demo.create_synthetic_thermal_scene(256, 320)
        assert scene.shape == (256, 320)
        assert scene.dtype == np.uint8
        assert scene.max() > 0

    def test_generate_grid_tiles(self):
        demo = _import_demo()
        tiles = demo.generate_grid_tiles(640, 512, 4, 4)
        assert len(tiles) == 16
        # All tiles cover the full 640×512 without overlap
        coverage = np.zeros((512, 640), dtype=np.int32)
        for x0, y0, x1, y1 in tiles:
            coverage[y0:y1, x0:x1] += 1
        assert coverage.min() == 1
        assert coverage.max() == 1

    def test_rank_tiles_by_contrast_returns_all_indices(self):
        demo = _import_demo()
        lr = demo.create_synthetic_thermal_scene(256, 320)
        tiles = demo.generate_grid_tiles(640, 512, 4, 4)
        ranked = demo.rank_tiles_by_contrast(lr, tiles)
        assert sorted(ranked) == list(range(16))

    def test_fsrcnn_forward_on_synthetic_lr(self):
        """SR model runs cleanly on a synthetic LR patch, CPU only."""
        ckpt_path = _ROOT / "models" / "sr" / "fsrcnn_large_x2_stable_step9d3_v1.pt"
        if not ckpt_path.is_file():
            pytest.skip("FSRCNN checkpoint not available")

        demo = _import_demo()
        device = torch.device("cpu")
        model = demo.load_fsrcnn(ckpt_path, device)

        lr = demo.create_synthetic_thermal_scene(256, 320)
        patch_lr = torch.from_numpy(lr[64:96, 80:112].astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = model(patch_lr)
        assert out.shape == (1, 1, 64, 64)
        assert torch.isfinite(out).all()
