import importlib.util
import tempfile
import unittest
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_small_objects.py"
spec = importlib.util.spec_from_file_location("eval_small", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestDetectionEvaluation(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(module.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(module.iou((0, 0, 10, 10), (11, 11, 20, 20)), 0.0)

    def test_size_bins(self):
        self.assertEqual(module.category((0, 0, 31, 31)), "small")
        self.assertEqual(module.category((0, 0, 32, 32)), "medium")
        self.assertEqual(module.category((0, 0, 96, 96)), "large")

    def test_one_to_one_and_threshold(self):
        gt = [{"cls": 0, "box": (0, 0, 20, 20)}, {"cls": 0, "box": (0, 0, 20, 20)}]
        detections = [{"cls": 0, "box": (0, 0, 20, 20), "conf": 0.2}]
        self.assertEqual(len(module.match_gt(gt, detections, 0.1)), 1)
        self.assertEqual(len(module.match_gt(gt, detections, 0.25)), 0)

    def test_wrong_class_does_not_match(self):
        gt = [{"cls": 0, "box": (0, 0, 20, 20)}]
        detections = [{"cls": 1, "box": (0, 0, 20, 20), "conf": 0.9}]
        self.assertEqual(len(module.match_gt(gt, detections, 0.1)), 0)

    def test_label_read(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sample.txt"
            p.write_text("0 0.5 0.5 0.2 0.2\n")
            labels = module.read_labels(p, 320, 256)
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0]["size"], "medium")


if __name__ == "__main__":
    unittest.main()
