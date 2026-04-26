import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_module(module_name: str, relative_path: str):
    root = Path(__file__).resolve().parents[1]
    module_path = root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


depth_quality = _load_module("depth_quality", "evaluation/depth_quality.py")


class DepthQualityTests(unittest.TestCase):
    def test_compute_depth_metrics_perfect_prediction(self):
        gt = np.full((20, 20), 5000, dtype=np.uint16)   # 1.0m
        pred = np.full((20, 20), 5000, dtype=np.uint16)  # 1.0m

        metrics = depth_quality.compute_depth_metrics(pred, gt)
        self.assertEqual(metrics["n_valid"], 400)
        self.assertAlmostEqual(metrics["abs_rel"], 0.0, places=8)
        self.assertAlmostEqual(metrics["rmse"], 0.0, places=8)
        self.assertAlmostEqual(metrics["mae"], 0.0, places=8)
        self.assertAlmostEqual(metrics["delta_1"], 100.0, places=8)
        self.assertAlmostEqual(metrics["delta_2"], 100.0, places=8)
        self.assertAlmostEqual(metrics["delta_3"], 100.0, places=8)

    def test_compute_depth_metrics_returns_nan_when_too_few_valid_pixels(self):
        gt = np.zeros((10, 10), dtype=np.uint16)
        pred = np.zeros((10, 10), dtype=np.uint16)
        gt[:9, :9] = 5000
        pred[:9, :9] = 5000

        metrics = depth_quality.compute_depth_metrics(pred, gt)
        self.assertEqual(metrics["n_valid"], 81)
        self.assertTrue(math.isnan(metrics["abs_rel"]))
        self.assertTrue(math.isnan(metrics["rmse"]))
        self.assertTrue(math.isnan(metrics["delta_1"]))
        self.assertTrue(math.isnan(metrics["delta_2"]))
        self.assertTrue(math.isnan(metrics["delta_3"]))

    def test_compute_depth_metrics_known_values(self):
        gt = np.full((20, 20), 5000, dtype=np.uint16)    # 1.0m
        pred = np.full((20, 20), 7500, dtype=np.uint16)  # 1.5m

        metrics = depth_quality.compute_depth_metrics(pred, gt)
        self.assertAlmostEqual(metrics["abs_rel"], 0.5, places=8)
        self.assertAlmostEqual(metrics["rmse"], 0.5, places=8)
        self.assertAlmostEqual(metrics["mae"], 0.5, places=8)
        self.assertAlmostEqual(metrics["delta_1"], 0.0, places=8)
        self.assertAlmostEqual(metrics["delta_2"], 100.0, places=8)
        self.assertAlmostEqual(metrics["delta_3"], 100.0, places=8)

    def test_create_error_heatmap_marks_invalid_pixels_black(self):
        pred = np.array([[0, 5000], [10000, 10000]], dtype=np.uint16)
        gt = np.array([[5000, 5000], [0, 10000]], dtype=np.uint16)

        heatmap = depth_quality.create_error_heatmap(pred, gt)
        self.assertEqual(heatmap.shape, (2, 2, 3))
        self.assertTrue(np.array_equal(heatmap[0, 0], np.array([0, 0, 0], dtype=np.uint8)))
        self.assertTrue(np.array_equal(heatmap[1, 0], np.array([0, 0, 0], dtype=np.uint8)))

    def test_create_comparison_image_output_shape(self):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        gt_depth = np.full((4, 5), 5000, dtype=np.uint16)
        pred_depth = np.full((4, 5), 5000, dtype=np.uint16)

        comp = depth_quality.create_comparison_image(rgb, gt_depth, pred_depth)
        self.assertEqual(comp.shape, (8, 10, 3))

    def test_evaluate_dataset_writes_summary_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "dataset"
            pred_dir = tmp_path / "pred"
            out_dir = tmp_path / "out"
            rgb_dir = dataset / "rgb"
            depth_dir = dataset / "depth"
            rgb_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            pred_dir.mkdir(parents=True)

            ts = "1305031102.175304"
            (dataset / "associations.txt").write_text(
                f"{ts} rgb/{ts}.png {ts} depth/{ts}.png\n"
            )

            rgb = np.zeros((20, 20, 3), dtype=np.uint8)
            gt_depth = np.full((20, 20), 5000, dtype=np.uint16)
            pred_depth = np.full((20, 20), 5000, dtype=np.uint16)

            depth_quality.cv2.imwrite(str(rgb_dir / f"{ts}.png"), rgb)
            depth_quality.cv2.imwrite(str(depth_dir / f"{ts}.png"), gt_depth)
            depth_quality.cv2.imwrite(str(pred_dir / f"{ts}.png"), pred_depth)

            depth_quality.evaluate_dataset(
                dataset_path=str(dataset),
                pred_depth_dir=str(pred_dir),
                output_dir=str(out_dir),
                sample_every=1,
            )

            self.assertTrue((out_dir / "depth_metrics.csv").exists())
            self.assertTrue((out_dir / "depth_summary.txt").exists())
            self.assertTrue((out_dir / "comparison_0000.jpg").exists())


if __name__ == "__main__":
    unittest.main()
