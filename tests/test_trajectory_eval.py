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


trajectory_eval = _load_module("trajectory_eval", "evaluation/trajectory_eval.py")


def _pose(tx=0.0, ty=0.0, tz=0.0, yaw_deg=0.0):
    yaw = math.radians(yaw_deg)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.array(
        [[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]], dtype=np.float64
    )
    t[:3, 3] = [tx, ty, tz]
    return t


class TrajectoryEvalTests(unittest.TestCase):
    def test_quaternion_to_rotation_matrix_identity(self):
        r = trajectory_eval.quaternion_to_rotation_matrix(0.0, 0.0, 0.0, 1.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-8)

    def test_quaternion_to_rotation_matrix_normalizes_input(self):
        r = trajectory_eval.quaternion_to_rotation_matrix(0.0, 0.0, 0.0, 2.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-8)

    def test_load_tum_trajectory_skips_comments_and_short_lines(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("# comment\n")
            f.write("bad line\n")
            f.write("1.0 0 0 0 0 0 0 1\n")
            f.write("2.0 1 2 3 0 0 0 1\n")
            temp_path = f.name

        traj = trajectory_eval.load_tum_trajectory(temp_path)
        self.assertEqual(set(traj.keys()), {1.0, 2.0})
        np.testing.assert_allclose(traj[1.0][:3, 3], [0, 0, 0], atol=1e-8)
        np.testing.assert_allclose(traj[2.0][:3, 3], [1, 2, 3], atol=1e-8)

    def test_associate_trajectories_uses_nearest_and_threshold(self):
        gt = {1.00: np.eye(4), 2.00: np.eye(4)}
        est = {0.99: np.eye(4), 2.03: np.eye(4)}
        matches = trajectory_eval.associate_trajectories(gt, est, max_diff=0.02)
        self.assertEqual(matches, [(1.0, 0.99)])

    def test_umeyama_alignment_recovers_similarity_transform(self):
        data = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
        yaw = math.radians(30.0)
        r_true = np.array(
            [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
        )
        s_true = 1.7
        t_true = np.array([2.0, -0.5, 0.25])
        model = s_true * r_true @ data + t_true.reshape(3, 1)

        r, t, s = trajectory_eval.umeyama_alignment(model, data)
        np.testing.assert_allclose(r, r_true, atol=1e-6)
        np.testing.assert_allclose(t, t_true, atol=1e-6)
        self.assertAlmostEqual(s, s_true, places=6)

    def test_compute_ate_returns_nan_with_insufficient_matches(self):
        gt = {1.0: _pose(0, 0, 0), 2.0: _pose(1, 0, 0)}
        est = {1.0: _pose(0, 0, 0), 2.0: _pose(1, 0, 0)}
        ate = trajectory_eval.compute_ate(gt, est)
        self.assertTrue(math.isnan(ate["ate_rmse"]))
        self.assertEqual(ate["n_matched"], 2)

    def test_compute_ate_aligns_scaled_rotated_translated_path(self):
        gt = {1.0: _pose(0, 0, 0), 2.0: _pose(1, 0, 0), 3.0: _pose(0, 2, 0)}
        s = 2.5
        r = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        t = np.array([5.0, -2.0, 1.0])

        est = {}
        for ts, pose in gt.items():
            p = pose[:3, 3]
            p_est = s * (r @ p) + t
            est[ts] = _pose(*p_est)

        ate = trajectory_eval.compute_ate(gt, est)
        self.assertLess(ate["ate_rmse"], 1e-6)
        self.assertEqual(ate["n_matched"], 3)

    def test_compute_rpe_returns_nan_when_not_enough_pairs(self):
        gt = {1.0: _pose(0, 0, 0)}
        est = {1.0: _pose(0, 0, 0)}
        rpe = trajectory_eval.compute_rpe(gt, est, delta=1)
        self.assertTrue(math.isnan(rpe["rpe_trans"]))
        self.assertTrue(math.isnan(rpe["rpe_rot"]))

    def test_compute_rpe_translation_error(self):
        gt = {1.0: _pose(0, 0, 0), 2.0: _pose(1, 0, 0), 3.0: _pose(2, 0, 0)}
        est = {1.0: _pose(0, 0, 0), 2.0: _pose(1.1, 0, 0), 3.0: _pose(2.2, 0, 0)}
        rpe = trajectory_eval.compute_rpe(gt, est, delta=1)
        self.assertAlmostEqual(rpe["rpe_trans_rmse"], 0.1, places=6)
        self.assertAlmostEqual(rpe["rpe_rot_rmse"], 0.0, places=6)
        self.assertEqual(rpe["n_pairs"], 2)

    def test_compute_rpe_rotation_error(self):
        gt = {1.0: _pose(), 2.0: _pose(), 3.0: _pose()}
        est = {1.0: _pose(yaw_deg=0), 2.0: _pose(yaw_deg=10), 3.0: _pose(yaw_deg=20)}
        rpe = trajectory_eval.compute_rpe(gt, est, delta=1)
        self.assertAlmostEqual(rpe["rpe_rot_rmse"], 10.0, places=4)
        self.assertAlmostEqual(rpe["rpe_trans_rmse"], 0.0, places=6)

    def test_evaluate_trajectory_writes_results_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "gt.txt"
            est_path = Path(tmp) / "est.txt"
            out_dir = Path(tmp) / "out"

            gt_path.write_text(
                "\n".join(
                    [
                        "1.0 0 0 0 0 0 0 1",
                        "2.0 1 0 0 0 0 0 1",
                        "3.0 0 2 0 0 0 0 1",
                    ]
                )
                + "\n"
            )
            est_path.write_text(
                "\n".join(
                    [
                        "1.0 5 -2 1 0 0 0 1",
                        "2.0 5.0 0.5 1 0 0 0 1",
                        "3.0 0.0 -2.0 1 0 0 0 1",
                    ]
                )
                + "\n"
            )

            ate, rpe = trajectory_eval.evaluate_trajectory(str(gt_path), str(est_path), str(out_dir))
            self.assertIn("ate_rmse", ate)
            self.assertIn("rpe_trans_rmse", rpe)
            self.assertTrue((out_dir / "trajectory_results.txt").exists())


if __name__ == "__main__":
    unittest.main()
