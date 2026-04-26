#!/usr/bin/env python3
"""
Trajectory Evaluation (ATE/RPE)
================================
Pseudo RGB-D SLAM Pipeline | BotLabs Dynamic Assignment

Compares estimated camera trajectory against TUM ground truth.
Implements the standard TUM evaluation metrics:

1. ATE (Absolute Trajectory Error):
   - Align estimated trajectory to ground truth using Umeyama alignment (SE(3))
   - Report RMSE of aligned position errors
   - Shows overall drift/accuracy of the SLAM system

2. RPE (Relative Pose Error):
   - Compute per-frame relative motion error
   - Decomposes into translational (m) and rotational (deg) components
   - Shows local consistency of pose estimation

Reference: Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems" (IROS 2012)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


def load_tum_trajectory(filepath: str) -> Dict[float, np.ndarray]:
    """Load trajectory in TUM format: timestamp tx ty tz qx qy qz qw.
    
    Returns:
        Dict mapping timestamp → 4×4 SE(3) transformation matrix
    """
    trajectory = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
            
            # Build 4×4 SE(3) matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            
            trajectory[ts] = T
    
    return trajectory


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3×3 rotation matrix."""
    # Normalize
    n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ])
    return R


def associate_trajectories(
    gt: Dict[float, np.ndarray],
    est: Dict[float, np.ndarray],
    max_diff: float = 0.02,
) -> List[Tuple[float, float]]:
    """Associate ground truth and estimated trajectory by closest timestamps.
    
    Args:
        gt: Ground truth trajectory {timestamp: SE3}
        est: Estimated trajectory {timestamp: SE3}
        max_diff: Maximum allowed time difference (seconds)
    
    Returns:
        List of (gt_timestamp, est_timestamp) pairs
    """
    gt_times = sorted(gt.keys())
    est_times = sorted(est.keys())
    
    matches = []
    est_idx = 0
    
    for gt_t in gt_times:
        while est_idx < len(est_times) - 1 and \
              abs(est_times[est_idx + 1] - gt_t) < abs(est_times[est_idx] - gt_t):
            est_idx += 1
        
        if est_idx < len(est_times) and abs(est_times[est_idx] - gt_t) < max_diff:
            matches.append((gt_t, est_times[est_idx]))
    
    return matches


def umeyama_alignment(
    model: np.ndarray,
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Umeyama alignment: find rotation R, translation t, scale s such that
    
    model ≈ s * R @ data + t
    
    This is the standard alignment used for ATE computation.
    Minimizes: Σ ||model_i - (s * R @ data_i + t)||²
    
    Umeyama, "Least-squares estimation of transformation parameters
    between two point patterns", IEEE TPAMI 1991.
    
    Args:
        model: (3, N) reference points (ground truth)
        data: (3, N) points to align (estimated)
    
    Returns:
        R (3×3), t (3,), s (float)
    """
    assert model.shape == data.shape
    n = model.shape[1]
    
    # Centroids
    mu_model = model.mean(axis=1, keepdims=True)
    mu_data = data.mean(axis=1, keepdims=True)
    
    # Center the data
    model_c = model - mu_model
    data_c = data - mu_data
    
    # Covariance
    sigma_sq = np.sum(data_c ** 2) / n
    cov = (model_c @ data_c.T) / n
    
    # SVD
    U, D, Vt = np.linalg.svd(cov)
    
    # Handle reflection
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    
    # Rotation
    R = U @ S @ Vt
    
    # Scale
    s = np.trace(np.diag(D) @ S) / sigma_sq
    
    # Translation
    t = mu_model.flatten() - s * R @ mu_data.flatten()
    
    return R, t, s


def compute_ate(
    gt: Dict[float, np.ndarray],
    est: Dict[float, np.ndarray],
    matches: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Compute Absolute Trajectory Error (ATE).
    
    1. Extract translation components from matched poses
    2. Align using Umeyama (SE(3) + scale)
    3. Compute RMSE of position errors after alignment
    
    Returns:
        Dictionary with ATE statistics
    """
    if matches is None:
        matches = associate_trajectories(gt, est)
    
    if len(matches) < 3:
        return {'ate_rmse': float('nan'), 'n_matched': len(matches)}
    
    # Extract positions (3×N)
    gt_positions = np.array([gt[gt_t][:3, 3] for gt_t, _ in matches]).T
    est_positions = np.array([est[est_t][:3, 3] for _, est_t in matches]).T
    
    # Umeyama alignment
    R, t, s = umeyama_alignment(gt_positions, est_positions)
    
    # Apply alignment
    est_aligned = s * R @ est_positions + t.reshape(3, 1)
    
    # Compute errors
    errors = np.linalg.norm(gt_positions - est_aligned, axis=0)
    
    return {
        'ate_rmse': float(np.sqrt(np.mean(errors ** 2))),
        'ate_mean': float(np.mean(errors)),
        'ate_median': float(np.median(errors)),
        'ate_std': float(np.std(errors)),
        'ate_min': float(np.min(errors)),
        'ate_max': float(np.max(errors)),
        'scale': float(s),
        'n_matched': len(matches),
    }


def compute_rpe(
    gt: Dict[float, np.ndarray],
    est: Dict[float, np.ndarray],
    matches: Optional[List[Tuple[float, float]]] = None,
    delta: int = 1,
) -> Dict[str, float]:
    """Compute Relative Pose Error (RPE).
    
    For each pair of frames (i, i+delta):
      RPE_i = (GT_i⁻¹ · GT_{i+delta})⁻¹ · (Est_i⁻¹ · Est_{i+delta})
    
    Decomposes into translational and rotational components.
    
    Returns:
        Dictionary with RPE statistics
    """
    if matches is None:
        matches = associate_trajectories(gt, est)
    
    if len(matches) < delta + 1:
        return {'rpe_trans': float('nan'), 'rpe_rot': float('nan')}
    
    trans_errors = []
    rot_errors = []
    
    for i in range(len(matches) - delta):
        gt_t1, est_t1 = matches[i]
        gt_t2, est_t2 = matches[i + delta]
        
        # Relative motion in GT
        gt_rel = np.linalg.inv(gt[gt_t1]) @ gt[gt_t2]
        
        # Relative motion in estimated
        est_rel = np.linalg.inv(est[est_t1]) @ est[est_t2]
        
        # Error = GT_rel⁻¹ · Est_rel
        error = np.linalg.inv(gt_rel) @ est_rel
        
        # Translational error (meters)
        trans_error = np.linalg.norm(error[:3, 3])
        trans_errors.append(trans_error)
        
        # Rotational error (degrees)
        # θ = arccos((tr(R) - 1) / 2)
        R_error = error[:3, :3]
        trace = np.clip((np.trace(R_error) - 1) / 2, -1, 1)
        rot_error = np.degrees(np.arccos(trace))
        rot_errors.append(rot_error)
    
    return {
        'rpe_trans_rmse': float(np.sqrt(np.mean(np.array(trans_errors) ** 2))),
        'rpe_trans_mean': float(np.mean(trans_errors)),
        'rpe_rot_rmse': float(np.sqrt(np.mean(np.array(rot_errors) ** 2))),
        'rpe_rot_mean': float(np.mean(rot_errors)),
        'n_pairs': len(trans_errors),
    }


def evaluate_trajectory(
    gt_file: str,
    est_file: str,
    output_dir: str = './trajectory_eval',
):
    """Full trajectory evaluation pipeline."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trajectories
    print(f'Loading ground truth: {gt_file}')
    gt = load_tum_trajectory(gt_file)
    print(f'  → {len(gt)} poses')
    
    print(f'Loading estimated: {est_file}')
    est = load_tum_trajectory(est_file)
    print(f'  → {len(est)} poses')
    
    # Associate
    matches = associate_trajectories(gt, est)
    print(f'Matched: {len(matches)} pairs')
    
    # ATE
    ate = compute_ate(gt, est, matches)
    
    # RPE
    rpe = compute_rpe(gt, est, matches)
    
    # Print results
    print('\n' + '='*60)
    print('TRAJECTORY EVALUATION RESULTS')
    print('='*60)
    print(f'\nAbsolute Trajectory Error (ATE):')
    print(f'  RMSE:   {ate["ate_rmse"]:.4f} m')
    print(f'  Mean:   {ate["ate_mean"]:.4f} m')
    print(f'  Median: {ate["ate_median"]:.4f} m')
    print(f'  Std:    {ate["ate_std"]:.4f} m')
    print(f'  Scale:  {ate["scale"]:.4f}')
    print(f'  Matched: {ate["n_matched"]} poses')
    
    print(f'\nRelative Pose Error (RPE, δ=1):')
    print(f'  Trans RMSE: {rpe["rpe_trans_rmse"]:.4f} m/frame')
    print(f'  Rot RMSE:   {rpe["rpe_rot_rmse"]:.4f} deg/frame')
    print(f'  Pairs: {rpe["n_pairs"]}')
    
    print(f'\nReference (ORB-SLAM2 published, fr1/desk):')
    print(f'  Real RGB-D ATE RMSE: ~0.016 m')
    print(f'  Our degradation: {ate["ate_rmse"]/0.016:.1f}× vs real sensor')
    print('='*60)
    
    # Save results
    results_path = out_dir / 'trajectory_results.txt'
    with open(results_path, 'w') as f:
        f.write('Trajectory Evaluation Results\n')
        f.write(f'Ground truth: {gt_file}\n')
        f.write(f'Estimated: {est_file}\n\n')
        f.write(f'ATE RMSE: {ate["ate_rmse"]:.4f} m\n')
        f.write(f'ATE Mean: {ate["ate_mean"]:.4f} m\n')
        f.write(f'RPE Trans RMSE: {rpe["rpe_trans_rmse"]:.4f} m/frame\n')
        f.write(f'RPE Rot RMSE: {rpe["rpe_rot_rmse"]:.4f} deg/frame\n')
        f.write(f'Scale: {ate["scale"]:.4f}\n')
        f.write(f'Matched poses: {ate["n_matched"]}\n')
    
    print(f'\nResults saved to: {results_path}')
    
    return ate, rpe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trajectory Evaluation (ATE/RPE)')
    parser.add_argument('--gt', required=True, help='Ground truth trajectory (TUM format)')
    parser.add_argument('--est', required=True, help='Estimated trajectory (TUM format)')
    parser.add_argument('--output', default='./trajectory_eval', help='Output directory')
    args = parser.parse_args()
    
    evaluate_trajectory(args.gt, args.est, args.output)
